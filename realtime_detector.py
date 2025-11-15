"""
Real-time Insider Trading Detector for Polymarket
Polls Data API every 20 seconds, processes new trades incrementally
"""
import logging
import time
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any

from config import settings
from database.models import init_db, get_session, Market, Trade, SuspiciousTransaction, Wallet
from data_ingestion.historical_loader import HistoricalDataLoader
from prefiltering.timing_detector import TimingDetector
from prefiltering.volume_detector import VolumeDetector
from prefiltering.gains_detector import GainsDetector
from enrichment.wallet_profiler import WalletProfiler
from enrichment.market_context import MarketContextEnricher
from llm_analysis.claude_analyzer import ClaudeAnalyzer

# Configure minimal logging - only show important events
logging.basicConfig(
    level=logging.WARNING,  # Only warnings and errors by default
    format='%(message)s'  # Clean format without timestamps/module names
)

# Set our logger to INFO but silence noisy libraries
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Silence noisy libraries
logging.getLogger('data_ingestion.historical_loader').setLevel(logging.WARNING)
logging.getLogger('llm_analysis.openai_analyzer').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.ERROR)


class RealtimeInsiderDetector:
    """Real-time polling-based insider trading detector"""

    def __init__(self):
        # Initialize database
        self.engine = init_db()
        self.session = get_session(self.engine)

        # Initialize data loader
        self.loader = HistoricalDataLoader()

        # Initialize detectors
        self.timing_detector = TimingDetector()
        self.volume_detector = VolumeDetector()
        self.gains_detector = GainsDetector()

        # Initialize enrichment
        self.wallet_profiler = WalletProfiler()
        self.context_enricher = MarketContextEnricher()

        # Initialize LLM
        self.llm_analyzer = ClaudeAnalyzer()

        # In-memory market cache
        self.markets_cache = {}

    def bootstrap_database(self):
        """One-time: Load initial data from all active markets"""
        print("🔄 Bootstrapping database...")

        try:
            # Get all active markets with recent trades
            markets = self.loader.load_closed_markets_with_outcomes(days_back=30)

            # Store markets
            for market in markets:
                self._upsert_market(market)
                self.markets_cache[market['market_id']] = market

            # Fetch trades for each market (limit to avoid overwhelming)
            all_trades = []
            for market in markets[:30]:  # Limit to 30 markets for bootstrap
                trades_df = self.loader.load_market_trades(market['market_id'])
                if not trades_df.empty:
                    all_trades.append(trades_df)

            # Combine all trades
            if all_trades:
                bootstrap_df = pd.concat(all_trades, ignore_index=True)

                # Store trades in DB
                for _, trade in bootstrap_df.iterrows():
                    self._store_trade(trade.to_dict())

                # Set cursor to latest trade
                latest_timestamp = bootstrap_df['timestamp'].max()
                self._update_cursor(latest_timestamp)

                print(f"✅ Bootstrap complete: {len(markets)} markets, {len(bootstrap_df)} trades")
            else:
                print("⚠️  No trades found during bootstrap")
                self._update_cursor(datetime.now(timezone.utc))

            return len(markets)

        except Exception as e:
            print(f"❌ Bootstrap failed: {e}")
            raise

    def _get_cursor(self) -> datetime:
        """Get the last processed trade timestamp"""
        try:
            # Query the latest trade timestamp from DB
            latest_trade = self.session.query(Trade).order_by(Trade.timestamp.desc()).first()

            if latest_trade:
                return latest_trade.timestamp
            else:
                # No trades yet, start from now
                return datetime.now(timezone.utc)

        except Exception as e:
            return datetime.now(timezone.utc)

    def _update_cursor(self, timestamp: datetime):
        """Update cursor - just for logging, actual cursor is max(Trade.timestamp)"""
        logger.debug(f"Cursor updated to {timestamp}")

    def _fetch_new_trades(self, cursor: datetime) -> List[Dict]:
        """Fetch trades newer than cursor from Data API"""
        try:
            import requests

            # Fetch recent trades
            response = requests.get(
                'https://data-api.polymarket.com/trades',
                params={'limit': 200},  # Fetch more to ensure we don't miss any
                timeout=30
            )

            if response.status_code != 200:
                return []

            all_trades = response.json()

            # Filter for trades newer than cursor
            new_trades = []
            for trade in all_trades:
                trade_time = pd.to_datetime(trade.get('timestamp'), unit='s')  # Unix timestamp in seconds
                if trade_time > cursor:
                    new_trades.append(trade)

            return new_trades

        except Exception as e:
            return []

    def _store_trade(self, trade_dict: Dict):
        """Store a trade in the database"""
        try:
            tx_hash = trade_dict.get('trade_id') or trade_dict.get('transaction_hash')

            # Check if already exists by tx_hash
            existing = self.session.query(Trade).filter_by(
                tx_hash=tx_hash
            ).first()

            if existing:
                return  # Skip duplicates

            # Ensure wallet exists (create if needed)
            wallet_address = trade_dict.get('wallet_address')
            wallet = self.session.query(Wallet).filter_by(address=wallet_address).first()
            if not wallet:
                wallet = Wallet(
                    address=wallet_address,
                    first_seen=trade_dict.get('timestamp'),
                    last_seen=trade_dict.get('timestamp')
                )
                self.session.add(wallet)
                self.session.flush()  # Flush to make wallet available for FK

            # Create new trade (id auto-increments)
            trade = Trade(
                market_id=trade_dict.get('market_id'),
                wallet_address=wallet_address,
                timestamp=trade_dict.get('timestamp'),
                size=trade_dict.get('size', 0),
                price=trade_dict.get('price', 0),
                outcome=trade_dict.get('outcome'),
                side=trade_dict.get('side'),
                tx_hash=tx_hash
            )

            self.session.add(trade)
            self.session.commit()

        except Exception as e:
            self.session.rollback()

    def _upsert_market(self, market_dict: Dict):
        """Insert or update market in database"""
        try:
            existing = self.session.query(Market).filter_by(
                id=market_dict['market_id']
            ).first()

            if existing:
                # Update
                existing.question = market_dict.get('question')
                existing.outcome = market_dict.get('outcome')
                existing.volume = market_dict.get('volume', 0)
            else:
                # Insert
                market = Market(
                    id=market_dict['market_id'],
                    question=market_dict.get('question'),
                    end_date=market_dict.get('end_date'),
                    outcome=market_dict.get('outcome'),
                    volume=market_dict.get('volume', 0),
                    category=market_dict.get('category')
                )
                self.session.add(market)

            self.session.commit()

        except Exception as e:
            self.session.rollback()

    def _get_wallet_history(self, wallet_address: str, limit: int = 100) -> pd.DataFrame:
        """Get recent trades for a wallet from DB"""
        try:
            trades = self.session.query(Trade).filter_by(
                wallet_address=wallet_address
            ).order_by(Trade.timestamp.desc()).limit(limit).all()

            if not trades:
                return pd.DataFrame()

            data = [{
                'market_id': t.market_id,
                'wallet_address': t.wallet_address,
                'timestamp': t.timestamp,
                'size': t.size,
                'price': t.price,
                'outcome': t.outcome,
                'side': t.side
            } for t in trades]

            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error getting wallet history: {e}")
            return pd.DataFrame()

    def _run_detectors(self, trades_df: pd.DataFrame, threshold: float = 0.1) -> List[Dict]:
        """Run detection on new trades"""
        suspicious = []

        # Get market outcomes (none for active markets, but needed for structure)
        market_outcomes = {}

        for idx, trade in trades_df.iterrows():
            trade_dict = trade.to_dict()
            market_id = trade_dict['market_id']
            wallet = trade_dict['wallet_address']

            # Get market info
            if market_id not in self.markets_cache:
                continue

            market = self.markets_cache[market_id]

            # Get wallet history from DB
            wallet_trades = self._get_wallet_history(wallet)

            # Run 3 detectors (network disabled)
            timing_result = self.timing_detector.analyze_trade(trade_dict, market)
            volume_result = self.volume_detector.analyze_trade_volume(
                trade_dict, wallet_trades, market
            )
            gains_result = self.gains_detector.calculate_win_rate(
                wallet_trades, market_outcomes
            )

            # Calculate score (network = 0)
            total_score = (
                timing_result['score'] * 0.4 +
                volume_result['score'] * 0.3 +
                gains_result['score'] * 0.3
            )

            if total_score >= threshold:
                suspicious_trade = {
                    **trade_dict,
                    'timing_score': timing_result['score'],
                    'volume_score': volume_result['score'],
                    'gains_score': gains_result['score'],
                    'network_score': 0.0,
                    'total_score': total_score,
                    'flags': (
                        timing_result.get('flags', []) +
                        volume_result.get('flags', []) +
                        gains_result.get('flags', [])
                    )
                }
                suspicious.append(suspicious_trade)

        return suspicious

    def _enrich_and_analyze(self, trade: Dict) -> Dict:
        """Enrich trade with context and run LLM analysis"""
        try:
            wallet = trade['wallet_address']
            market_id = trade['market_id']

            # Get context from DB
            wallet_trades = self._get_wallet_history(wallet)
            market = self.markets_cache.get(market_id, {})

            # Build market outcomes dict (empty for active/real-time markets)
            market_outcomes = {}
            for m in self.markets_cache.values():
                if m.get('outcome'):
                    market_outcomes[m['market_id']] = m['outcome']

            # Create wallet profile
            wallet_profile = self.wallet_profiler.create_profile(
                wallet,
                wallet_trades,
                list(self.markets_cache.values()),
                market_outcomes
            )

            # Build enriched context
            enriched = {
                'transaction': trade,
                'wallet_profile': wallet_profile,
                'market': market
            }

            # Format for LLM (COMMENTED - causes 'market_activity' error)
            # formatted_context = self.context_enricher.format_for_llm(enriched)
            formatted_context = f"""
            Trade: {trade.get('size')} @ {trade.get('price')} on {trade.get('outcome')}
            Market: {market.get('question', 'Unknown')}
            Wallet: {wallet[:10]}...
            Detection Score: {trade.get('total_score', 0):.2f}
            Flags: {', '.join(trade.get('flags', []))}
            """

            # Analyze with LLM
            analysis = self.llm_analyzer.analyze_transaction(enriched, formatted_context)

            return analysis

        except Exception as e:
            return {
                'suspicion_score': 0,
                'confidence': 'low',
                'reasoning': f"Analysis failed: {str(e)}",
                'recommendation': 'manual_review',
                'error': str(e)
            }

    def _save_suspicious(self, trade: Dict, analysis: Dict):
        """Save suspicious trade to database"""
        try:
            # Find the Trade record by tx_hash to get the trade_id (Trade.id)
            tx_hash = trade.get('trade_id') or trade.get('transaction_hash')
            trade_record = self.session.query(Trade).filter_by(tx_hash=tx_hash).first()

            if not trade_record:
                logger.warning(f"Trade record not found for tx_hash {tx_hash}, skipping suspicious save")
                return

            suspicious = SuspiciousTransaction(
                trade_id=trade_record.id,
                market_id=trade['market_id'],
                wallet_address=trade['wallet_address'],
                timing_score=trade.get('timing_score', 0),
                volume_score=trade.get('volume_score', 0),
                gains_score=trade.get('gains_score', 0),
                network_score=trade.get('network_score', 0),
                total_score=trade.get('total_score', 0),
                flags=trade.get('flags', []),
                llm_analyzed=True,
                llm_suspicion_score=analysis.get('suspicion_score', 0),
                llm_reasoning=analysis.get('reasoning', ''),
                llm_analysis_date=datetime.now()
            )

            self.session.add(suspicious)
            self.session.commit()

        except Exception as e:
            self.session.rollback()

    def process_iteration(self):
        """Single iteration: fetch new trades and process them"""
        try:
            # Get cursor
            cursor = self._get_cursor()

            # Fetch new trades
            new_trades_raw = self._fetch_new_trades(cursor)

            if not new_trades_raw:
                return 0, 0

            # Parse trades
            new_trades = []
            for trade in new_trades_raw:
                parsed = {
                    'market_id': trade.get('conditionId'),
                    'wallet_address': trade.get('taker') or trade.get('proxyWallet'),
                    'timestamp': pd.to_datetime(trade.get('timestamp'), unit='s'),  # Unix timestamp in seconds
                    'size': float(trade.get('size', 0)),
                    'price': float(trade.get('price', 0)),
                    'outcome': trade.get('outcome'),
                    'side': trade.get('side'),
                    'trade_id': trade.get('transactionHash') or trade.get('id'),
                    'question': trade.get('title')
                }
                new_trades.append(parsed)

            # Store trades + update market cache
            for trade in new_trades:
                # IMPORTANT: Create market BEFORE storing trade (to avoid FK violation)
                market_id = trade['market_id']
                if market_id not in self.markets_cache:
                    market = {
                        'market_id': market_id,
                        'question': trade.get('question'),
                        'end_date': None,
                        'outcome': None,
                        'volume': 0,
                        'category': None
                    }
                    self._upsert_market(market)
                    self.markets_cache[market_id] = market

                # Now store the trade (market exists in DB)
                self._store_trade(trade)

            # Run detection on new trades
            trades_df = pd.DataFrame(new_trades)
            suspicious = self._run_detectors(trades_df, threshold=0.05)

            # Process each suspicious trade
            for trade in suspicious:
                print(
                    f"\n🚨 SUSPICIOUS TRADE DETECTED\n"
                    f"   Wallet: {trade['wallet_address']}\n"
                    f"   Market: {trade.get('question', 'Unknown')}\n"
                    f"   Size: {trade.get('size')} @ ${trade.get('price')}\n"
                    f"   Detection Score: {trade['total_score']:.2f}/1.0\n"
                    f"   Flags: {', '.join(trade.get('flags', []))}"
                )

                # Enrich and analyze with LLM
                analysis = self._enrich_and_analyze(trade)

                # Only show LLM result if non-zero or high confidence
                if analysis['suspicion_score'] > 0:
                    print(
                        f"   🤖 LLM Analysis: {analysis['suspicion_score']}/100\n"
                        f"   Reasoning: {analysis['reasoning'][:200]}...\n"
                        f"   Recommendation: {analysis['recommendation'].upper()}"
                    )

                # Save to DB
                self._save_suspicious(trade, analysis)

            # Update cursor
            if new_trades:
                latest = max(t['timestamp'] for t in new_trades)
                self._update_cursor(latest)

            return len(new_trades), len(suspicious)

        except Exception as e:
            print(f"⚠️  Iteration error: {str(e)[:100]}")
            return 0, 0

    def run_forever(self):
        """Main loop: run forever, poll every 20 seconds"""
        print("=" * 60)
        print("🔍 Polymarket Insider Trading Detector - LIVE")
        print("=" * 60)

        # Check if bootstrap needed
        trade_count = self.session.query(Trade).count()
        if trade_count == 0:
            self.bootstrap_database()
        else:
            print(f"✓ Loaded {trade_count} existing trades from database\n")

        print("👀 Monitoring trades (Ctrl+C to stop)...\n")

        iteration = 0
        total_suspicious = 0

        while True:
            iteration += 1
            start_time = time.time()

            try:
                # Process new trades
                trades_processed, suspicious_found = self.process_iteration()

                total_suspicious += suspicious_found

                # Only show stats if something interesting happened
                if suspicious_found > 0:
                    elapsed = time.time() - start_time
                    print(f"\n[Iteration #{iteration} complete - {elapsed:.1f}s]")

                # Sleep until next iteration (20 seconds)
                elapsed = time.time() - start_time
                sleep_time = max(0, 20 - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print(f"\n\n{'='*60}")
                print(f"✓ Detector stopped - {total_suspicious} suspicious trades found")
                print(f"{'='*60}")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                time.sleep(20)

        self.session.close()


def main():
    """Entry point"""
    detector = RealtimeInsiderDetector()
    detector.run_forever()


if __name__ == "__main__":
    main()
