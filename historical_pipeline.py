"""
Main Pipeline - Orchestrates the complete insider trading detection pipeline
Data Ingestion → Prefiltering → Enrichment → LLM Analysis
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

from config import settings
from database.models import init_db, get_session, Market, Wallet, Trade, SuspiciousTransaction
from data_ingestion.gamma_client import GammaClient
from data_ingestion.historical_loader import HistoricalDataLoader
from prefiltering.timing_detector import TimingDetector
from prefiltering.volume_detector import VolumeDetector
from prefiltering.gains_detector import GainsDetector
# Network detector removed - not used
from enrichment.wallet_profiler import WalletProfiler
from enrichment.market_context import MarketContextEnricher
from llm_analysis.claude_analyzer import ClaudeAnalyzer

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsiderTradingDetectionPipeline:
    """Main pipeline for detecting insider trading"""

    def __init__(self):
        # Initialize database
        self.engine = init_db()
        self.session = get_session(self.engine)
        
        # Initialize clients and detectors
        self.gamma_client = GammaClient()
        self.historical_loader = HistoricalDataLoader()
        
        # Initialize detectors
        self.timing_detector = TimingDetector()
        self.volume_detector = VolumeDetector()
        self.gains_detector = GainsDetector()
        # Network detector not used
        
        # Initialize enrichment
        self.wallet_profiler = WalletProfiler()
        self.context_enricher = MarketContextEnricher()
        
        # Initialize LLM analyzer
        self.claude_analyzer = ClaudeAnalyzer()

    def run_full_pipeline(
        self,
        days_back: int = 30,
        prefilter_threshold: float = 0.5,
        analyze_with_llm: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete detection pipeline

        Args:
            days_back: Number of days of historical data to analyze
            prefilter_threshold: Minimum score to flag as suspicious
            analyze_with_llm: Whether to run LLM analysis on flagged transactions

        Returns:
            Pipeline results summary
        """
        logger.info("=" * 80)
        logger.info("STARTING INSIDER TRADING DETECTION PIPELINE")
        logger.info("=" * 80)
        
        results = {
            'start_time': datetime.utcnow(),
            'markets_analyzed': 0,
            'trades_analyzed': 0,
            'suspicious_trades': 0,
            'llm_analyzed': 0,
            'high_risk_cases': 0
        }

        try:
            # Step 1: Data Ingestion
            logger.info("\n[1/5] DATA INGESTION")
            markets, trades_df = self._ingest_data(days_back)
            results['markets_analyzed'] = len(markets)
            results['trades_analyzed'] = len(trades_df)
            
            if trades_df.empty:
                logger.warning("No trades to analyze. Exiting pipeline.")
                return results

            # Step 2: Prefiltering
            logger.info("\n[2/5] PREFILTERING - Running 4 detectors")
            suspicious_trades = self._prefilter_trades(
                trades_df,
                markets,
                prefilter_threshold
            )
            results['suspicious_trades'] = len(suspicious_trades)
            
            if not suspicious_trades:
                logger.info("No suspicious trades detected. Pipeline complete.")
                return results

            # Step 3: Enrichment
            logger.info("\n[3/5] ENRICHMENT - Adding context")
            enriched_transactions = self._enrich_transactions(
                suspicious_trades,
                trades_df,
                markets
            )

            # Step 4: LLM Analysis
            if analyze_with_llm:
                logger.info("\n[4/5] LLM ANALYSIS - OpenAI o1 reasoning")
                llm_results = self._analyze_with_llm(enriched_transactions)
                results['llm_analyzed'] = len(llm_results)
                results['high_risk_cases'] = sum(
                    1 for r in llm_results if r['suspicion_score'] >= 70
                )

            # Step 5: Save Results
            logger.info("\n[5/5] SAVING RESULTS")
            self._save_results(enriched_transactions, llm_results if analyze_with_llm else [])

            results['end_time'] = datetime.utcnow()
            results['duration_seconds'] = (results['end_time'] - results['start_time']).total_seconds()

            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Markets analyzed: {results['markets_analyzed']}")
            logger.info(f"Trades analyzed: {results['trades_analyzed']}")
            logger.info(f"Suspicious trades: {results['suspicious_trades']}")
            logger.info(f"LLM analyzed: {results['llm_analyzed']}")
            logger.info(f"High risk cases: {results['high_risk_cases']}")
            logger.info(f"Duration: {results['duration_seconds']:.1f}s")
            logger.info("=" * 80)

            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

    def _ingest_data(self, days_back: int) -> tuple[List[Dict], pd.DataFrame]:
        """Ingest market and trade data"""
        logger.info(f"Fetching closed markets from last {days_back} days...")

        # Get closed markets
        markets = self.historical_loader.load_closed_markets_with_outcomes(days_back)
        logger.info(f"Found {len(markets)} closed markets")

        # Save markets to database
        self._save_markets(markets)

        # Load trades for these markets
        all_trades = []
        for i, market in enumerate(markets):  # No limit - analyze all
            logger.info(f"Loading trades for market {i+1}/{len(markets)}: {market['question'][:50]}...")
            market_trades = self.historical_loader.load_market_trades(market['market_id'])
            if not market_trades.empty:
                all_trades.append(market_trades)

        trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
        logger.info(f"Loaded {len(trades_df)} total trades")

        # Save trades and wallets to database
        self._save_trades(trades_df)

        return markets, trades_df

    def _prefilter_trades(
        self,
        trades_df: pd.DataFrame,
        markets: List[Dict],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Run prefiltering detectors on all trades"""
        
        suspicious = []
        market_dict = {m['market_id']: m for m in markets}
        market_outcomes = {m['market_id']: m['outcome'] for m in markets if m.get('outcome')}
        
        # Build network graph (COMMENTED OUT FOR PERFORMANCE)
        # logger.info("Building trade network...")
        # trade_graph = self.network_detector.build_trade_network(trades_df)

        logger.info(f"Analyzing {len(trades_df)} trades...")

        for idx, trade in trades_df.iterrows():
            trade_dict = trade.to_dict()
            market_id = trade_dict['market_id']
            wallet = trade_dict['wallet_address']

            if market_id not in market_dict:
                continue

            market = market_dict[market_id]
            wallet_trades = trades_df[trades_df['wallet_address'] == wallet]

            # Run detectors (3 out of 4, network commented out)
            timing_result = self.timing_detector.analyze_trade(trade_dict, market)
            volume_result = self.volume_detector.analyze_trade_volume(
                trade_dict, wallet_trades, market
            )
            gains_result = self.gains_detector.calculate_win_rate(
                wallet_trades, market_outcomes
            )
            # network_result = self.network_detector.analyze_wallet_connections(
            #     wallet, trade_graph
            # )
            network_result = {'score': 0.0, 'flags': []}  # Disabled for performance

            # Calculate total score (network weight redistributed)
            # New weights: Timing=40%, Volume=30%, Gains=30%, Network=0%
            total_score = (
                timing_result['score'] * 0.4 +
                volume_result['score'] * 0.3 +
                gains_result['score'] * 0.3 +
                network_result['score'] * 0.0
            )
            
            if total_score >= threshold:
                suspicious_trade = {
                    **trade_dict,
                    'timing_score': timing_result['score'],
                    'volume_score': volume_result['score'],
                    'gains_score': gains_result['score'],
                    'network_score': network_result['score'],
                    'total_score': total_score,
                    'flags': (
                        timing_result.get('flags', []) +
                        volume_result.get('flags', []) +
                        gains_result.get('flags', []) +
                        network_result.get('flags', [])
                    )
                }
                suspicious.append(suspicious_trade)
        
        logger.info(f"Flagged {len(suspicious)} suspicious trades (threshold: {threshold})")
        return suspicious

    def _enrich_transactions(
        self,
        suspicious_trades: List[Dict],
        all_trades: pd.DataFrame,
        markets: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Enrich suspicious transactions with context"""
        
        enriched = []
        market_dict = {m['market_id']: m for m in markets}
        market_outcomes = {m['market_id']: m['outcome'] for m in markets if m.get('outcome')}
        
        for trade in suspicious_trades:
            wallet = trade['wallet_address']
            market_id = trade['market_id']
            
            # Get wallet profile
            wallet_trades = all_trades[all_trades['wallet_address'] == wallet]
            wallet_profile = self.wallet_profiler.create_profile(
                wallet, wallet_trades, markets, market_outcomes
            )
            
            # Get market context
            market_trades = all_trades[all_trades['market_id'] == market_id]
            enriched_context = self.context_enricher.enrich_transaction(
                trade,
                market_dict.get(market_id, {}),
                market_trades,
                wallet_profile
            )
            
            enriched.append(enriched_context)
        
        logger.info(f"Enriched {len(enriched)} transactions with context")
        return enriched

    def _analyze_with_llm(
        self,
        enriched_transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze with OpenAI o1 LLM"""
        
        logger.info(f"Analyzing {len(enriched_transactions)} transactions with OpenAI o1...")

        results = []
        for i, context in enumerate(enriched_transactions):
            logger.info(f"Analyzing {i+1}/{len(enriched_transactions)}...")

            formatted = self.context_enricher.format_for_llm(context)
            analysis = self.claude_analyzer.analyze_transaction(context, formatted)
            results.append(analysis)
        
        return results

    def _save_markets(self, markets: List[Dict]):
        """Save markets to database with full metadata"""
        logger.info(f"Saving {len(markets)} markets to database...")

        saved_count = 0
        updated_count = 0

        for market_data in markets:
            try:
                # Check if market already exists
                existing = self.session.query(Market).filter_by(
                    id=market_data['market_id']
                ).first()

                if existing:
                    # Update existing market with new data
                    if market_data.get('outcome'):
                        existing.outcome = market_data['outcome']
                        existing.resolved = True
                    if market_data.get('volume'):
                        existing.volume = market_data['volume']
                    updated_count += 1
                else:
                    # Create new market
                    market = Market(
                        id=market_data['market_id'],
                        question=market_data.get('question', 'Unknown'),
                        description=market_data.get('description', ''),
                        category=market_data.get('category'),
                        end_date=pd.to_datetime(market_data['end_date']) if market_data.get('end_date') else None,
                        resolved=market_data.get('closed', False) and market_data.get('outcome') is not None,
                        outcome=market_data.get('outcome'),
                        volume=float(market_data.get('volume', 0))
                    )
                    self.session.add(market)
                    saved_count += 1

            except Exception as e:
                logger.warning(f"Error saving market {market_data.get('market_id')}: {e}")
                continue

        self.session.commit()
        logger.info(f"Saved {saved_count} new markets, updated {updated_count} existing markets")

    def _save_trades(self, trades_df: pd.DataFrame):
        """Save trades and wallets to database"""
        if trades_df.empty:
            return

        logger.info(f"Saving {len(trades_df)} trades to database...")

        # First, save unique wallets
        unique_wallets = trades_df['wallet_address'].unique()
        saved_wallets = 0

        for wallet_addr in unique_wallets:
            try:
                existing = self.session.query(Wallet).filter_by(
                    address=wallet_addr
                ).first()

                if not existing:
                    wallet = Wallet(address=wallet_addr)
                    self.session.add(wallet)
                    saved_wallets += 1
            except Exception as e:
                logger.warning(f"Error saving wallet {wallet_addr}: {e}")
                continue

        self.session.commit()
        logger.info(f"Saved {saved_wallets} new wallets")

        # Now save trades
        saved_trades = 0
        for _, trade_row in trades_df.iterrows():
            try:
                tx_hash = trade_row.get('transaction_hash')

                # Check if trade already exists (by tx_hash)
                existing = None
                if tx_hash:
                    existing = self.session.query(Trade).filter_by(
                        tx_hash=tx_hash
                    ).first()

                if not existing:
                    trade = Trade(
                        tx_hash=tx_hash,
                        market_id=trade_row['market_id'],
                        wallet_address=trade_row['wallet_address'],
                        timestamp=trade_row['timestamp'],
                        size=float(trade_row.get('size', 0)),
                        price=float(trade_row.get('price', 0)),
                        outcome=trade_row.get('outcome'),
                        side=trade_row.get('side')
                    )
                    self.session.add(trade)
                    saved_trades += 1
            except Exception as e:
                logger.warning(f"Error saving trade: {e}")
                continue

        self.session.commit()
        logger.info(f"Saved {saved_trades} new trades to database")

        # Update wallet statistics
        self._update_wallet_statistics(trades_df)

    def _update_wallet_statistics(self, trades_df: pd.DataFrame):
        """Update wallet statistics based on trades"""
        if trades_df.empty:
            return

        logger.info("Updating wallet statistics...")

        unique_wallets = trades_df['wallet_address'].unique()
        updated_count = 0

        for wallet_addr in unique_wallets:
            try:
                wallet = self.session.query(Wallet).filter_by(address=wallet_addr).first()
                if not wallet:
                    continue

                # Get all trades for this wallet
                wallet_trades = trades_df[trades_df['wallet_address'] == wallet_addr]

                # Update statistics
                wallet.total_trades = len(wallet_trades)
                wallet.total_volume = float(wallet_trades['size'].sum() if 'size' in wallet_trades.columns else 0)
                wallet.first_seen = wallet_trades['timestamp'].min()
                wallet.last_seen = wallet_trades['timestamp'].max()
                wallet.unique_markets = wallet_trades['market_id'].nunique() if 'market_id' in wallet_trades.columns else 0

                updated_count += 1

            except Exception as e:
                logger.warning(f"Error updating wallet {wallet_addr}: {e}")
                continue

        self.session.commit()
        logger.info(f"Updated statistics for {updated_count} wallets")

    def _save_results(
        self,
        enriched_transactions: List[Dict],
        llm_results: List[Dict]
    ):
        """Save suspicious transactions with LLM analysis to database"""
        logger.info(f"Saving {len(enriched_transactions)} suspicious transactions to database...")

        saved_count = 0
        for i, enriched in enumerate(enriched_transactions):
            try:
                transaction = enriched['transaction']
                tx_hash = transaction.get('transaction_hash')
                market_id = transaction['market_id']
                wallet_address = transaction['wallet_address']

                # Find the corresponding trade_id from database
                trade_id = None
                if tx_hash:
                    trade = self.session.query(Trade).filter_by(tx_hash=tx_hash).first()
                    if trade:
                        trade_id = trade.id

                # Check if suspicious transaction already exists
                existing = None
                if trade_id:
                    existing = self.session.query(SuspiciousTransaction).filter_by(
                        trade_id=trade_id
                    ).first()

                if existing:
                    continue

                # Get LLM analysis if available
                llm_analysis = llm_results[i] if i < len(llm_results) else {}

                # Convert pandas Timestamps to ISO strings for JSON serialization
                import json
                def serialize_for_json(obj):
                    """Convert pandas Timestamps and other non-JSON types"""
                    if pd.isna(obj):
                        return None
                    elif hasattr(obj, 'isoformat'):  # datetime/Timestamp
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: serialize_for_json(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_for_json(v) for v in obj]
                    else:
                        return obj

                serialized_context = serialize_for_json(enriched)

                # Ensure flags is a plain Python list
                flags = enriched['anomalies']['flags']
                if not isinstance(flags, list):
                    flags = list(flags) if hasattr(flags, '__iter__') else [str(flags)]

                suspicious_tx = SuspiciousTransaction(
                    trade_id=trade_id,
                    market_id=market_id,
                    wallet_address=wallet_address,
                    timing_score=float(enriched['anomalies']['detection_scores']['timing_score']),
                    volume_score=float(enriched['anomalies']['detection_scores']['volume_score']),
                    gains_score=float(enriched['anomalies']['detection_scores']['gains_score']),
                    network_score=float(enriched['anomalies']['detection_scores']['network_score']),
                    total_score=float(enriched['anomalies']['detection_scores']['total_score']),
                    flags=flags,  # JSON field - ensure it's a plain list
                    llm_analyzed=bool(llm_analysis) if llm_analysis else False,
                    llm_suspicion_score=float(llm_analysis.get('suspicion_score', 0)) if llm_analysis else None,
                    llm_reasoning=llm_analysis.get('reasoning', '') if llm_analysis else None,
                    context=serialized_context  # Store serialized context as JSON
                )

                self.session.add(suspicious_tx)
                saved_count += 1

            except Exception as e:
                logger.error(f"Error saving suspicious transaction: {e}")
                continue

        self.session.commit()
        logger.info(f"Saved {saved_count} suspicious transactions to database")

    def close(self):
        """Clean up resources"""
        self.gamma_client.close()
        self.historical_loader.close()
        self.session.close()


def main():
    """Main entry point with CLI arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Historical Polymarket Insider Trading Analysis'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=30,
        help='Number of days of historical data to analyze (default: 30)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='Detection threshold 0.0-1.0 (default: 0.05)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM analysis (faster)'
    )

    args = parser.parse_args()

    pipeline = InsiderTradingDetectionPipeline()

    try:
        print(f"\n{'='*80}")
        print(f"HISTORICAL PIPELINE CONFIGURATION")
        print(f"{'='*80}")
        print(f"Days back: {args.days_back}")
        print(f"Threshold: {args.threshold}")
        print(f"LLM Analysis: {'No' if args.no_llm else 'Yes'}")
        print(f"{'='*80}\n")

        results = pipeline.run_full_pipeline(
            days_back=args.days_back,
            prefilter_threshold=args.threshold,
            analyze_with_llm=not args.no_llm
        )

        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)
        for key, value in results.items():
            print(f"{key}: {value}")

    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
