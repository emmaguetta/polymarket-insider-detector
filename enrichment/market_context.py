"""
Market Context - Enriches suspicious transactions with market context
Provides comprehensive market data and timeline for LLM analysis
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketContextEnricher:
    """Enriches transactions with relevant market context"""

    def enrich_transaction(
        self,
        suspicious_trade: Dict[str, Any],
        market_data: Dict[str, Any],
        all_market_trades: pd.DataFrame,
        wallet_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive context for a suspicious transaction

        Args:
            suspicious_trade: The flagged trade
            market_data: Market information
            all_market_trades: All trades on this market
            wallet_profile: Wallet profile from profiler

        Returns:
            Enriched context dictionary
        """
        context = {
            'transaction': self._format_transaction(suspicious_trade),
            'market': self._format_market_info(market_data),
            'market_activity': self._analyze_market_activity(all_market_trades, suspicious_trade),
            'wallet': wallet_profile,
            'timeline': self._create_timeline(suspicious_trade, all_market_trades, market_data),
            'anomalies': self._summarize_anomalies(suspicious_trade)
        }

        return context

    def _format_transaction(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Format transaction details"""
        return {
            'timestamp': trade.get('timestamp'),
            'wallet_address': trade.get('wallet_address'),
            'side': trade.get('side', 'BUY'),
            'outcome': trade.get('outcome', 'YES'),
            'size_usd': trade.get('size', 0),
            'price': trade.get('price', 0),
            'market_id': trade.get('market_id')
        }

    def _format_market_info(self, market: Dict[str, Any]) -> Dict[str, Any]:
        """Format market information"""
        return {
            'question': market.get('question', 'Unknown'),
            'description': market.get('description', ''),
            'category': market.get('category', 'Unknown'),
            'end_date': market.get('end_date'),
            'resolved': market.get('resolved', False),
            'outcome': market.get('outcome'),
            'total_volume': market.get('volume', 0),
            'created_date': market.get('created_date')
        }

    def _analyze_market_activity(
        self,
        all_trades: pd.DataFrame,
        suspicious_trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze overall market trading activity"""
        if all_trades.empty:
            return {}

        trade_time = pd.to_datetime(suspicious_trade['timestamp'])

        # Activity before suspicious trade
        trades_before = all_trades[all_trades['timestamp'] < trade_time]
        trades_after = all_trades[all_trades['timestamp'] >= trade_time]

        analysis = {
            'total_traders': all_trades['wallet_address'].nunique(),
            'total_trades': len(all_trades),
            'total_volume': all_trades['size'].sum() if 'size' in all_trades.columns else 0,
            'trades_before_suspicious': len(trades_before),
            'trades_after_suspicious': len(trades_after)
        }

        # Volume analysis
        if 'size' in all_trades.columns:
            analysis['avg_trade_size'] = all_trades['size'].mean()
            analysis['median_trade_size'] = all_trades['size'].median()
            analysis['suspicious_trade_percentile'] = (
                (all_trades['size'] < suspicious_trade.get('size', 0)).sum() / len(all_trades) * 100
            )

        # Price movement analysis
        if 'price' in all_trades.columns and not all_trades.empty:
            all_trades_sorted = all_trades.sort_values('timestamp')
            analysis['price_movement'] = {
                'initial_price': float(all_trades_sorted['price'].iloc[0]),
                'final_price': float(all_trades_sorted['price'].iloc[-1]),
                'price_at_suspicious_trade': suspicious_trade.get('price', 0),
                'max_price': float(all_trades['price'].max()),
                'min_price': float(all_trades['price'].min())
            }

        return analysis

    def _create_timeline(
        self,
        suspicious_trade: Dict[str, Any],
        all_trades: pd.DataFrame,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create timeline of events around suspicious trade"""
        trade_time = pd.to_datetime(suspicious_trade['timestamp'])
        
        timeline = {
            'suspicious_trade_timestamp': trade_time.isoformat()
        }

        # Market creation to suspicious trade
        if 'created_date' in market_data:
            created = pd.to_datetime(market_data['created_date'])

            # Handle timezone differences
            if trade_time.tz is None and created.tz is not None:
                trade_time_local = trade_time.tz_localize('UTC')
            elif trade_time.tz is not None and created.tz is None:
                created = created.tz_localize('UTC')
                trade_time_local = trade_time
            else:
                trade_time_local = trade_time

            timeline['market_created'] = created.isoformat()
            timeline['days_market_active_before_trade'] = (trade_time_local - created).days

        # Suspicious trade to resolution
        if 'end_date' in market_data and market_data['end_date']:
            end_date = pd.to_datetime(market_data['end_date'])

            # Handle timezone differences
            if trade_time.tz is None and end_date.tz is not None:
                trade_time_local = trade_time.tz_localize('UTC')
            elif trade_time.tz is not None and end_date.tz is None:
                end_date = end_date.tz_localize('UTC')
                trade_time_local = trade_time
            else:
                trade_time_local = trade_time

            timeline['market_resolution'] = end_date.isoformat()
            timeline['hours_before_resolution'] = (end_date - trade_time_local).total_seconds() / 3600

        # Trading activity timeline around suspicious trade
        window_hours = 24
        window_start = trade_time - timedelta(hours=window_hours)
        window_end = trade_time + timedelta(hours=window_hours)

        window_trades = all_trades[
            (all_trades['timestamp'] >= window_start) &
            (all_trades['timestamp'] <= window_end)
        ]

        if not window_trades.empty:
            timeline['activity_24h_window'] = {
                'trades_before_24h': len(window_trades[window_trades['timestamp'] < trade_time]),
                'trades_after_24h': len(window_trades[window_trades['timestamp'] >= trade_time]),
                'volume_before_24h': window_trades[window_trades['timestamp'] < trade_time]['size'].sum() if 'size' in window_trades.columns else 0,
                'volume_after_24h': window_trades[window_trades['timestamp'] >= trade_time]['size'].sum() if 'size' in window_trades.columns else 0
            }

        return timeline

    def _summarize_anomalies(self, suspicious_trade: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize detected anomalies from prefiltering"""
        anomalies = {
            'detection_scores': {
                'timing_score': suspicious_trade.get('timing_score', 0),
                'volume_score': suspicious_trade.get('volume_score', 0),
                'gains_score': suspicious_trade.get('gains_score', 0),
                'network_score': suspicious_trade.get('network_score', 0),
                'total_score': suspicious_trade.get('total_score', 0)
            },
            'flags': suspicious_trade.get('flags', [])
        }

        return anomalies

    def format_for_llm(self, enriched_context: Dict[str, Any]) -> str:
        """
        Format enriched context as a readable string for LLM

        Args:
            enriched_context: The enriched context dictionary

        Returns:
            Formatted string for LLM
        """
        transaction = enriched_context['transaction']
        market = enriched_context['market']
        activity = enriched_context['market_activity']
        wallet = enriched_context['wallet']
        timeline = enriched_context['timeline']
        anomalies = enriched_context['anomalies']

        formatted = f"""
## SUSPICIOUS TRANSACTION ANALYSIS

### Transaction Details
- Wallet: {transaction['wallet_address']}
- Timestamp: {transaction['timestamp']}
- Market: {market['question']}
- Category: {market['category']}
- Trade Size: ${transaction['size_usd']:,.2f}
- Position: {transaction['outcome']} at price {transaction['price']}

### Market Context
- Total Market Volume: ${activity.get('total_volume', 0):,.2f}
- Total Traders: {activity.get('total_traders', 0)}
- Average Trade Size: ${activity.get('avg_trade_size', 0):,.2f}
- Market Status: {'Resolved' if market['resolved'] else 'Active'}
{f"- Outcome: {market['outcome']}" if market.get('outcome') else ""}

### Timeline
{f"- Hours Before Resolution: {timeline.get('hours_before_resolution', 'N/A')}" if 'hours_before_resolution' in timeline else ""}
{f"- Days Since Market Created: {timeline.get('days_market_active_before_trade', 'N/A')}" if 'days_market_active_before_trade' in timeline else ""}

### Wallet Profile
- Total Trades: {wallet.get('basic_stats', {}).get('total_trades', 0)}
- Account Age: {wallet.get('basic_stats', {}).get('account_age_days', 0)} days
- Win Rate: {wallet.get('performance', {}).get('win_rate', 0):.1%}
- Average Trade Size: ${wallet.get('basic_stats', {}).get('avg_trade_size', 0):,.2f}
- Unique Markets: {wallet.get('basic_stats', {}).get('unique_markets', 0)}

### Detected Anomalies
- Timing Score: {anomalies['detection_scores']['timing_score']:.2f}
- Volume Score: {anomalies['detection_scores']['volume_score']:.2f}
- Gains Score: {anomalies['detection_scores']['gains_score']:.2f}
- Network Score: {anomalies['detection_scores']['network_score']:.2f}
- Total Suspicion Score: {anomalies['detection_scores']['total_score']:.2f}

Flags: {', '.join(anomalies['flags'])}
"""

        return formatted.strip()
