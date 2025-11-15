"""
Wallet Profiler - Creates comprehensive profiles of wallets
Enriches wallet data with historical context and behavior patterns
"""
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WalletProfiler:
    """Creates detailed profiles of trading wallets"""

    def create_profile(
        self,
        wallet_address: str,
        wallet_trades: pd.DataFrame,
        all_markets: List[Dict[str, Any]],
        market_outcomes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Create a comprehensive wallet profile

        Args:
            wallet_address: The wallet address
            wallet_trades: All trades for this wallet
            all_markets: List of all markets
            market_outcomes: Known market outcomes

        Returns:
            Comprehensive wallet profile
        """
        profile = {
            'wallet_address': wallet_address,
            'basic_stats': self._calculate_basic_stats(wallet_trades),
            'trading_behavior': self._analyze_trading_behavior(wallet_trades),
            'market_preferences': self._analyze_market_preferences(wallet_trades, all_markets),
            'timing_patterns': self._analyze_timing_patterns(wallet_trades),
            'performance': self._calculate_performance(wallet_trades, market_outcomes),
            'risk_profile': self._analyze_risk_profile(wallet_trades),
            'activity_timeline': self._create_activity_timeline(wallet_trades)
        }

        return profile

    def _calculate_basic_stats(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic trading statistics"""
        if trades.empty:
            return {
                'total_trades': 0,
                'first_trade': None,
                'last_trade': None,
                'account_age_days': 0,
                'unique_markets': 0,
                'total_volume': 0.0,
                'avg_trade_size': 0.0
            }

        first_trade = trades['timestamp'].min()
        last_trade = trades['timestamp'].max()
        account_age = (last_trade - first_trade).days

        return {
            'total_trades': len(trades),
            'first_trade': first_trade.isoformat(),
            'last_trade': last_trade.isoformat(),
            'account_age_days': account_age,
            'unique_markets': trades['market_id'].nunique(),
            'total_volume': trades['size'].sum() if 'size' in trades.columns else 0.0,
            'avg_trade_size': trades['size'].mean() if 'size' in trades.columns else 0.0,
            'median_trade_size': trades['size'].median() if 'size' in trades.columns else 0.0
        }

    def _analyze_trading_behavior(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading behavior patterns"""
        if trades.empty:
            return {}

        behavior = {
            'trades_per_day': len(trades) / max(1, (trades['timestamp'].max() - trades['timestamp'].min()).days),
            'most_active_hour': int(trades['timestamp'].dt.hour.mode()[0]) if not trades.empty else None,
            'weekend_ratio': len(trades[trades['timestamp'].dt.dayofweek >= 5]) / len(trades) if len(trades) > 0 else 0
        }

        # Outcome preference
        if 'outcome' in trades.columns:
            outcome_counts = trades['outcome'].value_counts()
            behavior['outcome_preference'] = outcome_counts.to_dict()
            behavior['outcome_bias'] = outcome_counts.max() / len(trades) if len(trades) > 0 else 0

        # Side preference (BUY vs SELL)
        if 'side' in trades.columns:
            side_counts = trades['side'].value_counts()
            behavior['side_preference'] = side_counts.to_dict()

        return behavior

    def _analyze_market_preferences(
        self,
        trades: pd.DataFrame,
        all_markets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze which types of markets the wallet prefers"""
        if trades.empty:
            return {}

        # Create market lookup (use 'market_id' which is our standard key)
        market_dict = {m['market_id']: m for m in all_markets}

        # Get categories of traded markets
        categories = []
        for market_id in trades['market_id'].unique():
            if market_id in market_dict:
                category = market_dict[market_id].get('category', 'Unknown')
                categories.append(category)

        if not categories:
            return {}

        category_series = pd.Series(categories)
        category_counts = category_series.value_counts()

        return {
            'top_categories': category_counts.head(5).to_dict(),
            'category_diversity': category_series.nunique(),
            'focuses_on_single_category': category_counts.max() / len(categories) > 0.7 if len(categories) > 0 else False
        }

    def _analyze_timing_patterns(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when the wallet trades"""
        if trades.empty:
            return {}

        # Calculate time between trades
        trades_sorted = trades.sort_values('timestamp')
        time_diffs = trades_sorted['timestamp'].diff()

        patterns = {
            'avg_time_between_trades_hours': time_diffs.mean().total_seconds() / 3600 if not time_diffs.empty else 0,
            'median_time_between_trades_hours': time_diffs.median().total_seconds() / 3600 if not time_diffs.empty else 0,
            'trades_in_bursts': (time_diffs < timedelta(minutes=5)).sum() if not time_diffs.empty else 0
        }

        # Hour of day distribution
        hour_dist = trades['timestamp'].dt.hour.value_counts().sort_index()
        patterns['hour_distribution'] = hour_dist.to_dict()

        return patterns

    def _calculate_performance(
        self,
        trades: pd.DataFrame,
        market_outcomes: Dict[str, str]
    ) -> Dict[str, Any]:
        """Calculate trading performance metrics"""
        if trades.empty:
            return {
                'win_rate': 0.0,
                'total_wins': 0,
                'total_losses': 0
            }

        # Filter to resolved markets
        resolved_trades = trades[trades['market_id'].isin(market_outcomes.keys())].copy()

        if resolved_trades.empty:
            return {
                'win_rate': 0.0,
                'total_wins': 0,
                'total_losses': 0,
                'resolved_trades': 0
            }

        # Calculate wins
        resolved_trades['won'] = resolved_trades.apply(
            lambda row: market_outcomes.get(row['market_id']) == row.get('outcome', row.get('side', 'YES')),
            axis=1
        )

        wins = resolved_trades['won'].sum()
        total = len(resolved_trades)

        return {
            'win_rate': wins / total if total > 0 else 0.0,
            'total_wins': int(wins),
            'total_losses': int(total - wins),
            'resolved_trades': total,
            'unresolved_trades': len(trades) - total
        }

    def _analyze_risk_profile(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk-taking behavior"""
        if trades.empty or 'size' not in trades.columns:
            return {}

        sizes = trades['size']

        return {
            'max_single_trade': float(sizes.max()),
            'min_single_trade': float(sizes.min()),
            'std_dev_trade_size': float(sizes.std()),
            'coefficient_of_variation': float(sizes.std() / sizes.mean()) if sizes.mean() > 0 else 0,
            'large_trades_count': int((sizes > sizes.quantile(0.9)).sum()),
            'position_sizing_consistency': float(1 - (sizes.std() / sizes.mean())) if sizes.mean() > 0 else 0
        }

    def _create_activity_timeline(
        self,
        trades: pd.DataFrame,
        num_periods: int = 10
    ) -> List[Dict[str, Any]]:
        """Create timeline of trading activity"""
        if trades.empty:
            return []

        trades_sorted = trades.sort_values('timestamp')
        
        # Divide into time periods
        min_time = trades_sorted['timestamp'].min()
        max_time = trades_sorted['timestamp'].max()
        period_length = (max_time - min_time) / num_periods

        timeline = []
        for i in range(num_periods):
            period_start = min_time + (period_length * i)
            period_end = period_start + period_length

            period_trades = trades_sorted[
                (trades_sorted['timestamp'] >= period_start) &
                (trades_sorted['timestamp'] < period_end)
            ]

            timeline.append({
                'period': i + 1,
                'start': period_start.isoformat(),
                'end': period_end.isoformat(),
                'trade_count': len(period_trades),
                'total_volume': period_trades['size'].sum() if 'size' in period_trades.columns else 0
            })

        return timeline

    def compare_to_baseline(
        self,
        wallet_profile: Dict[str, Any],
        baseline_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare wallet profile to baseline/average trader

        Args:
            wallet_profile: The wallet's profile
            baseline_stats: Average statistics across all traders

        Returns:
            Comparison metrics
        """
        comparison = {}

        # Compare win rate
        if 'performance' in wallet_profile and 'win_rate' in baseline_stats:
            wallet_wr = wallet_profile['performance'].get('win_rate', 0)
            baseline_wr = baseline_stats.get('win_rate', 0.5)
            comparison['win_rate_vs_baseline'] = wallet_wr - baseline_wr
            comparison['win_rate_multiplier'] = wallet_wr / baseline_wr if baseline_wr > 0 else 0

        # Compare trade volume
        if 'basic_stats' in wallet_profile and 'avg_volume' in baseline_stats:
            wallet_vol = wallet_profile['basic_stats'].get('avg_trade_size', 0)
            baseline_vol = baseline_stats.get('avg_volume', 0)
            comparison['volume_vs_baseline'] = wallet_vol - baseline_vol
            comparison['volume_multiplier'] = wallet_vol / baseline_vol if baseline_vol > 0 else 0

        return comparison
