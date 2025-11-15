"""
Gains Detector - Identifies wallets with abnormally high win rates
Detects suspicious patterns of consistent wins or perfect timing
"""
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from config import settings

logger = logging.getLogger(__name__)


class GainsDetector:
    """Detects abnormal gain patterns and win rates"""

    def __init__(self, win_rate_threshold: float = None, min_trades: int = None):
        self.win_rate_threshold = win_rate_threshold or settings.win_rate_threshold
        self.min_trades = min_trades or settings.min_trades_for_analysis
        
    def calculate_win_rate(
        self,
        wallet_trades: pd.DataFrame,
        market_outcomes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Calculate win rate for a wallet

        Args:
            wallet_trades: DataFrame with wallet's trades
            market_outcomes: Dict mapping market_id to outcome (YES/NO)

        Returns:
            Win rate analysis
        """
        result = {
            'win_rate': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'unresolved_trades': 0,
            'suspicious': False,
            'score': 0.0,
            'flags': []
        }

        if wallet_trades.empty:
            return result

        # Filter to trades with known outcomes
        resolved_trades = wallet_trades[
            wallet_trades['market_id'].isin(market_outcomes.keys())
        ].copy()

        if resolved_trades.empty:
            result['unresolved_trades'] = len(wallet_trades)
            return result

        # Determine if each trade won
        resolved_trades['won'] = resolved_trades.apply(
            lambda row: self._did_trade_win(row, market_outcomes),
            axis=1
        )

        winning_trades = resolved_trades['won'].sum()
        total_trades = len(resolved_trades)
        
        result['total_trades'] = total_trades
        result['winning_trades'] = winning_trades
        result['losing_trades'] = total_trades - winning_trades
        result['unresolved_trades'] = len(wallet_trades) - total_trades

        if total_trades >= self.min_trades:
            win_rate = winning_trades / total_trades
            result['win_rate'] = win_rate

            # Statistical significance test
            p_value = self._binomial_test(winning_trades, total_trades, 0.5)
            result['p_value'] = p_value

            # Score based on win rate and statistical significance
            if win_rate >= self.win_rate_threshold:
                result['suspicious'] = True
                result['score'] = self._calculate_win_rate_score(win_rate, p_value)
                result['flags'].append('high_win_rate')
                
                if win_rate >= 0.85:
                    result['flags'].append('very_high_win_rate')
                if win_rate >= 0.95:
                    result['flags'].append('extremely_high_win_rate')
                if p_value < 0.01:
                    result['flags'].append('statistically_significant_win_rate')

        return result

    def analyze_roi(
        self,
        wallet_trades: pd.DataFrame,
        market_outcomes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze Return on Investment

        Args:
            wallet_trades: DataFrame with trades including size and price
            market_outcomes: Market outcomes

        Returns:
            ROI analysis
        """
        result = {
            'total_roi': 0.0,
            'avg_roi_per_trade': 0.0,
            'suspicious': False,
            'score': 0.0,
            'flags': [],
            'details': {}
        }

        if wallet_trades.empty or 'size' not in wallet_trades.columns:
            return result

        # Calculate PnL for each trade
        trades_with_outcomes = []
        total_invested = 0
        total_returned = 0

        for _, trade in wallet_trades.iterrows():
            market_id = trade['market_id']
            if market_id not in market_outcomes:
                continue

            size = trade['size']
            price = trade.get('price', 0.5)
            outcome = trade.get('outcome', 'YES')
            
            total_invested += size
            
            # Simplified PnL calculation
            if self._did_trade_win(trade, market_outcomes):
                # Win: get back investment plus profit
                pnl = size / price  # Simplified
                total_returned += pnl
            else:
                # Loss: lose investment
                total_returned += 0

        if total_invested > 0:
            roi = (total_returned - total_invested) / total_invested
            result['total_roi'] = roi
            result['avg_roi_per_trade'] = roi / len(wallet_trades) if len(wallet_trades) > 0 else 0
            result['details']['total_invested'] = total_invested
            result['details']['total_returned'] = total_returned

            # High ROI is suspicious
            if roi > 0.5:  # 50%+ ROI
                result['suspicious'] = True
                result['score'] = min(1.0, roi)
                result['flags'].append('high_roi')
                
            if roi > 1.0:  # 100%+ ROI
                result['flags'].append('very_high_roi')
                result['score'] = 0.9
                
            if roi > 2.0:  # 200%+ ROI
                result['flags'].append('extremely_high_roi')
                result['score'] = 1.0

        return result

    def detect_perfect_timing_streak(
        self,
        wallet_trades: pd.DataFrame,
        market_outcomes: Dict[str, str],
        streak_length: int = 5
    ) -> Dict[str, Any]:
        """
        Detect consecutive winning trades (perfect timing)

        Args:
            wallet_trades: Wallet trades sorted by time
            market_outcomes: Market outcomes
            streak_length: Minimum streak length to flag

        Returns:
            Streak analysis
        """
        result = {
            'max_win_streak': 0,
            'current_streak': 0,
            'suspicious': False,
            'score': 0.0,
            'flags': []
        }

        if wallet_trades.empty:
            return result

        # Sort by timestamp
        trades_sorted = wallet_trades.sort_values('timestamp')
        
        # Calculate wins
        trades_sorted['won'] = trades_sorted.apply(
            lambda row: self._did_trade_win(row, market_outcomes) if row['market_id'] in market_outcomes else None,
            axis=1
        )

        # Find streaks
        max_streak = 0
        current_streak = 0

        for won in trades_sorted['won']:
            if won is None:
                continue
            
            if won:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        result['max_win_streak'] = max_streak
        result['current_streak'] = current_streak

        if max_streak >= streak_length:
            result['suspicious'] = True
            result['score'] = min(1.0, max_streak / 10.0)
            result['flags'].append('winning_streak')
            
            if max_streak >= 8:
                result['flags'].append('long_winning_streak')
            if max_streak >= 10:
                result['flags'].append('extraordinary_winning_streak')

        return result

    def analyze_first_trade_success_rate(
        self,
        wallet_trades: pd.DataFrame,
        market_outcomes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze if wallet's first trades on markets have abnormally high success

        Args:
            wallet_trades: Wallet trades
            market_outcomes: Market outcomes

        Returns:
            First trade analysis
        """
        result = {
            'first_trade_win_rate': 0.0,
            'suspicious': False,
            'score': 0.0,
            'flags': []
        }

        if wallet_trades.empty:
            return result

        # Get first trade on each market
        first_trades = wallet_trades.sort_values('timestamp').groupby('market_id').first()
        
        # Calculate win rate for first trades
        wins = 0
        total = 0

        for market_id, trade in first_trades.iterrows():
            if market_id in market_outcomes:
                total += 1
                if self._did_trade_win(trade, market_outcomes):
                    wins += 1

        if total >= self.min_trades:
            win_rate = wins / total
            result['first_trade_win_rate'] = win_rate

            if win_rate >= 0.7:
                result['suspicious'] = True
                result['score'] = win_rate
                result['flags'].append('high_first_trade_success_rate')

        return result

    def _did_trade_win(
        self,
        trade: pd.Series,
        market_outcomes: Dict[str, str]
    ) -> bool:
        """Determine if a trade won based on market outcome"""
        market_id = trade['market_id']
        if market_id not in market_outcomes:
            return False
        
        outcome = market_outcomes[market_id]
        position = trade.get('outcome', trade.get('side', 'YES'))
        
        return outcome == position

    def _binomial_test(
        self,
        successes: int,
        trials: int,
        p: float = 0.5
    ) -> float:
        """
        Perform binomial test for statistical significance

        Args:
            successes: Number of wins
            trials: Total trades
            p: Expected probability (0.5 for random)

        Returns:
            P-value
        """
        return stats.binom_test(successes, trials, p, alternative='greater')

    def _calculate_win_rate_score(
        self,
        win_rate: float,
        p_value: float
    ) -> float:
        """
        Calculate suspicion score based on win rate and statistical significance

        Args:
            win_rate: The win rate
            p_value: Statistical p-value

        Returns:
            Score between 0 and 1
        """
        # Base score from win rate
        base_score = (win_rate - 0.5) / 0.5  # Normalize 0.5-1.0 to 0-1
        
        # Adjust by statistical significance
        if p_value < 0.05:
            base_score *= 1.2
        if p_value < 0.01:
            base_score *= 1.3
            
        return min(1.0, max(0.0, base_score))
