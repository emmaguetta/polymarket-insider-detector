"""
Timing Detector - Identifies trades with suspicious timing
Detects transactions placed shortly before market resolution or major events
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class TimingDetector:
    """Detects suspicious timing patterns in trades"""

    def __init__(self, threshold_hours: int = None):
        self.threshold_hours = threshold_hours or settings.timing_threshold_hours
        
    def analyze_trade(
        self,
        trade: Dict[str, Any],
        market: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a single trade for timing suspiciousness

        Args:
            trade: Trade data with timestamp
            market: Market data with resolution time

        Returns:
            Analysis result with score and flags
        """
        result = {
            'suspicious': False,
            'score': 0.0,
            'flags': [],
            'details': {}
        }

        trade_time = pd.to_datetime(trade['timestamp'])

        # Check if market has end_date/resolution time
        if 'end_date' in market and market['end_date']:
            end_time = pd.to_datetime(market['end_date'])

            # Make timezone-aware timestamps comparable
            if trade_time.tz is None and end_time.tz is not None:
                trade_time = trade_time.tz_localize('UTC')
            elif trade_time.tz is not None and end_time.tz is None:
                end_time = end_time.tz_localize('UTC')

            hours_before_end = (end_time - trade_time).total_seconds() / 3600
            
            result['details']['hours_before_resolution'] = hours_before_end
            
            # Score based on proximity to resolution
            if 0 < hours_before_end <= self.threshold_hours:
                # Exponential scoring - closer = more suspicious
                result['score'] = self._calculate_timing_score(hours_before_end)
                result['suspicious'] = True
                result['flags'].append('trade_before_resolution')
                
                if hours_before_end <= 6:
                    result['flags'].append('very_close_to_resolution')
                if hours_before_end <= 1:
                    result['flags'].append('extremely_close_to_resolution')

        return result

    def analyze_wallet_timing_pattern(
        self,
        wallet_trades: pd.DataFrame,
        markets: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze overall timing patterns for a wallet

        Args:
            wallet_trades: DataFrame with all trades for a wallet
            markets: List of market data

        Returns:
            Pattern analysis with score
        """
        if wallet_trades.empty:
            return {'score': 0.0, 'flags': [], 'details': {}}

        # Create market lookup
        market_dict = {m['condition_id']: m for m in markets}
        
        suspicious_count = 0
        total_analyzed = 0
        timing_scores = []

        for _, trade in wallet_trades.iterrows():
            market_id = trade.get('market_id')
            if market_id not in market_dict:
                continue
                
            market = market_dict[market_id]
            trade_dict = trade.to_dict()
            
            analysis = self.analyze_trade(trade_dict, market)
            total_analyzed += 1
            
            if analysis['suspicious']:
                suspicious_count += 1
                timing_scores.append(analysis['score'])

        result = {
            'score': 0.0,
            'flags': [],
            'details': {
                'total_trades': total_analyzed,
                'suspicious_timing_trades': suspicious_count,
                'suspicious_ratio': suspicious_count / total_analyzed if total_analyzed > 0 else 0
            }
        }

        if suspicious_count > 0:
            # Score based on both ratio and average timing score
            ratio_score = suspicious_count / total_analyzed
            avg_timing_score = np.mean(timing_scores)
            result['score'] = (ratio_score + avg_timing_score) / 2
            
            if ratio_score > 0.5:
                result['flags'].append('consistent_suspicious_timing')
            if ratio_score > 0.7:
                result['flags'].append('highly_consistent_suspicious_timing')
            if suspicious_count >= 3 and ratio_score > 0.3:
                result['flags'].append('multiple_suspicious_timing_trades')

        return result

    def detect_first_trade_pattern(
        self,
        trade: Dict[str, Any],
        wallet: Dict[str, Any],
        market: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Special detection for first trade by a wallet on a specific market

        Args:
            trade: Trade data
            wallet: Wallet data with history
            market: Market data

        Returns:
            Detection result
        """
        result = {
            'is_first_trade': False,
            'is_new_wallet': False,
            'suspicious': False,
            'score': 0.0,
            'flags': []
        }

        # Check if this is the wallet's first trade overall
        if wallet.get('total_trades', 0) <= 1:
            result['is_new_wallet'] = True
            result['flags'].append('brand_new_wallet')

        # Check timing of first trade
        trade_time = pd.to_datetime(trade['timestamp'])
        wallet_first_seen = pd.to_datetime(wallet.get('first_seen')) if 'first_seen' in wallet else None
        
        if wallet_first_seen:
            account_age_hours = (trade_time - wallet_first_seen).total_seconds() / 3600
            
            if account_age_hours < 24:
                result['flags'].append('very_new_account')
                if account_age_hours < 1:
                    result['flags'].append('account_created_immediately_before_trade')

        # If new wallet + suspicious timing = very suspicious
        if result['is_new_wallet']:
            timing_analysis = self.analyze_trade(trade, market)
            if timing_analysis['suspicious']:
                result['suspicious'] = True
                result['score'] = min(1.0, timing_analysis['score'] * 1.5)
                result['flags'].append('new_wallet_suspicious_timing')

        return result

    def _calculate_timing_score(self, hours_before: float) -> float:
        """
        Calculate suspicion score based on hours before resolution
        Exponential function: closer to resolution = higher score
        
        Args:
            hours_before: Hours before market resolution
            
        Returns:
            Score between 0 and 1
        """
        if hours_before <= 0:
            return 0.0
            
        # Exponential decay from threshold
        normalized = hours_before / self.threshold_hours
        score = 1.0 - normalized
        
        # Boost score for very close trades
        if hours_before <= 6:
            score = min(1.0, score * 1.3)
        if hours_before <= 1:
            score = min(1.0, score * 1.5)
            
        return max(0.0, min(1.0, score))
