"""
Volume Detector - Identifies trades with abnormal volumes
Detects unusual trade sizes, new accounts with large trades, etc.
"""
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from config import settings

logger = logging.getLogger(__name__)


class VolumeDetector:
    """Detects abnormal trading volumes and patterns"""

    def __init__(self, anomaly_threshold: float = None):
        self.anomaly_threshold = anomaly_threshold or settings.volume_anomaly_threshold
        self.isolation_forest = None
        
    def analyze_trade_volume(
        self,
        trade: Dict[str, Any],
        wallet_history: pd.DataFrame,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze if a trade's volume is abnormal

        Args:
            trade: Current trade data
            wallet_history: Historical trades for this wallet
            market_data: Market information

        Returns:
            Analysis result with score and flags
        """
        result = {
            'suspicious': False,
            'score': 0.0,
            'flags': [],
            'details': {}
        }

        trade_size = trade.get('size', 0)
        result['details']['trade_size'] = trade_size

        # Check against wallet's historical average
        if not wallet_history.empty and 'size' in wallet_history.columns:
            wallet_avg = wallet_history['size'].mean()
            wallet_std = wallet_history['size'].std()
            
            result['details']['wallet_avg_size'] = wallet_avg
            result['details']['wallet_std'] = wallet_std
            
            if wallet_std > 0:
                z_score = (trade_size - wallet_avg) / wallet_std
                result['details']['z_score'] = z_score
                
                if abs(z_score) > self.anomaly_threshold:
                    result['suspicious'] = True
                    result['score'] = min(1.0, abs(z_score) / 5.0)
                    result['flags'].append('abnormal_volume_for_wallet')
                    
                    if z_score > 3:
                        result['flags'].append('extremely_large_trade')
                    if z_score > 5:
                        result['flags'].append('unprecedented_trade_size')

        # Check for large absolute amounts
        if trade_size > 10000:  # $10K+
            result['flags'].append('large_absolute_amount')
            result['suspicious'] = True
            result['score'] = max(result['score'], 0.6)
            
        if trade_size > 50000:  # $50K+
            result['flags'].append('very_large_absolute_amount')
            result['score'] = max(result['score'], 0.8)

        # Check new wallet with large trade
        if wallet_history.empty or len(wallet_history) == 0:
            if trade_size > 1000:
                result['flags'].append('new_wallet_large_first_trade')
                result['suspicious'] = True
                result['score'] = max(result['score'], 0.7)

        return result

    def analyze_single_trade_wallet(
        self,
        wallet: Dict[str, Any],
        trade: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze if a wallet only trades once (very suspicious pattern)

        Args:
            wallet: Wallet data
            trade: The single trade

        Returns:
            Analysis result
        """
        result = {
            'is_single_trade_wallet': False,
            'suspicious': False,
            'score': 0.0,
            'flags': []
        }

        total_trades = wallet.get('total_trades', 0)
        
        if total_trades == 1:
            result['is_single_trade_wallet'] = True
            result['suspicious'] = True
            result['flags'].append('wallet_only_one_trade')
            
            # High score if it's also a large trade
            trade_size = trade.get('size', 0)
            if trade_size > 1000:
                result['score'] = 0.9
                result['flags'].append('single_trade_large_amount')
            else:
                result['score'] = 0.6

        elif total_trades <= 3 and wallet.get('unique_markets', 0) == 1:
            result['flags'].append('very_few_trades_single_market')
            result['suspicious'] = True
            result['score'] = 0.5

        return result

    def analyze_volume_spike(
        self,
        wallet_trades: pd.DataFrame,
        lookback_window: int = 20
    ) -> Dict[str, Any]:
        """
        Detect sudden spikes in trading volume

        Args:
            wallet_trades: DataFrame with wallet's trade history
            lookback_window: Number of recent trades to analyze

        Returns:
            Analysis result
        """
        result = {
            'spike_detected': False,
            'score': 0.0,
            'flags': [],
            'details': {}
        }

        if wallet_trades.empty or 'size' not in wallet_trades.columns:
            return result

        if len(wallet_trades) < lookback_window:
            lookback_window = len(wallet_trades)

        recent_trades = wallet_trades.tail(lookback_window)
        sizes = recent_trades['size'].values

        if len(sizes) < 3:
            return result

        # Calculate rolling statistics
        mean = np.mean(sizes[:-1])  # Exclude last trade
        std = np.std(sizes[:-1])
        last_size = sizes[-1]

        result['details']['recent_avg'] = mean
        result['details']['recent_std'] = std
        result['details']['last_trade_size'] = last_size

        if std > 0:
            spike_ratio = (last_size - mean) / std
            result['details']['spike_ratio'] = spike_ratio

            if spike_ratio > 2.0:
                result['spike_detected'] = True
                result['score'] = min(1.0, spike_ratio / 5.0)
                result['flags'].append('volume_spike_detected')
                
                if spike_ratio > 3.0:
                    result['flags'].append('major_volume_spike')

        return result

    def train_isolation_forest(
        self,
        historical_trades: pd.DataFrame,
        features: List[str] = None
    ):
        """
        Train an Isolation Forest model on historical trade data

        Args:
            historical_trades: DataFrame with historical trades
            features: List of feature columns to use
        """
        if features is None:
            features = ['size', 'price']
        
        # Filter to available features
        available_features = [f for f in features if f in historical_trades.columns]
        
        if not available_features or historical_trades.empty:
            logger.warning("Cannot train Isolation Forest: insufficient data")
            return

        X = historical_trades[available_features].dropna()
        
        if len(X) < 100:
            logger.warning("Insufficient data for Isolation Forest training")
            return

        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(X)
        logger.info(f"Trained Isolation Forest on {len(X)} trades")

    def detect_anomaly_isolation_forest(
        self,
        trade: Dict[str, Any],
        features: List[str] = None
    ) -> Dict[str, Any]:
        """
        Use Isolation Forest to detect anomalous trades

        Args:
            trade: Trade data
            features: Feature columns to use

        Returns:
            Anomaly detection result
        """
        result = {
            'is_anomaly': False,
            'anomaly_score': 0.0,
            'flags': []
        }

        if self.isolation_forest is None:
            logger.warning("Isolation Forest not trained")
            return result

        if features is None:
            features = ['size', 'price']

        # Prepare trade data
        X = pd.DataFrame([trade])[features].dropna()
        
        if X.empty:
            return result

        # Predict
        prediction = self.isolation_forest.predict(X)[0]
        anomaly_score = -self.isolation_forest.score_samples(X)[0]

        result['anomaly_score'] = anomaly_score

        if prediction == -1:  # Anomaly
            result['is_anomaly'] = True
            result['flags'].append('isolation_forest_anomaly')
            
        return result
