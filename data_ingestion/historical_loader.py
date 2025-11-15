"""
Historical data loader for Polymarket transactions
Loads and parses historical trading data
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON
from .gamma_client import GammaClient

logger = logging.getLogger(__name__)


class HistoricalDataLoader:
    """Loads and processes historical Polymarket trading data"""

    def __init__(self):
        from config import settings

        self.gamma_client = GammaClient()

        # Initialize CLOB client (unauthenticated - no private key needed)
        logger.info("Initializing CLOB client (public access)")
        self.clob_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=POLYGON
        )

    def load_market_trades(
        self,
        market_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load all trades for a specific market

        Args:
            market_id: The market condition ID
            start_date: Start date for trade history
            end_date: End date for trade date

        Returns:
            DataFrame with trade history
        """
        try:
            # Fetch trades directly from CLOB using condition_id
            logger.info(f"Fetching trades for market {market_id}")

            trades_raw = self._fetch_market_trades_from_clob(market_id)

            if not trades_raw:
                logger.info(f"No trades found for market {market_id}")
                return pd.DataFrame()

            # Parse trades into DataFrame (Data API format)
            trades = []
            for trade in trades_raw:
                trades.append({
                    'market_id': market_id,
                    'wallet_address': trade.get('taker') or trade.get('proxyWallet'),
                    'timestamp': pd.to_datetime(trade.get('timestamp'), unit='s'),  # Unix timestamp in seconds
                    'size': float(trade.get('size', 0)),
                    'price': float(trade.get('price', 0)),
                    'outcome': trade.get('outcome'),
                    'side': trade.get('side'),
                    'trade_id': trade.get('id'),
                    'maker': trade.get('maker'),
                    'taker': trade.get('taker'),
                    'transaction_hash': trade.get('transactionHash')
                })

            df = pd.DataFrame(trades)

            if not df.empty:
                df = df.sort_values('timestamp')
                logger.info(f"Loaded {len(df)} trades for market {market_id}")

            return df

        except Exception as e:
            logger.error(f"Error loading trades for market {market_id}: {e}")
            return pd.DataFrame()

    def load_wallet_trades(
        self,
        wallet_address: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Load all trades for a specific wallet

        Args:
            wallet_address: The wallet address
            limit: Maximum number of trades to fetch

        Returns:
            DataFrame with wallet trade history
        """
        try:
            # Fetch trades using CLOB client
            trades = []
            
            # Note: Actual API might differ, adjust based on py-clob-client docs
            # This is a placeholder for the actual implementation
            
            logger.warning(
                "Wallet trade fetching not fully implemented. "
                "Requires py-clob-client API exploration."
            )
            
            df = pd.DataFrame(trades)
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp')
                logger.info(f"Loaded {len(df)} trades for wallet {wallet_address}")
            
            return df

        except Exception as e:
            logger.error(f"Error loading trades for wallet {wallet_address}: {e}")
            return pd.DataFrame()

    def load_closed_markets_with_outcomes(
        self,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Load markets with recent trading activity

        Strategy: Fetch recent trades from Data API, extract unique markets,
        then get market details from Gamma API

        Args:
            days_back: Number of days to look back (used for filtering trades)

        Returns:
            List of markets with recent trades
        """
        try:
            import requests

            # Fetch recent trades to find active markets (no auth needed!)
            logger.info("Fetching recent trades to identify active markets...")
            trades_resp = requests.get(
                'https://data-api.polymarket.com/trades',
                params={'limit': 1000},  # Get last 1000 trades
                timeout=30
            )

            if trades_resp.status_code != 200:
                logger.error(f"Failed to fetch recent trades: {trades_resp.status_code}")
                return []

            recent_trades = trades_resp.json()

            # Extract unique condition IDs
            condition_ids = set()
            for trade in recent_trades:
                cond_id = trade.get('conditionId')
                if cond_id:
                    condition_ids.add(cond_id)

            logger.info(f"Found {len(condition_ids)} unique markets with recent trades")

            # Build market metadata from trades themselves (no need for Gamma API!)
            markets_with_trades = []
            markets_dict = {}

            # Extract market info from trades
            for trade in recent_trades:
                cond_id = trade.get('conditionId')
                if cond_id and cond_id not in markets_dict:
                    markets_dict[cond_id] = {
                        'market_id': cond_id,
                        'question': trade.get('title'),
                        'slug': trade.get('slug'),
                        'event_slug': trade.get('eventSlug'),
                        'end_date': None,  # Not available in trades
                        'outcome': None,  # Active markets don't have outcomes yet
                        'volume': 0,  # Would need to calculate from trades
                        'category': None  # Not in trade data
                    }

            markets_with_trades = list(markets_dict.values())[:20]  # Limit to 20

            logger.info(f"Loaded {len(markets_with_trades)} markets with trade data")
            return markets_with_trades

        except Exception as e:
            logger.error(f"Error loading markets with trades: {e}")
            return []

    def aggregate_wallet_statistics(
        self,
        wallet_address: str,
        trades_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate statistics for a wallet based on trade history

        Args:
            wallet_address: The wallet address
            trades_df: DataFrame with wallet trades

        Returns:
            Dictionary with wallet statistics
        """
        if trades_df.empty:
            return {
                'wallet': wallet_address,
                'total_trades': 0,
                'total_volume': 0,
                'win_rate': 0,
                'avg_trade_size': 0,
                'first_trade': None,
                'last_trade': None,
                'unique_markets': 0
            }

        stats = {
            'wallet': wallet_address,
            'total_trades': len(trades_df),
            'total_volume': trades_df['size'].sum() if 'size' in trades_df else 0,
            'avg_trade_size': trades_df['size'].mean() if 'size' in trades_df else 0,
            'first_trade': trades_df['timestamp'].min(),
            'last_trade': trades_df['timestamp'].max(),
            'unique_markets': trades_df['market_id'].nunique() if 'market_id' in trades_df else 0
        }

        # Calculate win rate if outcome data available
        if 'outcome' in trades_df and 'position' in trades_df:
            wins = (trades_df['outcome'] == trades_df['position']).sum()
            stats['win_rate'] = wins / len(trades_df) if len(trades_df) > 0 else 0

        return stats

    def _infer_outcome(self, market: Dict[str, Any]) -> Optional[str]:
        """
        Infer the winning outcome from outcomePrices

        Args:
            market: Market data from Gamma API

        Returns:
            Winning outcome string or None if not resolved
        """
        try:
            import json

            # Get outcomes and prices
            outcomes_str = market.get('outcomes')
            prices_str = market.get('outcomePrices')

            if not outcomes_str or not prices_str:
                return None

            # Parse JSON strings
            outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str

            if not outcomes or not prices or len(outcomes) != len(prices):
                return None

            # Find the outcome with price closest to 1.0 (winner)
            max_price = 0.0
            winning_outcome = None

            for i, price in enumerate(prices):
                price_float = float(price)
                if price_float > max_price and price_float > 0.9:  # Must be > 0.9 to be considered resolved
                    max_price = price_float
                    winning_outcome = outcomes[i]

            return winning_outcome

        except Exception as e:
            logger.warning(f"Error inferring outcome for market: {e}")
            return None

    def _extract_token_ids(self, market: Dict[str, Any]) -> List[str]:
        """Extract token IDs from market data"""
        token_ids = []
        
        if 'tokens' in market:
            for token in market['tokens']:
                if 'token_id' in token:
                    token_ids.append(token['token_id'])
        
        return token_ids

    def _fetch_market_trades_from_clob(
        self,
        condition_id: str,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Fetch trades for a market from Data API (public, no auth required)

        Args:
            condition_id: The market condition ID
            limit: Maximum number of trades to fetch

        Returns:
            List of trade dictionaries
        """
        try:
            import requests

            # Use public Data API endpoint (no authentication needed!)
            url = 'https://data-api.polymarket.com/trades'
            params = {
                'market': condition_id,
                'limit': limit
            }

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 200:
                trades = response.json()
                logger.info(f"Fetched {len(trades)} trades for {condition_id}")
                return trades
            else:
                logger.error(f"Failed to fetch trades: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error fetching trades from Data API: {e}")
            return []

    def _fetch_token_trades(
        self,
        token_id: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[Dict[str, Any]]:
        """Fetch trades for a specific token"""
        # Placeholder - requires actual API implementation
        logger.warning(f"Token trade fetching for {token_id} not fully implemented")
        return []

    def close(self):
        """Close all clients"""
        self.gamma_client.close()
