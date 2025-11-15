"""
Gamma API Client for Polymarket
Fetches markets, events, and trading data
"""
import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from config import settings

logger = logging.getLogger(__name__)


class GammaClient:
    """Client for interacting with Polymarket Gamma API"""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.gamma_api_base_url
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "PolymarketInsiderDetector/1.0"
        })

    def get_markets(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        archived: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Gamma API

        Args:
            active: Filter for active markets
            closed: Filter for closed markets
            archived: Filter for archived markets
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of market dictionaries
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if active is not None:
            params["active"] = str(active).lower()
        if closed is not None:
            params["closed"] = str(closed).lower()
        if archived is not None:
            params["archived"] = str(archived).lower()

        try:
            response = self.session.get(
                f"{self.base_url}/markets",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched {len(data)} markets from Gamma API")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching markets: {e}")
            raise

    def get_market_by_id(self, market_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific market by ID

        Args:
            market_id: The market condition ID

        Returns:
            Market dictionary or None if not found
        """
        try:
            response = self.session.get(
                f"{self.base_url}/markets/{market_id}",
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched market {market_id}")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching market {market_id}: {e}")
            return None

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        archived: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch events from Gamma API

        Args:
            limit: Maximum number of results
            offset: Pagination offset
            archived: Filter for archived events

        Returns:
            List of event dictionaries
        """
        params = {
            "limit": limit,
            "offset": offset
        }
        
        if archived is not None:
            params["archived"] = str(archived).lower()

        try:
            response = self.session.get(
                f"{self.base_url}/events",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched {len(data)} events from Gamma API")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error fetching events: {e}")
            raise

    def search_markets(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search markets by query string

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching markets
        """
        try:
            # Note: Actual endpoint might differ, this is based on common patterns
            response = self.session.get(
                f"{self.base_url}/markets",
                params={"search": query, "limit": limit},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Search '{query}' returned {len(data)} markets")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Error searching markets: {e}")
            return []

    def get_all_markets_paginated(
        self,
        active: Optional[bool] = None,
        closed: Optional[bool] = None,
        max_results: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all markets using pagination

        Args:
            active: Filter for active markets
            closed: Filter for closed markets
            max_results: Maximum total results to fetch

        Returns:
            List of all market dictionaries
        """
        all_markets = []
        offset = 0
        limit = 100

        while True:
            markets = self.get_markets(
                active=active,
                closed=closed,
                limit=limit,
                offset=offset
            )
            
            if not markets:
                break
                
            all_markets.extend(markets)
            
            if max_results and len(all_markets) >= max_results:
                all_markets = all_markets[:max_results]
                break
                
            if len(markets) < limit:
                break
                
            offset += limit

        logger.info(f"Fetched total of {len(all_markets)} markets")
        return all_markets

    def close(self):
        """Close the session"""
        self.session.close()
