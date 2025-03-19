"""
Support and Resistance Zone Detector for Binance Trading Bot.
Identifies key price levels where market reversals or consolidations are likely to occur.
Combines multiple methods including clustering of peaks/valleys and order book analysis.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from scipy.signal import argrelextrema
from sklearn.cluster import DBSCAN

class SupportResistanceDetector:
    """
    Detects support and resistance zones using multiple methods:
    1. Historical price extrema clustering
    2. Volume profile analysis
    3. Order book depth (if available)
    
    These zones are critical for setting stop losses, take profits,
    and identifying potential reversal points.
    """
    
    def __init__(self, symbol: str, timeframe: str = '1h', data_dir: str = 'data/zones'):
        """
        Initialize the support and resistance zone detector.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Data timeframe (e.g., '1h', '4h', '1d')
            data_dir: Directory for caching SR zones
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_dir = data_dir
        
        # Cache file path for SR zones
        self.cache_file = os.path.join(data_dir, f'sr_zones_{symbol}_{timeframe}.json')
        
        # Create data directory if needed
        os.makedirs(data_dir, exist_ok=True)
        
        # Store detected zones
        self.resistance_zones = []  # List of (price_level, strength) tuples
        self.support_zones = []     # List of (price_level, strength) tuples
        
        # Last update time
        self.last_update = datetime.now()
        
        # Minimum number of hours between full recalculations
        self.update_interval_hours = 4
        
        # Config parameters
        self.extrema_window = 10    # Window for local extrema detection
        self.price_tolerance = 0.01 # Percent tolerance for clustering prices
        self.strength_decay = 0.8   # How fast zone strength decays with price distance
        
        # Load cached zones if available
        self._load_cached_zones()
        
        # Logger setup
        self.logger = logging.getLogger('support_resistance_detector')
        
    def _load_cached_zones(self) -> bool:
        """
        Load cached support and resistance zones from disk.
        
        Returns:
            bool: True if zones were loaded successfully
        """
        try:
            if os.path.exists(self.cache_file):
                import json
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    
                self.support_zones = data.get('support_zones', [])
                self.resistance_zones = data.get('resistance_zones', [])
                self.last_update = datetime.fromisoformat(data.get('last_update', datetime.now().isoformat()))
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading cached zones: {e}")
            return False
            
    def _save_cached_zones(self) -> None:
        """Save current support and resistance zones to disk."""
        try:
            import json
            data = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'support_zones': self.support_zones,
                'resistance_zones': self.resistance_zones,
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving cached zones: {e}")
            
    def detect_zones(self, df: pd.DataFrame, order_book: Optional[Dict] = None, force_update: bool = False) -> Dict[str, List[Tuple[float, float]]]:
        """
        Detect support and resistance zones using historical price data and optionally order book.
        
        Args:
            df: DataFrame with OHLCV data
            order_book: Optional order book data (bids/asks)
            force_update: Force full recalculation even if cache is recent
            
        Returns:
            Dict with 'support' and 'resistance' zones as (price, strength) tuples
        """
        # Check if we need to update zones
        hours_since_update = (datetime.now() - self.last_update).total_seconds() / 3600
        
        if force_update or hours_since_update >= self.update_interval_hours or not self.support_zones:
            # Full recalculation
            self._detect_zones_from_extrema(df)
            self._detect_zones_from_volume_profile(df)
            
            # Add order book levels if available
            if order_book is not None:
                self._incorporate_order_book(order_book)
                
            # Update timestamp and cache zones
            self.last_update = datetime.now()
            self._save_cached_zones()
        else:
            # Just refine existing zones with new data
            self._refine_zones_with_recent_data(df.iloc[-100:])
            
            # Add order book levels if available
            if order_book is not None:
                self._incorporate_order_book(order_book)
        
        return {
            'support': self.support_zones,
            'resistance': self.resistance_zones
        }
        
    def _detect_zones_from_extrema(self, df: pd.DataFrame) -> None:
        """
        Detect support and resistance levels based on historical price extrema.
        
        Args:
            df: DataFrame with OHLCV data
        """
        # Ensure we have enough data
        if len(df) < self.extrema_window * 3:
            self.logger.warning(f"Insufficient data for extrema detection: {len(df)} rows")
            return
            
        # Extract highs and lows
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Find local maxima and minima
        high_idx = argrelextrema(highs, np.greater_equal, order=self.extrema_window)[0]
        low_idx = argrelextrema(lows, np.less_equal, order=self.extrema_window)[0]
        
        # Extract price levels
        resistance_prices = highs[high_idx]
        support_prices = lows[low_idx]
        
        # Apply DBSCAN clustering to identify price zones (more precise than simple binning)
        resistance_zones = self._cluster_price_levels(resistance_prices)
        support_zones = self._cluster_price_levels(support_prices)
        
        # Get current price
        current_price = closes[-1]
        
        # Filter zones based on current price
        self.resistance_zones = [(price, strength) for price, strength in resistance_zones 
                                if price > current_price * 1.001]  # Just above current price
        
        self.support_zones = [(price, strength) for price, strength in support_zones 
                             if price < current_price * 0.999]    # Just below current price
        
    def _cluster_price_levels(self, prices: np.ndarray) -> List[Tuple[float, float]]:
        """
        Cluster similar price levels to identify zones.
        
        Args:
            prices: Array of price levels
            
        Returns:
            List of (price_level, strength) tuples
        """
        if len(prices) < 3:
            return [(p, 0.5) for p in prices]  # Not enough data for clustering
            
        # Normalize prices for clustering
        X = prices.reshape(-1, 1)
        
        # Determine epsilon parameter based on price volatility
        price_range = np.max(X) - np.min(X)
        eps = price_range * self.price_tolerance
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(X)
        
        # Process clusters
        zones = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points (not part of any cluster)
                continue
                
            # Get cluster points
            cluster_mask = clustering.labels_ == label
            cluster_points = X[cluster_mask].flatten()
            
            # Calculate zone price (weighted average of the cluster)
            zone_price = np.mean(cluster_points)
            
            # Calculate zone strength based on number of points in cluster
            zone_strength = min(1.0, 0.3 + 0.1 * len(cluster_points))
            
            zones.append((float(zone_price), float(zone_strength)))
            
        # Add unclustered (noise) points with lower strength
        noise_mask = clustering.labels_ == -1
        for noise_price in X[noise_mask]:
            zones.append((float(noise_price), 0.3))  # Lower strength for unclustered points
            
        return zones
        
    def _detect_zones_from_volume_profile(self, df: pd.DataFrame) -> None:
        """
        Enhance zone detection using volume profile analysis.
        
        Args:
            df: DataFrame with OHLCV data
        """
        if 'volume' not in df.columns or len(df) < 100:
            return
            
        # Create price bins
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / 100  # 100 bins across price range
        
        # Create price bins and initialize volume profile
        bins = np.arange(df['low'].min(), df['high'].max() + bin_size, bin_size)
        volume_profile = np.zeros(len(bins) - 1)
        
        # Calculate volume for each bin using typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        for i, (_, row) in enumerate(df.iterrows()):
            # Find which bin this candle belongs to
            price = typical_price.iloc[i]
            bin_idx = np.digitize(price, bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += row['volume']
                
        # Find local maxima in volume profile (high volume nodes)
        volume_nodes = argrelextrema(volume_profile, np.greater_equal, order=3)[0]
        
        # Extract corresponding prices
        high_volume_prices = [(bins[i] + bins[i+1])/2 for i in volume_nodes]
        
        # Strengthen existing zones if they match high volume nodes
        current_price = df['close'].iloc[-1]
        
        for vol_price in high_volume_prices:
            # Strengthen resistance zones
            for i, (price, strength) in enumerate(self.resistance_zones):
                # Check if volume node is close to a known resistance
                if abs(price - vol_price) / price < self.price_tolerance and vol_price > current_price:
                    # Strengthen this zone
                    new_strength = min(1.0, strength + 0.2)
                    self.resistance_zones[i] = (price, new_strength)
                    break
            else:
                # If the node is above current price and not matched, add as new resistance
                if vol_price > current_price:
                    self.resistance_zones.append((float(vol_price), 0.5))
                    
            # Strengthen support zones
            for i, (price, strength) in enumerate(self.support_zones):
                # Check if volume node is close to a known support
                if abs(price - vol_price) / price < self.price_tolerance and vol_price < current_price:
                    # Strengthen this zone
                    new_strength = min(1.0, strength + 0.2)
                    self.support_zones[i] = (price, new_strength)
                    break
            else:
                # If the node is below current price and not matched, add as new support
                if vol_price < current_price:
                    self.support_zones.append((float(vol_price), 0.5))
                    
        # Sort zones by price
        self.resistance_zones.sort()
        self.support_zones.sort(reverse=True)  # Highest support first
        
    def _incorporate_order_book(self, order_book: Dict) -> None:
        """
        Enhance zones by incorporating order book data.
        
        Args:
            order_book: Order book data with bids and asks
        """
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return
            
        try:
            # Extract order book data
            bids = order_book['bids']  # List of [price, quantity] pairs
            asks = order_book['asks']  # List of [price, quantity] pairs
            
            # Normalize quantities
            total_bid_qty = sum(float(b[1]) for b in bids)
            total_ask_qty = sum(float(a[1]) for a in asks)
            
            if total_bid_qty == 0 or total_ask_qty == 0:
                return
                
            # Identify large orders (walls)
            threshold_pct = 0.05  # Orders that represent more than 5% of total are significant
            
            # Check for bid walls (supports)
            for b in bids:
                price = float(b[0])
                qty = float(b[1])
                qty_pct = qty / total_bid_qty
                
                if qty_pct > threshold_pct:
                    # Check if price is close to existing support
                    for i, (s_price, s_strength) in enumerate(self.support_zones):
                        if abs(price - s_price) / price < self.price_tolerance:
                            # Strengthen existing support
                            new_strength = min(1.0, s_strength + qty_pct)
                            self.support_zones[i] = (s_price, new_strength)
                            break
                    else:
                        # Add new support zone
                        self.support_zones.append((price, min(1.0, 0.4 + qty_pct)))
                        
            # Check for ask walls (resistances)
            for a in asks:
                price = float(a[0])
                qty = float(a[1])
                qty_pct = qty / total_ask_qty
                
                if qty_pct > threshold_pct:
                    # Check if price is close to existing resistance
                    for i, (r_price, r_strength) in enumerate(self.resistance_zones):
                        if abs(price - r_price) / price < self.price_tolerance:
                            # Strengthen existing resistance
                            new_strength = min(1.0, r_strength + qty_pct)
                            self.resistance_zones[i] = (r_price, new_strength)
                            break
                    else:
                        # Add new resistance zone
                        self.resistance_zones.append((price, min(1.0, 0.4 + qty_pct)))
                        
            # Keep zones sorted
            self.resistance_zones.sort()
            self.support_zones.sort(reverse=True)
            
            # Limit to top 10 strongest zones for each
            self.resistance_zones = sorted(self.resistance_zones, key=lambda x: x[1], reverse=True)[:10]
            self.support_zones = sorted(self.support_zones, key=lambda x: x[1], reverse=True)[:10]
            
        except Exception as e:
            self.logger.error(f"Error incorporating order book data: {e}")
            
    def _refine_zones_with_recent_data(self, recent_df: pd.DataFrame) -> None:
        """
        Refine existing zones with recent price action.
        
        Args:
            recent_df: DataFrame with recent OHLCV data
        """
        if len(recent_df) < 10:
            return
            
        # Get current price
        current_price = recent_df['close'].iloc[-1]
        
        # Check if price has broken through any zones
        for i, (price, strength) in enumerate(self.resistance_zones):
            # If price closed significantly above resistance, weaken it
            if current_price > price * 1.01:
                new_strength = max(0.1, strength - 0.2)
                self.resistance_zones[i] = (price, new_strength)
                
        for i, (price, strength) in enumerate(self.support_zones):
            # If price closed significantly below support, weaken it
            if current_price < price * 0.99:
                new_strength = max(0.1, strength - 0.2)
                self.support_zones[i] = (price, new_strength)
                
        # Remove very weak zones
        self.resistance_zones = [(p, s) for p, s in self.resistance_zones if s >= 0.2]
        self.support_zones = [(p, s) for p, s in self.support_zones if s >= 0.2]
        
    def get_nearest_zones(self, price: float, n: int = 3) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get the nearest support and resistance zones to the current price.
        
        Args:
            price: Current price
            n: Number of zones to return for each type
            
        Returns:
            Dictionary with nearest support and resistance zones
        """
        # Calculate distance-weighted strength
        res_with_distance = []
        for r_price, r_strength in self.resistance_zones:
            if r_price > price:  # Only resistances above current price
                distance_pct = (r_price - price) / price
                # Decay strength with distance
                adjusted_strength = r_strength * (self.strength_decay ** distance_pct)
                res_with_distance.append((r_price, adjusted_strength, distance_pct))
                
        sup_with_distance = []
        for s_price, s_strength in self.support_zones:
            if s_price < price:  # Only supports below current price
                distance_pct = (price - s_price) / price
                # Decay strength with distance
                adjusted_strength = s_strength * (self.strength_decay ** distance_pct)
                sup_with_distance.append((s_price, adjusted_strength, distance_pct))
                
        # Sort by adjusted strength (not just distance)
        nearest_res = sorted(res_with_distance, key=lambda x: x[1], reverse=True)[:n]
        nearest_sup = sorted(sup_with_distance, key=lambda x: x[1], reverse=True)[:n]
        
        return {
            'resistance': [(p, s) for p, s, _ in nearest_res],
            'support': [(p, s) for p, s, _ in nearest_sup]
        }
        
    def calculate_zone_context(self, price: float) -> Dict[str, Any]:
        """
        Calculate trading context based on support/resistance zones.
        
        Args:
            price: Current price
            
        Returns:
            Dict with zone-based analysis and indicators
        """
        nearest = self.get_nearest_zones(price, n=3)
        
        resistances = nearest['resistance']
        supports = nearest['support']
        
        # Check for strong overhead resistance or support beneath
        strong_resistance = False
        strong_support = False
        
        resistance_proximity = 1.0
        support_proximity = 1.0
        
        # Calculate proximity to nearest zones (as percentage of price)
        if resistances:
            nearest_res_price, nearest_res_strength = resistances[0]
            resistance_proximity = (nearest_res_price - price) / price
            strong_resistance = nearest_res_strength > 0.7 and resistance_proximity < 0.03
            
        if supports:
            nearest_sup_price, nearest_sup_strength = supports[0]
            support_proximity = (price - nearest_sup_price) / price
            strong_support = nearest_sup_strength > 0.7 and support_proximity < 0.03
            
        # Calculate zone strength ratio (support vs resistance)
        support_strength = sum(s for _, s in supports)
        resistance_strength = sum(s for _, s in resistances)
        
        zone_strength_ratio = 0.5  # Neutral default
        if support_strength + resistance_strength > 0:
            zone_strength_ratio = support_strength / (support_strength + resistance_strength)
            
        # Identify if price is in no-man's land (far from both S/R)
        in_no_mans_land = resistance_proximity > 0.05 and support_proximity > 0.05
        
        # Determine if price is in a zone
        in_resistance_zone = any(abs(price - r_price) / price < 0.005 for r_price, _ in resistances)
        in_support_zone = any(abs(price - s_price) / price < 0.005 for s_price, _ in supports)
        
        return {
            'nearest_resistance': resistances[0] if resistances else None,
            'nearest_support': supports[0] if supports else None,
            'strong_resistance': strong_resistance,
            'strong_support': strong_support,
            'resistance_proximity': resistance_proximity,
            'support_proximity': support_proximity,
            'zone_strength_ratio': zone_strength_ratio,  # >0.5 means stronger support, <0.5 means stronger resistance
            'in_no_mans_land': in_no_mans_land,
            'in_resistance_zone': in_resistance_zone,
            'in_support_zone': in_support_zone,
            'buy_zone_strength': zone_strength_ratio if not in_resistance_zone else 0.3,
            'sell_zone_strength': (1 - zone_strength_ratio) if not in_support_zone else 0.3
        }
        
    def get_optimal_stop_loss(self, price: float, side: str) -> Optional[float]:
        """
        Get optimal stop loss price based on support/resistance zones.
        
        Args:
            price: Entry price
            side: 'BUY' or 'SELL'
            
        Returns:
            Optimal stop loss price or None if not enough data
        """
        nearest = self.get_nearest_zones(price, n=3)
        
        if side.upper() == 'BUY':
            # For buy orders, place stop below nearest support
            supports = nearest['support']
            if supports:
                # Find strongest support below entry
                supports_below = [(p, s) for p, s in supports if p < price]
                if supports_below:
                    strongest_support = max(supports_below, key=lambda x: x[1])
                    # Place stop slightly below support
                    return strongest_support[0] * 0.995
                    
            # If no supports found, use default percentage
            return price * 0.97  # 3% below entry
            
        elif side.upper() == 'SELL':
            # For sell orders, place stop above nearest resistance
            resistances = nearest['resistance']
            if resistances:
                # Find strongest resistance above entry
                resistances_above = [(p, s) for p, s in resistances if p > price]
                if resistances_above:
                    strongest_resistance = max(resistances_above, key=lambda x: x[1])
                    # Place stop slightly above resistance
                    return strongest_resistance[0] * 1.005
                    
            # If no resistances found, use default percentage
            return price * 1.03  # 3% above entry
            
        return None
        
    def get_optimal_take_profit(self, price: float, side: str) -> List[Tuple[float, float]]:
        """
        Get optimal take profit levels based on support/resistance zones.
        
        Args:
            price: Entry price
            side: 'BUY' or 'SELL'
            
        Returns:
            List of (price, weight) tuples for take profit levels
        """
        if side.upper() == 'BUY':
            # For buy orders, target resistances above
            take_profits = []
            
            for r_price, r_strength in self.resistance_zones:
                if r_price > price * 1.005:  # At least 0.5% above entry
                    # Calculate reward/risk ratio
                    price_distance = (r_price - price) / price
                    
                    # Weight based on strength and distance
                    tp_weight = r_strength * min(1.0, 0.05 / price_distance)
                    
                    take_profits.append((r_price, tp_weight))
                    
            # Sort by weight and return top 3
            return sorted(take_profits, key=lambda x: x[1], reverse=True)[:3]
            
        elif side.upper() == 'SELL':
            # For sell orders, target supports below
            take_profits = []
            
            for s_price, s_strength in self.support_zones:
                if s_price < price * 0.995:  # At least 0.5% below entry
                    # Calculate reward/risk ratio
                    price_distance = (price - s_price) / price
                    
                    # Weight based on strength and distance
                    tp_weight = s_strength * min(1.0, 0.05 / price_distance)
                    
                    take_profits.append((s_price, tp_weight))
                    
            # Sort by weight and return top 3
            return sorted(take_profits, key=lambda x: x[1], reverse=True)[:3]
            
        return [] 