import logging
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class Position:
    """
    Represents a trading position with all relevant attributes.
    Provides a consistent interface for position information across the application.
    """
    
    def __init__(self, 
                symbol: str, 
                entry_price: float, 
                quantity: float,
                entry_time: Optional[datetime] = None):
        """
        Initialize a new position.
        
        Args:
            symbol: The trading symbol for this position
            entry_price: The price at which the position was entered
            quantity: The quantity of the asset in the position
            entry_time: The timestamp when the position was entered
        """
        self.symbol = symbol
        self.entry_price = float(entry_price)
        self.quantity = float(quantity)
        self.entry_time = entry_time or datetime.now()
        
        # Position tracking attributes
        self.current_price = self.entry_price
        self.highest_price = self.entry_price
        self.lowest_price = self.entry_price
        self.trailing_stop_level = None
        self.take_profit_level = None
        self.stop_loss_level = None
        self.unrealized_pnl = 0.0
        self.unrealized_pnl_pct = 0.0
        self.direction = 'long'  # Default is long, could be 'short'
        self.status = 'open'
        self.close_reason = None
        self.close_price = None
        self.close_time = None
        
        # Additional metadata
        self.id = f"{symbol}_{entry_time.strftime('%Y%m%d%H%M%S')}" if entry_time else f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.tags = []
        
    def update(self, current_price: float) -> None:
        """
        Update position with current price.
        
        Args:
            current_price: The current market price
        """
        self.current_price = float(current_price)
        
        # Update highest and lowest prices
        if self.current_price > self.highest_price:
            self.highest_price = self.current_price
        
        if self.current_price < self.lowest_price:
            self.lowest_price = self.current_price
            
        # Calculate P&L
        self.calculate_pnl()
        
    def calculate_pnl(self) -> None:
        """Calculate unrealized profit and loss for the position."""
        if self.entry_price > 0:
            pnl = self.quantity * (self.current_price - self.entry_price)
            self.unrealized_pnl = pnl
            self.unrealized_pnl_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100
            
            if self.direction == 'short':
                self.unrealized_pnl = -pnl
                self.unrealized_pnl_pct = -self.unrealized_pnl_pct
                
    def close(self, close_price: float, reason: str = None) -> None:
        """
        Close the position.
        
        Args:
            close_price: The price at which the position is closed
            reason: The reason for closing the position
        """
        self.close_price = float(close_price)
        self.close_time = datetime.now()
        self.close_reason = reason
        self.status = 'closed'
        
        # Calculate final P&L
        if self.entry_price > 0:
            pnl = self.quantity * (self.close_price - self.entry_price)
            pnl_pct = ((self.close_price - self.entry_price) / self.entry_price) * 100
            
            if self.direction == 'short':
                pnl = -pnl
                pnl_pct = -pnl_pct
                
            self.unrealized_pnl = pnl
            self.unrealized_pnl_pct = pnl_pct
            
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert position to dictionary for storage or serialization.
        
        Returns:
            Dictionary representation of the position
        """
        return {
            'id': self.id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_time': self.entry_time,
            'current_price': self.current_price,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'trailing_stop_level': self.trailing_stop_level,
            'take_profit_level': self.take_profit_level,
            'stop_loss_level': self.stop_loss_level,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'direction': self.direction,
            'status': self.status,
            'close_reason': self.close_reason,
            'close_price': self.close_price,
            'close_time': self.close_time,
            'tags': self.tags
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """
        Create a Position instance from dictionary data.
        
        Args:
            data: Dictionary containing position data
            
        Returns:
            Position instance
        """
        position = cls(
            symbol=data.get('symbol', ''),
            entry_price=data.get('entry_price', 0.0),
            quantity=data.get('quantity', 0.0),
            entry_time=data.get('entry_time')
        )
        
        # Set all other attributes from data
        for key, value in data.items():
            if hasattr(position, key) and key not in ['symbol', 'entry_price', 'quantity', 'entry_time']:
                setattr(position, key, value)
                
        return position
    
    def get(self, attribute: str, default: Any = None) -> Any:
        """
        Get an attribute value with a fallback default.
        Provides dict-like access to position attributes.
        
        Args:
            attribute: The attribute name to get
            default: The default value if attribute doesn't exist
            
        Returns:
            The attribute value or default
        """
        return getattr(self, attribute, default)
        
    def __str__(self) -> str:
        """String representation of the position."""
        return (f"Position({self.symbol}, entry={self.entry_price:.2f}, current={self.current_price:.2f}, "
                f"pnl={self.unrealized_pnl_pct:.2f}%, status={self.status})")
                
    def __repr__(self) -> str:
        """Detailed representation of the position."""
        return (f"Position(symbol={self.symbol}, entry_price={self.entry_price}, quantity={self.quantity}, "
                f"entry_time={self.entry_time}, status={self.status})") 