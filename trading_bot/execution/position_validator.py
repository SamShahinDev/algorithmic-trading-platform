# File: trading_bot/execution/position_validator.py
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PositionValidator:
    """
    Validates position snapshots and transitions
    Prevents invalid states from propagating
    """
    
    # Valid NQ contract patterns
    VALID_CONTRACTS = ['NQ', 'ENQ', 'MNQ']
    
    # Price sanity bounds for NQ (adjust as needed)
    MIN_VALID_PRICE = 10000  # NQ below 10k is suspicious
    MAX_VALID_PRICE = 30000  # NQ above 30k is suspicious
    
    @staticmethod
    def is_valid_position(pos: Optional[Dict]) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive position validation
        
        Returns:
            (is_valid, error_message)
        """
        if not pos:
            return True, None  # None/empty is valid (means flat)
        
        # Required fields validation
        required_fields = ['size', 'averagePrice', 'contractId']
        missing_fields = [f for f in required_fields if f not in pos]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Size validation
        size = pos.get('size', 0)
        if size == 0:
            return False, "Position has zero size"
        if abs(size) > 10:  # Sanity check - 10+ NQ contracts is unusual
            return False, f"Position size suspicious: {size}"
        
        # Price validation
        avg_price = pos.get('averagePrice', 0)
        if avg_price <= 0:
            return False, f"Invalid price: {avg_price}"
        if avg_price < PositionValidator.MIN_VALID_PRICE:
            return False, f"Price too low: {avg_price}"
        if avg_price > PositionValidator.MAX_VALID_PRICE:
            return False, f"Price too high: {avg_price}"
        
        # Contract validation
        contract_id = pos.get('contractId', '')
        if not any(valid in contract_id for valid in PositionValidator.VALID_CONTRACTS):
            return False, f"Invalid contract: {contract_id}"
        
        # Type validation
        pos_type = pos.get('type')
        if pos_type not in [1, 2]:  # 1=LONG, 2=SHORT
            return False, f"Invalid position type: {pos_type}"
        
        return True, None
    
    @staticmethod
    def validate_position_transition(old_pos: Optional[Dict], 
                                   new_pos: Optional[Dict]) -> Tuple[str, bool]:
        """
        Validate position state transitions
        
        Returns:
            (transition_type, is_suspicious)
        """
        old_size = old_pos.get('size', 0) if old_pos else 0
        new_size = new_pos.get('size', 0) if new_pos else 0
        
        # Flat to flat - Normal
        if old_size == 0 and new_size == 0:
            return "flat_to_flat", False
        
        # Opening position - Normal
        if old_size == 0 and new_size != 0:
            return "position_opened", False
        
        # Closing position - Normal
        if old_size != 0 and new_size == 0:
            return "position_closed", False
        
        # Size increase same direction - Normal
        if old_size > 0 and new_size > old_size:
            return "long_increased", False
        if old_size < 0 and new_size < old_size:
            return "short_increased", False
        
        # Size decrease same direction - Normal
        if old_size > 0 and 0 < new_size < old_size:
            return "long_decreased", False
        if old_size < 0 and old_size < new_size < 0:
            return "short_decreased", False
        
        # Position flip - SUSPICIOUS
        if (old_size > 0 and new_size < 0) or (old_size < 0 and new_size > 0):
            logger.error(f"SUSPICIOUS: Position flipped from {old_size} to {new_size}")
            return "position_flipped", True
        
        # Unknown transition - SUSPICIOUS
        logger.error(f"Unknown position transition: {old_size} -> {new_size}")
        return "unknown_transition", True
    
    @staticmethod
    def create_position_snapshot(pos: Dict) -> Dict:
        """Create a validated position snapshot for logging"""
        return {
            'timestamp': datetime.now().isoformat(),
            'size': pos.get('size', 0),
            'side': 'LONG' if pos.get('type') == 1 else 'SHORT',
            'avg_price': pos.get('averagePrice', 0),
            'contract': pos.get('contractId', ''),
            'id': pos.get('id', 'unknown'),
            'valid': PositionValidator.is_valid_position(pos)[0]
        }