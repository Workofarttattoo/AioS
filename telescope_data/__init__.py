# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Telescope Suite Data Collection Package
Handles data collection from multiple sources for all 7 prediction tools.
"""

from .career_collector import CareerDataCollector
from .health_collector import HealthDataCollector
from .market_collector import MarketDataCollector

__all__ = [
    'CareerDataCollector',
    'HealthDataCollector',
    'MarketDataCollector',
]
