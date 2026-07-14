"""
Feature Aggregator Module

Provides sliding window aggregation for OHLCV data.
"""

from .feature_aggregator import FeatureAggregator, AggregationScheduler

__all__ = ['FeatureAggregator', 'AggregationScheduler']
