"""
Density Analyzer Module.
Estimates spatial crowd density and generates heatmaps.
"""

from app.analytics.density import DensityEstimator, DensityResult, DensityGrid

# Alias matching master architecture
DensityAnalyzer = DensityEstimator

__all__ = ["DensityEstimator", "DensityAnalyzer", "DensityResult", "DensityGrid"]

