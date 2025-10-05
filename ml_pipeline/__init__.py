"""
ExoAI Hunter - ML Pipeline Package
NASA Space Apps Challenge 2025
"""

from .data_processor import ExoplanetDataProcessor
from .exoplanet_model import ExoAIHunterModel, create_exoai_model
from .train_model import ExoAITrainer

__version__ = "1.0.0"
__all__ = [
    "ExoplanetDataProcessor",
    "ExoAIHunterModel", 
    "create_exoai_model",
    "ExoAITrainer"
]
