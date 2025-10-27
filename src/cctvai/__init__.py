"""CCTVAI - Behaviour aware video analytics framework."""
from .config import CCTVAIConfig, default_config
from .pipeline import CCTVAI

__all__ = ["CCTVAI", "CCTVAIConfig", "default_config"]
