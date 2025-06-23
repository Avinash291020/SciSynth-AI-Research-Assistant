"""Model caching system for efficient model reuse."""
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from transformers.pipelines import pipeline
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelCache:
    _instance = None
    _models: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_sentence_transformer(cls, model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
        """Get or load a sentence transformer model."""
        key = f"sentence_transformer_{model_name}"
        if key not in cls._models:
            try:
                # Completely avoid device movement to prevent meta tensor errors
                logger.info(f"Use pytorch device_name: cpu")
                logger.info(f"Load pretrained SentenceTransformer: {model_name}")
                
                # Create model and keep it on CPU only
                model = SentenceTransformer(model_name)
                
                # Don't move to device - keep on CPU to avoid meta tensor issues
                cls._models[key] = model
                
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                # Fallback: try without device specification
                try:
                    cls._models[key] = SentenceTransformer(model_name)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise e2
        return cls._models[key]
    
    @classmethod
    def get_text_generator(cls, model_name: str = 'google/flan-t5-base') -> Any:
        """Get or load a text generation model."""
        key = f"text_generator_{model_name}"
        if key not in cls._models:
            try:
                # Keep everything on CPU to avoid device issues
                logger.info(f"Use pytorch device_name: cpu")
                logger.info(f"Load pretrained text generator: {model_name}")
                
                # Create pipeline and keep on CPU only
                pipeline_obj = pipeline('text2text-generation', model=model_name)
                
                # Don't move to device - keep on CPU to avoid meta tensor issues
                cls._models[key] = pipeline_obj
                
            except Exception as e:
                logger.error(f"Error loading text generator: {e}")
                # Fallback: try without device specification
                try:
                    cls._models[key] = pipeline('text2text-generation', model=model_name)
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise e2
        return cls._models[key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models."""
        cls._models.clear() 