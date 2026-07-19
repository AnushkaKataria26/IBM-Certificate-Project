import logging
from pathlib import Path
from lime.lime_text import LimeTextExplainer
from src.preprocessing.clean_text import clean_text

logger = logging.getLogger(__name__)


def explain_instance(text: str, predict_fn, num_features: int = 10) -> list[tuple[str, float]]:
    """
    Explain a single instance using LIME.
    
    Args:
        text (str): Raw input text.
        predict_fn (callable): Function that takes list of strings and returns probabilities.
        num_features (int): Maximum number of features to include in explanation.
        
    Returns:
        A list of (token, weight) tuples sorted by absolute weight descending.
    """
    # Instantiate explainer with fixed seed for perfect consistency
    _explainer = LimeTextExplainer(class_names=["real", "fake"], random_state=42)
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        raise ValueError("Input contains no usable text content after preprocessing.")
        
    # Generate explanation
    exp = _explainer.explain_instance(
        cleaned_text, 
        predict_fn, 
        num_features=num_features,
        num_samples=100
    )
    
    # exp.as_list() returns a list of (word, weight) for the predicted class
    explanation_list = exp.as_list()
    
    # Sort by absolute weight descending
    explanation_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # LIME might return fewer than num_features if the document is very short, 
    # but we just return what it gives (up to num_features).
    return explanation_list
