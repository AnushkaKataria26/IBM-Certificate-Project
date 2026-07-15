import sys
import numpy as np

# Mock unified_predict_proba
def unified_predict_proba(texts):
    return np.array([[0.4, 0.6] for _ in texts])

from src.serving.explain_lime import explain_instance
from src.preprocessing.clean_text import clean_text

text = "A new study has found that drinking two cups of coffee a day can increase workplace productivity by up to 20% according to a researcher."

print("Cleaned:", clean_text(text))

res = explain_instance(text, unified_predict_proba, num_features=20)
print("LIME tokens:", res)
