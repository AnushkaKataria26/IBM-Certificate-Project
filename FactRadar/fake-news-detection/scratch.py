from src.preprocessing.clean_text import clean_text

text = "A new study has found that drinking two cups of coffee a day can increase workplace productivity by up to 20% according to a researcher."
print("Original:", text)
cleaned = clean_text(text)
print("Cleaned:", cleaned)

