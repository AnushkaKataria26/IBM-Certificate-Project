from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

prompt = """The following article has been classified as 'real' with a confidence of 60.0%.
The top contributing words to this classification are:
- productivity (score: 0.5)
- researcher (score: 0.3)

Article text:
A new study has found that drinking two cups of coffee a day can increase workplace productivity by up to 20% according to a researcher.

Based on these words and the article text, write a short paragraph explaining why the article received this classification.
Explanation:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
