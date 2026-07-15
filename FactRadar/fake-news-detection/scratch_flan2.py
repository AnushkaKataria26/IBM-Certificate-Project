from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

prompt = """The following article has been classified as 'fake' with a confidence of 56.3%.
The top contributing words to this classification are:
- six (score: 0.122)
- drinking (score: -0.054)

Article text:
A new study has found that drinking two cups of coffee a day can increase workplace productivity by up to 20% according to a researcher.

Based on these words and the article text, write a short paragraph explaining why the article received this classification.
Explanation:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print("FLAN-T5 Output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
