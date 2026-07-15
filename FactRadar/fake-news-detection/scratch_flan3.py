from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

prompt = """Explain why the article was classified as fake.
Article: A new study has found that drinking two cups of coffee a day can increase workplace productivity by up to 20% according to a researcher.
Keywords that led to this classification: six, drinking, cup, 20, productivity, researcher, two, scientific, day, increase
Explanation: The article was classified as fake because"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print("FLAN-T5 Output 1:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

prompt2 = """Context: An article was classified as 'fake'. The keywords responsible for this classification are: six, drinking, 20, productivity.
Question: Why was this article classified as 'fake'?
Answer:"""

inputs2 = tokenizer(prompt2, return_tensors="pt")
outputs2 = model.generate(**inputs2, max_new_tokens=128)
print("FLAN-T5 Output 2:")
print(tokenizer.decode(outputs2[0], skip_special_tokens=True))
