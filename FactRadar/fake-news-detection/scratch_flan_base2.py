from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = """You are an AI designed to explain text classifications.
The following article was classified as 'real' with 56.3% confidence.

Article:
The city mayor announced a new initiative today to build five new public parks in the downtown area over the next two years to improve green spaces.

The key words that led to this classification are: mayor, public, parks, initiative.

Write a short, professional explanation of why the article might be considered 'real'.
Explanation:"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2, do_sample=True, temperature=0.7)
print("FLAN-BASE EXPLANATION:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
