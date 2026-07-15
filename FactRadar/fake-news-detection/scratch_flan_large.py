import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = "Explain why this article was classified as 'real' with 56.3% confidence.\nContext: The article text is: The city mayor announced a new initiative today to build five new public parks in the downtown area over the next two years to improve green spaces.\nQuestion: Why was this article classified as 'real'?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print("FLAN-LARGE EXPLANATION:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

prompt_rag = "Instructions: Assess whether the retrieved evidence supports or contradicts the flagged article. You must respond in one of three categories only: 'consistent', 'contradictory', or 'insufficient_evidence'. Include a one sentence justification. State your verdict keyword exactly.\nArticle: The city mayor announced a new initiative today...\nEvidence: No reference evidence was found.\nAnswer:"
inputs = tokenizer(prompt_rag, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
print("FLAN-LARGE RAG:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
