from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = "Explain why this article was classified as 'real' with 56.3% confidence.\nContext: The article text is: The city mayor announced a new initiative today to build five new public parks in the downtown area over the next two years to improve green spaces.\nQuestion: Why was this article classified as 'real'?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128, repetition_penalty=1.2, temperature=0.7, do_sample=True)
print("FLAN-BASE EXPLANATION:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

prompt_rag = "Instructions: Read the article and evidence. Decide if the evidence supports ('consistent'), contradicts ('contradictory'), or is not enough ('insufficient_evidence') for the article.\nArticle: The city mayor announced a new initiative today to build five new public parks.\nEvidence: No reference evidence was found.\nVerdict (choose one: consistent, contradictory, insufficient_evidence):"
inputs = tokenizer(prompt_rag, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32, repetition_penalty=1.2, temperature=0.1, do_sample=True)
print("FLAN-BASE RAG:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
