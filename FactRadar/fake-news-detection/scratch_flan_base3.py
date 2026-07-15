from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = """Instructions: Write a one-sentence explanation of why the article was classified based on the key words.
Classification: real
Key words: mayor, public, parks, initiative
Explanation:"""
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.2, do_sample=True, temperature=0.7)
print("FLAN-BASE EXPLANATION:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
