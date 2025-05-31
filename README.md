!pip install transformers

from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained tokenizer and model (from HuggingFace)
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Defines the text generation
def generate_text(prompt, max_length=50, temperature=1.0):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
# Prompt to train AI model to search for stuff
prompt = "Write a brief paragraph about the future of Artificial Intelligence"
generated = generate_text(prompt)
print(generated)

# Final prompt to define response
generated = generate_text(prompt, max_length=100, temperature=1.2)
