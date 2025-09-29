from transformers import AutoTokenizer

# Load the Qwen3 tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-4B") # Replace with the specific Qwen3 model you are using

# Encode text
text = "This is an example sentence for Qwen3 tokenization."
tokens = tokenizer.encode(text)
print(f"Tokens: {len(tokens)}")

# Decode tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")