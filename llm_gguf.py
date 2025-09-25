# python3 -m venv ~/.venvs/myenv
# source ~/.venvs/myenv/bin/activate
# pip install <your-package-name>

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
save_dir = "EXAONE-3.5-2.4B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
