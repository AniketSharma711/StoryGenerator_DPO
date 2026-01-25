# DPO Story Generator

A GPT-2 based story generator fine-tuned using Direct Preference Optimization (DPO) to align with human preferences.

## Model Weights
The trained model weights are hosted on Hugging Face:
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/trembl1nghands/Story_Generator_DPO_GPT2)

## How to Run
1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the generator (it will automatically download the weights):
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model_id = "trembl1nghands/Story_Generator_DPO_GPT2"
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForCausalLM.from_pretrained(model_id)

   inputs = tokenizer("Once upon a time", return_tensors="pt")
   outputs = model.generate(**inputs, max_new_tokens=50)
   print(tokenizer.decode(outputs[0]))
