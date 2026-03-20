# GPT-2 Storyteller: Alignment via Direct Preference Optimization (DPO)

This project explores **LLM Alignment**. Instead of just fine-tuning GPT-2 on a dataset (SFT), I used **Direct Preference Optimization (DPO)** to steer the model toward generating more coherent, creative, and "human-preferred" stories.

## The Concept
Most models are trained to predict the next word, but they don't always follow human preferences. I implemented DPO—a more stable alternative to RLHF—to optimize the model directly on a dataset of "preferred" vs "rejected" story completions.

## Tech Stack
- **Base Model:** GPT-2 (DistilGPT2/Small)
- **Framework:** Hugging Face `transformers` & `trl` (Transformer Reinforcement Learning)
- **Optimization:** Direct Preference Optimization (DPO)
- **Compute:** Fine-tuned using [specify if you used Colab/Kaggle/Local GPU]

## What does it contain?
- `train_dpo.py`: The script I used for the alignment loop.
- `generate.py`: A clean interface to test the aligned model vs the base model.
- `requirements.txt`: All the dependencies to get it running.

## How to use it
The trained weights are hosted on Hugging Face, so you don't need to retrain it from scratch to see the results.

1. **Clone & Install:**
   ```bash
   git clone [https://github.com/AniketSharma711/GPT2-Story-DPO.git](https://github.com/AniketSharma711/GPT2-Story-DPO.git)
   pip install -r requirements.txt
