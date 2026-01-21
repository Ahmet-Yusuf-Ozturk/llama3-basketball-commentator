# 🏀 Automated Sports Narration for EuroLeague (DSAI 585)

> **"Watching with Ears":** A Generative AI pipeline capable of producing real-time, bilingual (Turkish/English) basketball commentary for visually impaired fans.

![License](https://img.shields.io/badge/license-MIT-blue) ![Model](https://img.shields.io/badge/Model-Llama--3--8B-green) ![Framework](https://img.shields.io/badge/Framework-Unsloth-orange) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-yellow)](https://huggingface.co/YOUR_HF_USERNAME/euroleague-narrator)

## 📌 Project Overview
The intersection of Sports Analytics and Generative AI offers a unique opportunity to enhance the spectator experience. While data visualization tools are common, they fail to convey the emotional flow of a game to those who cannot see it.

Motivated by the challenges faced by visually impaired fans (inspired by **1907 UNIFEB Boğaziçi**), this project aims to bridge that gap. We fine-tuned **Llama-3-8B** to transform dry, structured game logs into enthusiastic, broadcast-quality narration in real-time.

**Key Features:**
* **Bilingual Support:** Generates idiomatic commentary in both **Turkish** and **English**.
* **Style Transfer:** Moves beyond robotic templates to capture the excitement of a "Buzzer Beater" or a "Monster Block."
* **Social Accessibility:** Designed to help visually impaired fans follow the rapid pace of EuroLeague basketball.

## ⚙️ Methodology

### 1. Synthetic Data Generation (The Gemini Pipeline)
Due to the scarcity of paired (Game Log $\to$ Broadcast Transcript) data, we implemented a synthetic generation pipeline.
* **Engine:** Google Gemini.
* **Strategy:** Two-Shot Learning (Few-Shot) to ensure structural consistency.
* **Dataset Size:** **369** high-quality pairs of JSON events and emotional commentary.
* **Verification:** Manual double-checking to remove inconsistencies.

### 2. Fine-Tuning with Unsloth
We utilized **Unsloth** to accelerate the fine-tuning of Llama-3-8B on a T4 GPU (Google Colab).
* **Technique:** LoRA (Low-Rank Adaptation).
* **Optimization:** 4-bit Quantization for memory efficiency.
* **Training:** Learned to map specific JSON keys (`time`, `action`, `result`) to narrative tension.

## 📊 Evaluation & Results

We evaluated the model on **20 independent test cases** using an **LLM-as-a-Judge** framework to compare the Fine-Tuned Model against the Base Model.

| Metric | Base Llama-3 | Fine-Tuned Adapter (Ours) |
| :--- | :--- | :--- |
| **Win Rate** | ~5% | **95%** |
| **Format Adherence** | **FAIL**: Often outputted Python code or meta-descriptions like *"The program takes..."* | **PASS**: Consistently outputted clean commentary tracks. |
| **Linguistic Quality** | **FAIL**: Grammatically broken Turkish (*"maalesef bir fail"*). | **PASS**: Mastered domain-specific register (*"son saniyede gönderiyor"*, *"isabet"*). |

### ⚠️ Identified Limitations
* **The Truncation Bottleneck:** About 50% of outputs were cut off mid-sentence (e.g., *"Motley fa..."*) due to uncalibrated `max_tokens` or EOS token sensitivity during inference.
* **Translation Artifacts:** In complex Turkish sentences, the model sometimes appeared to generate English logic first and translate it, leading to minor structural disruptions.

## 🚀 Installation & Usage

### 1. Prerequisites
* Python 3.10+
* GPU (T4 or better recommended)
* `unsloth`, `xformers`, `torch`

### 2. Quick Start
Since the model weights are hosted on Hugging Face (to avoid GitHub size limits), you must download them separately.

```bash
# 1. Clone the repo
git clone [https://github.com/Ahmet-Yusuf-Ozturk/llama3-basketball-commentator.git](https://github.com/Ahmet-Yusuf-Ozturk/llama3-basketball-commentator.git)
cd llama3-basketball-commentator

# 2. Download Model Weights (from Hugging Face)
# Clone the weights into the 'adapter' folder
git clone [https://huggingface.co/AhmetYusufOzturk/llama3-basketball-commentator](https://huggingface.co/AhmetYusufOzturk/llama3-basketball-commentator) adapter

# 3. Install Dependencies
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

### 3. Inference Script
We provide a script to generate commentary from new JSON data.

```python
from unsloth import FastLanguageModel

# Load Model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
# Load your fine-tuned adapter
model.load_adapter("adapter")

# Generate Commentary
def generate_narration(json_input, lang="Turkish"):
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{json_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    return tokenizer.batch_decode(outputs)[0]
```

## 🔮 Future Work
1.  **Sliding Context Window:** Currently, events are treated in isolation. Future versions will use a history window to build narrative tension (e.g., mentioning a player is "heating up").
2.  **RAG Integration:** Connecting to a real-time roster database to enable specific commentator personas (e.g., **"Ertem Şener Style"**) by retrieving player background stats on the fly.
3.  **Fixing Truncation:** Adjusting `max_tokens` and EOS calibration to ensure full sentence completion.

## 📜 Credits
* **Author:** Ahmet Yusuf Öztürk
* **Course:** DSAI 585 - Generative AI
* **Date:** January 17, 2026

## ⚖️ License
MIT License
