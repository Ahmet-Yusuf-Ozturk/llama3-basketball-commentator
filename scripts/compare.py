import torch
import pandas as pd
from unsloth import FastLanguageModel

# --- CONFIGURATION ---
ADAPTER_PATH = "llama3_basketball_adapter" # Or path to your Drive folder
base_model_name = "unsloth/llama-3-8b-bnb-4bit"

# --- 1. DEFINE THE 20 TEST CASES ---
# We mix English (EN) and Turkish (TR) and different event types
test_cases = [
    # CLUTCH / BUZZER BEATERS
    {"lang": "English", "json": '{"time": "00:01", "team": "Fenerbahçe", "player": "Nigel Hayes-Davis", "action": "3pt_shot", "result": "make"}'},
    {"lang": "Turkish", "json": '{"time": "00:00", "team": "Anadolu Efes", "player": "Shane Larkin", "action": "drive_layup", "result": "make"}'},
    
    # REGULAR PLAY (EARLY GAME)
    {"lang": "English", "json": '{"time": "09:45", "team": "Fenerbahçe", "player": "Scottie Wilbekin", "action": "turnover", "result": "steal"}'},
    {"lang": "Turkish", "json": '{"time": "08:12", "team": "Fenerbahçe", "player": "Sertaç Şanlı", "action": "block", "result": "success"}'},
    
    # MISSES
    {"lang": "English", "json": '{"time": "04:20", "team": "Anadolu Efes", "player": "Will Clyburn", "action": "3pt_shot", "result": "miss"}'},
    {"lang": "Turkish", "json": '{"time": "02:15", "team": "Fenerbahçe", "player": "Marko Guduric", "action": "free_throw", "result": "miss"}'},
    
    # ... (Add 14 more to reach 20) ...
]

# --- 2. FUNCTION TO RUN INFERENCE ---
def run_inference(model, tokenizer, items, model_type="Fine-Tuned"):
    results = []
    FastLanguageModel.for_inference(model)
    
    for item in items:
        # Prompt Construction
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a basketball commentator. Language: {item['lang']}.<|eot_id|><|start_header_id|>user<|end_header_id|>

Input Data: {item['json']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
        
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, 
            use_cache=True, 
            temperature=0.6, 
            do_sample=True
        )
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # Clean Output
        if "assistant" in decoded: decoded = decoded.split("assistant")[0]
        decoded = decoded.strip()
        
        results.append({
            "Model": model_type,
            "Language": item['lang'],
            "Input": item['json'],
            "Output": decoded
        })
    return results

# --- 3. RUN TEST ON BASE MODEL (CONTROL GROUP) ---
print("🔵 Loading BASE Model (No Adapter)...")
model_base, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

print("Running Inference on Base Model...")
base_results = run_inference(model_base, tokenizer, test_cases, model_type="Base Llama-3")

# Free up memory to load the next model
del model_base
torch.cuda.empty_cache()

# --- 4. RUN TEST ON FINE-TUNED MODEL (EXPERIMENTAL GROUP) ---
print("\n🟢 Loading FINE-TUNED Model (With Adapter)...")
model_ft, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name, # Load base first
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
# NOW load the adapter on top
model_ft.load_adapter(ADAPTER_PATH) 

print("Running Inference on Fine-Tuned Model...")
ft_results = run_inference(model_ft, tokenizer, test_cases, model_type="Fine-Tuned")

# --- 5. SAVE COMPARISON ---
all_results = base_results + ft_results
df = pd.DataFrame(all_results)
df.to_csv("model_comparison_results.csv", index=False)
print("\n✅ Testing Complete! Results saved to 'model_comparison_results.csv'")

# Display first few rows
print(df[["Model", "Language", "Output"]].head())