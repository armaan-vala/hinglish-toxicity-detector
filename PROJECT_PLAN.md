# Hinglish Toxic Comment Detector — Complete Project Plan

## Project Summary
Pre-trained LLM ko QLoRA se fine-tune karke Hinglish toxic comments detect karna.
Dataset: `hinglish_toxic_dataset.csv` (30,875 rows — offensive/not_offensive)

---

## IMPORTANT: 2 Jagah Kaam Hoga

Is project me kaam DO jagah hoga. Ye samajhna zaroori hai:

### Part 1: Google Colab (Browser me — Training)
- **Kyu:** Fine-tuning ke liye GPU chahiye. Tumhare laptop me GPU nahi hai (ya kam hai). Colab FREE GPU deta hai.
- **Kya hoga:** Model load, quantize, LoRA lagao, train, save
- **Output:** Fine-tuned model files download hongi

### Part 2: Local / Claude Code (VS Code me — Deployment)
- **Kyu:** FastAPI app banana, UI banana, Render pe deploy karna — ye sab local me hoga
- **Kya hoga:** Colab se aaye model files ko load karke FastAPI app banana aur deploy karna

```
Google Colab (Training)          Local (Deployment)
========================         ========================
Dataset upload karo              FastAPI app banao
Model load + quantize            Colab se model files lao
LoRA fine-tune karo              UI banao
Model save karo                  Render pe deploy karo
Model files download karo   →    Model load karke predict karo
```

---

## STEP-BY-STEP EXECUTION PLAN

### PHASE 1: Google Colab — Model Fine-Tuning
**Location:** Google Colab (browser me)
**Time:** ~1-2 hours

#### Step 1.1: Colab Setup
- Google Colab open karo: https://colab.research.google.com
- New Notebook banao
- Runtime → Change runtime type → GPU select karo (T4 free milega)
- Dataset upload karo (hinglish_toxic_dataset.csv)

#### Step 1.2: Libraries Install
```python
!pip install -q unsloth peft transformers trl datasets bitsandbytes accelerate
```

#### Step 1.3: Model Load + 4-bit Quantization
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=512,
    load_in_4bit=True,       # 4-bit quantization — 16GB model → 4GB me fit
)
```
**Concept — Quantization:**
- Normal model 16-bit me hota hai (bahut bada, GPU me fit nahi hota)
- 4-bit me compress karte hain (size 4x chhota, thodi accuracy loss but worth it)
- Isse free Colab GPU (T4 = 15GB RAM) me bhi bade models chal jaate hain

#### Step 1.4: LoRA Adapters Add
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                    # LoRA rank — kitna bada adapter (16 = balanced)
    lora_alpha=32,           # scaling factor
    lora_dropout=0.05,       # overfitting se bachao
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # kaunse layers train karne hain
)
```
**Concept — LoRA (Low-Rank Adaptation):**
- Full model me crores of parameters hain — sab train karna expensive hai
- LoRA sirf chhote "adapter" layers add karta hai (1-2% parameters)
- Baaki sab FREEZE rehta hai — sirf adapters train hote hain
- Result almost utna hi acha hota hai jitna full fine-tuning

**Concept — QLoRA:**
- Q = Quantized (4-bit) + LoRA = QLoRA
- Matlab: pehle model ko 4-bit me compress karo, phir uske upar LoRA adapters lagao
- Isse chhoti GPU me bhi bade models fine-tune ho jaate hain

#### Step 1.5: Dataset Prepare
```python
from datasets import Dataset
import pandas as pd

df = pd.read_csv("hinglish_toxic_dataset.csv")

# Format data for instruction tuning
def format_prompt(row):
    return {
        "text": f"""### Instruction:
Classify this Hinglish comment as 'offensive' or 'not_offensive'.

### Input:
{row['text']}

### Response:
{row['label']}"""
    }

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_prompt)
dataset = dataset.train_test_split(test_size=0.1)
```

#### Step 1.6: Training
```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    max_seq_length=512,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
    ),
)

trainer.train()
```

#### Step 1.7: Save Model + Download
```python
# Save LoRA adapters
model.save_pretrained("toxic-detector-lora")
tokenizer.save_pretrained("toxic-detector-lora")

# Zip karke download
!zip -r toxic-detector-lora.zip toxic-detector-lora/
from google.colab import files
files.download("toxic-detector-lora.zip")
```

**Output:** `toxic-detector-lora.zip` download hoga tumhare laptop me

---

### PHASE 2: Local — FastAPI Deployment
**Location:** Local machine (VS Code / Claude Code)
**Time:** ~30 mins

#### Step 2.1: Project Setup
New folder banao (e.g., `hinglish-toxic-detector/`)
```
hinglish-toxic-detector/
├── app.py                  # FastAPI web app
├── requirements.txt        # Dependencies
├── model/                  # Colab se downloaded model files yahan paste karo
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
└── README.md
```

#### Step 2.2: FastAPI App
- Model load karo (quantized + LoRA adapters)
- Predict function banao
- Web UI banao (HTML — same style jaisa sentiment analysis me tha)
- API endpoint banao

#### Step 2.3: Deploy to Render
- GitHub pe push karo
- Render pe new Web Service banao
- Same process jaisa sentiment analysis me kiya tha

---

## KEY CONCEPTS SUMMARY (Interview ke liye)

### Quantization
- Model ke weights ko kam bits me store karna
- 32-bit → 16-bit → 8-bit → 4-bit
- Jitne kam bits, utna chhota model, utni kam RAM chahiye
- Tradeoff: thodi accuracy kam hoti hai par bahut RAM bachti hai

### LoRA (Low-Rank Adaptation)
- Full model freeze karo
- Chhote trainable "adapter" layers add karo
- Sirf 1-2% parameters train hote hain
- 100x kam memory use hoti hai vs full fine-tuning
- Results almost utne hi achhe

### QLoRA
- Quantization + LoRA combined
- Pehle 4-bit me compress → phir LoRA adapters add → train
- Isse 7B parameter model bhi free GPU pe train ho jata hai

### Fine-Tuning vs Training from Scratch
- Training from scratch: model ko zero se sab seekhana (bahut data + time chahiye)
- Fine-tuning: model already jaanta hai language, bas apna specific task seekhana hai
- Jaise: ek experienced chef ko sirf nayi recipe seekhani hai, cooking basics nahi

### Instruction Tuning
- Model ko "instruction → response" format me data dena
- "Classify this comment" → "offensive"
- Isse model samajhta hai ki kya karna hai

---

## GOOGLE COLAB GUIDE (For First-Time Users)

### Colab kya hai?
- Google ka free Jupyter Notebook — browser me chalta hai
- Free GPU milta hai (T4 — 15GB RAM)
- Google Drive se connected hai

### Kaise open kare?
1. Browser me jaao: https://colab.research.google.com
2. Google account se sign in karo
3. "New Notebook" click karo
4. Upar me: Runtime → Change Runtime Type → GPU → Save

### Kaise use kare?
- Har cell me code likho
- Shift+Enter se cell run karo
- Cell ke neeche output dikhega
- `!` se terminal commands run hote hain (e.g., `!pip install torch`)

### Dataset kaise upload kare?
Option 1 (Direct upload):
- Left sidebar me folder icon click karo
- Upload button click karo
- `hinglish_toxic_dataset.csv` select karo

Option 2 (Google Drive se):
```python
from google.colab import drive
drive.mount('/content/drive')
# Ab file path hoga: /content/drive/MyDrive/hinglish_toxic_dataset.csv
```

### Model kaise download kare?
Training ke baad model files Colab me save hongi. Download karne ke liye:
```python
from google.colab import files
files.download("toxic-detector-lora.zip")
```
Ye file tumhare laptop ke Downloads folder me aa jayegi.

### Important Tips:
- Colab session 12 hours baad disconnect hota hai (free tier)
- Tab band mat karo training ke dauran
- Training complete hote hi model download kar lo
- Har baar reconnect karne pe libraries dobara install karni padengi

---

## WORKFLOW SUMMARY

```
Step 1: Colab open karo, GPU select karo
Step 2: Libraries install karo
Step 3: Dataset upload karo (hinglish_toxic_dataset.csv)
Step 4: Model load karo (4-bit quantized)
Step 5: LoRA adapters lagao
Step 6: Dataset format karo (instruction tuning style)
Step 7: Train karo (3 epochs, ~30-60 min)
Step 8: Model save + download karo
Step 9: (Local) New project folder banao, model files paste karo
Step 10: (Local) FastAPI app banao
Step 11: (Local) GitHub push + Render deploy
Step 12: DONE — live URL share karo
```
