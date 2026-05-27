import os
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(title="Hinglish Toxic Comment Detector")

model = None
tokenizer = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
BASE_MODEL = "unsloth/Phi-3.5-mini-instruct"


def load_model():
    global model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
        )

    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.eval()


@app.on_event("startup")
async def startup():
    load_model()


class CommentRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: CommentRequest):
    prompt = f"""### Instruction:
Classify this Hinglish comment as 'offensive' or 'not_offensive'.

### Input:
{req.text}

### Response:
"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.1, do_sample=False)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = result.split("### Response:")[-1].strip().lower()

    if "not_offensive" in prediction:
        label = "not_offensive"
        confidence = "Safe"
    elif "offensive" in prediction:
        label = "offensive"
        confidence = "Toxic"
    else:
        label = prediction
        confidence = "Unknown"

    return {"label": label, "confidence": confidence, "text": req.text}


@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hinglish Toxicity Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            background: #1a1410;
            color: #f5ede4;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 580px;
            animation: fadeUp 0.6s ease-out;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .header {
            text-align: center;
            margin-bottom: 36px;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 700;
            color: #f5ede4;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }

        .header p {
            font-size: 14px;
            color: #a89888;
        }

        .badge {
            display: inline-block;
            background: rgba(212, 160, 90, 0.12);
            border: 1px solid rgba(212, 160, 90, 0.25);
            color: #d4a05a;
            font-size: 11px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            margin-bottom: 16px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .card {
            background: #231e18;
            border: 1px solid #3a3028;
            border-radius: 16px;
            padding: 32px;
            transition: border-color 0.3s ease;
        }

        .card:focus-within {
            border-color: #d4a05a;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: #1a1410;
            border: 1px solid #3a3028;
            border-radius: 10px;
            padding: 16px;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            color: #f5ede4;
            resize: vertical;
            outline: none;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }

        textarea::placeholder { color: #6b5d4f; }

        textarea:focus {
            border-color: #d4a05a;
            box-shadow: 0 0 0 3px rgba(212, 160, 90, 0.1);
        }

        .btn {
            width: 100%;
            padding: 14px;
            margin-top: 16px;
            background: linear-gradient(135deg, #d4a05a, #c48a3a);
            border: none;
            border-radius: 10px;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            font-weight: 600;
            color: #1a1410;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(212, 160, 90, 0.3);
        }

        .btn:active { transform: translateY(0); }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(26, 20, 16, 0.3);
            border-top-color: #1a1410;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            margin: 0 auto;
        }

        .btn.loading .btn-text { display: none; }
        .btn.loading .spinner { display: inline-block; }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            animation: slideIn 0.4s ease-out;
            display: none;
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result.safe {
            background: rgba(76, 145, 95, 0.1);
            border: 1px solid rgba(76, 145, 95, 0.3);
        }

        .result.toxic {
            background: rgba(190, 70, 60, 0.1);
            border: 1px solid rgba(190, 70, 60, 0.3);
        }

        .result-icon {
            font-size: 32px;
            margin-bottom: 8px;
        }

        .result-label {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .result.safe .result-label { color: #4c915f; }
        .result.toxic .result-label { color: #be463c; }

        .result-detail {
            font-size: 13px;
            color: #a89888;
        }

        .examples {
            margin-top: 24px;
        }

        .examples-title {
            font-size: 12px;
            font-weight: 600;
            color: #6b5d4f;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        .example-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .chip {
            background: #1a1410;
            border: 1px solid #3a3028;
            border-radius: 8px;
            padding: 8px 14px;
            font-size: 13px;
            color: #a89888;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chip:hover {
            border-color: #d4a05a;
            color: #d4a05a;
            background: rgba(212, 160, 90, 0.05);
        }

        .footer {
            text-align: center;
            margin-top: 28px;
            font-size: 12px;
            color: #4a3f34;
        }

        .tech-stack {
            display: flex;
            justify-content: center;
            gap: 16px;
            margin-top: 8px;
        }

        .tech-item {
            font-size: 11px;
            color: #5a4f44;
            padding: 4px 10px;
            background: rgba(212, 160, 90, 0.05);
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge">QLoRA Fine-Tuned</div>
            <h1>Hinglish Toxicity Detector</h1>
            <p>Detect offensive content in Hindi-English mixed text</p>
        </div>

        <div class="card">
            <textarea id="input" placeholder="Type a Hinglish comment here..."></textarea>
            <button class="btn" id="analyzeBtn" onclick="analyze()">
                <span class="btn-text">Analyze Comment</span>
                <div class="spinner"></div>
            </button>

            <div class="result" id="result">
                <div class="result-icon" id="resultIcon"></div>
                <div class="result-label" id="resultLabel"></div>
                <div class="result-detail" id="resultDetail"></div>
            </div>

            <div class="examples">
                <div class="examples-title">Try an example</div>
                <div class="example-chips">
                    <div class="chip" onclick="tryExample(this)">Bhai tu kamaal ka insaan hai</div>
                    <div class="chip" onclick="tryExample(this)">Abe chal hat bewakoof</div>
                    <div class="chip" onclick="tryExample(this)">Yaar ye movie must watch hai</div>
                </div>
            </div>
        </div>

        <div class="footer">
            <p>Built with Phi-3.5 Mini + QLoRA</p>
            <div class="tech-stack">
                <span class="tech-item">FastAPI</span>
                <span class="tech-item">Transformers</span>
                <span class="tech-item">PEFT</span>
            </div>
        </div>
    </div>

    <script>
        const input = document.getElementById('input');
        const btn = document.getElementById('analyzeBtn');
        const result = document.getElementById('result');

        function tryExample(el) {
            input.value = el.textContent;
            input.focus();
        }

        async function analyze() {
            const text = input.value.trim();
            if (!text) return;

            btn.classList.add('loading');
            btn.disabled = true;
            result.style.display = 'none';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();

                const isSafe = data.label === 'not_offensive';
                result.className = 'result ' + (isSafe ? 'safe' : 'toxic');
                document.getElementById('resultIcon').textContent = isSafe ? '\\u2714' : '\\u2718';
                document.getElementById('resultLabel').textContent = isSafe ? 'Not Offensive' : 'Offensive';
                document.getElementById('resultDetail').textContent =
                    isSafe ? 'This comment appears to be safe.'
                           : 'This comment has been flagged as potentially offensive.';
                result.style.display = 'block';
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        }

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                analyze();
            }
        });
    </script>
</body>
</html>
"""
