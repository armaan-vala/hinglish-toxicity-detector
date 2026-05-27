import torch
import re
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_MODEL = "armaan-vala/hinglish-toxic-merged"

print("Loading merged model...")
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL, torch_dtype=torch.float16, device_map="auto")
else:
    model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL, torch_dtype=torch.float16, device_map="cpu")

model.eval()
print("Model ready!")


def predict(text):
    if not text or not text.strip():
        return "Enter a comment first.", ""

    prompt = f"""### Instruction:
Classify this Hinglish comment as 'offensive' or 'not_offensive'.

### Input:
{text.strip()}

### Response:
"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = result.split("### Response:")[-1].strip().lower()

    if re.search(r"not[_\s-]?offen", raw):
        return "Not Offensive", "This comment appears to be safe."
    elif re.search(r"offen", raw):
        return "Offensive", "This comment has been flagged as potentially offensive."
    else:
        return "Uncertain", f"Raw output: {raw[:100]}"


css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    background: #1a1410 !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 620px !important;
    margin: 0 auto !important;
}
.badge-row { text-align: center; margin-bottom: 8px; }
.badge {
    display: inline-block;
    background: rgba(212, 160, 90, 0.12);
    border: 1px solid rgba(212, 160, 90, 0.25);
    color: #d4a05a; font-size: 11px; font-weight: 600;
    padding: 4px 14px; border-radius: 20px;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.main-title {
    text-align: center; color: #f5ede4;
    font-size: 28px; font-weight: 700;
    margin-bottom: 4px; letter-spacing: -0.5px;
}
.sub-title {
    text-align: center; color: #a89888;
    font-size: 14px; margin-bottom: 24px;
}
.tech-footer {
    text-align: center; margin-top: 24px; font-size: 12px; color: #4a3f34;
}
.tech-tags {
    display: flex; justify-content: center; gap: 12px; margin-top: 8px;
}
.tech-tag {
    font-size: 11px; color: #5a4f44; padding: 4px 10px;
    background: rgba(212, 160, 90, 0.05); border-radius: 6px;
}

textarea, input[type="text"] {
    background: #231e18 !important;
    border: 1px solid #3a3028 !important;
    color: #f5ede4 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #d4a05a !important;
    box-shadow: 0 0 0 3px rgba(212, 160, 90, 0.1) !important;
}
label { color: #a89888 !important; }

button.primary {
    background: linear-gradient(135deg, #d4a05a, #c48a3a) !important;
    border: none !important; color: #1a1410 !important;
    font-weight: 600 !important; border-radius: 10px !important;
    font-size: 15px !important; padding: 12px !important;
}
button.primary:hover {
    box-shadow: 0 6px 20px rgba(212, 160, 90, 0.3) !important;
}

.gr-examples button, .gr-sample-btn {
    background: #1a1410 !important;
    border: 1px solid #3a3028 !important;
    color: #a89888 !important; border-radius: 8px !important;
}
.gr-examples button:hover, .gr-sample-btn:hover {
    border-color: #d4a05a !important; color: #d4a05a !important;
}

footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as demo:
    gr.HTML('<div class="badge-row"><span class="badge">QLoRA Fine-Tuned</span></div>')
    gr.HTML('<div class="main-title">Hinglish Toxicity Detector</div>')
    gr.HTML('<div class="sub-title">Detect offensive content in Hindi-English mixed text</div>')

    input_text = gr.Textbox(
        label="Enter Hinglish Comment",
        placeholder="Type a Hinglish comment here...",
        lines=3,
    )
    btn = gr.Button("Analyze Comment", variant="primary")

    verdict = gr.Textbox(label="Verdict", interactive=False)
    detail = gr.Textbox(label="Details", interactive=False)

    gr.Examples(
        examples=[
            ["Bhai tu kamaal ka insaan hai"],
            ["Abe saale gadhe kahika nikal yahan se"],
            ["Yaar ye movie bohot achhi thi must watch"],
            ["Tujhe toh jootey maarne chahiye bewakoof"],
            ["Bahut achha kaam kiya bhai, proud of you"],
        ],
        inputs=input_text,
    )

    gr.HTML("""
    <div class="tech-footer">
        <p>Built with Phi-3.5 Mini + QLoRA</p>
        <div class="tech-tags">
            <span class="tech-tag">Transformers</span>
            <span class="tech-tag">PEFT</span>
            <span class="tech-tag">Gradio</span>
        </div>
    </div>
    """)

    btn.click(fn=predict, inputs=input_text, outputs=[verdict, detail])
    input_text.submit(fn=predict, inputs=input_text, outputs=[verdict, detail])

demo.launch()
