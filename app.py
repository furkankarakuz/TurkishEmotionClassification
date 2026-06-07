import gradio as gr
from transformers import pipeline

MODEL_ID = "furkankarakuz/turkish-emotion-classifier"

# Loaded once from the Hub at startup. top_k=None -> scores for all classes.
pipe = pipeline("text-classification", model=MODEL_ID, top_k=None)

EMOJI = {"anger": "😠", "disgust": "🤢", "fear": "😨", "joy": "😊", "sadness": "😢", "surprise": "😲"}


def classify(text: str) -> dict:
    """Take raw text and return a probability for each emotion (for gr.Label)."""
    if not text or not text.strip():
        return {}
    scores = pipe(text)[0]  # [{'label': 'joy', 'score': 0.98}, ...]
    return {f'{EMOJI.get(d["label"], "")} {d["label"]}': d["score"] for d in scores}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=3, label="Text", placeholder="Type a Turkish sentence..."),
    outputs=gr.Label(num_top_classes=6, label="Emotion"),
    title="Turkish Emotion Classifier",
    description=("Based on dbmdz/bert-base-turkish-cased. Six emotion classes: anger, disgust, fear, joy, sadness, surprise. Pass raw text — no preprocessing needed."),
    examples=[["Bu film gerçekten harikaydı, çok keyif aldım!"], ["Bugün başıma gelenler beni çok üzdü."], ["Bu duruma gerçekten çok sinirlendim."], ["Karşımda onu görünce çok şaşırdım!"]],
)

if __name__ == "__main__":
    demo.launch()
