!pip install --upgrade transformers gradio

from transformers import pipeline
import gradio as gr

# Load FLAN-T5 model via instruction-style pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-base")

# Chat function (single-turn)
def chatbot(message, history):
    try:
        result = pipe(message, max_new_tokens=100)[0]["generated_text"]
        return result.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# Launch Gradio chatbot
gr.ChatInterface(
    fn=chatbot,
    examples=[
        "Explain recursion in simple terms.",
        "Translate this to French: Hello, how are you?",
        "Summarize: AI is transforming the future of work."
    ],
    title="FLAN-T5 Instruction Bot"
).launch(share=True, debug=True)
