import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)

ui = gr.Interface(fn=lambda x: x, inputs="text", outputs="text")

if __name__ == "__main__":
    ui.launch()
