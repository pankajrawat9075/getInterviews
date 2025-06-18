import gradio as gr
from utilities import gradio_utility

model_selection_ui = gradio_utility.create_model_selection_ui()
resume_ui = gradio_utility.create_resume_ui()

with gr.Blocks() as demo:
    model_selection_ui.render()

with demo.route("Second Page"):
    resume_ui.render()

if __name__ == "__main__":
    demo.launch()

