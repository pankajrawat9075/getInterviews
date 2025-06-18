import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv()

import nest_asyncio
nest_asyncio.apply()
# Mapping of companies to their models
company_models = {
    "OpenAI": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    "Mistral": ["mistral-medium", "mistral-small", "mistral-tiny"],
    "Google": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.0-pro"]
}


# Function to update the model dropdown based on selected company
def update_models(company):
    print(f"inside update_models: {company}")
    return gr.update(choices=company_models[company], value=company_models[company][0])

# On submit, handle the selection and API key
def handle_submit(company, model, api_key):
    if not api_key:
        return "❌ Please enter an API key."
    os.environ["MODEL_COMPANY"] = company
    os.environ["MODEL"] = model
    os.environ["API_KEY"] = api_key

    return f"✅ Selected {company} - {model} with key: {api_key[:4]}... (received)"


def create_model_selection_ui():
    with gr.Blocks() as model_selection_ui:
        gr.Markdown("## 👋 Welcome to get Interviews! Select your model and enter API key")

        company_dd = gr.Dropdown(
            label="Select Model Company",
            choices=list(company_models.keys()),
            value="OpenAI"
        )

        model_dd = gr.Dropdown(
            label="Select Model",
            choices=company_models["OpenAI"],
            value=company_models["OpenAI"][0]
        )

        api_key_input = gr.Textbox(
            label="Enter API Key",
            type="password",
            placeholder="sk-..."
        )

        output = gr.Textbox(label="Status")

        print(company_dd, model_dd)
        print(company_dd.value, model_dd.value)
        # When company changes, update model dropdown
        company_dd.change(fn=update_models, inputs=company_dd, outputs=model_dd)

        # Handle final submit
        submit_btn = gr.Button("Submit")
        submit_btn.click(
            fn=handle_submit,
            inputs=[company_dd, model_dd, api_key_input],
            outputs=output
        )
    
    return model_selection_ui

def create_resume_ui():
    with gr.Blocks() as resume_ui:
        gr.Markdown("## 👋 Welcome to upload resume page.")

    return resume_ui