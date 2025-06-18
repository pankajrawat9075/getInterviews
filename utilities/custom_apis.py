from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os
load_dotenv()




def get_llm():
    company = os.environ["MODEL_COMPANY"]
    model = os.environ["MODEL"]
    api_key = os.environ["API_KEY"]

    if company == "OpenAI":
        os.environ["OPENAI_API_KEY"] = api_key

        llm = ChatOpenAI(
            model=model,
            temperature=0.0,
        )

        return llm

    elif company == "Anthropic":
        os.environ["ANTHROPIC_API_KEY"] = api_key

        llm = ChatAnthropic(
        model_name=model,
        temperature=0.0,
        timeout=100, # Increase for complex tasks
        )

        return llm

    elif company == "Mistral":
        # add it
        pass

    elif company == "Google":
        os.environ["GOOGLE_API_KEY"] = api_key

        llm = ChatGoogleGenerativeAI(model=model)

        return llm

    else:
        print("model company or model")


