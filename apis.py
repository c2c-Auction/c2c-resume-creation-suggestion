# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


import numpy as np


def groq(input, api_key):
    chat = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=api_key  # Use the API key provided from the frontend
    )
    return chat.invoke(input)

def final(Input, api_key):
    print("calling from groq")
    return groq(Input, api_key).content


