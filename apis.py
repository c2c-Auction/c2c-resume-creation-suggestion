# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

groq_keys = ["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","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","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","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","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","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"]
gemmini_api_key = ["AIzaSyD3kNDlEpRsF-Mb14oQfZNaqPF6ECnvKrA","AIzaSyDS1oMtsTh91nRaKYE97O5f8VZ10VGJkwY","AIzaSyBuy0CZ3j7nO_slJwOqWG_DOe__T3Jo4lw","AIzaSyDckSoZM8Fagz3aOlE2EBybI_a8ruIOhcE","AIzaSyDckSoZM8Fagz3aOlE2EBybI_a8ruIOhcE","AIzaSyBiy03reDB-TzwGfRKiXp7ihjhf1C-95-4"]
#542,eurotech ,y3,c2c,y1sh,billionare

import numpy as np
def keys():
    number = np.random.randint(len(groq_keys))
    models = np.random.randint(0,2)
    return number,models
# def gemini(input,key):
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest",google_api_key=key, temperature=0)
#     return llm.invoke(input)

def groq(input,key):
    chat = ChatGroq(
        temperature=0,
        model="llama3-70b-8192",
        api_key=key
    )
    
    return chat.invoke(input)
def final(Input):
       

    print("calling from groq")
    # return groq(Input,groq_keys[keys()[0]])
    return groq(Input,groq_keys[keys()[0]]).content


