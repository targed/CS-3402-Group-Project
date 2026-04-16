import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyCKAUmIXLt_WPfIkMHfwBn9HcibV57eREI"))

for model in client.models.list():
    print(model.name)