import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
assert key, "GOOGLE_API_KEY not found in .env"

genai.configure(api_key=key)

# show what this key can actually use
print("Available models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print("  ", m.name)

model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
model = genai.GenerativeModel(model_name)
print(f"\nTesting {model_name} ...")
print(model.generate_content("Say OK if you can hear me.").text)

import duckdb, pandas, langgraph
print("duckdb", duckdb.__version__, "| pandas", pandas.__version__)
print("Setup complete.")