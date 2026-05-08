# Libraries
import os
from dotenv import load_dotenv
from openai import OpenAI

# API_KEY
load_dotenv()
foundry = os.getenv("FOUNDRY_KEY")

# ENDPOINT FICA FIXO CADU!
# AQUI VOCE SO TROCA MODEL_NAME E DEPLOYMENT_NAME! ELES SEMPRE SÃO IGUAIS!
endpoint = "https://bgai-foundry.cognitiveservices.azure.com/openai/v1/"
model_name = "Kimi-K2.5" #Kimi-K2.5 // grok-4-20-non-reasoning // gpt-5.4-mini
deployment_name = "Kimi-K2.5" #Kimi-K2.5 // grok-4-20-non-reasoning // gpt-5.4-mini
api_key = foundry

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

completion = client.chat.completions.create(
    model=deployment_name,
    messages=[{"role": "user", "content": "Explain RUST to me!"}],
)

print(completion.choices[0].message)