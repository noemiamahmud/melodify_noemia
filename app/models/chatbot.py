from openai import OpenAI

client = OpenAI()

def home():
    json_message = None


    completion = client.chat.completions.create(
            model="gpt-3.5-turbo"
        )
    
    
    
