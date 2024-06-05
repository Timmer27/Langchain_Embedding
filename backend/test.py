import requests

# Define the API endpoint
ollama_url = 'http://192.168.1.203:11434/api/generate'

# Example payload (customize as per your requirements)
payload = {
    'model': 'platypus-kor:latest',
    'prompt': 'who are you?',
    'stream': True,
}

try:
    response = requests.post(ollama_url, json=payload)
    print(';response',response)
    response_data = response.text

    print(';response_data',response_data, type(response_data))

    # Extract the generated content
    generated_text = response_data.get('message', {}).get('content', 'Error: No content received')

    print(f"Generated text: {generated_text}")
except requests.RequestException as e:
    print(f"Error: {e}")
