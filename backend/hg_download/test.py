import os

# Define the file name and content
abs_path = os.path.dirname(os.path.abspath(__file__))
file_name = os.path.join(abs_path, "Modelfile")
test = 'sdfsdf'
content = f'''FROM {test}

TEMPLATE """{{ if .System }}
<|start_header_id|>{{ .System }}<|end_header_id|>
{{ end }}
<|start_header_id|>Human:
{{ .Prompt }}<|end_header_id|>
<s>Assistant:
"""

SYSTEM """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
'''

# Create the file and write the content to it
with open(file_name, "w") as file:
    file.write(content)

# Delete the file
os.remove(file_name)
