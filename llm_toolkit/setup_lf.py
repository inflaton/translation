import os, json, sys, yaml
from dotenv import find_dotenv, load_dotenv

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.llm_utils import *
from llm_toolkit.translation_utils import *

org_name = os.getenv("ORG_NAME")
model_name = os.getenv("MODEL_NAME")
chat_template = os.getenv("CHAT_TEMPLATE")
filename = os.getenv("YAML")
data_path = os.getenv("DATA_PATH")

print(org_name, model_name, chat_template, filename, data_path)

if not filename:
    print("Error: Environment variable YAML not set")
    sys.exit(1)

if not os.path.exists(filename):
    print(f"Error: File {filename} not found")
    sys.exit(1)

file = open(filename)
yaml_content = file.read()
file.close()

keys = ["ORG_NAME", "MODEL_NAME", "CHAT_TEMPLATE"]
for key in keys:
    yaml_content = yaml_content.replace(key, os.getenv(key))

# print(f"YAML content:\n{yaml_content}")
parts = filename.split("/")
parts[-1] = "models"
parts.append(f"{os.getenv('MODEL_NAME')}.yaml")
filename = "/".join(parts)
print(f"Writing to {filename}")

# Create the parent directory if it doesn't exist
os.makedirs(os.path.dirname(filename), exist_ok=True)

file = open(filename, "w")
file.write(yaml_content)
file.close()

y = yaml.safe_load(open(filename))
print(f"{filename}:\n", json.dumps(y, indent=2))

dataset = load_alpaca_data(data_path)
print_row_details(dataset, [0, -1])
