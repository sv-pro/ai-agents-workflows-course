import os
import google.generativeai as genai

# --- Configuration ---
# It's recommended to set your API key as an environment variable for security.
# You can also pass it directly to genai.configure(api_key="YOUR_API_KEY")
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=api_key)

# --- Model Initialization ---
# Create an instance of the generative model.
# You can choose different models, for example, 'gemini-pro'.
available_models = [m.name for m in genai.list_models()]
# print(available_models)


desired_model = 'models/gemini-2.5-pro'
if desired_model not in available_models:
    raise ValueError(f"Model {desired_model} is not available. Use one of the following: {available_models}")

model = genai.GenerativeModel(desired_model)

# --- Generate Content ---
# The prompt for the model.
prompt = "Hello Google AI, please introduce yourself in a short paragraph."

try:
    # Generate content based on the prompt.
    response = model.generate_content(prompt)

    # --- Print the Response ---
    # The generated text is in the 'text' attribute of the response.
    print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")