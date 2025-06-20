import os
from google.generativeai import GenerativeModel
from config import GEMINI_API_KEY

class BaseAgent:
    def __init__(self, model_name="gemini-2.5-flash-preview-05-20", temperature=0.8, top_p=0.95):
        # 🔐 Validate API key before configuring the client
        if not GEMINI_API_KEY or not isinstance(GEMINI_API_KEY, str):
            raise ValueError("Missing or invalid GEMINI_API_KEY. Check your configuration.")

        # 🌐 Set the API key globally for the Google GenAI client
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

        self.temperature = temperature
        self.top_p = top_p

        # 🧠 Initialize the selected Gemini model
        self.model = GenerativeModel(model_name)

    def run(self, prompt: str) -> str:
        # 🧾 Ensure prompt is a non-empty string
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string.")

        # 📡 Send the prompt to the Gemini model with the configured generation parameters
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

        # 🚫 Check if the model returned a valid text response
        if not response or not response.text:
            raise RuntimeError("The model returned an empty response.")

        return response.text
