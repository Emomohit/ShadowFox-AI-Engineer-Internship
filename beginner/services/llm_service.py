"""
LLM Service Layer for StudyPilot AI.
Provides unified, decoupled interface to Google Gemini models or Mock Provider.
Encapsulates API credentials, error handling, rate limiting, and timeouts.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Try importing google.genai (modern SDK)
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from shared.mock_provider import MockLLMProvider


class LLMServiceError(Exception):
    """Custom exception raised when an LLM provider fails."""
    pass


class LLMService:
    """Handles communication with LLM APIs with robust error and fallback handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: str = "gemini",
        use_mock: bool = False
    ):
        load_dotenv()
        self.provider = os.getenv("LLM_PROVIDER", provider).lower()
        self.api_key = api_key if api_key is not None else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")

        # Determine mock status
        if use_mock is True or self.provider == "mock":
            self.use_mock = True
        elif not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            if use_mock is False:
                # User explicitly requested live mode but key is absent
                raise LLMServiceError(
                    "Missing valid Gemini API key. Set GEMINI_API_KEY in your .env file or enter it in the app sidebar."
                )
            self.use_mock = True
        else:
            self.use_mock = False

        self._client = None
        if not self.use_mock:
            if not GENAI_AVAILABLE:
                raise LLMServiceError(
                    "The 'google-genai' SDK is not installed. Please run `pip install google-genai`."
                )
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise LLMServiceError(f"Failed to initialize Gemini Client: {str(e)}")

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """
        Sends a prompt to the configured LLM and returns generated text.
        Raises LLMServiceError on provider failures.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if self.use_mock:
            mock = MockLLMProvider(model_name="mock-model")
            return mock.generate(prompt=prompt, system_instruction=system_instruction)

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
                max_output_tokens=2048,
            )
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            if not response.text:
                raise LLMServiceError("Model returned an empty response. The content may have triggered safety filters.")
            return response.text
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                raise LLMServiceError("API quota or rate limit exceeded. Please wait a moment or check your API billing limits.")
            elif "INVALID_ARGUMENT" in err_msg or "API key not valid" in err_msg:
                raise LLMServiceError("Invalid API key provided. Please verify your GEMINI_API_KEY.")
            raise LLMServiceError(f"LLM API Error: {err_msg}")
