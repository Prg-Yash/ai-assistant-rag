# gemini_llm.py
import os
from typing import Any
from pathway.xpacks.llm import llms

# lazy-import google generative ai to fail only at runtime if missing
def _get_genai():
    import google.generativeai as genai
    return genai

class GeminiChat(llms.LLM):
    """
    Minimal wrapper so Pathway's SummaryQuestionAnswerer can call Gemini for generation.
    It implements a 'complete(prompt: str) -> str' method which Pathway LLMs expect.
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None):
        super().__init__()
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment (.env).")
        self.genai = _get_genai()
        # configure SDK
        try:
            self.genai.configure(api_key=self.api_key)
        except Exception:
            # some versions use a different configure name; ignore if it fails
            pass

    def complete(self, prompt: str) -> str:
        """
        Called by Pathway to generate text. We try common Gemini SDK methods.
        """
        try:
            # Preferred pattern used earlier in examples:
            model = self.genai.GenerativeModel(self.model_name)
            # generate_content returns an object with .text in some versions
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None)
            if text:
                return text
            # some return object in a 'candidates' or 'output' field
            if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                return getattr(resp.candidates[0], "content", str(resp))
            return str(resp)
        except Exception:
            # fallback attempt to call another likely function
            try:
                resp = self.genai.generate_text(model=self.model_name, prompt=prompt)
                return getattr(resp, "text", str(resp))
            except Exception as e:
                # Final fallback: raise descriptive error so you can debug
                raise RuntimeError(f"Gemini generation failed: {e}")
