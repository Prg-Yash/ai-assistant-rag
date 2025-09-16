# app.py
import logging
import os
from dotenv import load_dotenv

import pathway as pw
from pathway.xpacks.llm.question_answering import SummaryQuestionAnswerer
from pathway.xpacks.llm.servers import QASummaryRestServer
from pydantic import BaseModel, ConfigDict, InstanceOf

from gemini_llm import GeminiChat

# uncomment if you have a Pathway Scale license:
# pw.set_license_key("your-license-key")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
load_dotenv()

class App(BaseModel):
    question_answerer: InstanceOf[SummaryQuestionAnswerer]
    host: str = "0.0.0.0"
    port: int = 8000
    with_cache: bool = True
    terminate_on_error: bool = False
    model_config = ConfigDict(extra="forbid")


if __name__ == "__main__":
    # Load YAML file into a config dict
    with open("app.yaml") as f:
        config = pw.load_yaml(f)

    # Create the Pydantic App using Pathway's YAML -> builds SummaryQuestionAnswerer
    app = App(**config)

    # Replace the internal LLM with Gemini wrapper
    try:
        gem = GeminiChat(model="gemini-1.5-flash")
        app.question_answerer.llm = gem
        print("Replaced LLM with GeminiChat.")
    except Exception as e:
        print("WARNING: Could not instantiate GeminiChat:", e)
        print("Continuing with the LLM defined in app.yaml (if any).")

    # Start REST server that exposes an endpoint (e.g., /query)
    server = QASummaryRestServer(app.host, app.port, app.question_answerer)
    server.run(
        with_cache=app.with_cache,
        terminate_on_error=app.terminate_on_error,
        cache_backend=pw.persistence.Backend.filesystem("Cache"),
    )
