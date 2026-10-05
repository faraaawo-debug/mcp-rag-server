"""Single entry point to the LLMs: Mistral (API), Groq (API) or Ollama (local).
The rest of the code calls chat() without knowing which provider answers."""
import logging
import os
import time

import config

logger = logging.getLogger(__name__)

_clients = {}


def _api_key(variable_name):
    """API keys are read from environment variables, never from the code.
    On Windows, if the variable was not passed to the process (an MCP client only passes a
    restricted list of variables to the server it launches), it is read directly from the
    Windows user environment variables."""
    key = os.environ.get(variable_name)
    if not key and os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as registry:
                key = winreg.QueryValueEx(registry, variable_name)[0]
        except OSError:
            key = None
    if not key:
        raise RuntimeError(f"Environment variable {variable_name} is missing (see the README).")
    return key


def _client(provider):
    if provider not in _clients:
        if provider == "mistral":
            from mistralai.client import Mistral
            _clients[provider] = Mistral(api_key=_api_key("MISTRAL_API_KEY"))
        elif provider == "groq":
            from groq import Groq
            # Retries are handled by chat(), the same way for every provider
            _clients[provider] = Groq(api_key=_api_key("GROQ_API_KEY"), max_retries=0)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    return _clients[provider]


def _call(messages, provider, model, json_mode):
    if provider == "ollama":
        import ollama
        result = ollama.chat(
            model=model, messages=messages,
            format="json" if json_mode else "", options={"temperature": 0},
        )
        return result["message"]["content"]

    response_format = {"type": "json_object"} if json_mode else None
    if provider == "mistral":
        result = _client("mistral").chat.complete(
            model=model, messages=messages, temperature=0, response_format=response_format)
    else:
        result = _client(provider).chat.completions.create(
            model=model, messages=messages, temperature=0, response_format=response_format)
    return result.choices[0].message.content


def chat(messages, provider, model, json_mode=False):
    """Sends a conversation to the LLM (temperature 0) and returns the text of its answer.
    If the API answers "too many requests" (429) or is temporarily unavailable (5xx),
    waits longer and longer (2, 4, 8... s) before trying again."""
    for attempt in range(config.LLM_MAX_ATTEMPTS):
        try:
            return _call(messages, provider, model, json_mode).strip()
        except Exception as error:
            code = getattr(error, "status_code", None)
            temporary = code == 429 or (code is not None and code >= 500)
            if not temporary or attempt == config.LLM_MAX_ATTEMPTS - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning(f"{provider}/{model}: error {code}, retrying in {wait} s")
            time.sleep(wait)
