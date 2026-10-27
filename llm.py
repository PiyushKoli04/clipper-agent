"""Pluggable LLM providers (single file).

``OllamaProvider`` (default, fully offline), ``GeminiProvider`` (free API key),
and ``GroqProvider`` (fast cloud inference) share one tiny interface:
``complete(prompt, system) -> str``.

Set ``LLM_FALLBACK`` in ``.env`` to automatically retry on a second provider
when the primary one returns a transient error (429 / 503 / connection failure).

Add a provider by subclassing :class:`LLMProvider` and registering it in ``PROVIDERS``.
"""
from __future__ import annotations

import abc
import time

import requests

import config

log = config.log.getChild("llm")


class LLMError(RuntimeError):
    """Raised when a provider is unreachable, misconfigured or returns an error."""


class LLMProvider(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        """Return the raw model text for ``prompt`` (JSON mode requested where supported)."""


class OllamaProvider(LLMProvider):
    """Local Ollama server (https://ollama.com). Run ``ollama pull llama3.1:8b`` once."""

    name = "ollama"

    def __init__(self, url: str | None = None, model: str | None = None, timeout: int | None = None):
        self.url = (url or config.OLLAMA_URL).rstrip("/")
        self.model = model or config.OLLAMA_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "stream": False, "format": "json",
                "options": {"temperature": temperature, "num_ctx": 16384}}
        try:
            r = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        except requests.ConnectionError as e:
            raise LLMError(f"Cannot reach Ollama at {self.url}. Is it running? (`ollama serve`)") from e
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        if r.status_code == 404:
            raise LLMError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
        if not r.ok:
            raise LLMError(f"Ollama error {r.status_code}: {r.text[:300]}")
        return r.json().get("message", {}).get("content", "")


class GeminiProvider(LLMProvider):
    """Google Gemini via the REST API (free key: https://aistudio.google.com/apikey)."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int | None = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        if not self.api_key:
            raise LLMError("GEMINI_API_KEY is not set (add it to .env)")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body: dict = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        for attempt in range(3):  # free tier rate-limits: back off on 429/503
            try:
                r = requests.post(url, json=body, headers={"x-goog-api-key": self.api_key}, timeout=self.timeout)
            except requests.RequestException as e:
                raise LLMError(f"Gemini request failed: {e}") from e
            if r.status_code in (429, 503) and attempt < 2:
                retry_header = r.headers.get("retry-after")
                try:
                    wait = max(10, int(float(retry_header))) if retry_header else 10
                except (ValueError, TypeError):
                    wait = 10
                log.warning("Gemini rate limit (%d) hit — waiting %ds before retry (attempt %d/3)...", r.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            break
        if not r.ok:
            raise LLMError(f"Gemini error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected Gemini response: {r.text[:300]}") from e


class GroqProvider(LLMProvider):
    """Groq cloud inference (https://console.groq.com). Uses the OpenAI-compatible API.

    Default model: ``openai/gpt-oss-120b`` — 120B parameter model with strong
    JSON-following ability, running on Groq's LPU hardware for very fast inference.
    """

    name = "groq"

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int | None = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        if not self.api_key:
            raise LLMError("GROQ_API_KEY is not set (add it to .env)")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }
        for attempt in range(3):
            try:
                r = requests.post(
                    self.API_URL, json=body,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise LLMError(f"Groq request failed: {e}") from e
            if r.status_code in (429, 503) and attempt < 2:
                retry_header = r.headers.get("retry-after")
                try:
                    wait = max(10, int(float(retry_header))) if retry_header else 10
                except (ValueError, TypeError):
                    wait = 10
                log.warning("Groq rate limit (%d) hit — waiting %ds before retry (attempt %d/3)...", r.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            break
        if not r.ok:
            raise LLMError(f"Groq error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected Groq response: {r.text[:300]}") from e


class NvidiaNemotronProvider(LLMProvider):
    """NVIDIA NIM hosted inference (https://build.nvidia.com).

    Default model: ``nvidia/llama-3.1-nemotron-70b-instruct`` — 70B parameter
    instruction-tuned model heavily optimized for helpfulness and alignment.
    """

    name = "nemotron"
    API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: int | None = None):
        self.api_key = api_key or config.NVIDIA_API_KEY
        self.model = model or config.NVIDIA_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        if not self.api_key:
            raise LLMError("NVIDIA_API_KEY is not set (add it to .env from https://build.nvidia.com)")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body: dict = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": 4096,
        }
        for attempt in range(3):
            try:
                r = requests.post(
                    self.API_URL, json=body,
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    timeout=self.timeout,
                )
            except requests.RequestException as e:
                raise LLMError(f"NVIDIA Nemotron request failed: {e}") from e
            if r.status_code in (429, 503) and attempt < 2:
                retry_header = r.headers.get("retry-after")
                try:
                    wait = max(10, int(float(retry_header))) if retry_header else 10
                except (ValueError, TypeError):
                    wait = 10
                log.warning("NVIDIA Nemotron rate limit (%d) hit — waiting %ds before retry (attempt %d/3)...", r.status_code, wait, attempt + 1)
                time.sleep(wait)
                continue
            break
        if not r.ok:
            raise LLMError(f"NVIDIA Nemotron error {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"Unexpected NVIDIA Nemotron response: {r.text[:300]}") from e


class FallbackProvider(LLMProvider):
    """Wraps a priority chain of providers (e.g. Gemini -> Groq -> NVIDIA Nemotron).

    If the current provider fails (after retries and rate limit backoffs), the provider
    **permanently hands over** to the next provider in the chain for all remaining
    requests in the session.
    """

    name = "fallback"

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = providers
        self._current_idx = 0

    @property
    def active_name(self) -> str:
        return self.providers[self._current_idx].name

    def complete(self, prompt: str, system: str | None = None, temperature: float = 0.4) -> str:
        errors = []
        while self._current_idx < len(self.providers):
            current = self.providers[self._current_idx]
            try:
                return current.complete(prompt, system=system, temperature=temperature)
            except LLMError as err:
                log.warning(
                    "Provider %s failed: %s",
                    current.name, err
                )
                errors.append(f"{current.name}: {err}")
                self._current_idx += 1
                if self._current_idx < len(self.providers):
                    next_p = self.providers[self._current_idx]
                    log.warning(
                        "Permanently handing over to %s for remaining requests",
                        next_p.name
                    )

        raise LLMError(
            "All providers in fallback chain failed:\n" + "\n".join(f"  {e}" for e in errors)
        )


PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "nemotron": NvidiaNemotronProvider,
    "nvidia": NvidiaNemotronProvider,
}


def get_provider(name: str | None = None) -> LLMProvider:
    """Instantiate the requested provider or a multi-tier fallback chain.

    Defaults to a chain: [primary] -> [groq] -> [nemotron].
    If primary or groq fails, requests automatically hand over to NVIDIA Nemotron.
    """
    primary_key = (name or config.LLM_PROVIDER or "gemini").lower()
    if primary_key not in PROVIDERS:
        raise LLMError(f"Unknown LLM provider '{primary_key}'. Choose from: {', '.join(PROVIDERS)}")

    # Build chain: primary -> fallback(s) -> nemotron
    chain_keys = [primary_key]
    
    # Check configured fallback
    fb = (config.LLM_FALLBACK or "").lower().strip()
    if fb:
        for item in fb.split(","):
            item = item.strip()
            if item in PROVIDERS and item not in chain_keys:
                chain_keys.append(item)

    # Automatically add Groq and Nemotron to chain if not already in it
    for auto in ["groq", "nemotron"]:
        if auto not in chain_keys and auto in PROVIDERS:
            chain_keys.append(auto)

    instances = [PROVIDERS[k]() for k in chain_keys]
    if len(instances) == 1:
        return instances[0]

    log.info("LLM Provider chain initialized: %s", " -> ".join(p.name for p in instances))
    return FallbackProvider(instances)
