"""
Pluggable LLM backends for Nanotoxicology RAG.
- ollama: Local models via Ollama (recommended for local use; no Python model loading).
- openai: OpenAI API (cloud).
- transformers: Hugging Face in-process (heavy; requires torch/transformers).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


def get_llm_backend(
    backend: str = "ollama",
    *,
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-3.5-turbo",
    transformers_model: str = "meta-llama/Llama-2-7b-chat-hf",
    ollama_model: str = "tinyllama",
    ollama_base_url: str = "http://localhost:11434",
    device: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
) -> "LLMBackend":
    """Build an LLM backend from string name and options."""
    if backend.lower() == "ollama":
        return OllamaBackend(model=ollama_model, base_url=ollama_base_url.rstrip("/"))
    if backend.lower() == "openai":
        if not openai_api_key:
            raise ValueError("openai_api_key required for backend='openai'")
        return OpenAIBackend(api_key=openai_api_key, model=openai_model)
    if backend.lower() in ("transformers", "local", "llama"):
        return TransformersBackend(
            model_name_or_path=transformers_model,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
    raise ValueError(f"Unknown backend: {backend}. Use 'ollama', 'openai', or 'transformers'.")


class LLMBackend(ABC):
    """Abstract base for RAG LLM backends."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from a list of chat messages.
        messages: list of {"role": "system"|"user"|"assistant", "content": "..."}
        Returns the generated text (assistant reply).
        """
        pass


class OpenAIBackend(LLMBackend):
    """Use OpenAI API (e.g. gpt-3.5-turbo, gpt-4)."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,
        do_sample: bool = True,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (response.choices[0].message.content or "").strip()


class OllamaBackend(LLMBackend):
    """
    Use a local model via Ollama (https://ollama.com).
    No data leaves your machine; no torch/transformers in Python.
    Install Ollama, then: ollama pull tinyllama  (or llama2, phi, etc.)
    """

    def __init__(self, model: str = "tinyllama", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,
        do_sample: bool = True,
    ) -> str:
        import urllib.request
        import json as _json

        url = f"{self.base_url}/api/chat"
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
            },
        }
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = _json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama request failed. Is Ollama running? Try: ollama serve  and  ollama pull {self.model}"
            ) from e
        reply = out.get("message") or {}
        return (reply.get("content") or "").strip()


class TransformersBackend(LLMBackend):
    """
    Use a Hugging Face causal LM (e.g. Llama 2 7B Chat) for local inference.
    Uses the model's chat template when available; otherwise builds a simple prompt.
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-chat-hf",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        dtype = None
        if torch_dtype:
            dtype = getattr(torch, torch_dtype.replace("torch.", ""), torch.float16)

        model_kwargs = {"trust_remote_code": True}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif load_in_4bit:
                model_kwargs["load_in_4bit"] = True
        else:
            model_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        if self.device == "cpu" and not (load_in_8bit or load_in_4bit):
            self.model = self.model.to(self.device)

        # Prefer chat template; many models have it
        self._use_chat_template = self.tokenizer.chat_template is not None

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.4,
        do_sample: bool = True,
    ) -> str:
        import torch

        if self._use_chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: concat system + user as a single prompt
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}\n\n")
                elif role == "user":
                    parts.append(f"User: {content}\n\n")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}\n\n")
            parts.append("Assistant: ")
            prompt = "".join(parts)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with __import__("torch").no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
                top_p=0.95 if do_sample else None,
            )

        full = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Return only the new reply (after the last "Assistant:" or similar)
        if "Assistant:" in full:
            full = full.split("Assistant:")[-1]
        if "assistant" in full.lower() and "\n" in full:
            # Some templates add "assistant\n"
            for sep in ["\nassistant\n", "\nAssistant\n", "\n\n"]:
                if sep in full:
                    full = full.split(sep)[-1]
                    break
        return full.strip()
