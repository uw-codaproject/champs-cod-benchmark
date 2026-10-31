from __future__ import annotations

import json
import os
import queue
import re
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
from torch.utils.data import Dataset

from .config import ModelConfig, PromptConfig

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover
    BitsAndBytesConfig = None

try:
    from transformers.utils.quantization_config import Mxfp4Config
except ImportError:  # pragma: no cover
    Mxfp4Config = None

try:
    from openai import OpenAI as APIClient
except ImportError:  # pragma: no cover
    APIClient = None


class ModelRunner:
    """Runs text generation locally via HF or via remote API client."""

    def __init__(self, config: ModelConfig, prompt_config: PromptConfig):
        self.config = config
        self.prompt_config = prompt_config
        self.system_prompt = prompt_config.system_prompt
        self.tokenizer = None
        self.model = None
        self._pipeline = None
        self._api_client = None
        self._use_api = (config.provider or "").lower() == "api"
        self._gen_kwargs = {}

    def load(self, device: Optional[int | str] = None) -> None:
        if self._use_api:
            self._load_api_client()
            self._gen_kwargs = {
                "max_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            return

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.name, revision=self.config.revision
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        dtype = getattr(torch, self.config.dtype, torch.bfloat16)
        is_gpt_oss = "gpt-oss" in (self.config.name or "").lower()
        is_awq = "awq" in (self.config.name or "").lower()

        quantization_config = None
        if self.config.use_4bit and not is_gpt_oss and not is_awq:
            if BitsAndBytesConfig is None:
                raise ImportError("bitsandbytes is required for 4-bit quantization.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        device_map = device if device is not None else ("auto" if torch.cuda.is_available() else None)
        self.model = self._load_model_with_fallback(
            dtype,
            quantization_config,
            is_gpt_oss=is_gpt_oss,
            is_awq=is_awq,
            device_map=device_map,
        )
        # temperature<=0 => greedy decoding (do_sample=False) for fully
        # deterministic output; sampling params are ignored in that mode.
        if self.config.temperature and self.config.temperature > 0:
            self._gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "return_full_text": False,
            }
        else:
            self._gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "do_sample": False,
                "return_full_text": False,
            }
        self._pipeline = hf_pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            eos_token_id=self.tokenizer.eos_token_id
        )

    def predict_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """
        Serial prompt runner used by orchestrators; parallelization happens in predict_batches.
        """
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        if self._use_api:
            return self._predict_api(prompt, system_prompt=sys_prompt)
        return self._predict_hf(prompt, system_prompt=sys_prompt)

    def predict_batches(
        self, prompts: List[str], system_prompts: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        # For API calls, run prompts in parallel to improve throughput.
        if self._use_api:
            if self._api_client is None:
                raise RuntimeError("ModelRunner.load() must be called first.")
            generations: List[Optional[str]] = [None] * len(prompts)
            prompt_texts: List[Optional[str]] = [None] * len(prompts)

            def run_one(idx: int, p: str, s: str) -> None:
                try:
                    gen, ptxt = self._predict_api(p, system_prompt=s)
                    generations[idx] = gen
                    prompt_texts[idx] = ptxt
                except Exception as e:
                    print(f"[ModelRunner] API worker failed: {e}")
                    generations[idx] = ""
                    prompt_texts[idx] = ""

            sys_prompts = system_prompts if system_prompts else [self.system_prompt] * len(prompts)
            with ThreadPoolExecutor(max_workers=self.config.inference_batch_size) as executor:
                for idx, (p, s) in enumerate(zip(prompts, sys_prompts)):
                    executor.submit(run_one, idx, p, s)
            return [g or "" for g in generations], [p or "" for p in prompt_texts]

        if self._pipeline is None:
            raise RuntimeError("ModelRunner.load() must be called first.")

        sys_prompts = system_prompts if system_prompts else [self.system_prompt] * len(prompts)
        prompt_texts = []
        for p, s in zip(prompts, sys_prompts):
            messages = [
                {"role": "system", "content": s},
                {"role": "user", "content": p},
            ]
            ptxt = self._build_prompt_text(messages)
            prompt_texts.append(ptxt)

        dataset = ListDataset(prompt_texts)
        generations: List[str] = []
        # pipeline supports inference_batch_size and handles the loop over the dataset.
        for out in self._pipeline(
            dataset, batch_size=self.config.inference_batch_size, **self._gen_kwargs
        ):
            # Assume standard HF text-gen output (list of dicts) or direct dict fallback.
            res = out[0] if isinstance(out, list) and out else out
            gen = res.get("generated_text", "") if isinstance(res, dict) else ""
            generations.append(self._extract_json_only(gen))

        return generations, prompt_texts

    def _load_model_with_fallback(
        self, dtype, quantization_config, is_gpt_oss=False, is_awq=False, device_map=None
    ):
        """
        Some models (e.g., GPT-OSS) are already quantized with MxFP4 and reject
        BitsAndBytesConfig. Try with user-provided quantization; on specific
        incompatibility errors, retry without it.
        """
        if is_gpt_oss or is_awq:
            # GPT-OSS (MxFP4) or AWQ models: do not pass bitsandbytes configs.
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.config.name,
                    revision=self.config.revision,
                    dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True,
                )
            except Exception as exc:  # pragma: no cover - env-specific fallback
                import sys
                print(
                    f"[ERROR] GPT-OSS/AWQ quantized load failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                raise

        try:
            return AutoModelForCausalLM.from_pretrained(
                self.config.name,
                revision=self.config.revision,
                dtype=dtype,
                device_map=device_map,
                quantization_config=quantization_config,
            )
        except ValueError as exc:
            msg = str(exc)
            if "Mxfp4Config" in msg or "Mxfp4" in msg or "is quantized with" in msg:
                # Retry without BitsAndBytesConfig; model provides its own quantization.
                return AutoModelForCausalLM.from_pretrained(
                    self.config.name,
                    revision=self.config.revision,
                    dtype=dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
            raise

    def _load_api_client(self) -> None:
        api_key = self.config.api_key or os.getenv(self.config.api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"API key missing. Set env '{self.config.api_key_env}' or pass ModelConfig.api_key."
            )
        client_kwargs = {"api_key": api_key}
        if self.config.api_base:
            client_kwargs["base_url"] = self.config.api_base
        self._api_client = APIClient(**client_kwargs)

    def _predict_hf(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        if self._pipeline is None:
            raise RuntimeError("ModelRunner.load() must be called first.")
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._build_prompt_text(messages)
        result = self._pipeline(prompt_text, **self._gen_kwargs)
        if isinstance(result, list) and result:
            generated = result[0].get("generated_text", "")
            return self._extract_json_only(generated), prompt_text
        return "", prompt_text

    def _build_prompt_text(self, messages: List[dict]) -> str:
        """
        Build a model-ready prompt with the tokenizer chat template.
        GPT-OSS requires its harmony format; use add_generation_prompt=True.
        """
        is_gpt_oss = "gpt-oss" in (self.config.name or "").lower()
        if is_gpt_oss:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium"
            )
        else:
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt_text

    def _predict_api(self, prompt: str, system_prompt: Optional[str] = None) -> Tuple[str, str]:
        """Single API call used by both serial and threaded paths."""
        if self._api_client is None:
            raise RuntimeError("ModelRunner.load() must be called first.")
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self._render_messages(messages)
        try:
            response = self._api_client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                # temperature=self.config.temperature,
                # top_p=self.config.top_p,
                # max_completion_tokens=self.config.max_new_tokens,
            )
        except Exception as exc:  # pragma: no cover - network failures
            raise RuntimeError(f"API generation failed: {exc}") from exc
        content = ""
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") if message else ""
        if not content:
            print(f"[ModelRunner] API returned empty content for choice 0.")
        return self._extract_json_only(content), prompt_text

    def _render_messages(self, messages: List[dict]) -> str:
        """Human-readable representation of chat messages (for logging/exports)."""
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"<{role}>\n{content}")
        return "\n\n".join(parts)

    def _extract_json_only(self, text: str) -> str:
        """
        Best-effort enforcement: return only the first JSON object found.
        Falls back to original text if no valid JSON is present.
        """
        if not text:
            return ""
        if "gpt-oss" in (self.config.name or "").lower():
            text = self._extract_harmony_final(text)
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                parsed, end = decoder.raw_decode(text[idx:])
                return json.dumps(parsed)
            except json.JSONDecodeError:
                continue
        return text

    def _extract_harmony_final(self, text: str) -> str:
        """
        Extract the assistant final channel content from GPT-OSS Harmony output.
        Falls back to the original text if no final marker is found.
        """
        if not text:
            return ""
        marker = "assistantfinal"
        if marker in text:
            return text.split(marker, 1)[1].lstrip()
        # Handle template-style tags if present.
        tag = "<|start|>assistant<|message|>"
        if tag in text:
            tail = text.split(tag, 1)[1]
            return tail.lstrip().removeprefix("final").lstrip()
        return text


class ModelRunnerPool:
    """Manages a pool of ModelRunner instances across multiple GPUs."""

    def __init__(self, model_cfg: ModelConfig, prompt_cfg: PromptConfig):
        self.model_cfg = model_cfg
        self.prompt_cfg = prompt_cfg
        self.runners: queue.Queue[ModelRunner] = queue.Queue()

    def load(self) -> None:
        """Initialize and load model instances into the pool."""
        gpu_count = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

        num_instances = self.model_cfg.num_models
        if num_instances <= 0:
            num_instances = max(1, gpu_count)

        if self.model_cfg.provider == "api":
            print(f"[RunnerPool] Initializing {num_instances} API runner slots...")
            for _ in range(num_instances):
                runner = self._make_runner()
                runner.load()
                self.runners.put(runner)
            return

        # Local provider: determine how many instances to load.
        print(
            f"[RunnerPool] Pre-loading {num_instances} model instances across {gpu_count} GPUs..."
        )
        for i in range(num_instances):
            device = i % gpu_count if gpu_count > 0 else None
            runner = self._make_runner()
            runner.load(device=device)
            self.runners.put(runner)

    def get(self) -> ModelRunner:
        return self.runners.get()

    def put(self, runner: ModelRunner) -> None:
        self.runners.put(runner)

    def _make_runner(self) -> ModelRunner:
        # The pool holds generic runners; system prompts are swapped at inference.
        return ModelRunner(self.model_cfg, self.prompt_cfg)

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]