#!/usr/bin/env python3
"""Generate a training corpus from a local llama.cpp / OpenAI-compatible server.

This is sequence-level distillation: the teacher writes high-quality text and
Mariana learns from it.  It works across tokenizers -- the 35B teacher may use a
248k BPE vocabulary while Mariana is byte-level -- which full logit distillation
cannot do without a shared tokenizer.

The teacher's top-k logprobs are still useful: this script can generate
several candidates per prompt and keep the one with the highest mean
log-probability, and it can reject low-confidence samples.  The output is plain
UTF-8 text plus a small manifest, ready for ``--data``.

Providers
---------
    local     llama.cpp / any OpenAI-compatible local server (default)
    deepseek  DeepSeek's OpenAI-compatible API (chat endpoint, API key)

Examples
--------
    # local llama.cpp
    python scripts/distill_teacher.py --url http://127.0.0.1:8080 \
        --out trained-model/teacher-corpus.txt --samples-per-prompt 4 \
        --max-tokens 384 --workers 2

    # DeepSeek V4.1 Flash (DEEPSEEK_API_KEY in the environment)
    python scripts/distill_teacher.py --provider deepseek \
        --model deepseek-flash \
        --out trained-model/teacher-corpus-deepseek.txt \
        --samples-per-prompt 4 --max-tokens 512 --workers 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request


DEEPSEEK_URL = "https://api.deepseek.com"
# DeepSeek's API docs: deepseek-flash is served by DeepSeek-V4.1-Flash-1017.
DEEPSEEK_MODEL = "deepseek-flash"

DEFAULT_PROMPTS = [
    "Write a clear, factual explanation of how photosynthesis works.",
    "Explain the difference between TCP and UDP with a practical example.",
    "Write a short Python function that merges two sorted lists and explain its complexity.",
    "A user asks for help debugging a failing SQL query. Write the assistant's reply.",
    "Summarize the causes of the French Revolution in three paragraphs.",
    "Write a dialogue between a physicist and a historian about the nature of time.",
    "Explain Bayes' theorem using a medical-test example.",
    "Write a compact tutorial on gradient descent for a beginner.",
    "Describe how a modern CPU pipeline handles a branch misprediction.",
    "Write a product announcement for a battery-powered hiking lantern.",
    "Explain the role of mitochondria in a cell.",
    "Draft a polite email declining a meeting invitation.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between correlation and causation with examples.",
    "Write a mathematical proof that the square root of 2 is irrational.",
    "Give a step-by-step recipe for a simple lentil soup.",
    "Describe the plot of Hamlet in five sentences.",
    "Explain how transformers use attention in neural networks.",
    "Write a SQL query that finds duplicate email addresses in a users table.",
    "Draft a checklist for reviewing a pull request.",
    "Explain the greenhouse effect to a curious 12-year-old.",
    "Write a short fable about a fox and a compass.",
    "Explain why the sky is blue, including Rayleigh scattering.",
    "Write a short guide to git rebase versus git merge.",
    "Describe the water cycle for a middle-school science class.",
]


class TeacherClient:
    """Minimal HTTP client for llama.cpp's native and OpenAI endpoints."""

    def __init__(self, url: str, api: str = "chat", model: str | None = None,
                 api_key: str | None = None, timeout: float = 180.0,
                 retries: int = 3, system: str | None = None,
                 provider: str = "local"):
        self.base = url.rstrip("/")
        self.root = self.base.removesuffix("/v1")
        self.provider = provider
        # DeepSeek documents https://api.deepseek.com/chat/completions (no /v1).
        self.openai_base = (
            self.root if provider == "deepseek"
            else (self.base if self.base.endswith("/v1") else self.base + "/v1")
        )
        self.api = api
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY")
        self.timeout = timeout
        self.retries = retries
        self.system = system or "Answer directly and concisely. Do not show hidden reasoning."
        discovered = model or self._discover_model()
        if discovered is None:
            if provider == "deepseek":
                raise SystemExit(
                    "Could not discover a DeepSeek model; pass --model deepseek-flash "
                    "or another model ID."
                )
            discovered = "local"
        self.model = discovered

    def _discover_model(self) -> str:
        for endpoint in (
            f"{self.openai_base}/models",
            f"{self.root}/models",
            f"{self.root}/v1/models",
        ):
            try:
                data = self._request(endpoint, None, method="GET")
                if isinstance(data, dict):
                    items = data.get("data") or data.get("models") or []
                    if items:
                        items = sorted(
                            items,
                            key=lambda item: item.get("created", 0),
                            reverse=True,
                        )
                        return str(items[0].get("id") or items[0].get("model") or "local")
            except Exception:
                continue
        return None

    def _request(self, url: str, payload: dict | None, method: str = "POST"):
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method=method)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
            except Exception as exc:  # network, HTTP, JSON
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(1.5 ** attempt + random.random() * 0.3)
        raise RuntimeError(f"teacher request failed: {url}: {last_error}")

    def complete(self, prompt: str, max_tokens: int, temperature: float,
                 top_p: float, n_probs: int = 8, seed: int | None = None,
                 stop: list[str] | None = None) -> dict:
        if self.api == "llama":
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n_probs": n_probs,
                "cache_prompt": True,
            }
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop
            data = self._request(f"{self.root}/completion", payload)
            return self._parse_llama(data)

        if self.api == "chat":
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if seed is not None:
                payload["seed"] = seed
            if stop:
                payload["stop"] = stop
            data = self._request(f"{self.openai_base}/chat/completions", payload)
            return self._parse_chat(data)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": n_probs,
            "echo": False,
        }
        if seed is not None:
            payload["seed"] = seed
        if stop:
            payload["stop"] = stop
        data = self._request(f"{self.openai_base}/completions", payload)
        return self._parse_openai(data)

    @staticmethod
    def _mean_logprob(values):
        vals = [float(v) for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    @classmethod
    def _parse_openai(cls, data: dict) -> dict:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("teacher returned no choices")
        choice = choices[0]
        logprobs = choice.get("logprobs") or {}
        values = []
        if isinstance(logprobs, dict):
            content = logprobs.get("content") or []
            values = [
                item.get("logprob") for item in content
                if isinstance(item, dict) and item.get("logprob") is not None
            ]
        elif isinstance(logprobs, list):
            values = [
                item.get("logprob") if isinstance(item, dict) else item
                for item in logprobs
            ]
        return {
            "text": choice.get("text") or "",
            "mean_logprob": cls._mean_logprob(values),
            "finish_reason": choice.get("finish_reason"),
            "raw": data,
        }

    @classmethod
    def _parse_chat(cls, data: dict) -> dict:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("teacher returned no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        logprobs = choice.get("logprobs") or {}
        values = []
        if isinstance(logprobs, dict):
            values = [
                item.get("logprob") for item in (logprobs.get("content") or [])
                if isinstance(item, dict) and item.get("logprob") is not None
            ]
        return {
            "text": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or "",
            "mean_logprob": cls._mean_logprob(values),
            "finish_reason": choice.get("finish_reason"),
            "raw": data,
        }

    @classmethod
    def _parse_llama(cls, data: dict) -> dict:
        probs = data.get("completion_probabilities") or []
        values = []
        for item in probs:
            if not isinstance(item, dict):
                continue
            logprob = item.get("logprob")
            if logprob is None and item.get("prob") is not None:
                logprob = math.log(max(float(item["prob"]), 1e-9))
            values.append(logprob)
        return {
            "text": data.get("content") or "",
            "mean_logprob": cls._mean_logprob(values),
            "finish_reason": data.get("stop_type"),
            "raw": data,
        }


def _load_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    with open(path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise SystemExit(f"no prompts found in {path}")
    return prompts


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    return text.strip()


def _strip_think(text: str) -> str:
    """Remove reasoning-model ``<think>...</think>`` blocks (and unclosed tags)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text.replace("</think>", "").strip()


def sample_once(client: TeacherClient, prompt: str, args) -> dict:
    """Generate and filter one sample; returns ``{text}`` or ``{error}``."""
    best = None
    for _ in range(max(1, args.candidates)):
        result = client.complete(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            n_probs=args.n_probs,
            seed=args.seed,
            stop=args.stop,
        )
        score = result["mean_logprob"]
        score = float("-inf") if score is None else score
        if best is None or score > best[0]:
            best = (score, result)

    result = best[1]
    completion = _clean_text(result["text"])
    reasoning = (result.get("reasoning") or "").strip()
    if args.keep_think:
        if reasoning:
            completion = f"<think>{reasoning}</think>\n{completion}"
    else:
        completion = _strip_think(completion)
    if not completion:
        return {"error": "empty completion"}
    if args.drop_truncated and result.get("finish_reason") == "length":
        return {"error": "truncated"}

    if args.include_prompt:
        text = prompt.rstrip() + args.prompt_separator + completion
    else:
        text = completion
    if len(text) < args.min_chars:
        return {"error": "too short"}
    if args.max_chars and len(text) > args.max_chars:
        return {"error": "too long"}
    if (
        args.min_mean_logprob is not None
        and result["mean_logprob"] is not None
        and result["mean_logprob"] < args.min_mean_logprob
    ):
        return {"error": "low teacher confidence"}

    return {
        "text": text,
        "prompt": prompt,
        "mean_logprob": result["mean_logprob"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="distill_teacher",
        description="Generate a distillation corpus from a local llama.cpp server.",
    )
    p.add_argument("--provider", type=str, default="local",
                   choices=["local", "deepseek"],
                   help="local llama.cpp/OpenAI-compatible server, or DeepSeek API")
    p.add_argument("--url", type=str, default=None,
                   help="teacher base URL (default: local 127.0.0.1:8080 or "
                        "https://api.deepseek.com)")
    p.add_argument("--api", type=str, default="chat",
                   choices=["chat", "openai", "llama"],
                   help="chat = /v1/chat/completions (clean answers); "
                        "openai = /v1/completions (top-k logprobs); "
                        "llama = native /completion")
    p.add_argument("--model", type=str, default=None,
                   help="teacher model name; use 'auto' to query /v1/models "
                        "(DeepSeek default: deepseek-flash or $DEEPSEEK_MODEL)")
    p.add_argument("--api-key", type=str, default=None,
                   help="API key (default: $DEEPSEEK_API_KEY for deepseek, "
                        "$LLAMA_API_KEY for local)")
    p.add_argument("--system", type=str, default=None,
                   help="system message for --api chat")
    p.add_argument("--prompts", type=str, default=None,
                   help="one prompt per line (default: built-in diverse prompts)")
    p.add_argument("--out", type=str, default="trained-model/teacher-corpus.txt")
    p.add_argument("--samples-per-prompt", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0,
                   help="stop after this many accepted samples (0 = unlimited)")
    p.add_argument("--max-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--candidates", type=int, default=1,
                   help="teacher samples per prompt; keep the highest mean logprob")
    p.add_argument("--n-probs", type=int, default=8,
                   help="top-k teacher logprobs requested")
    p.add_argument("--min-mean-logprob", type=float, default=None,
                   help="reject samples below this mean teacher logprob")
    p.add_argument("--min-chars", type=int, default=32)
    p.add_argument("--max-chars", type=int, default=0)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--stop", nargs="*", default=None)
    p.add_argument("--no-prompt", action="store_true",
                   help="write only teacher completions, not the prompt")
    p.add_argument("--prompt-separator", type=str, default="\n\n",
                   help="text between prompt and answer (default: blank line)")
    p.add_argument("--drop-truncated", action="store_true",
                   help="drop samples that hit max-tokens before finishing")
    p.add_argument("--keep-think", action="store_true",
                   help="keep <think>...</think> reasoning blocks (default: strip)")
    p.add_argument("--append", action="store_true",
                   help="append to --out and skip already-written samples")
    p.add_argument("--dry-run", action="store_true")
    return p


def _resolve_teacher(args):
    """Resolve provider defaults into ``(url, model, api_key)``."""
    if args.provider == "deepseek":
        if args.api != "chat":
            raise SystemExit(
                "DeepSeek uses the chat-completions API; use --api chat."
            )
        url = args.url or DEEPSEEK_URL
        model = args.model or os.environ.get("DEEPSEEK_MODEL") or DEEPSEEK_MODEL
        if model == "auto":
            model = None
        key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise SystemExit(
                "Set DEEPSEEK_API_KEY or pass --api-key for --provider deepseek."
            )
        return url, model, key

    url = args.url or "http://127.0.0.1:8080"
    model = None if args.model == "auto" else args.model
    key = args.api_key or os.environ.get("LLAMA_API_KEY")
    return url, model, key


def main() -> None:
    args = build_parser().parse_args()
    args.include_prompt = not args.no_prompt
    args.keep_think = bool(args.keep_think)
    if args.seed is not None:
        random.seed(args.seed)

    args.url, args.model, args.api_key = _resolve_teacher(args)
    prompts = _load_prompts(args.prompts)
    client = TeacherClient(
        args.url, api=args.api, model=args.model,
        api_key=args.api_key, timeout=args.timeout, system=args.system,
        provider=args.provider,
    )

    if args.api == "chat" and (args.candidates > 1 or args.min_mean_logprob is not None):
        print(
            "Note: the chat endpoint does not expose logprobs; candidate scoring "
            "and --min-mean-logprob are inactive. Use --api openai for those.",
            file=sys.stderr,
        )

    if args.dry_run:
        print(
            f"provider: {args.provider} | teacher: {client.model} "
            f"at {args.url} ({args.api})"
        )
        print(f"prompts: {len(prompts)} | output: {args.out}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    seen_path = args.out + ".seen.json"
    seen: set[str] = set()
    if args.append and os.path.exists(seen_path):
        try:
            with open(seen_path, "r", encoding="utf-8") as f:
                seen = set(json.load(f))
        except Exception:
            seen = set()

    tasks = [
        prompt
        for prompt in prompts
        for _ in range(max(1, args.samples_per_prompt))
    ]
    if args.max_samples:
        # Keep a small buffer for filtered/duplicate samples instead of
        # submitting the whole (possibly large) prompt set.
        tasks = tasks[: max(args.max_samples * 3, args.workers * 2)]

    accepted = filtered = duplicates = errors = 0
    t0 = time.time()
    mode = "a" if args.append else "w"

    with open(args.out, mode, encoding="utf-8") as out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(sample_once, client, prompt, args) for prompt in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    errors += 1
                    print(f"  error: {exc}", file=sys.stderr)
                    continue
                if "error" in result:
                    filtered += 1
                    continue
                text = result["text"]
                digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                out.write(text)
                if not text.endswith("\n"):
                    out.write("\n")
                out.write("\n")
                accepted += 1
                if accepted % 10 == 0:
                    print(f"  accepted {accepted} samples ...")
                if args.max_samples and accepted >= args.max_samples:
                    for pending in futures:
                        pending.cancel()
                    break

    elapsed = time.time() - t0
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f)

    manifest = {
        "out": args.out,
        "provider": args.provider,
        "teacher_url": args.url,
        "teacher_model": client.model,
        "api": args.api,
        "accepted": accepted,
        "filtered": filtered,
        "duplicates": duplicates,
        "errors": errors,
        "seconds": round(elapsed, 2),
        "args": {k: v for k, v in vars(args).items() if k != "api_key"},
    }
    manifest_path = args.out + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Wrote {accepted} samples to {args.out} "
        f"({filtered} filtered, {duplicates} duplicates, {errors} errors, "
        f"{elapsed:.1f}s)"
    )
    print(f"Manifest: {manifest_path}")
    print("\nSuggested training run:")
    print(
        f"  python -m mariana train --data {args.out} --moe \\\n"
        f"      --checkpoint trained-model/mariana-moe.pt \\\n"
        f"      --steps 20000 --eval-every 1000 --no-gen"
    )


if __name__ == "__main__":
    main()
