# chatgpt_simple.py
# Features:
#   - ask_chatgpt(): einfache vollständige Antwort (Blocking)
#   - stream_chatgpt(): Streaming (tokenweise, wie im Chat)
#   - Timeout, Retries (exponentielles Backoff), Proxy-Support
#   - Optionaler JSON-Output (für Pipes/Skripting)
#   - .env-Support (python-dotenv) für OPENAI_API_KEY
#
# CLI-Beispiele:
#   python chatgpt_simple.py "Dein Prompt"
#   python chatgpt_simple.py --stream --model gpt-4o "Dein Prompt"
#   python chatgpt_simple.py --timeout 20 --retries 4 --proxy http://127.0.0.1:8080 "Test"
#   python chatgpt_simple.py --json "Sehr kurze JSON-taugliche Antwort."
#
# Voraussetzungen:
#   pip install --upgrade openai python-dotenv
#   .env mit: OPENAI_API_KEY=sk-....

from __future__ import annotations
from openai import OpenAI
import os
import argparse
import time
import json
from typing import Optional, Dict, Any

import httpx
from dotenv import load_dotenv

# -------------------------
# Helpers
# -------------------------

def _build_http_client(timeout: float, proxy: Optional[str]) -> httpx.Client:
    """httpx.Client mit Timeout und optional Proxy erstellen."""
    return httpx.Client(timeout=timeout, proxies=proxy)

def _get_client(timeout: float, proxy: Optional[str]) -> OpenAI:
    """
    OpenAI-Client mit Timeout/Proxy.
    API-Key wird aus .env (falls vorhanden) und/oder Umgebung geladen.
    """
    load_dotenv()  # .env einlesen, falls vorhanden
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Kein API-Key gefunden. Setze OPENAI_API_KEY als Umgebungsvariable "
            "oder lege eine .env-Datei mit OPENAI_API_KEY=... an."
        )
    http_client = _build_http_client(timeout=timeout, proxy=proxy)
    return OpenAI(api_key=api_key, http_client=http_client)

def _extract_text_fallback(resp) -> str:
    """Robuste Textextraktion für die Responses API."""
    if hasattr(resp, "output_text"):
        return resp.output_text
    parts = []
    try:
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if isinstance(t, str):
                    parts.append(t)
    except Exception:
        return str(resp)
    return "".join(parts) if parts else str(resp)

def _extract_usage(resp) -> Dict[str, Any]:
    """Usage-/Token-Metadaten extrahieren (falls vorhanden)."""
    usage = getattr(resp, "usage", None)
    if not usage:
        return {}
    try:
        return dict(usage)
    except Exception:
        try:
            return usage.model_dump()
        except Exception:
            return {}

# -------------------------
# Public API
# -------------------------

def ask_chatgpt(
    prompt: str,
    model: str = "gpt-4.1",
    *,
    timeout: float = 30.0,
    retries: int = 3,
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Blocking-Anfrage: Prompt → gesamte Antwort.
    Rückgabe: {model, prompt, answer, usage}
    """
    attempt = 0
    backoff = 1.5
    last_exc: Optional[Exception] = None

    while attempt <= retries:
        try:
            client = _get_client(timeout=timeout, proxy=proxy)
            resp = client.responses.create(model=model, input=prompt)
            answer = _extract_text_fallback(resp)
            usage = _extract_usage(resp)
            return {"model": model, "prompt": prompt, "answer": answer, "usage": usage}
        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            time.sleep(max(0.5, backoff ** attempt))
            attempt += 1

    raise RuntimeError(f"API-Aufruf fehlgeschlagen nach {retries+1} Versuch(en): {last_exc}")

def stream_chatgpt(
    prompt: str,
    model: str = "gpt-4.1",
    *,
    timeout: float = 30.0,
    retries: int = 3,
    proxy: Optional[str] = None,
    print_live: bool = True,
) -> Dict[str, Any]:
    """
    Streaming-Anfrage: tokenweise Ausgabe (wenn print_live=True).
    Rückgabe: {model, prompt, answer, usage}
    """
    attempt = 0
    backoff = 1.5
    last_exc: Optional[Exception] = None

    while attempt <= retries:
        try:
            client = _get_client(timeout=timeout, proxy=proxy)
            final_text_chunks = []

            with client.responses.stream(model=model, input=prompt) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            final_text_chunks.append(delta)
                            if print_live:
                                print(delta, end="", flush=True)
                    elif event.type == "response.error":
                        err = getattr(event, "error", None)
                        raise RuntimeError(f"API-Fehler: {err}")
                stream.until_done()

                try:
                    final_resp = stream.get_final_response()
                    answer = _extract_text_fallback(final_resp)
                    usage = _extract_usage(final_resp)
                except Exception:
                    answer = "".join(final_text_chunks)
                    usage = {}

            if print_live and (not answer.endswith("\n")):
                print()
            return {"model": model, "prompt": prompt, "answer": answer, "usage": usage}

        except Exception as e:
            last_exc = e
            if attempt == retries:
                break
            time.sleep(max(0.5, backoff ** attempt))
            attempt += 1

    raise RuntimeError(f"Streaming fehlgeschlagen nach {retries+1} Versuch(en): {last_exc}")

# -------------------------
# CLI
# -------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ChatGPT-CLI mit Streaming/Timeout/Retry/Proxy/JSON und .env-Support.")
    p.add_argument("prompt", nargs="*", help="Prompt-Text (oder leer lassen für einen Default).")
    p.add_argument("--model", default="gpt-4.1",
                   help="Modellname, z. B. gpt-4.1, gpt-4o, gpt-4o-mini, o4-mini …")
    p.add_argument("--stream", action="store_true",
                   help="Antwort tokenweise streamen (wie im Chat).")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="HTTP-Timeout in Sekunden (Default: 30).")
    p.add_argument("--retries", type=int, default=3,
                   help="Anzahl Retries bei Fehlern (Default: 3).")
    p.add_argument("--proxy", type=str, default=None,
                   help="Proxy-URL, z. B. http://127.0.0.1:8080")
    p.add_argument("--json", action="store_true",
                   help="Gibt das Ergebnis als JSON auf stdout aus.")
    return p

def _print_result(result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result.get("answer", ""))

def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    prompt = "Sag kurz Hallo von der OpenAI API." if not args.prompt else " ".join(args.prompt)

    if args.stream:
        result = stream_chatgpt(
            prompt,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            proxy=args.proxy,
            print_live=not args.json,
        )
        if args.json:
            _print_result(result, as_json=True)
    else:
        result = ask_chatgpt(
            prompt,
            model=args.model,
            timeout=args.timeout,
            retries=args.retries,
            proxy=args.proxy,
        )
        _print_result(result, as_json=args.json)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
