"""Run a fixed set of prompts against an Ollama HTTP server and persist results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "https://flagstone-bundle-monopoly.ngrok-free.dev"
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_OUTPUT_PATH = "inference_results.json"


def get_test_prompts() -> list[str]:
    """Return 10 diverse prompts for inference testing."""
    return [
        "Объясни простыми словами, что такое машинное обучение, в 2 предложениях.",
        "Составь список из 5 полезных привычек для продуктивной учебы.",
        "Перефразируй фразу: 'Сегодня отличная погода для прогулки' в нейтральном стиле.",
        "Напиши короткое поздравление с днем рождения на 1-2 предложения.",
        "Дай 3 идеи, как снизить стресс перед экзаменом.",
        "Чем отличается HTTP от HTTPS? Ответь кратко.",
        "Приведи пример SQL-запроса для выбора всех пользователей старше 18 лет.",
        "Сгенерируй 3 варианта названия для IT-стартапа в сфере кибербезопасности.",
        "Кратко опиши, что делает функция Python len().",
        "Дай рецепт бутерброда с сыром в 3 шага.",
    ]


def build_generate_url(base_url: str) -> str:
    """Build the Ollama generate endpoint URL from a base URL."""
    return f"{base_url.rstrip('/')}/api/generate"


def query_ollama(
    *,
    base_url: str,
    model: str,
    prompt: str,
    timeout_seconds: int = 60,
) -> str:
    """Send one prompt to Ollama and return generated text.

    Args:
        base_url: Public base URL of the Ollama server.
        model: Model identifier used by Ollama.
        prompt: User prompt sent to the model.
        timeout_seconds: HTTP timeout in seconds.

    Returns:
        Model text response from Ollama.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status.
        requests.RequestException: For networking and timeout failures.
        ValueError: If Ollama response JSON has no `response` field.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(
        build_generate_url(base_url),
        json=payload,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    data = response.json()
    answer = data.get("response")
    if not isinstance(answer, str):
        raise ValueError("Ollama response JSON does not contain a string field 'response'.")
    return answer.strip()


def run_inference_batch(
    *,
    base_url: str,
    model: str,
    prompts: list[str],
    timeout_seconds: int,
) -> list[dict[str, str]]:
    """Run a list of prompts against Ollama and collect prompt/response pairs.

    Args:
        base_url: Public base URL of the Ollama server.
        model: Model identifier used by Ollama.
        prompts: Prompts to execute sequentially.
        timeout_seconds: HTTP timeout in seconds for each request.

    Returns:
        List of dictionaries with `prompt` and `response` keys.

    Raises:
        RuntimeError: If any prompt cannot be processed successfully.
    """
    results: list[dict[str, str]] = []
    for index, prompt in enumerate(prompts, start=1):
        try:
            response = query_ollama(
                base_url=base_url,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Failed on prompt #{index}: {prompt}") from exc
        results.append({"prompt": prompt, "response": response})
    return results


def save_results(
    *,
    output_path: Path,
    base_url: str,
    model: str,
    results: list[dict[str, str]],
) -> None:
    """Persist inference results and metadata to a JSON file.

    Args:
        output_path: Target JSON path.
        base_url: Public base URL of the Ollama server.
        model: Model identifier used by Ollama.
        results: Prompt/response pairs.
    """
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": base_url,
        "model": model,
        "result_count": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference batch execution."""
    parser = argparse.ArgumentParser(
        description="Send 10 test prompts to Ollama and save responses.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output JSON report",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds",
    )
    return parser.parse_args()


def main() -> int:
    """Run end-to-end inference batch and write results to disk."""
    args = parse_args()
    prompts = get_test_prompts()
    results = run_inference_batch(
        base_url=args.base_url,
        model=args.model,
        prompts=prompts,
        timeout_seconds=args.timeout,
    )
    save_results(
        output_path=Path(args.output),
        base_url=args.base_url,
        model=args.model,
        results=results,
    )
    print(f"Saved {len(results)} responses to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
