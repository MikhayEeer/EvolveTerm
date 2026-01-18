"""
Script to compare ranking-function inference prompts in the playground environment.

Two modes are supported and should be run separately:
1) rf: direct ranking function inference
2) rf_template: template selection (type/depth)

Usage:
```bash
python src/prompts_playground/rf_inference_compare.py --mode rf --input-path results/aeval/extract_v2_0110_glm47
python src/prompts_playground/rf_inference_compare.py --mode rf_template --input-path results/aeval/extract_v2_0110_glm47
```
"""

import sys
import time
import csv
import json
import datetime
import traceback
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

# Add src to sys.path to allow imports from evolve_term
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Project root for data loading
PROJECT_ROOT = SRC_ROOT.parent

from evolve_term.llm_client import APILLMClient, LLMUnavailableError
from evolve_term.prompts_loader import PromptRepository
from evolve_term.utils import parse_llm_yaml


class PlaygroundPredictor:
    """
    Lightweight predictor for playground ranking-function experiments.
    """

    def __init__(self, llm_client: APILLMClient, prompt_repo: PromptRepository):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo

    def infer_ranking(
        self,
        code: str,
        invariants: List[str],
        references: List[Any],
        prompt_version: str,
        llm_params: Dict[str, Any] | None = None,
    ) -> Tuple[str, object]:
        prompt_name = f"rf/{prompt_version}"
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            invariants=json.dumps(invariants, ensure_ascii=False, indent=2),
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2),
        )
        if llm_params:
            prompt.update(llm_params)
        response = self.llm_client.complete(prompt)
        return response, parse_llm_yaml(response)


class TrackingLLMClient(APILLMClient):
    """
    A subclass of APILLMClient that tracks usage metrics and latency.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_metadata = {}

    def complete(self, prompt: str | dict[str, str], retry: int = 3, retry_delay: float = 2.0) -> str:
        self.call_count += 1
        request_overrides: dict = {}
        messages = []

        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            supported_params = [
                "response_format",
                "max_tokens",
                "temperature",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
                "stop",
                "seed",
            ]
            for param in supported_params:
                if param in prompt:
                    val = prompt[param]
                    if param == "max_tokens":
                        try:
                            val = int(val)
                        except Exception:
                            pass
                    request_overrides[param] = val

            if prompt.get("system"):
                messages.append({"role": "system", "content": prompt["system"]})
            if prompt.get("user"):
                messages.append({"role": "user", "content": prompt["user"]})

        last_exception = None
        start_time = time.time()

        for attempt in range(retry + 1):
            try:
                call_args = {
                    "model": self.model,
                    "messages": messages,
                    **self.payload_template,
                    **request_overrides,
                }

                response = self.client.chat.completions.create(**call_args)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                usage = object()
                if hasattr(response, "usage") and response.usage:
                    usage = response.usage

                self.last_metadata = {
                    "latency_ms": round(latency_ms, 2),
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                    "model": response.model,
                    "system_fingerprint": getattr(response, "system_fingerprint", ""),
                    "call_args": str(call_args),
                }

                if not response.choices:
                    raise LLMUnavailableError("LLM provider returned no choices")
                message = response.choices[0].message
                content = getattr(message, "content", None) or ""
                return content

            except Exception as exc:
                last_exception = exc
                if attempt < retry:
                    print(f"\n[Warning] LLM request failed (Attempt {attempt + 1}/{retry + 1}): {exc}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue

        raise LLMUnavailableError(
            f"LLM provider error after {retry} retries: {last_exception}"
        ) from last_exception


import argparse

DEFAULT_CONFIG_TAG = "default"
PROMPT_SYSTEM_SUFFIX = ".system.txt"
PROMPT_USER_SUFFIX = ".user.txt"


def discover_rf_prompts(prompts_dir: Path, mode: str) -> List[str]:
    rf_dir = prompts_dir / "rf"
    if not rf_dir.exists():
        return []
    system_files = {
        path.name[:-len(PROMPT_SYSTEM_SUFFIX)]
        for path in rf_dir.glob(f"*{PROMPT_SYSTEM_SUFFIX}")
    }
    user_files = {
        path.name[:-len(PROMPT_USER_SUFFIX)]
        for path in rf_dir.glob(f"*{PROMPT_USER_SUFFIX}")
    }
    candidates = sorted(system_files & user_files)
    if mode == "rf_template":
        return [name for name in candidates if name.startswith("rf_template")]
    return [name for name in candidates if not name.startswith("rf_template")]


def parse_cli_list(values: List[str] | None) -> List[str]:
    if not values:
        return []
    items: List[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                items.append(part)
    return items


def normalize_config_tag(value: str | None) -> str:
    tag = (value or "").strip()
    return tag if tag else DEFAULT_CONFIG_TAG


def load_existing_csv_rows(csv_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not csv_path.exists():
        return [], []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader), (reader.fieldnames or [])
    except Exception as exc:
        print(f"[Warning] Failed to read existing CSV {csv_path}: {exc}")
        return [], []


def _extract_info(data: object) -> dict:
    if isinstance(data, dict):
        if isinstance(data.get("ranking"), dict):
            return data["ranking"]
        if isinstance(data.get("configuration"), dict):
            return data["configuration"]
        return data
    if isinstance(data, str):
        return {"function": data}
    return {}


def _extract_ranking_fields(data: object) -> tuple[str | None, str | None, dict]:
    info = _extract_info(data)
    ranking = info.get("function") or info.get("ranking_function")
    if not ranking and isinstance(data, dict):
        ranking = data.get("function") or data.get("ranking_function")
    ranking_type = info.get("type") or (data.get("type") if isinstance(data, dict) else None)
    if isinstance(ranking, str):
        ranking = ranking.strip()
    return ranking, ranking_type, info


def _extract_template_fields(data: object) -> tuple[str | None, int | None, dict]:
    info = _extract_info(data)
    template_type = info.get("type") or info.get("template_type")
    depth = info.get("depth") or info.get("template_depth")
    if isinstance(depth, str):
        try:
            depth = int(depth)
        except Exception:
            pass
    return template_type, depth, info


def _load_loops_from_yaml(path: Path) -> tuple[List[Dict[str, Any]], str | None, str | None]:
    try:
        content = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[Warning] Failed to parse YAML {path}: {exc}")
        return [], None, None

    if not isinstance(content, dict):
        return [], None, None

    source_path = content.get("source_path")
    source_file = content.get("source_file")

    loops: List[Dict[str, Any]] = []
    if isinstance(content.get("loops"), list):
        for idx, item in enumerate(content.get("loops") or []):
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            loop_id = item.get("id") or item.get("loop_id") or (idx + 1)
            loops.append({"loop_id": loop_id, "code": code, "invariants": []})
    elif isinstance(content.get("invariants_result"), list):
        for idx, item in enumerate(content.get("invariants_result") or []):
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "").strip()
            if not code:
                continue
            loop_id = item.get("loop_id") or item.get("id") or (idx + 1)
            invs = item.get("invariants") if isinstance(item.get("invariants"), list) else []
            loops.append({"loop_id": loop_id, "code": code, "invariants": invs})

    return loops, source_path, source_file


def run_experiments() -> None:
    parser = argparse.ArgumentParser(description="Run ranking-function inference experiments.")
    parser.add_argument("--mode", type=str, default="rf", choices=["rf", "rf_template"],
                        help="Run direct RF or RF template inference (default: rf)")
    parser.add_argument("--config-tag", type=str, default="default",
                        help="Tag in llm_config.json to select which LLM model to use (default: 'default')")
    parser.add_argument("--input-path", type=str,
                        default="results",
                        help="Path to extract YAML file or directory relative to project root (default: 'results')")
    parser.add_argument("--config-tags", nargs="*", default=None,
                        help="Config tags to run in batch (space/comma-separated). Overrides --config-tag when set.")
    parser.add_argument("--prompt-version", type=str, default=None,
                        help="Prompt base name under prompts/rf (default: rf_direct or rf_template)")
    parser.add_argument("--prompt-batch", action="store_true",
                        help="Batch all prompts under playground/prompts/rf/ (filtered by mode)")
    parser.add_argument("--overwrite-used-prompts", action="store_true",
                        help="Overwrite results for prompts already recorded in the output CSV")
    parser.add_argument("--temp-strategy", type=str, default="all",
                        choices=["all", "deterministic", "balanced", "creative"],
                        help="Which temperature strategy to run (default: all)")
    args = parser.parse_args()

    config_tags = parse_cli_list(args.config_tags)
    if not config_tags:
        config_tags = [args.config_tag]
    seen_tags = set()
    config_tags = [tag for tag in config_tags if not (tag in seen_tags or seen_tags.add(tag))]

    mode = args.mode
    default_prompt = "rf_direct" if mode == "rf" else "rf_template"
    prompt_version = args.prompt_version or default_prompt

    # 1. Setup Environment
    playground_dir = CURRENT_FILE.parent
    prompts_dir = playground_dir / "prompts"
    output_dir = playground_dir / "output" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Define Experiments
    params_deterministic = {"temperature": 0.0, "top_p": 1.0}
    params_balanced = {"temperature": 0.7, "top_p": 0.9}
    params_creative = {"temperature": 1.0, "top_p": 1.0}

    strategy_catalog = {
        "deterministic": {"label": "Deterministic", "params": params_deterministic, "description": "temp=0"},
        "balanced": {"label": "Balanced", "params": params_balanced, "description": "temp=0.7"},
        "creative": {"label": "Creative", "params": params_creative, "description": "temp=1.0"},
    }
    if args.temp_strategy == "all":
        selected_strategies = ["deterministic", "balanced", "creative"]
    else:
        selected_strategies = [args.temp_strategy]

    def build_experiments(prompt_name: str, prefix: str | None = None) -> List[Dict[str, Any]]:
        exp_prefix = prefix or prompt_name
        result = []
        for strategy_name in selected_strategies:
            strategy = strategy_catalog[strategy_name]
            result.append({
                "name": f"{exp_prefix}_{strategy['label']}",
                "prompt_version": prompt_name,
                "params": strategy["params"],
                "description": strategy["description"],
            })
        return result

    experiments = build_experiments(prompt_version, prefix=prompt_version)
    if args.prompt_batch:
        prompt_versions = discover_rf_prompts(prompts_dir, mode)
        if not prompt_versions:
            print(f"[Error] No rf prompts found in: {prompts_dir / 'rf'}")
            return
        print("Prompt batch enabled. Prompts to run:")
        print(", ".join(prompt_versions))
        experiments = []
        for pv in prompt_versions:
            experiments.extend(build_experiments(pv))
    experiment_names_in_run = {exp["name"] for exp in experiments}

    # 3. Batch Processing Setup
    input_rel_path = args.input_path
    target_path = PROJECT_ROOT / input_rel_path
    if Path(input_rel_path).is_absolute():
        target_path = Path(input_rel_path)

    target_files = []
    target_root = target_path
    if target_path.is_file():
        target_files = [target_path]
        target_root = target_path.parent
    elif target_path.is_dir():
        yaml_files = list(target_path.rglob("*.yml"))
        yaml_files.extend(target_path.rglob("*.yaml"))
        target_files = sorted({p.resolve() for p in yaml_files})
        target_root = target_path
    else:
        print(f"[Error] Target path not found: {target_path}")
        return

    print(f"Found {len(target_files)} YAML files to process in: {target_path}")

    if mode == "rf":
        csv_columns = [
            "timestamp",
            "experiment_name",
            "target_file",
            "source_path",
            "source_file",
            "loop_id",
            "prompt_version",
            "config_tag",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "ranking_function",
            "ranking_type",
            "parsed_ranking",
            "raw_response_snippet",
        ]
    else:
        csv_columns = [
            "timestamp",
            "experiment_name",
            "target_file",
            "source_path",
            "source_file",
            "loop_id",
            "prompt_version",
            "config_tag",
            "model",
            "temperature",
            "top_p",
            "max_tokens",
            "latency_ms",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "template_type",
            "template_depth",
            "parsed_configuration",
            "raw_response_snippet",
        ]

    # 4. Run Loop per config tag
    for config_tag in config_tags:
        try:
            print(f"\nInitializing LLM Client with config tag: '{config_tag}'")
            llm_client = TrackingLLMClient(config_name="llm_config.json", config_tag=config_tag)
        except Exception as e:
            print(f"Failed to initialize LLM Client: {e}")
            continue

        prompt_repo = PromptRepository(root=prompts_dir)
        predictor = PlaygroundPredictor(llm_client=llm_client, prompt_repo=prompt_repo)

        for idx, target_file in enumerate(target_files):
            print(f"\n[{idx + 1}/{len(target_files)}] Processing: {target_file.name} ...")

            loops, source_path, source_file = _load_loops_from_yaml(target_file)
            if not loops:
                print(f"  -> Skipping (no loops found): {target_file.name}")
                continue

            try:
                rel_path = target_file.relative_to(target_root)
            except ValueError:
                rel_path = Path(target_file.name)

            file_csv_path = output_dir / rel_path.parent / (rel_path.name + ".csv")
            file_csv_path.parent.mkdir(parents=True, exist_ok=True)

            file_exists = file_csv_path.exists()
            existing_rows, existing_fieldnames = load_existing_csv_rows(file_csv_path) if file_exists else ([], [])
            needs_header_upgrade = file_exists and "config_tag" not in existing_fieldnames
            used_experiments = {
                row.get("experiment_name", "").strip()
                for row in existing_rows
                if row.get("experiment_name")
                and normalize_config_tag(row.get("config_tag")) == config_tag
            }
            if used_experiments and not args.overwrite_used_prompts:
                experiments_to_run = [
                    exp for exp in experiments
                    if exp["name"] not in used_experiments
                ]
                skipped = sorted(
                    {exp["name"] for exp in experiments} & used_experiments
                )
                if skipped:
                    print(f"  -> Skipping experiments already recorded: {', '.join(skipped)}")
                if not experiments_to_run:
                    print("  -> Skipping file (all experiments already recorded). Use --overwrite-used-prompts to rerun.")
                    continue
            else:
                experiments_to_run = experiments

            file_mode = "a"
            kept_rows = []
            if file_exists and (args.overwrite_used_prompts or needs_header_upgrade):
                if args.overwrite_used_prompts:
                    kept_rows = [
                        row for row in existing_rows
                        if not (
                            row.get("experiment_name", "").strip() in experiment_names_in_run
                            and normalize_config_tag(row.get("config_tag")) == config_tag
                        )
                    ]
                else:
                    kept_rows = existing_rows
                file_mode = "w"

            with open(file_csv_path, mode=file_mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                if file_mode == "w":
                    writer.writeheader()
                    if kept_rows:
                        writer.writerows(kept_rows)
                elif not file_exists:
                    writer.writeheader()

                for exp in experiments_to_run:
                    exp_name = exp["name"]
                    p_version = exp["prompt_version"]
                    llm_params = exp.get("params", {})

                    print(f"  -> Experiment: {exp_name} ", end="")
                    for loop_entry in loops:
                        loop_code = loop_entry["code"]
                        loop_id = loop_entry["loop_id"]
                        invs = loop_entry.get("invariants") or []

                        try:
                            response, parsed = predictor.infer_ranking(
                                code=loop_code,
                                invariants=invs,
                                references=[],
                                prompt_version=p_version,
                                llm_params=llm_params,
                            )
                            meta = llm_client.last_metadata
                            raw_snippet = (response or "").strip().replace("\n", " ")[:200]

                            row = {
                                "timestamp": datetime.datetime.now().isoformat(),
                                "experiment_name": exp_name,
                                "target_file": target_file.name,
                                "source_path": source_path or "",
                                "source_file": source_file or "",
                                "loop_id": loop_id,
                                "prompt_version": p_version,
                                "config_tag": config_tag,
                                "model": meta.get("model", "unknown"),
                                "temperature": llm_params.get("temperature", ""),
                                "top_p": llm_params.get("top_p", ""),
                                "max_tokens": llm_params.get("max_tokens", ""),
                                "latency_ms": meta.get("latency_ms", 0),
                                "prompt_tokens": meta.get("prompt_tokens", 0),
                                "completion_tokens": meta.get("completion_tokens", 0),
                                "total_tokens": meta.get("total_tokens", 0),
                                "raw_response_snippet": raw_snippet,
                            }

                            if mode == "rf":
                                ranking, ranking_type, info = _extract_ranking_fields(parsed)
                                row.update({
                                    "ranking_function": ranking or "",
                                    "ranking_type": ranking_type or "",
                                    "parsed_ranking": json.dumps(info, ensure_ascii=False),
                                })
                            else:
                                template_type, depth, info = _extract_template_fields(parsed)
                                row.update({
                                    "template_type": template_type or "",
                                    "template_depth": depth if depth is not None else "",
                                    "parsed_configuration": json.dumps(info, ensure_ascii=False),
                                })

                            writer.writerow(row)
                            f.flush()

                        except Exception as e:
                            print(f"| Failed on loop {loop_id}: {e}")
                            traceback.print_exc()
                    print("| Done")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    run_experiments()
