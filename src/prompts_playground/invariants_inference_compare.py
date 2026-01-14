"""
Script to compare different invariant inference prompts in the playground environment.
Function:
1. Define different experiments with varying prompt versions in the `experiments` list.
2. Load a target code file for invariant inference.
Usage:
```bash
python src/prompts_playground/invariants_inference_compare.py
```
"""

import sys
import time
import csv
import json
import datetime
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Add src to sys.path to allow imports from evolve_term
CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

# Project root for data loading
PROJECT_ROOT = SRC_ROOT.parent

from evolve_term.llm_client import APILLMClient, LLMUnavailableError
from evolve_term.prompts_loader import PromptRepository
from evolve_term.predict import Predictor

from evolve_term.utils import parse_acsl_invariants, parse_llm_yaml

class PlaygroundPredictor(Predictor):
    """
    Subclass of Predictor to support flexible parsing logic for playground experiments
    and passing LLM parameters (temp, top_p, etc.).
    """
    def infer_invariants(self, code: str, 
                         references: List[Any], 
                         prompt_version: str = "acsl_cot", 
                         llm_params: Dict[str, Any] = None) -> List[str]:
        # using prompt_version to select prompt template
        # TODO: Add support for different prompt versions, or batch use different prompts
        prompt_name = f"invariants/{prompt_version}"
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        # Apply default max_tokens if not specified
        # default max_tokens is 2048
        if "max_tokens" not in (llm_params or {}):
            if prompt_version.endswith("_cot") or prompt_version.endswith("_cot_fewshot"):
                prompt["max_tokens"] = 4096
        
        # Merge LLM params into prompt dictionary so TrackingLLMClient can pick them up
        if llm_params:
            prompt.update(llm_params)

        response = self.llm_client.complete(prompt)
        
        # Flexible Parsing Logic
        data = parse_llm_yaml(response)
        invariants = []

        if isinstance(data, dict):
             # Strategy 1: Standard "invariants" key
            if "invariants" in data and isinstance(data["invariants"], list):
                invariants = data["invariants"]
            # Strategy 2: Singular "invariant" key
            elif "invariant" in data and isinstance(data["invariant"], list):
                invariants = data["invariant"]
            # Other strategies can be added here
        elif isinstance(data, list):
            # Strategy 3: Response is a direct list
            invariants = data
            
        print("[Debug] Playground Invariant End...\n")
        if not invariants:
            invariants = parse_acsl_invariants(response)
        if not invariants:
            print(f"[Debug] Invariant Parsing Failed or Empty. Raw Response:\n{response}\n")
            return []
        return [str(item) for item in invariants if str(item).strip()]

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
            # Handle dictionary prompt
            # Extract common LLM params to pass to the API
            supported_params = ["response_format", "max_tokens", "temperature", "top_p", "frequency_penalty", "presence_penalty", "stop", "seed"]
            for param in supported_params:
                if param in prompt:
                    val = prompt[param]
                    # Ensure numeric types are correct if needed, but OpenAI SDK handles mixed types well usually
                    if param == "max_tokens":
                         try:
                            val = int(val)
                         except: pass
                    request_overrides[param] = val
            
            if prompt.get("system"):
                messages.append({"role": "system", "content": prompt["system"]})
            if prompt.get("user"):
                messages.append({"role": "user", "content": prompt["user"]})
        
        last_exception = None
        start_time = time.time()
        
        for attempt in range(retry + 1):
            try:
                # Prepare args
                call_args = {
                    "model": self.model,
                    "messages": messages,
                    **self.payload_template, # default args like temp, etc.
                    **request_overrides
                }
                
                # Update with any specific overrides if we want to support per-experiment settings here
                # (currently relying on payload_template or request_overrides)

                response = self.client.chat.completions.create(**call_args)
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                # Capture usage
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
                    "call_args": str(call_args) # for debugging
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
        
        raise LLMUnavailableError(f"LLM provider error after {retry} retries: {last_exception}") from last_exception

import argparse

DEFAULT_CONFIG_TAG = "default"
PROMPT_SYSTEM_SUFFIX = ".system.txt"
PROMPT_USER_SUFFIX = ".user.txt"

def discover_invariant_prompts(prompts_dir: Path) -> List[str]:
    invariant_dir = prompts_dir / "invariants"
    if not invariant_dir.exists():
        return []
    system_files = {
        path.name[:-len(PROMPT_SYSTEM_SUFFIX)]
        for path in invariant_dir.glob(f"*{PROMPT_SYSTEM_SUFFIX}")
    }
    user_files = {
        path.name[:-len(PROMPT_USER_SUFFIX)]
        for path in invariant_dir.glob(f"*{PROMPT_USER_SUFFIX}")
    }
    return sorted(system_files & user_files)

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
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader), (reader.fieldnames or [])
    except Exception as exc:
        print(f"[Warning] Failed to read existing CSV {csv_path}: {exc}")
        return [], []

def run_experiments():
    # 0. Parse Arguments
    parser = argparse.ArgumentParser(description="Run invariant inference experiments.")
    parser.add_argument("--config-tag", type=str, default="default", 
                        help="Tag in llm_config.json to select which LLM model to use (default: 'default')")
    parser.add_argument("--input-path", type=str, 
                        default="Loopy_dataset_InvarBenchmark/loop_invariants/code2inv",
                        help="Path to C file or directory relative to project data/ folder (default: 'Loopy_dataset_InvarBenchmark/loop_invariants/code2inv')")
    parser.add_argument("--config-tags", nargs="*", default=None,
                        help="Config tags to run in batch (space/comma-separated). Overrides --config-tag when set.")
    parser.add_argument("--prompt-batch", action="store_true",
                        help="Batch all prompts under playground/prompts/invariants/")
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

    # 1. Setup Environment
    playground_dir = CURRENT_FILE.parent
    prompts_dir = playground_dir / "prompts"
    output_dir = playground_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # 2. Define Experiments
    # Recommended Parameter Groups
    params_deterministic = {"temperature": 0.0, "top_p": 1.0}
    params_balanced = {"temperature": 0.7, "top_p": 0.9}
    params_creative = {"temperature": 1.0, "top_p": 1.0}

    strategy_catalog = {
        "deterministic": {"label": "Deterministic", "params": params_deterministic, "description": "temp=0"},
        "balanced": {"label": "Balanced", "params": params_balanced, "description": "temp=0.7"},
        "creative": {"label": "Creative", "params": params_creative, "description": "temp=1.0"}
    }
    if args.temp_strategy == "all":
        selected_strategies = ["deterministic", "balanced", "creative"]
    else:
        selected_strategies = [args.temp_strategy]

    def build_experiments(prompt_version: str, prefix: str | None = None) -> List[Dict[str, Any]]:
        exp_prefix = prefix or prompt_version
        result = []
        for strategy_name in selected_strategies:
            strategy = strategy_catalog[strategy_name]
            result.append({
                "name": f"{exp_prefix}_{strategy['label']}",
                "prompt_version": prompt_version,
                "params": strategy["params"],
                "description": strategy["description"]
            })
        return result

    experiments = build_experiments("acsl_cot", prefix="ACSL_CoT")
    if args.prompt_batch:
        prompt_versions = discover_invariant_prompts(prompts_dir)
        if not prompt_versions:
            print(f"[Error] No invariant prompts found in: {prompts_dir / 'invariants'}")
            return
        print("Prompt batch enabled. Prompts to run:")
        print(", ".join(prompt_versions))
        experiments = []
        for prompt_version in prompt_versions:
            experiments.extend(build_experiments(prompt_version))
    experiment_names_in_run = {exp["name"] for exp in experiments}

    # 3. Batch Processing Setup
    # Config: Directory or File (Relative to PROJECT_ROOT/data)
    # Default set to a folder for batch processing, or can be a specific file
    input_rel_path = args.input_path
    input_base = PROJECT_ROOT / "data"
    target_path = input_base / input_rel_path
    
    # Allow absolute paths too if user insists, though help text says relative to data
    if Path(input_rel_path).is_absolute():
        target_path = Path(input_rel_path)
    
    target_files = []
    if target_path.is_file():
        target_files = [target_path]
    elif target_path.is_dir():
        # Recursive search for .c files
        target_files = sorted(list(target_path.rglob("*.c")))
        # Optional: Limit for testing
        # target_files = target_files[:1] 
    else:
        print(f"[Error] Target path not found: {target_path}")
        return

    print(f"Found {len(target_files)} files to process in: {target_path}")

    csv_columns = [
        "timestamp", "experiment_name", "target_file", "prompt_version",
        "config_tag",
        "model", "temperature", "top_p", "max_tokens",
        "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
        "invariants_count", "parsed_invariants", "raw_response_snippet"
    ]

    # 4. Run Loop per config tag
    for config_tag in config_tags:
        # Initialize Components
        # Load config from default location, but we will wrap the client
        try:
            print(f"\nInitializing LLM Client with config tag: '{config_tag}'")
            llm_client = TrackingLLMClient(config_name="llm_config.json", config_tag=config_tag)
        except Exception as e:
            print(f"Failed to initialize LLM Client: {e}")
            continue

        # Use a custom prompt repository pointing to playground/prompts
        prompt_repo = PromptRepository(root=prompts_dir)
        
        # Use PlaygroundPredictor for flexible parsing
        predictor = PlaygroundPredictor(llm_client=llm_client, prompt_repo=prompt_repo)

        # 5. Run Loop Iterating over files
        for idx, target_file in enumerate(target_files):
            print(f"\n[{idx+1}/{len(target_files)}] Processing: {target_file.name} ...")
            
            try:
                code_content = target_file.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Skipping {target_file.name}: Read error {e}")
                continue

            # Determine Output CSV Path (Mirroring directory structure)
            try:
                rel_path = target_file.relative_to(input_base)
            except ValueError:
                rel_path = Path(target_file.name)
            
            # Example: output/Loopy_dataset.../code2inv/1.c.csv
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

            file_mode = 'a'
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
                file_mode = 'w'

            with open(file_csv_path, mode=file_mode, newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
                if file_mode == 'w':
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
                    
                    try:
                        # Run inference
                        invariants = predictor.infer_invariants(
                            code=code_content, 
                            references=[], 
                            prompt_version=p_version,
                            llm_params=llm_params
                        )
                        
                        # Collect metrics
                        meta = llm_client.last_metadata
                        
                        row = {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "experiment_name": exp_name,
                            "target_file": target_file.name,
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
                            "invariants_count": len(invariants),
                            "parsed_invariants": json.dumps(invariants, ensure_ascii=False),
                            "raw_response_snippet": "" 
                        }
                        
                        writer.writerow(row)
                        f.flush()
                        
                        print(f"| Found: {len(invariants)} | Latency: {row['latency_ms']}ms")

                    except Exception as e:
                        print(f"| Failed: {e}")
                        traceback.print_exc()

    print(f"\nAll experiments finished.")

if __name__ == "__main__":
    run_experiments()
