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

from evolve_term.utils import parse_llm_yaml

class PlaygroundPredictor(Predictor):
    """
    Subclass of Predictor to support flexible parsing logic for playground experiments
    and passing LLM parameters (temp, top_p, etc.).
    """
    def infer_invariants(self, code: str, references: List[Any], prompt_version: str = "acsl_cot", llm_params: Dict[str, Any] = None) -> List[str]:
        prompt_name = f"invariants/{prompt_version}"
        prompt = self.prompt_repo.render(
            prompt_name,
            code=code,
            references=json.dumps([ref.__dict__ for ref in references], ensure_ascii=False, indent=2)
        )
        # Apply default max_tokens if not specified
        if "max_tokens" not in (llm_params or {}):
            if prompt_version.endswith("_cot") or prompt_version.endswith("_cot_fewshot"):
                prompt["max_tokens"] = 8192
        
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

def run_experiments():
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

    experiments = [
        # Group 1: Deterministic Baseline
        {
            "name": "ACSL_CoT_Deterministic",
            "prompt_version": "acsl_cot",
            "params": params_deterministic,
            "description": "Baseline CoT, temp=0"
        },
        # Group 2: Balanced Exploration
        {
            "name": "ACSL_CoT_Balanced",
            "prompt_version": "acsl_cot",
            "params": params_balanced,
            "description": "Baseline CoT, temp=0.7"
        },
        # Group 3: Creative Exploration
        {
            "name": "ACSL_CoT_Creative",
            "prompt_version": "acsl_cot",
            "params": params_creative,
            "description": "Baseline CoT, temp=1.0"
        }
    ]

    # 4. Initialize Components
    # Load config from default location, but we will wrap the client
    try:
        llm_client = TrackingLLMClient(config_name="llm_config.json")
    except Exception as e:
        print(f"Failed to initialize LLM Client: {e}")
        return

    # Use a custom prompt repository pointing to playground/prompts
    prompt_repo = PromptRepository(root=prompts_dir)
    
    # Use PlaygroundPredictor for flexible parsing
    predictor = PlaygroundPredictor(llm_client=llm_client, prompt_repo=prompt_repo)

    # 3. Batch Processing Setup
    # Config: Directory or File (Relative to PROJECT_ROOT/data)
    # Default set to a folder for batch processing, or can be a specific file
    input_rel_path = "Loopy_dataset_InvarBenchmark/loop_invariants/code2inv" 
    input_base = PROJECT_ROOT / "data"
    target_path = input_base / input_rel_path
    
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
        "model", "temperature", "top_p", "max_tokens",
        "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
        "invariants_count", "parsed_invariants", "raw_response_snippet"
    ]

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

        with open(file_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            if not file_exists:
                writer.writeheader()

            for exp in experiments:
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
