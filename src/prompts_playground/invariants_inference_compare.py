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

class TrackingLLMClient(APILLMClient):
    """
    A subclass of APILLMClient that tracks usage metrics and latency.
    Since the base APILLMClient.complete only returns string, we override it 
    to capture the full response details.
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
            response_format = prompt.get("response_format")
            if response_format is not None:
                request_overrides["response_format"] = response_format
            
            max_tokens = prompt.get("max_tokens")
            if max_tokens is not None:
                try:
                    request_overrides["max_tokens"] = int(max_tokens)
                except (TypeError, ValueError):
                    pass
            
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

    # 3. Load Data (Target Code)
    # Configurable test case path (relative to PROJECT_ROOT / "data")
    test_case_rel_path = "Loopy_dataset_InvarBenchmark/loop_invariants/code2inv/1.c"
    target_file = PROJECT_ROOT / "data" / test_case_rel_path
    
    if not target_file.exists():
        print(f"[Error] Target file not found: {target_file}")
        return

    print(f"Loading target code from: {target_file}")
    code_content = target_file.read_text(encoding="utf-8")

    # Generate CSV filename based on test case path
    # Replace path separators with dashes for the filename
    safe_rel_path = test_case_rel_path.replace("/", "-").replace("\\", "-")
    csv_path = output_dir / f"comparison_results_{safe_rel_path}.csv"
    print(f"Results will be appended to: {csv_path}")

    # 2. Define Experiments
    # You can add more experiments here with different prompt_versions or params
    experiments = [
        {
            "name": "Validation_Run_ACSL_CoT",
            "prompt_version": "acsl_cot",
            "description": "Baseline CoT prompt copied from main repo"
        },
        # Example of another experiment (ensure the prompt file exists in playground/prompts/invariants/)
        # {
        #     "name": "Experimental_Prompt_V2",
        #     "prompt_version": "acsl_cot_v2", 
        #     "description": "Modified systematic analysis"
        # }
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
    
    predictor = Predictor(llm_client=llm_client, prompt_repo=prompt_repo)

    # 5. Run Loop
    results = []
    
    # Check if CSV exists to write headers
    file_exists = csv_path.exists()
    
    csv_columns = [
        "timestamp", "experiment_name", "target_file", "prompt_version", 
        "model", "latency_ms", "prompt_tokens", "completion_tokens", "total_tokens",
        "invariants_count", "parsed_invariants", "raw_response_snippet"
    ]

    with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        if not file_exists:
            writer.writeheader()

        for exp in experiments:
            exp_name = exp["name"]
            p_version = exp["prompt_version"]
            print(f"\n--- Running Experiment: {exp_name} (Prompt: {p_version}) ---")
            
            try:
                # Run inference
                # references passed as empty list per requirement
                invariants = predictor.infer_invariants(
                    code=code_content, 
                    references=[], 
                    prompt_version=p_version
                )
                
                # Collect metrics
                meta = llm_client.last_metadata
                
                # It's possible infer_invariants parsed the output but we want the raw output too?
                # predictor doesn't return raw response, but we can't easily get it without modifying Predictor.
                # However, since we are only using this locally, we can assume the last call to complete 
                # corresponds to this inference.
                # NOTE: Predictor calls 'complete', which returns 'content'.
                # But 'content' is not stored in 'meta' by our TrackingClient (it returns it).
                # We can't access the specific string output unless we capture it inside complete 
                # or modify Predictor to return it. 
                # For now let's log what we have.
                
                row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "experiment_name": exp_name,
                    "target_file": target_file.name,
                    "prompt_version": p_version,
                    "model": meta.get("model", "unknown"),
                    "latency_ms": meta.get("latency_ms", 0),
                    "prompt_tokens": meta.get("prompt_tokens", 0),
                    "completion_tokens": meta.get("completion_tokens", 0),
                    "total_tokens": meta.get("total_tokens", 0),
                    "invariants_count": len(invariants),
                    "parsed_invariants": json.dumps(invariants, ensure_ascii=False),
                    "raw_response_snippet": "" # Not easily accessible without changing Predictor signature, leaving empty for now
                }
                
                writer.writerow(row)
                f.flush() # Ensure write
                
                print(f"[Success] {exp_name}: Found {len(invariants)} invariants. (Latency: {row['latency_ms']}ms)")
                print(f"Invariants: {invariants}")

            except Exception as e:
                print(f"[Error] Experiment {exp_name} failed: {e}")
                traceback.print_exc()

    print(f"\nAll experiments finished. Results saved to: {csv_path}")

if __name__ == "__main__":
    run_experiments()
