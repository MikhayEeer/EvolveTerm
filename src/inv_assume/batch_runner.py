import os
import glob
import pandas as pd
from typing import Optional
from .pipeline import ASTInstrumentationPipeline
from .verifier import SeaHornVerifier

class BatchRunner:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 llm_config: str = "llm_config.json",
                 strategy: str = "simple",
                 enable_verification: bool = False,
                 verifier: Optional[SeaHornVerifier] = None,
                 verify_timeout: int = 60):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_verification = enable_verification
        self.verify_timeout = verify_timeout
        self.pipeline = ASTInstrumentationPipeline(
            llm_config=llm_config,
            strategy=strategy,
            verifier=verifier,
        )

        if enable_verification and self.pipeline.verifier is None:
            self.pipeline.verifier = SeaHornVerifier()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run(self):
        c_files = glob.glob(os.path.join(self.input_dir, "*.c"))
        print(f"Found {len(c_files)} C files in {self.input_dir}")
        
        results = []
        
        for idx, file_path in enumerate(c_files):
            file_name = os.path.basename(file_path)
            print(f"[{idx+1}/{len(c_files)}] Processing {file_name}...")
            
            # Record execution metadata
            result_entry = {
                "file": file_name,
                "status": "pending",
                "loops_found": 0,
                "invariants_generated": 0,
                "output_path": "",
                "verification_status": "n/a",
                "verification_verdict": "n/a",
                "verification_output": "",
                "error": ""
            }
            
            try:
                pipeline_result = self.pipeline.run(
                    file_path,
                    self.output_dir,
                    verify=self.enable_verification,
                    verify_timeout=self.verify_timeout,
                )
                result_entry["status"] = "instrumented"
                if isinstance(pipeline_result, dict):
                    result_entry["loops_found"] = pipeline_result.get("loops_count", 0)
                    result_entry["invariants_generated"] = pipeline_result.get("injections_count", 0)
                    result_entry["output_path"] = pipeline_result.get("output_path", "")
                    if self.enable_verification:
                        result_entry["verification_status"] = pipeline_result.get("verification_status", "n/a")
                        result_entry["verification_verdict"] = pipeline_result.get("verification_verdict", "n/a")
                        result_entry["verification_output"] = pipeline_result.get("verification_output", "")
                    
            except Exception as e:
                print(f"  Error processing {file_name}: {e}")
                result_entry["status"] = "error"
                result_entry["error"] = str(e)
            
            results.append(result_entry)

        # Save Summary
        df = pd.DataFrame(results)
        summary_path = os.path.join(self.output_dir, "batch_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"Batch processing complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch run invariant instrumentation and verification")
    parser.add_argument("--input_dir", required=True, help="Directory containing .c files")
    parser.add_argument("--output_dir", required=True, help="Directory to save instrumented files")
    parser.add_argument("--strategy", choices=["simple", "2stage"], default="simple",
                        help="Invariant generation strategy (simple=one-shot, 2stage=atom-filter-candidate)")
    parser.add_argument("--config", default="llm_config.json", help="LLM configuration file")
    parser.add_argument("--verify", action="store_true", help="Enable SeaHorn verification (Docker)")
    parser.add_argument("--seahorn-image", default="seahorn/seahorn-llvm14:nightly", help="SeaHorn docker image")
    parser.add_argument("--seahorn-timeout", type=int, default=60, help="SeaHorn timeout seconds")
    parser.add_argument("--docker-mount-root", default=None, help="Optional docker mount root (must contain output dir)")
    
    args = parser.parse_args()

    verifier = None
    if args.verify:
        verifier = SeaHornVerifier(
            docker_image=args.seahorn_image,
            docker_mount_root=args.docker_mount_root,
        )

    runner = BatchRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm_config=args.config,
        strategy=args.strategy,
        enable_verification=args.verify,
        verifier=verifier,
        verify_timeout=args.seahorn_timeout,
    )
    runner.run()
