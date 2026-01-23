import os
import glob
import pandas as pd
from typing import List, Dict
import time
from .pipeline import ASTInstrumentationPipeline
from .verifier import SeaHornVerifier

class BatchRunner:
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 llm_config: str = "llm_config.json",
                 enable_verification: bool = False,
                 verifier_command: str = "sea pf {file}"):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.enable_verification = enable_verification
        self.pipeline = ASTInstrumentationPipeline(llm_config=llm_config)
        
        if enable_verification:
            self.verifier = SeaHornVerifier(command_template=verifier_command)
        else:
            self.verifier = None

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run(self):
        c_files = glob.glob(os.path.join(self.input_dir, "*.c"))
        print(f"Found {len(c_files)} C files in {self.input_dir}")
        
        results = []
        
        for idx, file_path in enumerate(c_files):
            file_name = os.path.basename(file_path)
            print(f"[{idx+1}/{len(c_files)}] Processing {file_name}...")
            
            output_file = os.path.join(self.output_dir, file_name)
            
            # Record execution metadata
            result_entry = {
                "file": file_name,
                "status": "pending",
                "loops_found": 0,
                "invariants_generated": 0,
                "verification_status": "n/a",
                "verification_output": "",
                "error": ""
            }
            
            try:
                # 1. Pipeline Execution
                # We need to hook into pipeline to count loops/invariants if we want stats.
                # For now, let's just run it.
                # To get stats, we might want to modify ASTInstrumentationPipeline to return info.
                # Let's interact with pipeline components directly or update pipeline.py later.
                # Assuming pipeline simply runs for now:
                
                # We read file to count loops just for stats (inefficient but safe or rely on logs)
                # Let's assume pipeline runs successfully.
                
                # To capture detailed loop info, we would modify pipeline.run to return it.
                # I will modify pipeline.py in next step to return metadata.
                # For now assuming it returns None or path.
                
                pipeline_result = self.pipeline.run(file_path, output_file)
                result_entry["status"] = "instrumented"
                if isinstance(pipeline_result, dict):
                    result_entry["loops_found"] = pipeline_result.get("loops_count", 0)
                    result_entry["invariants_generated"] = pipeline_result.get("injections_count", 0)
                
                # 2. Verification (Optional)
                if self.enable_verification and self.verifier:
                    print(f"  Verifying {file_name}...")
                    v_status, v_out = self.verifier.verify(output_file)
                    result_entry["verification_status"] = v_status
                    result_entry["verification_output"] = v_out[-200:] # Log last 200 chars
                    print(f"  -> Result: {v_status}")
                    
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
    parser.add_argument("--verify", action="store_true", help="Enable SeaHorn verification")
    parser.add_argument("--cmd", default="sea pf {file}", help="Verification command template")
    
    args = parser.parse_args()
    
    runner = BatchRunner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        enable_verification=args.verify,
        verifier_command=args.cmd
    )
    runner.run()
