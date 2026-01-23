import subprocess
import tempfile
import sys
from pathlib import Path
from typing import List

class Z3Verifier:
    def __init__(self, llm_client, prompt_repo):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo

    def verify(self, code: str, invariants: List[str], ranking_function: str) -> str:
        prompt = self.prompt_repo.render(
            "z3_verification",
            code=code,
            invariants="\n".join(invariants),
            ranking_function=ranking_function
        )
        script_response = self.llm_client.complete(prompt)
        
        # Extract python code
        script = script_response
        if "```python" in script:
            script = script.split("```python")[1].split("```")[0].strip()
        elif "```" in script:
            script = script.split("```")[1].split("```")[0].strip()
            
        # Run in temp file
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tmp:
                tmp.write(script)
                tmp_path = tmp.name
            
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, 
                text=True, 
                timeout=10  # 10 seconds timeout for verification
            )
            output = (result.stdout or "").strip()
            error_output = (result.stderr or "").strip()
            
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)
            
            if "Verified" in output or "Verified" in error_output:
                return "Verified"
            elif "Failed" in output or "Failed" in error_output:
                # Extract the failure message
                lines = (output + "\n" + error_output).splitlines()
                for line in lines:
                    if "Failed" in line:
                        return line.strip()
                return "Failed"
            else:
                msg = error_output or output
                if result.returncode != 0 and not msg:
                    msg = f"Z3 script exited with code {result.returncode}"
                msg = msg or "Unknown verification output"
                return f"Error: {msg[:200]}..."  # Return partial output for debug
                
        except Exception as e:
            return f"Execution Error: {str(e)}"
