import subprocess
import shutil
import json
import os
from typing import Tuple, Optional

class SeaHornVerifier:
    def __init__(self, command_template: str = "sea pf {file}"):
        """
        command_template: Template string for the verification command.
        Example: "sea pf {file} --vac" or "docker run -v ... seahorn/seahorn sea pf ..."
        Default assumes 'sea' is in PATH.
        """
        self.command_template = command_template

    def verify(self, file_path: str, timeout: int = 60) -> Tuple[str, str]:
        """
        Runs the verification command on the file.
        Returns: (status, output)
        status: 'safe' (unsat), 'unsafe' (sat), 'unknown', 'error', 'timeout'
        """
        abs_path = os.path.abspath(file_path)
        cmd = self.command_template.format(file=abs_path)
        
        # Split command for subprocess unless it's a complex shell string
        # Simple splitting by space (caveat: paths with spaces need detailed handling, 
        # but for research code this often suffices or requires shell=True)
        cmd_args = cmd.split()
        
        try:
            result = subprocess.run(
                cmd_args, 
                capture_output=True, 
                text=True, 
                timeout=timeout
            )
            stdout = result.stdout
            stderr = result.stderr
            return self._parse_output(stdout, stderr)
            
        except subprocess.TimeoutExpired:
            return "timeout", "Verification timed out."
        except Exception as e:
            return "error", str(e)

    def _parse_output(self, stdout: str, stderr: str) -> Tuple[str, str]:
        """
        Parses SeaHorn output.
        SeaHorn typically outputs 'unsat' for SAFE and 'sat' for UNSAFE.
        """
        full_output = stdout + "\n" + stderr
        if "unsat" in stdout:
            return "safe", full_output
        elif "sat" in stdout:
            return "unsafe", full_output
        else:
            return "unknown", full_output
