import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

class SVMRankerClient:
    def __init__(self, tool_root: str | Path):
        self.tool_root = Path(tool_root)
        self.cli_script = self.tool_root / "src" / "CLIMain.py"

    def run(self, file_path: Path, mode: str = "lnested", depth: int = 1, timeout: int = 300) -> Optional[str]:
        """
        Run SVMRanker on the given file.
        
        Args:
            file_path: Path to the C file.
            mode: "lnested" or "lmulti" (or other modes supported by SVMRanker).
            depth: Depth bound.
            timeout: Timeout in seconds.
            
        Returns:
            The stdout output from SVMRanker if successful, None otherwise.
        """
        if not self.cli_script.exists():
            # Fail silently or log warning? For now, just return None so pipeline falls back.
            return None

        # Construct command
        # python3 src/CLIMain.py [mode] --depth_bound [depth] [file]
        cmd = [
            sys.executable,
            str(self.cli_script),
            mode,
            "--depth_bound",
            str(depth),
            str(file_path)
        ]

        try:
            # We run it with cwd set to tool_root because often tools depend on relative paths
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.tool_root)
            )
            
            if result.returncode != 0:
                return None
            
            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            return None
        except Exception as e:
            return None
