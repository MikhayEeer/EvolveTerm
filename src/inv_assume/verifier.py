import os
import shlex
import subprocess
from typing import Tuple, Optional, Sequence

class SeaHornVerifier:
    def __init__(
        self,
        command_template: Optional[str] = None,
        use_docker: bool = True,
        docker_image: str = "seahorn/seahorn-llvm14:nightly",
        docker_mount_root: Optional[str] = None,
        sea_args: Optional[Sequence[str]] = None,
    ):
        """
        command_template: Template string for the verification command.
        Example: "sea pf {file} --vac" or "docker run -v ... seahorn/seahorn sea pf ..."
        If command_template is set, it takes precedence over use_docker.
        """
        self.command_template = command_template
        self.use_docker = use_docker and command_template is None
        self.docker_image = docker_image
        self.docker_mount_root = docker_mount_root
        self.sea_args = list(sea_args) if sea_args is not None else ["pf"]

    def verify(self, file_path: str, timeout: int = 60) -> Tuple[str, str, str]:
        """
        Runs the verification command on the file.
        Returns: (status, output, verdict)
        status: 'safe' (unsat), 'unsafe' (sat), 'unknown', 'error', 'timeout'
        verdict: 'unsat', 'sat', or 'unknown'
        """
        abs_path = os.path.abspath(file_path)
        if self.command_template:
            cmd = self.command_template.format(file=abs_path)
            cmd_args = shlex.split(cmd)
        elif self.use_docker:
            cmd_args = self._build_docker_command(abs_path)
        else:
            cmd_args = ["sea"] + self.sea_args + [abs_path]
        
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
            return "timeout", "Verification timed out.", "unknown"
        except Exception as e:
            return "error", str(e), "unknown"

    def _parse_output(self, stdout: str, stderr: str) -> Tuple[str, str, str]:
        """
        Parses SeaHorn output.
        SeaHorn typically outputs 'unsat' for SAFE and 'sat' for UNSAFE.
        """
        full_output = stdout + "\n" + stderr
        if "unsat" in stdout:
            return "safe", full_output, "unsat"
        elif "sat" in stdout:
            return "unsafe", full_output, "sat"
        else:
            return "unknown", full_output, "unknown"

    def _build_docker_command(self, abs_path: str) -> list:
        mount_root = os.path.abspath(self.docker_mount_root or os.path.dirname(abs_path))
        rel_path = os.path.relpath(abs_path, mount_root)
        if rel_path.startswith(".."):
            raise ValueError(
                f"File '{abs_path}' is not under docker_mount_root '{mount_root}'"
            )
        container_file = os.path.join("/work", rel_path)
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{mount_root}:/work",
            "-w",
            "/work",
            self.docker_image,
            "sea",
            *self.sea_args,
            container_file,
        ]
