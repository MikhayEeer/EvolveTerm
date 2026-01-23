#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from evolve_term.verifiers.seahorn_verifier import SeaHornVerifier, DEFAULT_SEAHORN_IMAGE


SUMMARY_FIELDS = [
    "csv_path",
    "row_index",
    "experiment_name",
    "prompt_version",
    "config_tag",
    "model",
    "temperature",
    "top_p",
    "target_file",
    "source_path",
    "invariants_count",
    "invariants_used",
    "dropped_invariants",
    "status",
    "reason",
    "log_file",
]


def _parse_invariants(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except json.JSONDecodeError:
        pass
    try:
        import ast

        data = ast.literal_eval(text)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        return []
    return []


def _slugify(value: str, max_len: int = 80) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._-") or "row"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:max_len - 9]}_{digest}"


def _discover_csv_files(output_root: Path) -> List[Path]:
    csv_files: List[Path] = []
    if not output_root.exists():
        return csv_files
    for child in sorted(output_root.iterdir()):
        if child.is_dir() and child.name.startswith("Loo"):
            csv_files.extend(sorted(child.rglob("*.csv")))
    return csv_files


def _resolve_source_path(csv_path: Path, output_root: Path, data_root: Path) -> Optional[Path]:
    try:
        rel = csv_path.relative_to(output_root)
    except ValueError:
        return None
    rel_no_csv = rel.with_suffix("")
    return data_root / rel_no_csv


def _combine_output(stdout: str, stderr: str) -> str:
    out = (stdout or "").strip()
    err = (stderr or "").strip()
    if out and err:
        return f"{out}\n{err}"
    return out or err


def _run_seahorn(
    code: str,
    docker_image: str,
    timeout_seconds: int,
    sea_args: List[str],
    container_name: Optional[str],
) -> Tuple[str, str, List[str], Optional[int], str]:
    with tempfile.TemporaryDirectory(prefix="seahorn_csv_") as tmp_dir:
        src_path = Path(tmp_dir) / "input.c"
        src_path.write_text(code, encoding="utf-8")
        volume_dir = src_path.parent.resolve()

        cmd = ["docker", "run", "--rm", "-v", f"{volume_dir}:/src", "-w", "/src"]
        if container_name:
            cmd += ["--name", container_name]
        cmd += [docker_image, "sea", "pf", f"/src/{src_path.name}"]
        if sea_args:
            cmd.extend(sea_args)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError:
            return "Error", "docker_not_found", cmd, None, "docker: command not found"
        except subprocess.TimeoutExpired as exc:
            output = _combine_output(getattr(exc, "stdout", ""), getattr(exc, "stderr", ""))
            return "Error", f"timeout:{output}" if output else "timeout", cmd, None, output
        except Exception as exc:  # pragma: no cover - defensive
            return "Error", f"exception:{exc}", cmd, None, str(exc)

        output = _combine_output(result.stdout, result.stderr)
        if "unsat" in output:
            return "Verified", "", cmd, result.returncode, output
        if "sat" in output:
            return "Failed", "", cmd, result.returncode, output
        if result.returncode != 0:
            return "Error", f"exit_{result.returncode}", cmd, result.returncode, output
        return "Error", "unexpected_output", cmd, result.returncode, output


def _write_log(
    log_path: Path,
    metadata: Dict[str, object],
    seahorn_cmd: List[str],
    seahorn_output: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in metadata.items():
        lines.append(f"{key}: {value}")
    lines.append(f"seahorn_cmd: {' '.join(seahorn_cmd)}")
    lines.append("seahorn_output:")
    lines.append(seahorn_output.strip() if seahorn_output.strip() else "(empty)")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_instrumented(path: Path, code: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def _iter_rows(csv_path: Path) -> Iterable[Tuple[int, Dict[str, str]]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Loopy CSV invariants with SeaHorn (Docker) and write detailed logs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "src/prompts_playground/output",
        help="Root directory containing Loo* CSV outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root directory for benchmark C sources.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=PROJECT_ROOT / "results/seahorn_csv_verification/logs",
        help="Directory for seahorn log files.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results/seahorn_csv_verification/summary.csv",
        help="CSV summary output file.",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default=DEFAULT_SEAHORN_IMAGE,
        help="SeaHorn docker image.",
    )
    parser.add_argument(
        "--container-name",
        type=str,
        default="",
        help="Optional docker container name.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="SeaHorn timeout in seconds.",
    )
    parser.add_argument(
        "--sea-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments for `sea pf` (pass after --sea-args).",
    )
    parser.add_argument(
        "--invariants-column",
        type=str,
        default="parsed_invariants",
        help="CSV column containing parsed invariants.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum number of rows to process in total (0 = no limit).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows that already have a log file.",
    )
    parser.add_argument(
        "--save-instrumented",
        action="store_true",
        help="Save instrumented C code alongside logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and log metadata without running SeaHorn.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to summary CSV instead of overwriting.",
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    data_root = args.data_root.resolve()
    logs_root = args.logs_root.resolve()
    summary_path = args.summary.resolve()
    sea_args = list(args.sea_args)
    if sea_args and sea_args[0] == "--":
        sea_args = sea_args[1:]

    csv_files = _discover_csv_files(output_root)
    if not csv_files:
        print(f"No CSV files found under {output_root}")
        return 1

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists() or not args.append
    summary_mode = "a" if args.append else "w"
    processed = 0

    verifier = SeaHornVerifier(docker_image=args.docker_image, timeout_seconds=args.timeout)

    with open(summary_path, summary_mode, newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()

        for csv_path in csv_files:
            source_path = _resolve_source_path(csv_path, output_root, data_root)
            for row_idx, row in _iter_rows(csv_path):
                if args.max_rows and processed >= args.max_rows:
                    print(f"Reached max rows limit ({args.max_rows}).")
                    return 0
                processed += 1

                invariants_raw = row.get(args.invariants_column)
                invariants = _parse_invariants(invariants_raw)
                experiment = row.get("experiment_name", "")
                prompt_version = row.get("prompt_version", "")
                model = row.get("model", "")
                temp = row.get("temperature", "")
                top_p = row.get("top_p", "")
                config_tag = row.get("config_tag", "")
                target_file = row.get("target_file", "")

                log_dir = logs_root / csv_path.relative_to(output_root).with_suffix("")
                slug_parts = [experiment, prompt_version, model, f"t{temp}" if temp else "", f"p{top_p}" if top_p else ""]
                slug = _slugify("_".join([part for part in slug_parts if part]))
                log_path = log_dir / f"row_{row_idx:04d}_{slug}.log"

                if args.skip_existing and log_path.exists():
                    writer.writerow(
                        {
                            "csv_path": str(csv_path),
                            "row_index": row_idx,
                            "experiment_name": experiment,
                            "prompt_version": prompt_version,
                            "config_tag": config_tag,
                            "model": model,
                            "temperature": temp,
                            "top_p": top_p,
                            "target_file": target_file,
                            "source_path": str(source_path) if source_path else "",
                            "invariants_count": len(invariants),
                            "invariants_used": 0,
                            "dropped_invariants": 0,
                            "status": "Skipped",
                            "reason": "log_exists",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                if source_path is None or not source_path.exists():
                    writer.writerow(
                        {
                            "csv_path": str(csv_path),
                            "row_index": row_idx,
                            "experiment_name": experiment,
                            "prompt_version": prompt_version,
                            "config_tag": config_tag,
                            "model": model,
                            "temperature": temp,
                            "top_p": top_p,
                            "target_file": target_file,
                            "source_path": str(source_path) if source_path else "",
                            "invariants_count": len(invariants),
                            "invariants_used": 0,
                            "dropped_invariants": 0,
                            "status": "Error",
                            "reason": "missing_source",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                code = source_path.read_text(encoding="utf-8")

                cleaned_invs, dropped_invs = verifier._sanitize_invariants(invariants)
                dropped_count = len(dropped_invs)

                if not cleaned_invs:
                    metadata = {
                        "csv_path": str(csv_path),
                        "row_index": row_idx,
                        "experiment_name": experiment,
                        "prompt_version": prompt_version,
                        "config_tag": config_tag,
                        "model": model,
                        "temperature": temp,
                        "top_p": top_p,
                        "target_file": target_file,
                        "source_path": str(source_path),
                        "invariants_raw": invariants,
                        "dropped_invariants": dropped_invs,
                        "status": "Skipped",
                        "reason": "no_valid_invariants",
                    }
                    _write_log(log_path, metadata, [], "")
                    writer.writerow(
                        {
                            "csv_path": str(csv_path),
                            "row_index": row_idx,
                            "experiment_name": experiment,
                            "prompt_version": prompt_version,
                            "config_tag": config_tag,
                            "model": model,
                            "temperature": temp,
                            "top_p": top_p,
                            "target_file": target_file,
                            "source_path": str(source_path),
                            "invariants_count": len(invariants),
                            "invariants_used": 0,
                            "dropped_invariants": dropped_count,
                            "status": "Skipped",
                            "reason": "no_valid_invariants",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                loop_invs = {1: invariants}
                instrumented_code, report = verifier.instrument(code, loop_invs, None)
                if args.save_instrumented:
                    instrumented_path = log_dir / f"row_{row_idx:04d}_{slug}.c"
                    _save_instrumented(instrumented_path, instrumented_code)
                else:
                    instrumented_path = None

                if not report.get("instrumented_loop_ids"):
                    metadata = {
                        "csv_path": str(csv_path),
                        "row_index": row_idx,
                        "experiment_name": experiment,
                        "prompt_version": prompt_version,
                        "config_tag": config_tag,
                        "model": model,
                        "temperature": temp,
                        "top_p": top_p,
                        "target_file": target_file,
                        "source_path": str(source_path),
                        "invariants_raw": invariants,
                        "invariants_used": cleaned_invs,
                        "dropped_invariants": dropped_invs,
                        "instrumented_path": str(instrumented_path) if instrumented_path else "",
                        "report": report,
                        "status": "Skipped",
                        "reason": "no_loop_instrumented",
                    }
                    _write_log(log_path, metadata, [], "")
                    writer.writerow(
                        {
                            "csv_path": str(csv_path),
                            "row_index": row_idx,
                            "experiment_name": experiment,
                            "prompt_version": prompt_version,
                            "config_tag": config_tag,
                            "model": model,
                            "temperature": temp,
                            "top_p": top_p,
                            "target_file": target_file,
                            "source_path": str(source_path),
                            "invariants_count": len(invariants),
                            "invariants_used": len(cleaned_invs),
                            "dropped_invariants": dropped_count,
                            "status": "Skipped",
                            "reason": "no_loop_instrumented",
                            "log_file": str(log_path),
                        }
                    )
                    continue

                if args.dry_run:
                    status = "Skipped"
                    reason = "dry_run"
                    output = ""
                    cmd = []
                    returncode = None
                else:
                    status, reason, cmd, returncode, output = _run_seahorn(
                        instrumented_code,
                        args.docker_image,
                        args.timeout,
                        sea_args,
                        args.container_name or None,
                    )
                    if not output:
                        output = "(seahorn output is empty)"

                metadata = {
                    "csv_path": str(csv_path),
                    "row_index": row_idx,
                    "experiment_name": experiment,
                    "prompt_version": prompt_version,
                    "config_tag": config_tag,
                    "model": model,
                    "temperature": temp,
                    "top_p": top_p,
                    "target_file": target_file,
                    "source_path": str(source_path),
                    "invariants_raw": invariants,
                    "invariants_used": cleaned_invs,
                    "dropped_invariants": dropped_invs,
                    "instrumented_path": str(instrumented_path) if instrumented_path else "",
                    "report": report,
                    "status": status,
                    "reason": reason,
                    "returncode": returncode,
                }

                _write_log(log_path, metadata, cmd, output)

                writer.writerow(
                    {
                        "csv_path": str(csv_path),
                        "row_index": row_idx,
                        "experiment_name": experiment,
                        "prompt_version": prompt_version,
                        "config_tag": config_tag,
                        "model": model,
                        "temperature": temp,
                        "top_p": top_p,
                        "target_file": target_file,
                        "source_path": str(source_path),
                        "invariants_count": len(invariants),
                        "invariants_used": len(cleaned_invs),
                        "dropped_invariants": dropped_count,
                        "status": status,
                        "reason": reason,
                        "log_file": str(log_path),
                    }
                )

    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
