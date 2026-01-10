from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import glob
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline import default_prompt_dir, infer_invariants


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-stage LLM invariant synthesis with Z3/SeaHorn filtering."
    )
    parser.add_argument("--in", dest="input_paths", required=True, nargs="+", help="C file(s) or directory.")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory.")
    parser.add_argument("--loop-id", type=int, default=1, help="1-based loop id to target.")
    parser.add_argument("--llm-config", default="llm_config.json", help="LLM config file name or path.")
    parser.add_argument("--llm-tag", default="default", help="LLM config tag.")
    parser.add_argument("--prompts-dir", default=None, help="Prompt directory override.")
    parser.add_argument("--max-atoms", type=int, default=12, help="Maximum atoms kept after Z3 filtering.")
    parser.add_argument("--max-invariants", type=int, default=4, help="Maximum invariants kept after Z3 filtering.")
    parser.add_argument("--seahorn-timeout", type=int, default=60, help="SeaHorn timeout in seconds.")
    parser.add_argument("--no-seahorn", action="store_true", help="Skip SeaHorn checks.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories when --in is a dir.")
    return parser


def _collect_input_files(paths: List[str], recursive: bool) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.exists():
            if candidate.is_file():
                files.append(candidate)
                continue
            if candidate.is_dir():
                if recursive:
                    files.extend(sorted(candidate.rglob("*.c")))
                else:
                    files.extend(sorted(candidate.glob("*.c")))
                continue
        if any(ch in raw for ch in "*?[]"):
            for match in sorted(glob.glob(raw)):
                match_path = Path(match)
                if match_path.is_file():
                    files.append(match_path)
            continue
        raise FileNotFoundError(f"Input path not found: {raw}")

    seen = set()
    unique: List[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _output_dir_for_file(out_root: Path, file_path: Path, common_base: Path, multi: bool) -> Path:
    if not multi:
        return out_root
    rel = file_path.resolve().relative_to(common_base)
    rel_dir = rel.with_suffix("")
    return out_root / rel_dir


def _write_outputs(
    out_dir: Path,
    input_path: Path,
    report: dict,
    instrumented_code: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seahorn_logs_dir = out_dir / "seahorn_logs"
    seahorn_logs_dir.mkdir(parents=True, exist_ok=True)

    report["input_file"] = str(input_path)
    report["output_dir"] = str(out_dir)

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    instrumented_path = out_dir / "instrumented.c"
    instrumented_path.write_text(instrumented_code, encoding="utf-8")

    atoms_path = out_dir / "atoms.txt"
    atoms = report.get("stage1_atoms_filtered", [])
    atoms_path.write_text("\n".join(atoms) + ("\n" if atoms else ""), encoding="utf-8")

    invariants_path = out_dir / "invariants.txt"
    invariants = report.get("final_invariants") or report.get("instrumented_invariants", [])
    invariants_path.write_text("\n".join(invariants) + ("\n" if invariants else ""), encoding="utf-8")

    print(f"[OK] Report written to: {report_path}")
    print(f"[OK] Instrumented C written to: {instrumented_path}")
    print(f"[OK] Atoms written to: {atoms_path}")
    print(f"[OK] Invariants written to: {invariants_path}")
    print(f"[OK] SeaHorn logs: {seahorn_logs_dir}")


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    input_files = _collect_input_files(args.input_paths, args.recursive)
    if not input_files:
        raise FileNotFoundError("No C files found for processing.")

    prompt_dir = Path(args.prompts_dir) if args.prompts_dir else default_prompt_dir()

    llm_config_path = Path(args.llm_config)
    llm_config = str(llm_config_path.resolve()) if llm_config_path.exists() else str(args.llm_config)

    if len(input_files) == 1:
        common_base = input_files[0].resolve().parent
    else:
        common_base = Path(os.path.commonpath([str(p.resolve()) for p in input_files]))
        if common_base.is_file():
            common_base = common_base.parent

    multi = len(input_files) > 1

    for input_path in input_files:
        code = input_path.read_text(encoding="utf-8")
        seahorn_logs_dir = _output_dir_for_file(out_root, input_path, common_base, multi) / "seahorn_logs"
        report, instrumented_code = infer_invariants(
            code,
            llm_config=llm_config,
            llm_tag=args.llm_tag,
            prompt_dir=prompt_dir,
            loop_id=args.loop_id,
            max_atoms=args.max_atoms,
            max_invariants=args.max_invariants,
            seahorn_timeout=args.seahorn_timeout,
            enable_seahorn=not args.no_seahorn,
            seahorn_log_dir=seahorn_logs_dir,
        )

        out_dir = _output_dir_for_file(out_root, input_path, common_base, multi)
        _write_outputs(out_dir, input_path, report, instrumented_code)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
