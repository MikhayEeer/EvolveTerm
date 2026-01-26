from pathlib import Path
from typing import Optional
from .c_parser import CParser
from .inv_generator import InvariantGenerator
from .injector import Injector
from .verifier import SeaHornVerifier

class ASTInstrumentationPipeline:
    def __init__(
        self,
        llm_config: str = "llm_config.json",
        strategy: str = "simple",
        verifier: Optional[SeaHornVerifier] = None,
    ):
        self.parser = CParser()
        self.generator = InvariantGenerator(config_name=llm_config, strategy=strategy)
        self.injector = Injector()
        self.verifier = verifier

    def run(
        self,
        input_file: str,
        output_dir: str,
        verify: bool = False,
        verify_timeout: int = 60,
    ):
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not output_dir:
            raise ValueError("output_dir is required (use --output to set it).")
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"{input_path.name}.instrumented.c"

        with open(input_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # 1. Parse loops
        loops = self.parser.find_loops(source_code)
        
        # 2. Generate invariants and collect injections
        injections = []
        print(f"Found {len(loops)} loops.")
        
        for loop in loops:
            print(f"Generating invariant for loop at offset {loop['node'].start_byte} using strategy '{self.generator.strategy_name}'...")
            context = loop['code_context']
            try:
                invariant = self.generator.generate_invariant(context)
                print(f"  -> Generated: {invariant}")
                
                # Injection point is inside the loop body
                offset = loop['insertion_point']
                injections.append((offset, invariant))
            except Exception as e:
                print(f"  -> Error generating invariant: {e}")

        # Note: Adding header changes offsets! 
        # Strategy: 
        #   Calculate header length diff and adjust offsets? 
        #   OR: Inject Header LAST? No, header is top.
        #   OR: Inject invariants into original source, THEN add header. 
        # Let's do: Inject invariants -> New Code -> Add Header -> Final Code.
        
        # 4. Inject Invariants
        pass_1_code = self.injector.inject_invariants(source_code, injections)
        
        # 5. Add Header to the modified code
        final_code = self.injector.add_header(pass_1_code)

        # 6. Output
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_code)
        print(f"Instrumented code written to {output_path}")

        result = {
            "loops_count": len(loops),
            "injections_count": len(injections),
            "output_path": str(output_path),
        }

        if verify:
            verifier = self.verifier or SeaHornVerifier()
            print(f"Verifying {output_path} with SeaHorn...")
            v_status, v_out, v_verdict = verifier.verify(str(output_path), timeout=verify_timeout)
            result["verification_status"] = v_status
            result["verification_verdict"] = v_verdict
            result["verification_output"] = v_out[-200:]
            if v_verdict != "unknown":
                print(f"Verification result: {v_status} ({v_verdict})")
            else:
                print(f"Verification result: {v_status}")
            print("SeaHorn output:")
            print(v_out)

        return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inject invariants into C code using AST analysis.")
    parser.add_argument("input_path", help="Path to the C source file or directory")
    parser.add_argument("--output", required=True, help="Output directory for instrumented files")
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

    pipeline = ASTInstrumentationPipeline(
        llm_config=args.config,
        strategy=args.strategy,
        verifier=verifier,
    )

    input_path = Path(args.input_path)
    if input_path.is_dir():
        from .batch_runner import BatchRunner
        runner = BatchRunner(
            input_dir=str(input_path),
            output_dir=args.output,
            llm_config=args.config,
            strategy=args.strategy,
            enable_verification=args.verify,
            verifier=verifier,
            verify_timeout=args.seahorn_timeout,
        )
        runner.run()
    else:
        pipeline.run(
            str(input_path),
            args.output,
            verify=args.verify,
            verify_timeout=args.seahorn_timeout,
        )
