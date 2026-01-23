import os
from .c_parser import CParser
from .inv_generator import InvariantGenerator
from .injector import Injector

class ASTInstrumentationPipeline:
    def __init__(self, llm_config="llm_config.json", strategy="simple"):
        self.parser = CParser()
        self.generator = InvariantGenerator(config_name=llm_config, strategy=strategy)
        self.injector = Injector()

    def run(self, input_file: str, output_file: str = None):
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
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

        # 3. Add Header Definitions (if needed)
        source_with_header = self.injector.add_header(source_code)
        
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
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_code)
            print(f"Instrumented code written to {output_file}")
            return {
                "loops_count": len(loops),
                "injections_count": len(injections),
                "output_path": output_file
            }
        else:
            return {
                "loops_count": len(loops),
                "injections_count": len(injections),
                "code": final_code
            }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inject invariants into C code using AST analysis.")
    parser.add_argument("input_file", help="Path to the C source file")
    parser.add_argument("--output", help="Output file path", default=None)
    parser.add_argument("--strategy", choices=["simple", "2stage"], default="simple", 
                        help="Invariant generation strategy (simple=one-shot, 2stage=atom-filter-candidate)")
    parser.add_argument("--config", default="llm_config.json", help="LLM configuration file")
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else args.input_file + ".instrumented.c"
    
    pipeline = ASTInstrumentationPipeline(llm_config=args.config, strategy=args.strategy)
    pipeline.run(args.input_file, output_path)
