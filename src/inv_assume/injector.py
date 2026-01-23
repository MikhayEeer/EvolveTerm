from typing import List, Tuple

class Injector:
    def parse_header_present(self, source_code: str) -> bool:
        return "void __VERIFIER_assume" in source_code or "#include \"seahorn/seahorn.h\"" in source_code

    def add_header(self, source_code: str) -> str:
        """
        Ensures the source code has the necessary definitions for assume.
        """
        if self.parse_header_present(source_code):
            return source_code
            
        header = """
#ifndef _INJECTED_ASSUME_
#define _INJECTED_ASSUME_
extern void __VERIFIER_assume(int);
#define assume(X) __VERIFIER_assume(!!(X))
#endif
"""
        # Insert at top
        return header + source_code

    def inject_invariants(self, source_code: str, injections: List[Tuple[int, str]]) -> str:
        """
        injections: List of (byte_offset, invariant_string)
        """
        # Sort by offset descending so modification doesn't shift previous offsets
        sorted_injections = sorted(injections, key=lambda x: x[0], reverse=True)
        
        # We need to operate on bytes because tree-sitter returns byte offsets
        # If source_code is str, we assume utf-8
        code_bytes = source_code.encode("utf-8")
        
        for offset, invariant in sorted_injections:
            # construct injection string
            # We add a newline for readability
            injection_str = f"\n    assume({invariant});"
            injection_bytes = injection_str.encode("utf-8")
            
            # Splice
            code_bytes = code_bytes[:offset] + injection_bytes + code_bytes[offset:]
            
        return code_bytes.decode("utf-8")
