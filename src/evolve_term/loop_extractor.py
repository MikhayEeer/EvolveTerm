"""Loop extraction orchestrated via prompts + heuristics."""

from __future__ import annotations

import json
import re
import io
from typing import List

try:
    from pycparser import c_parser, c_ast, c_generator
    from pcpp import Preprocessor
    HAS_PYCPARSER = True

    class LoopVisitor(c_ast.NodeVisitor):
        def __init__(self):
            self.loops = []
            self.generator = c_generator.CGenerator()
            self.current_func_node = None
            self.current_func_name = None

        def visit_FuncDef(self, node):
            self.current_func_node = node
            self.current_func_name = node.decl.name
            self.generic_visit(node)
            self.current_func_node = None
            self.current_func_name = None

        def visit_FuncCall(self, node):
            if self.current_func_name and isinstance(node.name, c_ast.ID) and node.name.name == self.current_func_name:
                # Found recursion, treat function body as a loop
                # Avoid adding the same function multiple times
                func_code = self.generator.visit(self.current_func_node)
                if func_code not in self.loops:
                    self.loops.append(func_code)
            self.generic_visit(node)

        def visit_For(self, node):
            self.loops.append(self.generator.visit(node))
            self.generic_visit(node)

        def visit_While(self, node):
            self.loops.append(self.generator.visit(node))
            self.generic_visit(node)

        def visit_DoWhile(self, node):
            self.loops.append(self.generator.visit(node))
            self.generic_visit(node)

except ImportError:
    HAS_PYCPARSER = False
    LoopVisitor = None

from .llm_client import LLMClient
from .prompts_loader import PromptRepository


class CppToCConverter:
    """
    A helper class to convert simple C++ code to C99 compatible code
    so that pycparser can parse it.
    This is a heuristic conversion and not a full compiler frontend.
    """
    
    def __init__(self):
        self.replacements = [
            # Remove using namespace
            (r'using\s+namespace\s+\w+\s*;' , ''),
            # Remove template declarations (simple ones)
            (r'template\s*<[^>]*>', ''),
            # Replace bool types
            (r'\bbool\b', 'int'),
            (r'\btrue\b', '1'),
            (r'\bfalse\b', '0'),
            (r'\bnullptr\b', '0'),
            # Remove access modifiers
            (r'\b(public|private|protected)\s*:', ''),
            # Remove C++ keywords that might confuse C parser
            (r'\b(virtual|friend|explicit|constexpr|inline)\b', ''),
            # Remove std:: prefix
            (r'std::', ''),
            # Simple IO replacement (very basic)
            (r'cin\s*>>\s*(\w+);', r'scanf("%d", &\1);'),
            (r'cout\s*<<\s*([^;]+);', r'printf("\1");'), 
            # Remove includes that pycparser can't handle (standard C++ headers)
            (r'#include\s+<iostream>', ''),
            (r'#include\s+<vector>', ''),
            (r'#include\s+<algorithm>', ''),
            (r'#include\s+<cmath>', '#include <math.h>'),
            (r'#include\s+<cstring>', '#include <string.h>'),
            (r'#include\s+<cstdio>', '#include <stdio.h>'),
            (r'#include\s+<cstdlib>', '#include <stdlib.h>'),
        ]

    def preprocess(self, code: str) -> str:
        # 1. Basic Regex Replacements
        for pattern, repl in self.replacements:
            code = re.sub(pattern, repl, code)
        
        # 2. Run C Preprocessor (pcpp) to handle macros and defines
        if HAS_PYCPARSER:
            try:
                preprocessor = Preprocessor()
                preprocessor.parse(code)
                output = io.StringIO()
                preprocessor.write(output)
                code = output.getvalue()
            except Exception:
                # If pcpp fails, we just return the regex-processed code
                pass
        
        print(f"[Debug] Preprocessed C code for pycparser:\n{code}\n[Debug]CppToCConverter End\n")
        return code


class LoopExtractor:
    """Delegates loop extraction to the LLM with a heuristic fallback."""

    def __init__(self, llm_client: LLMClient, prompt_repo: PromptRepository):
        self.llm_client = llm_client
        self.prompt_repo = prompt_repo
        self.converter = CppToCConverter()
        # last LLM response and method used for extraction ('llm', 'heuristic', 'pycparser')
        self.last_response: str | None = None
        self.last_method: str | None = None

    def extract(self, code: str, max_loops: int = 5) -> List[str]:
        # 1. Try pycparser extraction first (Structural Analysis)
        if HAS_PYCPARSER:
            # No try-except block here, let it raise if parsing fails
            loops = self.extract_with_pycparser(code)
            if loops:
                self.last_method = "pycparser"
                return loops[:max_loops]
        
        # If pycparser is missing or found no loops, proceed to LLM
        # Note: If pycparser is present but fails (raises Exception), the program will crash here as requested.

        # 2. Try LLM extraction
        prompt = self.prompt_repo.render("loop_extraction", code=code)
        response = self.llm_client.complete(prompt)
        self.last_response = response
        
        # Try parsing LLM response
        loops = self._parse_response(response)
        
        # Verify extracted loops against original code
        verified_loops = []
        if loops:
            for loop in loops:
                if self._verify_loop_in_code(loop, code):
                    verified_loops.append(loop)
                else:
                    # Optional: Log that a hallucination was dropped
                    pass
        
        if verified_loops:
            self.last_method = "llm"
            return verified_loops[:max_loops]
            
        # 3. Fallback to heuristic if LLM failed or all extractions were hallucinations
        loops = self._heuristic_loops(code)
        self.last_method = "heuristic"
        return loops[:max_loops]

    def extract_with_pycparser(self, code: str) -> List[str]:
        """
        Extract loops using pycparser.
        First preprocesses the code to make it C99 compatible.
        """
        if not HAS_PYCPARSER:
            return []
            
        # Preprocess C++ to C
        c_code = self.converter.preprocess(code)
        
        parser = c_parser.CParser()
        # We might need to handle parse errors
        ast = parser.parse(c_code)
        visitor = LoopVisitor()
        visitor.visit(ast)
        return visitor.loops

    def _verify_loop_in_code(self, loop_snippet: str, original_code: str) -> bool:
        """Check if loop_snippet exists in original_code, ignoring whitespace."""
        def normalize(s: str) -> str:
            return "".join(s.split())
        
        return normalize(loop_snippet) in normalize(original_code)

    def _parse_response(self, response: str) -> List[str]:
        """Parse the custom delimiter-separated response."""
        if "NO_LOOPS_FOUND" in response:
            return []
            
        # Split by the custom delimiter
        parts = response.split("---LOOP_SEPARATOR---")
        
        # Clean up each part
        loops = []
        for part in parts:
            cleaned = part.strip()
            # Remove potential markdown fences if LLM ignored instructions
            if cleaned.startswith("```"):
                cleaned = cleaned.replace("```c", "").replace("```", "")
            cleaned = cleaned.strip()
            if cleaned:
                loops.append(cleaned)
                
        return loops

    def _heuristic_loops(self, code: str) -> List[str]:
        loops = re.findall(r"for\s*\(.*?\{.*?\}|while\s*\(.*?\{.*?\}", code, flags=re.DOTALL)
        if loops:
            return [re.sub(r"\s+", " ", loop.strip()) for loop in loops]
        return ["/* no loops detected */"]
