"""Handler for the 'feature' command."""
from __future__ import annotations
import re
import yaml
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..llm_client import build_llm_client
from ..prompts_loader import PromptRepository
from ..utils import LiteralDumper
from ..cli_utils import collect_files

console = Console()

class FeatureHandler:
    def __init__(self, llm_config: str):
        self.llm_client = build_llm_client(llm_config)
        # Force model to qwen-plus as requested
        self.llm_client.model = "qwen-plus"
        self.prompt_repo = PromptRepository()

    def run(self, input_path: Path, output: Optional[Path], recursive: bool):
        files = collect_files(input_path, recursive, {".c", ".cpp", ".cc", ".h", ".hpp"})
        if not files:
            console.print("[yellow]No source files found.[/yellow]")
            return

        # Determine output strategy
        output_dir = None
        if output:
            if output.suffix.lower() in {'.yml', '.yaml'}:
                # Single file output? Only if single input?
                # We usually output 1-to-1 yml
                console.print("[red]Output must be a directory for batch feature extraction.[/red]")
                return
            output_dir = output
            output_dir.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing features...", total=len(files))
            
            for f in files:
                progress.update(task, description=f"Analyzing {f.name}...")
                try:
                    result = self.analyze_file(f, input_path)
                    
                    # Save result
                    out_filename = f"{f.stem}_feature.yml"
                    if output_dir:
                        # preserve structure relative to input_path if possible
                        if input_path.is_dir():
                            rel = f.parent.relative_to(input_path)
                            target_dir = output_dir / rel
                        else:
                            target_dir = output_dir
                    else:
                        target_dir = f.parent
                    
                    target_dir.mkdir(parents=True, exist_ok=True)
                    out_file = target_dir / out_filename
                    
                    with open(out_file, 'w', encoding='utf-8') as yf:
                        yaml.dump(result, yf, Dumper=LiteralDumper, sort_keys=False, allow_unicode=True)
                        
                except Exception as e:
                    console.print(f"[red]Error analyzing {f.name}: {e}[/red]")

    def analyze_file(self, file_path: Path, base_dir: Path) -> Dict[str, Any]:
        code = file_path.read_text(encoding="utf-8")
        
        # 1. Static Analysis
        static_feats = self._analyze_static(code, file_path, base_dir)
        
        # 2. LLM Analysis
        # Determine if we should ask about recursion or generic summary
        # We ask LLM for: summary, recur_type, initial_sat_condition
        llm_feats = self._analyze_llm(code, static_feats)
        
        # 3. Merge
        merged = {**static_feats, **llm_feats}
        
        # Final adjustment of program_type
        # If loops > 0 -> loop. Else check recursion from LLM.
        if merged['loops_count'] > 0:
            merged['program_type'] = 'loop'
            # If confusingly also has recursion, we prioritize loop or keep both? 
            # User templates says "loop or recur". Usually distinct.
        elif merged['recur_type'] in ['tail', 'non-tail']:
            merged['program_type'] = 'recur'
        else:
            merged['program_type'] = 'unknown' # or just empty source

        # Fill loop_type based on static counts
        if merged['loops_count'] == 0:
            merged['loop_type'] = None
        else:
            if merged['loops_depth'] > 1:
                merged['loop_type'] = 'Nested'
            elif merged['loops_count'] > 1:
                merged['loop_type'] = 'Multiple'
            else:
                merged['loop_type'] = 'Linear'

        return merged

    def _analyze_static(self, code: str, file_path: Path, base_dir: Path) -> Dict[str, Any]:
        # Basic Clean for comments (C style)
        clean_code = self._remove_comments(code)
        
        lines = len(code.splitlines())
        lang = "C" if file_path.suffix.lower() == ".c" else "C++"
        
        # Operators
        array_op = '[' in clean_code and ']' in clean_code
        pointer_op = '*' in clean_code or '->' in clean_code or '&' in clean_code
        has_break = bool(re.search(r'\bbreak\s*;', clean_code))
        
        # Loops Analysis (Brace counting)
        loops_count, loops_depth, max_depth = self._analyze_loops_structure(clean_code)
        
        # Conditions Analysis
        # Variable count in conditions
        loop_cond_vars = self._count_loop_condition_vars(clean_code)
        
        # Infinite loop pattern
        # while(1), while(true), for(;;)
        # ignoring spaces
        no_space = re.sub(r'\s+', '', clean_code)
        always_true = ('while(1)' in no_space) or ('while(true)' in no_space) or ('for(;;)' in no_space)
        if not always_true:
            # Try slightly looser regex for "while (  1  )"
            if re.search(r'while\s*\(\s*(1|true)\s*\)', clean_code):
                always_true = True

        relative_path = str(file_path.relative_to(base_dir)) if base_dir.is_dir() else str(file_path.name)

        return {
            "source_path": relative_path,
            "language": lang,
            "program_type": "", # placeholder
            "recur_type": None, # placeholder
            "loop_type": "", # placeholder
            "loops_count": loops_count,
            "loops_depth": max_depth,
            "loop_condition_variables_count": loop_cond_vars,
            "has_break": has_break,
            "loop_condition_always_true": always_true,
            "initial_sat_condition": False, # placeholder
            "array_operator": array_op,
            "pointer_operator": pointer_op,
            "lines": lines,
            "summary": ""
        }

    def _analyze_llm(self, code: str, static_feats: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt_dict = self.prompt_repo.render("summary", code=code)
            # Add JSON enforcement hint if not in prompt
            prompt_dict["response_format"] = {"type": "json_object"}
            
            response_text = self.llm_client.complete(prompt_dict)
            
            # Simple JSON parse
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return {
                    "summary": data.get("summary", ""),
                    "recur_type": data.get("recur_type"),
                    "initial_sat_condition": data.get("initial_sat_condition", True)
                }
        except Exception as e:
            console.print(f"[yellow]LLM Analysis failed: {e}[/yellow]")
            
        return {
            "summary": "LLM analysis failed",
            "recur_type": None,
            "initial_sat_condition": True
        }

    def _remove_comments(self, code):
        # Remove // comments
        code = re.sub(r'//.*', '', code)
        # Remove /* */ comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code

    def _analyze_loops_structure(self, code: str):
        # Heuristic approach for loop depth and count
        # 1. Identify loop starts: while, for, do
        # 2. Track brace nesting
        
        # Tokenize roughly
        tokens = re.split(r'(\W+)', code)
        
        current_depth = 0
        max_depth = 0
        loops_found = 0
        # This keeps track of which depth level contains a loop head
        # loop_levels[d] = boolean (is there a loop active at depth d)
        # This is hard to do perfectly without AST.
        
        # Alternative simple approach:
        # Loop Count: just regex count keywords
        # Loop Depth: Track indentation? No.
        # Track braces. When we hit 'for', 'while', we record current brace depth as "start of loop".
        # But 'do' is tricky.
        
        # Regex for keywords
        # We'll just count total loops first
        matches = list(re.finditer(r'\b(for|while|do)\b', code))
        loops_count = 0
        
        # Filter 'do' that is part of 'do-while' -> wait, 'do' starts the loop, 'while' ends it.
        # 'while' can be a loop or a do-while tail.
        # This regex heuristic is imperfect.
        # But 'for' is always a loop. 'while' is loop or do-while end.
        
        # Better: iterate through code, track curly braces.
        # When we match a loop keyword, we mark that we are entering a loop.
        
        # Let's try a balance-based scanner
        # This is complex to implement in one shot.
        # Fallback: Max depth of any block? No, max depth of LOOPS.
        
        # Simplified Logic for depth:
        # 1. Remove strings strings
        # 2. Scan. If '{' depth++. If '}' depth--.
        # 3. If we see 'for', 'while', 'do', record (depth) in a list of loop_depths.
        # 4. But 'while' in 'do { } while()' is at depth 0 (relative to block).
        # We need to ignore 'while' if it follows '}'. 
        
        # Impl:
        # Just scan for loop keywords.
        # Calculate indentation depth? No.
        # Calculate brace depth at the keyword position.
        
        # NOTE: A nested loop is a loop inside braces of another loop.
        # My static analysis without AST is an approximation.
        
        # Approximation:
        # Total loops = count of "for" + count of "while" (not preceded by '}') + count of "do"
        # Depth:
        # We track "brace hierarchy".
        
        # Let's stick to a robust approximation:
        # Just count all "for" and "while" and "do".
        # Assume "do...while" counts 'do' as start. 'while' at end might be double counted.
        # Check if 'while' is preceded by '}' (ignoring space).
        
        # Scan code for keywords and braces
        search_pattern = r'(\bfor\b|\bwhile\b|\bdo\b|\{|\}|;)'
        token_iter = re.finditer(search_pattern, code)
        
        depth = 0
        loop_depths = [0] # base
        
        # We need to know if we are 'inside' a loop to calculate nesting.
        # Use a stack of "is_loop_scope".
        scope_stack = [False] # False means normal block, True means loop block
        
        current_max_depth = 0
        total_loops = 0
        
        last_t = ""
        
        for m in token_iter:
            token = m.group(1)
            
            if token == '{':
                depth += 1
                # Inherit loop status or if we just saw a loop keyword?
                # This is hard.
                # Let's simplify:
                # Max Loop Depth = Max number of OPEN loops at any point.
                pass 
                
            elif token == '}':
                depth = max(0, depth - 1)
                
            elif token in ['for', 'while', 'do']:
                # Heuristic: Check if 'while' is after '}' -> do-while end
                if token == 'while' and last_t == '}':
                    # Likely end of do-while
                    pass
                else:
                    total_loops += 1
                    # How to calc depth?
                    # We can't know if we are INSIDE another loop just by brace depth (could be if/else).
                    # We assume worst case for complexity: depth implies nesting? No.
                    # Correct way: Track if parent scopes are loops.
                    pass
            
            if token.strip():
                last_t = token
                
        # Given limitations, I will use a regex to find max nested braces, 
        # But restricting to determining if it is a Loop block is hard.
        
        # Fallback to simple counts for this demo script
        # loops_count = 'for' + 'while' (not after }) + 'do'
        
        c_for = len(re.findall(r'\bfor\b', code))
        # Find 'while' not preceded of '}'
        # We reverse string to use lookbehind or just regex
        # simpler: count 'do'
        c_do = len(re.findall(r'\bdo\b', code))
        # count 'while'
        c_while = len(re.findall(r'\bwhile\b', code))
        
        # adjustment: do..while uses 1 'do' and 1 'while'. We count the 'do'.
        # We try to subtract do-whiles from 'while' count.
        # number of 'do' usually equals number of 'while' endings.
        # So loops = for + while - do? No.
        # loops = for + (while_loops) + do_loops
        # do..while has 'do' and 'while'.
        # We count 'do' as 1 loop.
        # We want to count 'while(...){' as 1 loop.
        # 'while' in do-while is '...} while(...);'
        
        # Approx: Total = for + while + do. If do-while, we double counted.
        real_count = c_for + c_while # 'do' is usually paired with while. 
        # If I have 'while' loops, count is correct.
        # If I have 'do...while', 'while' is counted. 'do' is extra? Or 'do' is counted?
        # Let's assume 'do' is distinct.
        # Actually in C, 'while' appears in both 'while loop' and 'do loop'.
        # So identifying 'while' covers both, EXCEPT 'do' allows us to know it's a loop start.
        # But 'while' covers all.
        # Wait, `do { } while(0);`
        # `while(1) { }`
        # Counting `while` occurrences is good, but `do` helps confirmation.
        # Let's just use `c_for + c_while`. It finds all loops eventually.
        
        # Depth:
        # Hacky way: Remove all non-loop code blocks (if/else), calculate brace depth? Impossible.
        # Return 1 if count == 1, else 2 (Nested/Multi).
        # If regex finds `for .* \{ .* for .* \{` -> depth 2?
        
        max_d = 1 if real_count > 0 else 0
        if real_count > 1:
            # Check for nesting
            # Remove all code except braces and keywords
            mini = re.sub(r'[^\{\}forwhiled]', '', code) # simplified
            # If we see `for{for{` pattern?
            if re.search(r'(for|while|do)[^}]*?\{[^}]*?(for|while|do)', code, re.DOTALL):
                 max_d = 2 # At least 2
                 
        return real_count, max_d, max_d

    def _count_loop_condition_vars(self, code: str):
        # Extract content between parens for loops
        # for ( ...; ...; ...) -> take middle
        # while ( ... )
        vars_found = set()
        
        # While pattern
        for m in re.finditer(r'while\s*\((.*?)\)', code, re.DOTALL):
            cond = m.group(1)
            # tokenize
            ws = re.findall(r'\b[a-zA-Z_]\w*\b', cond)
            for w in ws:
                if w not in ['true', 'false', 'NULL', 'and', 'or', 'not']:
                    vars_found.add(w)
                    
        # For pattern: for(init; cond; step)
        for m in re.finditer(r'for\s*\(([^;]*);([^;]*);', code, re.DOTALL):
            cond = m.group(2)
            ws = re.findall(r'\b[a-zA-Z_]\w*\b', cond)
            for w in ws:
                 if w not in ['true', 'false', 'NULL']:
                    vars_found.add(w)
        
        return len(vars_found)
