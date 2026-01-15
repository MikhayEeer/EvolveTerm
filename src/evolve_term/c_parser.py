"""Enhanced C parser using tree-sitter for robust loop extraction."""

from __future__ import annotations

import tree_sitter
import tree_sitter_c
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoopInfo:
    """Information about a single loop extracted from source code."""

    loop_id: int
    loop_type: str  # "for", "while", "do_while"
    code: str  # Complete loop code
    start_byte: int
    end_byte: int
    nesting_depth: int
    parent_loop_id: Optional[int]
    body_start_byte: int  # Start of loop body (after opening brace or condition)
    has_braces: bool


class CParser:
    """Tree-sitter based C parser for loop extraction.

    Features:
    - Supports for, while, and do-while loops
    - Handles loops with and without braces
    - Tracks nesting depth and parent-child relationships
    - Provides precise byte offsets for code manipulation
    """

    def __init__(self):
        self.language = tree_sitter.Language(tree_sitter_c.language())
        self.parser = tree_sitter.Parser(self.language)

    def parse(self, source_code: str) -> tree_sitter.Tree:
        """Parse source code and return the AST tree."""
        return self.parser.parse(bytes(source_code, "utf8"))

    def find_all_loops(self, source_code: str) -> List[LoopInfo]:
        """Extract all loops with nesting information.

        Args:
            source_code: C source code as string

        Returns:
            List of LoopInfo objects ordered by appearance in source
        """
        tree = self.parse(source_code)
        loops: List[LoopInfo] = []
        self._traverse_loops(tree.root_node, loops, source_code, depth=0, parent_id=None)
        return loops

    def _traverse_loops(
        self,
        node: tree_sitter.Node,
        loops: List[LoopInfo],
        source_code: str,
        depth: int,
        parent_id: Optional[int],
    ) -> None:
        """Recursively traverse AST to find and extract loops."""
        loop_types = {
            "for_statement": "for",
            "while_statement": "while",
            "do_statement": "do_while",
        }

        if node.type in loop_types:
            loop_id = len(loops) + 1
            loop_type = loop_types[node.type]

            # Extract complete loop code
            code = source_code[node.start_byte : node.end_byte]

            # Analyze loop body
            has_braces, body_start_byte = self._analyze_body(node, source_code)

            loop_info = LoopInfo(
                loop_id=loop_id,
                loop_type=loop_type,
                code=code,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                nesting_depth=depth,
                parent_loop_id=parent_id,
                body_start_byte=body_start_byte,
                has_braces=has_braces,
            )
            loops.append(loop_info)

            # Continue traversal with increased depth for nested loops
            # We need to traverse children but only count nested loops
            for child in node.children:
                self._traverse_loops(child, loops, source_code, depth + 1, loop_id)
        else:
            # Not a loop node, continue traversal at same depth
            for child in node.children:
                self._traverse_loops(child, loops, source_code, depth, parent_id)

    def _analyze_body(self, loop_node: tree_sitter.Node, source_code: str) -> tuple[bool, int]:
        """Analyze loop body for braces and body start position.

        Args:
            loop_node: The loop AST node
            source_code: Original source code

        Returns:
            Tuple of (has_braces, body_start_byte)
        """
        body_node = loop_node.child_by_field_name("body")

        if not body_node:
            # No body found, return loop start as fallback
            return False, loop_node.start_byte

        if body_node.type == "compound_statement":
            # Body has braces { ... }
            # Find the opening brace position
            # compound_statement children: '{', statements..., '}'
            if body_node.children:
                # First child is '{', we want position after it
                opening_brace = body_node.children[0]
                return True, opening_brace.end_byte
            else:
                # Empty body {}
                return True, body_node.start_byte + 1
        else:
            # Single statement body without braces
            # The body IS the single statement
            return False, body_node.start_byte

    def get_loop_with_context(
        self, source_code: str, loop_info: LoopInfo, context_lines: int = 5
    ) -> str:
        """Get loop code with surrounding context (variable declarations, etc.).

        Args:
            source_code: Full source code
            loop_info: LoopInfo object
            context_lines: Number of lines to include before the loop

        Returns:
            Loop code with context
        """
        # Find the line start before the loop
        lines_before = source_code[: loop_info.start_byte].split("\n")
        context_start_line = max(0, len(lines_before) - context_lines)

        # Get all lines
        all_lines = source_code.split("\n")
        loop_end_line = source_code[: loop_info.end_byte].count("\n")

        # Extract context lines
        context_lines_list = all_lines[context_start_line : loop_end_line + 1]
        return "\n".join(context_lines_list)

    def wrap_body_with_braces(self, source_code: str, loop_info: LoopInfo) -> str:
        """Add braces to a loop body that doesn't have them.

        This is useful for instrumentation that requires compound statements.

        Args:
            source_code: Original source code
            loop_info: LoopInfo for the loop to modify

        Returns:
            Modified source code with braces added, or original if already has braces
        """
        if loop_info.has_braces:
            return source_code

        # Find the body node
        tree = self.parse(source_code)
        loop_node = self._find_node_at_byte(tree.root_node, loop_info.start_byte)
        if not loop_node:
            return source_code

        body_node = loop_node.child_by_field_name("body")
        if not body_node:
            return source_code

        # Wrap body in braces
        body_code = source_code[body_node.start_byte : body_node.end_byte]
        wrapped_body = f"{{ {body_code} }}"

        # Construct new source code
        new_source = (
            source_code[: body_node.start_byte] + wrapped_body + source_code[body_node.end_byte :]
        )
        return new_source

    def _find_node_at_byte(
        self, node: tree_sitter.Node, byte_offset: int
    ) -> Optional[tree_sitter.Node]:
        """Find the node at a specific byte offset."""
        if node.start_byte <= byte_offset < node.end_byte:
            if node.type in ["for_statement", "while_statement", "do_statement"]:
                return node
            for child in node.children:
                found = self._find_node_at_byte(child, byte_offset)
                if found:
                    return found
        return None