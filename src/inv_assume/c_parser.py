import tree_sitter
import tree_sitter_c
from typing import List, Dict, Any, Tuple

class CParser:
    def __init__(self):
        self.language = tree_sitter.Language(tree_sitter_c.language())
        self.parser = tree_sitter.Parser(self.language)

    def parse_code(self, source_code: str) -> tree_sitter.Tree:
        return self.parser.parse(bytes(source_code, "utf8"))

    def find_loops(self, source_code: str) -> List[Dict[str, Any]]:
        """
        Parse source code and find all loops (while/for).
        Returns a list of dicts with:
        - 'node': The AST node of the loop
        - 'insertion_point': byte offset where to insert assume
        - 'code_context': string content of the loop (or surrounding function) for LLM
        """
        tree = self.parse_code(source_code)
        root_node = tree.root_node
        loops = []
        
        # Traverse finding while_statement and for_statement
        # Simple recursive traversal
        self._traverse_for_loops(root_node, loops, source_code)
        return loops

    def _traverse_for_loops(self, node, loops: List, source_code: str):
        if node.type in ["while_statement", "for_statement"]:
            insertion_point = self._get_insertion_point(node)
            if insertion_point is not None:
                loops.append({
                    "type": node.type,
                    "node": node,
                    "insertion_point": insertion_point,
                    "code_context": node.text.decode("utf8"), # Provide the loop code as context
                    # Ideally we might want the whole function, but loop code is a good start
                    # To get full function context, we'd need to walk up parent nodes.
                })
        
        for child in node.children:
            self._traverse_for_loops(child, loops, source_code)

    def _get_insertion_point(self, loop_node) -> int:
        """
        Finds the insertion point inside the loop body.
        Expects the body to be a compound_statement (surrounded by {}).
        Returns the byte offset immediately after the opening '{'.
        """
        # In tree-sitter-c:
        # while_statement: child_by_field_name('body')
        # for_statement: child_by_field_name('body')
        
        body_node = loop_node.child_by_field_name('body')
        if not body_node:
            return None
            
        if body_node.type == 'compound_statement':
            # It has braces. We want to insert after the first child which is '{'
            # usually body_node.children[0] is '{'
            # We insert right after it.
             return body_node.start_byte + 1
        else:
            # Single statement body without braces. e.g. while(1) stmt;
            # We cannot easily insert assume without adding braces.
            # Current strategy: Skip or creating a complex injector that adds braces.
            # For simplicity in this demo: Skip if no braces or return start of body 
            # (but injection logic needs to handle adding braces then).
            # Let's Skip for now to keep AST manipulation simple as requested.
            return None
