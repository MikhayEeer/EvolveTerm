from src.evolve_term.llm_client import APILLMClient
from src.inv_assume.strategies.two_stage import TwoStageStrategy

class InvariantGenerator:
    def __init__(self, config_name: str = "llm_config.json", strategy: str = "simple"):
        self.config_name = config_name
        self.strategy_name = strategy
        self.llm_client = APILLMClient(config_name=config_name)
        
        self.strategy_impl = None
        if strategy == "2stage":
            self.strategy_impl = TwoStageStrategy(llm_config=config_name)

    def generate_invariant(self, code_context: str) -> str:
        """
        Input: C code snippet (loop).
        Output: C boolean expression (e.g. "i < n && x > 0").
        """
        if self.strategy_impl:
            return self.strategy_impl.generate(code_context)
            
        # Default Simple Strategy
        prompt = f"""
You are an expert in C program verification.
I will provide you with a C loop.
Your task is to infer a loop invariant that holds at the beginning of the loop body.
The invariant must be a valid C boolean expression.
Do NOT use Markdown blocks.
Do NOT use special quantifiers like forall/exists (unless expressed in C).
Do NOT explain. Just output the boolean expression string.
Examples:
Code: while(i < n) {{ i++; }}
Invariant: i <= n

Code:
{code_context}

Invariant:
"""
        response = self.llm_client.complete(prompt)
        # Basic cleanup
        invariant = response.strip()
        if invariant.startswith("```"):
            lines = invariant.split("\n")
            if len(lines) >= 3:
                # remove first and last lines typically
                invariant = "\n".join(lines[1:-1]).strip()
        
        # Remove trailing semi-colon if LLM added it
        if invariant.endswith(";"):
            invariant = invariant[:-1]
            
        return invariant
