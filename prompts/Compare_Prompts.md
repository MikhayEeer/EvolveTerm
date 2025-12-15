# Prompt Strategy Comparison

This document analyzes the differences between three prompt strategies for termination analysis: **EvolveTerm (Current)**, **Const Prompts (Reference)**, and **Loopy Templates (Reference)**.

## 1. Overview

| Feature | **EvolveTerm (Current)** | **Const Prompts** | **Loopy Templates** |
| :--- | :--- | :--- | :--- |
| **Goal** | End-to-end Ranking Function (RF) generation | Classification & Phase detection | Specific RF generation (Lexicographic/Standard) |
| **Input** | C Code + Invariants + RAG | Boogie Code (IR) | C Code |
| **Strategy** | **Chain of Thought (CoT)** + Verification | **Divide & Conquer** (Classify -> Solve) | **Template-based** (Specific prompts for specific types) |
| **Output** | JSON (`ranking_function`, `explanation`) | Tags (`[PHASE_NUM]`, `[RANKING_TYPE]`) | ACSL Code Block |

## 2. Detailed Analysis

### A. EvolveTerm (Current)
**Philosophy**: "One-shot reasoning with explicit verification."
- **Pros**: 
    - Integrated into a single pipeline.
    - "Chain of Thought" (CoT) forces the model to show its work (math verification), reducing hallucinations like "fake conservation laws".
    - Flexible JSON output.
- **Cons**: 
    - Can be overwhelmed if the loop logic is extremely complex (e.g., multi-phase nested loops) without prior classification.
- **Example Snippet**:
    ```plaintext
    ### Verification Analysis
    R(x) = n - i
    R(x_new) = n - (i + 1) = R(x) - 1
    Delta = 1 > 0. Verified.
    ```

### B. Const Prompts
**Philosophy**: "Classify first, solve later."
- **Pros**: 
    - **High Precision**: By asking "Is this Single, Nested, or Multi-phase?", it forces the LLM to understand the *structure* of termination before doing the math.
    - **Robustness**: Handles complex Boogie IR well.
- **Cons**: 
    - Requires multiple round-trips (Classify -> Generate).
    - Boogie syntax might be unfamiliar to some general-purpose LLMs (though fine for specialized ones).
- **Example Snippet**:
    ```plaintext
    [RANKING_TYPE]
    nested
    ```

### C. Loopy Templates
**Philosophy**: "Specific tools for specific jobs."
- **Pros**: 
    - **ACSL Standard**: Outputs standard ACSL, ready for tools like Frama-C.
    - **Lexicographic Focus**: Explicitly handles tuple-based ranking functions `(e1, e2, ...)`.
- **Cons**: 
    - Requires the user/system to know *which* template to use (Lexicographic vs Standard).
- **Example Snippet**:
    ```c
    /*@
        loop variant v1;
        loop variant v2;
    */
    ```

## 3. Evolution Strategy (Implemented)

Based on this analysis, EvolveTerm has adopted a hybrid approach:
1.  **Retain CoT**: The "Verification Analysis" section is crucial for correctness.
2.  **Adopt Classification**: The System Prompt now explicitly defines **Scalar** vs **Lexicographic** types, encouraging the model to "Classify" mentally before generating.
3.  **Support Tuples**: The output format now supports tuple syntax `(a, b)` to handle nested/lexicographic cases, inspired by Loopy Templates.