"""
LLM Prompts for Chameleon HumanEval Pipeline
Centrally defined as global variables for easy maintainability and tuning.
"""

# ============================================================================
# DISTORTION PROMPTS
# ============================================================================

DISTORTION_SYSTEM_PROMPT = """You are a semantic distortion expert for programming benchmarks. 
Your task is to create lexically distorted versions of HumanEval-style coding problem descriptions 
while preserving the required function behavior and all reference tests.

DATA SOURCE ARCHITECTURE:
You are processing a JSON-Lines (JSONL) row with the following fields:
- "task_id": Unique identifier (e.g., "HumanEval/20")
- "prompt": The problem stem. This contains:
    1. Python imports (optional)
    2. A SINGLE function definition line: "def <name>(...)->...:"
    3. A triple-quoted docstring block (\"\"\" ... \"\"\") indented under the function.
    **THIS IS THE ONLY FIELD YOU TARGET.** You must output a modified version of this "prompt" field.
- "entry_point": The function name to call (e.g., "find_closest_elements"). DO NOT TOUCH.
- "canonical_solution": The reference implementation. DO NOT TOUCH.
- "test": The unit test assertions. DO NOT TOUCH.

CRITICAL RULES:
1. TARGET SCOPE:
   - You are ONLY rewriting the natural-language description inside the triple-quoted docstring within the "prompt" field.

2. PRESERVATION (STRICT):
   - **Function Signature**: The `def function_name(...):` line MUST remain EXACTLY the same.
   - **Structure**: Imports, definition line, indentation, and docstring delimiters must happen EXACTLY as in the original.
   - **Examples**: Inside the docstring, NEVER modify lines starting with `>>>` or their corresponding output lines.
   - **Code**: NEVER modify any Python code, variable names, or logic.

3. DISTORTION (Natural Language Only):
   - Rewrite the explanation text inside the docstring.
   - Use synonyms, sentence restructuring, and paraphrasing.
   - NEVER start with preambles like "Here are...".
   - NEVER use markdown formatting (no **, ##, or ``` ).
   - Output exactly the requested number of distortions.

Your goal is to produce a valid replacement for the "prompt" field that tests if a model can still solve the problem when the description is phrased differently."""


def get_distortion_prompt_template(question: str, miu: float, n_distortions: int, rule: str) -> str:
    return f"""Distort this HumanEval-style programming task description {n_distortions} unique ways at μ={miu}.

DISTORTION RULE for μ={miu}: {rule}

ORIGINAL PROMPT: {question}

STRICT REQUIREMENTS:

SEMANTIC INVARIANCE: The logic, requirements, and constraints must remain 100% identical to the original. The distorted prompt must describe the exact same mathematical or algorithmic logic.

STRUCTURAL VARIATION: Do not just swap words for synonyms. Change the sentence structure, use negations of opposites (e.g., "even" becomes "not odd"), or describe the process from a different perspective (e.g., "return the largest" becomes "exclude all elements except the maximum").

TEMPLATE PRESERVATION: The HumanEval template MUST be preserved exactly: • Keep all Python import lines exactly as given. • Keep the function definition line (def ...(...)->...) exactly as given. • Keep the triple-quoted docstring delimiters (\"\"\" ... \"\"\") and indentation exactly as given.

DOCSTRING INTEGRITY: • Inside the docstring, you may ONLY modify the natural-language explanation sentences. • DO NOT modify lines that start with \">>>\", nor the example output lines that follow them.

CODE INTEGRITY: • If any Python code is present (signatures, type hints, examples, asserts), copy that code EXACTLY with NO modifications. • Do NOT change function names, argument names, return types, or literal values.

NO FORMATTING: NO leetspeak, NO random characters, NO markdown formatting (no **, ##, or ```).

CLEAN OUTPUT: NO preambles like "Here are...". Output only the numbered list.

TEST COMPATIBILITY: The reference tests for this task must still pass perfectly after these distortions.

FOLLOW THE μ={miu} RULE EXACTLY.

OUTPUT (exactly {n_distortions} numbered lines):

[distortion]

[distortion] ... {n_distortions}. [distortion]
"""


def get_batch_distortion_prompt_template(questions_text: str, miu: float, n_distortions: int, rule: str) -> str:
    return f"""⚠️ MANDATORY: You MUST output EXACTLY {n_distortions} UNIQUE distortions for EACH HumanEval prompt. NO MORE, NO LESS.

TASK: Distort each HumanEval-style programming task description at μ={miu} intensity level.

DISTORTION RULE for μ={miu}: {rule}
{questions_text}

═══════════════════════════════════════════════════════════════════════
ABSOLUTE REQUIREMENTS (VIOLATION = FAILURE):
═══════════════════════════════════════════════════════════════════════

1. QUANTITY: Output EXACTLY {n_distortions} distortions per prompt
   - Not {n_distortions - 1}, not {n_distortions + 1} — EXACTLY {n_distortions}
   - Each must be numbered 1 through {n_distortions}

2. UNIQUENESS: All {n_distortions} distortions MUST be different from each other
   - NO duplicates allowed
   - NO near-duplicates (different by just 1-2 words)
   - Each distortion must have substantial unique variation

3. PRESERVATION: The required function behavior and reference tests must remain valid
   - Do NOT change input-output examples or edge-case behavior
   - Do NOT introduce new requirements or remove existing ones

4. TEMPLATE PRESERVATION (HumanEval format):
   - Keep ALL Python import lines exactly as in the original prompt
   - Keep the SINGLE function definition line (def ...(...)->...) exactly as in the original
   - Keep the triple-quoted docstring delimiters (\"\"\" ... \"\"\") and indentation exactly as in the original
   - Inside the docstring, you may ONLY rewrite the natural-language explanation sentences
   - DO NOT modify any lines that start with ">>>" or the example output lines that follow them

5. FORBIDDEN:
   ❌ Do NOT modify any Python code (function signatures, type hints, asserts, examples)
   ❌ Do NOT change function names, argument names, return types, or literal values in code or examples
   ❌ NO random characters or symbols
   ❌ NO markdown (**bold**, ##headers, ```code```)
   ❌ NO preambles ("Here are", "Sure", "Certainly")
   ❌ NO explanations or meta-commentary

6. ALLOWED (description text ONLY):
   ✅ Synonyms and vocabulary changes
   ✅ Sentence restructuring
   ✅ Voice changes (active↔passive)
   ✅ Clause reordering

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT (EXACT):
═══════════════════════════════════════════════════════════════════════

Q1:
1. [first unique distortion]
2. [second unique distortion]
3. [third unique distortion]
... continue to {n_distortions}
{n_distortions}. [{n_distortions}th unique distortion]

Q2:
1. [first unique distortion]
... continue for all questions

BEGIN OUTPUT NOW:"""


def get_retry_distortion_prompt_template(q_text: str, miu: float, rule: str, existing_list: str, needed: int, existing_count: int) -> str:
    return f"""⚠️ CRITICAL: Generate EXACTLY {needed} NEW unique distortion(s) for this HumanEval programming prompt.

TASK PROMPT:
{q_text}

DISTORTION RULE for μ={miu}: {rule}
{existing_list}

═══════════════════════════════════════════════════════════════════════
REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════

1. Generate EXACTLY {needed} NEW distortion(s)
2. Each must be COMPLETELY DIFFERENT from the existing {existing_count} distortions above
3. Focus ONLY on the natural-language description; copy any Python code exactly with NO modifications
4. Use different synonyms, different sentence structures, different word orders
5. NO duplicates or near-duplicates of existing ones
6. Follow the μ={miu} rule: {rule}
7. Preserve the HumanEval template:
   - Keep all import lines, the function definition line, and the triple-quoted docstring delimiters exactly as in the original
   - Only modify the natural-language explanation sentences inside the docstring
   - Do NOT modify lines that start with ">>>" or the example output lines that follow them

FORBIDDEN:
❌ Do NOT modify any Python code (function signatures, type hints, asserts, examples)
❌ Do NOT change function names, argument names, return types, or literal values
❌ NO typos, misspellings, or character noise
❌ NO leetspeak (3→e, 0→o)
❌ NO markdown (**bold**, ##headers)
❌ NO preambles ("Here are", "Sure")
❌ DO NOT repeat or slightly modify any existing distortion

OUTPUT FORMAT (EXACT):
1. [first NEW unique distortion]
{"2. [second NEW unique distortion]" if needed > 1 else ""}
... up to {needed}

BEGIN OUTPUT NOW:"""


# ============================================================================
# VALIDATION PROMPTS
# ============================================================================

VALIDATION_MEGA_JUDGE_PROMPT = """You are an expert impartial judge of semantic equivalence in coding tasks. Your goal is to verify if a modified task description requires the exact same implementation as the original.

[ORIGINAL PROMPT]
{original}

[DISTORTED PROMPT]
{distorted}

EVALUATION CRITERIA:
Identical (Fully Equivalent): The distorted prompt describes the exact same algorithmic logic, input-output constraints, and edge cases. A solution that passes the original will pass this one without modification.

Partially Equivalent: The core task is the same, but a minor constraint was lost, a literal value was changed (e.g., 100 instead of 10), or the return type was slightly altered.

Not Equivalent: The fundamental logic has changed. Solving the distorted prompt would result in a different output or a failed test case compared to the original.

ANALYSIS RULES:
Compare mathematical operations (e.g., "even" vs "not odd").

Compare boundary conditions (e.g., "less than" vs "not exceeding").

Check for any loss of constraints (e.g., time complexity or specific data structures mentioned).

OUTPUT FORMAT:
Respond ONLY with a valid JSON object. No preamble or markdown fences.
{{"equivalence_level": "Identical" | "Partially Equivalent" | "Not Equivalent", "analysis": "A detailed step-by-step breakdown of why the prompts are or are not logically identical, highlighting specific differences in wording vs. logic.", "is_safe_to_test": boolean}}
"""


# ============================================================================
# GENERATION PROMPTS
# ============================================================================

GENERATION_SYSTEM_PROMPT = """You are an expert Python developer. Your task is to complete the implementation of a specific function based on the provided prompt.

STRICT GENERATION RULES:

NO MARKDOWN: Do not use code blocks (```python ... ```). Return only raw text.

NO EXPLANATIONS: Do not include any introductory text, comments, or post-implementation notes.

COMPLETE CODE: Start exactly with the function definition line (def ...).

STRICT LOGIC: The code must perfectly satisfy the logic described in the docstring, ensuring it passes all hidden unit tests.

TEMPLATE ADHERENCE: Use the exact function name and argument names provided in the prompt.

IMPORTS: Include any necessary imports at the very top if they are required for the logic (e.g., import math or from typing import List).

PYTHON CODE:"""


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

EVALUATION_SYSTEM_PROMPT = """You are an expert Python 3 programmer solving HumanEval-style coding tasks.
Your ONLY output should be valid Python code that solves the task.
NEVER explain, NEVER add extra text, NEVER show reasoning.
Do NOT wrap the code in markdown or backticks.
Keep the provided function signature(s) unchanged and do not add a main block or print statements unless explicitly required by the prompt."""

EVALUATION_USER_TEMPLATE = """═══════════════════════════════════════════════════════════════════════
HUMANEVAL PROGRAMMING TASK
═══════════════════════════════════════════════════════════════════════

INSTRUCTIONS:
• Read the task description and any provided code carefully.
• Use Python 3.
• Keep all given function signature(s), argument names, and return types unchanged.
• Do NOT modify any provided test code or example assertions.
• Do NOT add an "if __name__ == '__main__':" block.
• Do NOT print anything unless the prompt explicitly requires printing.
• Return only the solution code; NO explanations, comments, or markdown.

TASK DESCRIPTION:
{question}

ADDITIONAL CONTEXT (if present):
{choices_text}
═══════════════════════════════════════════════════════════════════════

WRITE YOUR SOLUTION BELOW (Python code only):"""
