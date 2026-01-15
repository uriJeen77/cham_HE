# distortion might include functions name dropping. align with all.


"""
Distortion configuration for HumanEval-style coding prompts.

All distortion-related constants, formulas, and rules for rephrasing
HumanEval problem descriptions are defined here to avoid duplication
across multiple files.
"""

import math
from typing import Dict, List

# ============================================================================
# MIU (μ) Distortion Rules
# ============================================================================

MIU_RULES: Dict[float, str] = {
    0.0: "NONE: Keep the HumanEval prompt exactly as it is. No changes allowed.",
    0.1: "MINIMAL: Adjust 1-2 words in the natural-language description with simple synonyms. Do NOT touch any code.",
    0.2: "LIGHT: Change 2-3 words in the description using synonyms. Keep all code and examples identical.",
    0.3: "MODERATE: Change 3-4 words or a short phrase in the description. Limited rephrasing; code remains unchanged.",
    0.4: "HEAVY LEXICAL: Extensive vocabulary changes in the description. Moderate restructuring; do not modify code or literal values.",
    0.5: "LIGHT MIXED: Combine lexical changes and mild sentence restructuring in the description only. Code stays exactly the same.",
    0.6: "MODERATE MIXED: Significant changes to wording and structure of the description. Preserve all semantics, code, and literals.",
    0.7: "HEAVY MIXED: Major restructuring of the natural-language description. Code, function signatures, and examples must remain identical.",
    0.8: "NEAR PARAPHRASE: Extensive restructuring of the description with different patterns while preserving the exact behavior.",
    0.9: "FULL PARAPHRASE: Completely rephrase the description while keeping function behavior, code, and examples semantically identical.",
}

# Default miu values for distortion levels
DEFAULT_MIU_VALUES: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Default number of distortions per question per miu level
DEFAULT_DISTORTIONS_PER_QUESTION: int = 10

# ============================================================================
# Temperature Calculation Formula
# ============================================================================

def calculate_temperature(miu: float) -> float:
    """
    Calculate optimal temperature for distortion based on miu value.

    The formula combines three components:
    1. Exponential base: Higher miu = more temperature
    2. Sigmoid activation: Sharper increase around miu=0.5
    3. Paraphrase boost: Extra temperature for high miu (>=0.7)

    Args:
        miu: Distortion intensity (0.0 to 1.0)

    Returns:
        Temperature value (clamped to 0.1-1.5 for API compatibility)
    """
    if miu == 0.0:
        return 0.1

    # Component 1: Exponential base
    exp_component = 0.3 + (miu ** 1.5) * 1.2

    # Component 2: Sigmoid activation around miu=0.5
    sig_component = 0.3 * (1 / (1 + math.exp(-10 * (miu - 0.5))))

    # Component 3: Paraphrase boost for high miu
    para_component = 0.2 * ((miu - 0.7) / 0.3) ** 2 if miu >= 0.7 else 0

    # Combine and clamp
    temperature = exp_component + sig_component + para_component
    return round(max(0.1, min(1.5, temperature)), 2)


# ============================================================================
# Prompt Templates
# Note: These are generic fallbacks. Benchmarks should provide their own specific prompts.
# ============================================================================
DISTORTION_SYSTEM_PROMPT = """You are a distortion expert. 
Your task is to create variations of given prompts while preserving their core logic and requirements."""



def get_distortion_prompt(question: str, miu: float, n_distortions: int) -> str:
    """
    Generic distortion prompt (fallback for benchmarks that don't provide custom prompts).
    For HumanEval, use the specific version in benchmarks.human_eval.prompts instead.
    
    Args:
        question: The original prompt text
        miu: Distortion intensity level
        n_distortions: Number of unique distortions to generate
    
    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    return f"""Distort this prompt {n_distortions} unique ways at μ={miu} intensity level.

DISTORTION RULE for μ={miu}: {rule}

ORIGINAL: {question}

REQUIREMENTS:
1. Preserve semantic logic.
2. Follow rule for μ={miu}.
3. Output list of {n_distortions} distortions.
"""



def get_batch_distortion_prompt(questions: List[Dict[str, str]], miu: float, n_distortions: int) -> str:
    """
    Generate a prompt for distorting multiple HumanEval prompts in one API call.

    Args:
        questions: List of dicts with 'text' key containing HumanEval-style prompts
        miu: Distortion intensity level
        n_distortions: Number of unique distortions per question

    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    q_text = "".join(f"\nQ{i}: {q['text']}\n" for i, q in enumerate(questions, 1))

    return f"""⚠️ MANDATORY: You MUST output EXACTLY {n_distortions} UNIQUE distortions for EACH HumanEval prompt. NO MORE, NO LESS.

TASK: Distort each HumanEval-style programming task description at μ={miu} intensity level.

DISTORTION RULE for μ={miu}: {rule}
{q_text}

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


def get_retry_distortion_prompt(question: Dict, miu: float, needed: int) -> str:
    """
    Generate a prompt for retrying distortions when we already have some filled.
    Includes existing distortions so the LLM knows what to avoid.

    Args:
        question: Dict with 'text', 'existing' (list of existing distortions), 'needed' (how many more)
        miu: Distortion intensity level
        needed: Number of NEW distortions needed

    Returns:
        Formatted prompt string
    """
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    q_text = question['text']
    existing = question.get('existing', [])

    existing_list = ""
    if existing:
        existing_list = "\n\n⚠️ ALREADY GENERATED (DO NOT REPEAT THESE OR SIMILAR):\n"
        for i, e in enumerate(existing, 1):
            existing_list += f"  {i}. {e}\n"

    return f"""⚠️ CRITICAL: Generate EXACTLY {needed} NEW unique distortion(s) for this HumanEval programming prompt.

TASK PROMPT:
{q_text}

DISTORTION RULE for μ={miu}: {rule}
{existing_list}

═══════════════════════════════════════════════════════════════════════
REQUIREMENTS:
═══════════════════════════════════════════════════════════════════════

1. Generate EXACTLY {needed} NEW distortion(s)
2. Each must be COMPLETELY DIFFERENT from the existing {len(existing)} distortions above
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
# Evaluation Prompt (for target model)
# ============================================================================

EVALUATION_SYSTEM_PROMPT = """You are an expert Python 3 programmer solving HumanEval-style coding tasks.
Your ONLY output should be valid Python code that solves the task.
NEVER explain, NEVER add extra text, NEVER show reasoning.
Do NOT wrap the code in markdown or backticks.
Keep the provided function signature(s) unchanged and do not add a main block or print statements unless explicitly required by the prompt."""


def get_evaluation_prompt(question: str, choices: Dict[str, str]) -> str:
    """
    Generate a prompt for solving a HumanEval-style programming task
    with the target model.

    This wraps the task description (and optional context) into a single
    user prompt that asks the model to output only Python code.

    Args:
        question: The (possibly distorted) HumanEval problem description
        choices: Optional dict of extra context. Only single-letter keys
                 (A, B, C, ...) are included in the formatted text.

    Returns:
        Formatted prompt string
    """
    # Normalize keys to uppercase for consistent display
    normalized_choices = {}
    for k, v in choices.items():
        key = k.strip().upper() if isinstance(k, str) else str(k).upper()
        # Handle keys like "A:" or "a:" - extract just the letter
        if key.endswith(':'):
            key = key[:-1]
        if len(key) == 1 and key in 'ABCDEFGH':
            normalized_choices[key] = v

    # Sort and format choices
    choices_text = "\n".join([f"{k}: {v}" for k, v in sorted(normalized_choices.items())])

    return f"""═══════════════════════════════════════════════════════════════════════
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


# ============================================================================
# API Configuration Defaults
# ============================================================================

API_DEFAULTS = {
    "mistral": {
        "base_url": "https://api.mistral.ai/v1/chat/completions",
        "default_model": "mistral-large-latest",
        "max_tokens": 8000,  # Increased default for longer responses
        "timeout": 120,
    },
    "openai": {
        "default_model": "gpt-5.1",
        "max_tokens": 4000,
        "timeout": 120,
    },
}

# Batch processing defaults
BATCH_DEFAULTS = {
    "questions_per_batch": 5,  # Questions per API call for distortion
    "workers_per_miu": 2,      # Parallel workers per miu level
    "max_retries": 3,          # Retry attempts for failed API calls
    "save_interval": 30,       # Seconds between auto-saves
}


def calculate_max_tokens(questions: list, n_distortions: int, miu: float = 0.5, buffer_pct: float = 0.20) -> int:
    """
    Calculate max_tokens dynamically based on prompt length and miu.

    Higher miu = more paraphrasing = potentially longer output.

    Length multiplier by miu:
    - miu 0.0-0.3: 1.2x (minimal changes, similar length)
    - miu 0.4-0.6: 1.4x (moderate changes)
    - miu 0.7-0.9: 1.6x (heavy paraphrasing, can be longer)

    Args:
        questions: List of prompt dicts with 'text' key
        n_distortions: Number of distortions per prompt
        miu: Distortion intensity (0.0-1.0)
        buffer_pct: Additional buffer percentage (default 20%)

    Returns:
        Recommended max_tokens for the API call
    """
    # Length multiplier based on miu (higher miu = more expansion allowed)
    if miu <= 0.3:
        length_multiplier = 1.2
    elif miu <= 0.6:
        length_multiplier = 1.4
    else:
        length_multiplier = 1.6

    # Estimate tokens (roughly 1.5 tokens per word for English)
    total_words = sum(len(q.get('text', '').split()) for q in questions)

    # Each question needs N distortions, each distortion ~multiplier * original length
    # Plus formatting overhead (~20 tokens per distortion for "Q1:", "1.", "2.", newlines etc.)
    estimated_output_words = total_words * length_multiplier * n_distortions
    formatting_overhead = len(questions) * n_distortions * 20  # Q markers + numbering + newlines

    # Convert words to tokens (1.5x for safety) and add buffer
    estimated_tokens = int((estimated_output_words * 1.5 + formatting_overhead) * (1 + buffer_pct))

    # Clamp between reasonable bounds
    min_tokens = 2000  # Minimum for any reasonable response
    max_tokens = 16000  # Mistral large supports up to 32k context

    return max(min_tokens, min(estimated_tokens, max_tokens))
