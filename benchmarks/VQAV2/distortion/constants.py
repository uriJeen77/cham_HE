"""
Distortion configuration for VQAv2.
Optimized for semantic preservation while maximizing syntactic diversity.
"""

from typing import Dict, List

# ============================================================================
# MIU (μ) Distortion Rules
# ============================================================================

BASE_CONSTRAINT = (
    "CRITICAL: The answer to the question MUST remain exactly the same. "
    "Do not assume new details. Do not change numbers, colors, or directions."
)

MIU_RULES: Dict[float, str] = {
    0.0: "Return the original question exactly as is.",
    0.1: f"{BASE_CONSTRAINT} Replace 1 word with a synonym (e.g., 'big' -> 'large'). Keep syntax identical.",
    0.2: f"{BASE_CONSTRAINT} Replace 2-3 words with synonyms. Keep sentence structure.",
    0.3: f"{BASE_CONSTRAINT} Change the sentence structure slightly (e.g., active to passive) but keep vocabulary similar.",
    0.4: f"{BASE_CONSTRAINT} Rephrase standard phrases (e.g., 'What kind of' -> 'Which type of').",
    0.5: f"{BASE_CONSTRAINT} Moderate paraphrase: Change both vocabulary and structure, but keep key entities (nouns) unchanged.",
    0.6: f"{BASE_CONSTRAINT} Significant rephrase: You may split or combine clauses, provided clarity is maintained.",
    0.7: f"{BASE_CONSTRAINT} High distortion: Use rare synonyms and complex sentence structures.",
    0.8: f"{BASE_CONSTRAINT} Intense distortion: Rewrite the question as if asked by a completely different speaker (e.g., formal vs casual).",
    0.9: f"{BASE_CONSTRAINT} Maximum distortion: Change every word possible while keeping the logic strictly identical.",
}

# ============================================================================
# Temperature Calculation (Safety First)
# ============================================================================

def calculate_temperature(miu: float) -> float:
    """
    Linear scaling with safety cap.
    VQA requires precision, so we never exceed 1.0 even at high miu.
    """
    if miu < 0.1: return 0.0
    
    # Base temp 0.2, scales up to 0.9 at miu=1.0
    temp = 0.2 + (miu * 0.7) 
    
    return round(min(1.0, temp), 2)

# ============================================================================
# System Prompts
# ============================================================================

DISTORTION_SYSTEM_PROMPT = """You are a data augmentor for a Visual Question Answering dataset.
Input: A question about a hidden image.
Task: Generate distinct rephrasings of the question.

CONSTRAINTS:
1. The Semantic Logic must be invariant (Answer must not change).
2. Do NOT add new visual constraints (e.g., don't change 'cat' to 'black cat').
3. Do NOT remove visual constraints (e.g., don't change 'black cat' to 'animal').
4. Output must end with a question mark.
"""

def get_distortion_prompt(question: str, miu: float, n_distortions: int) -> str:
    rule = MIU_RULES.get(miu, MIU_RULES[0.5])
    return f"""
Original Question: "{question}"

Instruction (Intensity μ={miu}):
{rule}

Output exactly {n_distortions} variations.
Format:
1. [Variation 1]
2. [Variation 2]
...
"""

# ============================================================================
# API Defaults
# ============================================================================

API_DEFAULTS = {
    "openai": {
        "default_model": "gpt-4o",
        "max_tokens": 300, # VQA questions are short
        "timeout": 60,
    }
}