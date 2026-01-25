"""
Distortion Validator for VQAv2 (Optimized)
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
from enum import Enum

class ValidationFailure(Enum):
    EMPTY = "empty"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    NO_QUESTION_MARK = "missing_question_mark"
    TYPE_MISMATCH = "question_type_mismatch"
    KEYWORD_LOST = "critical_keywords_lost"  # New check
    MARKDOWN = "markdown_formatting"
    PREAMBLE = "preamble_text"
    IDENTICAL = "identical_to_original"

@dataclass
class ValidationResult:
    is_valid: bool
    original: str
    distorted: str
    miu: float
    failures: List[ValidationFailure]
    details: Optional[str] = None

# Updated Heuristics
QUESTION_STARTERS = {
    "count": ["how many", "number of"], 
    # "what number" is handled specifically in the function logic below
    "yesno": ["is ", "are ", "does ", "do ", "can ", "could ", "has ", "have ", "was ", "were "], # Added spaces to avoid partial matches
    "what": ["what", "which"],
    "where": ["where"],
    "who": ["who"]
}

PREAMBLE_PATTERNS = [
    r'^here are', r'^below are', r'^the rephrased', r'^rephrased:', 
    r'^variation:', r'^alternative:', r'^sure[,!]', r'^certainly'
]

# Stopwords to ignore in keyword check
STOPWORDS = {"the", "a", "an", "is", "are", "do", "does", "in", "on", "of", "to", "at", "it", "this", "that", "with", "and", "or"}

def get_keywords(text: str) -> Set[str]:
    """Extracts significant words for overlap checking."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return {w for w in words if w not in STOPWORDS and len(w) > 2}

def detect_question_type(text: str) -> str:
    text = text.lower().strip()
    
    # Priority 1: Count (overrides "What" if asking about numbers)
    if "number" in text or text.startswith("how many"):
        return "count"
        
    # Priority 2: Standard starters
    for q_type, triggers in QUESTION_STARTERS.items():
        if q_type == "count": continue # Already checked
        for trigger in triggers:
            if text.startswith(trigger):
                return q_type
    return "other"

def clean_distortion(text: str) -> str:
    if not text: return ""
    text = str(text).strip()
    text = re.sub(r'\*\*|\*|`|##|#', '', text)
    text = re.sub(r'^(?:\d+[\.\)]|Q\d?[:\.]|Question\s*\d?[:\.]|New[:\.]|Rephrased[:\.])\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^["\']|["\']$', '', text)
    replacements = {'â€™': "'", 'â€œ': '"', 'â€': '"', 'â€"': '-'}
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text.strip()

def validate_distortion(
    original: str,
    distorted: str,
    miu: float
) -> ValidationResult:
    failures: List[ValidationFailure] = []
    distorted_clean = clean_distortion(distorted)

    if not distorted_clean:
        return ValidationResult(False, original, "", miu, [ValidationFailure.EMPTY])

    # Length Checks
    orig_words = original.split()
    dist_words = distorted_clean.split()
    
    # Min length: Allow distinct 2-word questions like "Who's that?"
    if len(dist_words) < 2: 
        failures.append(ValidationFailure.TOO_SHORT)
    
    if miu < 0.8 and len(dist_words) > (len(orig_words) * 2.5) + 5:
        failures.append(ValidationFailure.TOO_LONG)

    # Syntax Check
    if not distorted_clean.endswith('?'):
        if distorted_clean[-1].isalnum():
             distorted_clean += "?"
        else:
             failures.append(ValidationFailure.NO_QUESTION_MARK)

    # Identity Check
    if miu > 0 and distorted_clean.lower() == original.lower().strip():
        failures.append(ValidationFailure.IDENTICAL)

    # Preamble Detection
    if any(re.match(p, distorted_clean.lower()) for p in PREAMBLE_PATTERNS):
        failures.append(ValidationFailure.PREAMBLE)

    # Semantic Consistency Checks (Skipped for High Miu > 0.6)
    if miu < 0.6:
        # 1. Type Mismatch
        orig_type = detect_question_type(original)
        dist_type = detect_question_type(distorted_clean)
        
        if (orig_type == "count" and dist_type == "yesno") or \
           (orig_type == "yesno" and dist_type == "count"):
            failures.append(ValidationFailure.TYPE_MISMATCH)

        # 2. Keyword Preservation (Prevents "dog" -> "cat")
        # Ensure at least 50% of original significant keywords exist in distortion
        orig_keys = get_keywords(original)
        dist_keys = get_keywords(distorted_clean)
        
        if orig_keys:
            overlap = orig_keys.intersection(dist_keys)
            # If we lost too many keywords, it's suspicious
            if len(overlap) / len(orig_keys) < 0.4: 
                failures.append(ValidationFailure.KEYWORD_LOST)

    return ValidationResult(
        is_valid=len(failures) == 0,
        original=original,
        distorted=distorted_clean,
        miu=miu,
        failures=failures,
        details=str([f.value for f in failures]) if failures else None
    )

def parse_llm_response(text: str) -> List[str]:
    results = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        match = re.match(r'^(?:\d+[\.\)]|-|\*)\s*(.*)', line)
        if match:
            candidate = match.group(1).strip()
            # Lowered threshold to 2 chars to allow "Why?" etc.
            if len(candidate) > 2: 
                results.append(clean_distortion(candidate))
        elif len(line) > 5 and line.endswith('?'):
             results.append(clean_distortion(line))
             
    return results