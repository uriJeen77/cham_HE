import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import time

from dotenv import load_dotenv
from mistralai import Mistral

# Import internal modules (assuming running from root)
from chameleon.distortion.engine import DistortionEngine
from human_eval.execution import check_correctness

# Config
PROJECT_ROOT = Path(__file__).parent
HUMAN_EVAL_FILE = PROJECT_ROOT / "Projects" / "HumanEvalV3" / "original_data" / "human-eval-v2-20210705.jsonl"
OUTPUT_FILE = PROJECT_ROOT / "custom_pipeline_results.json"
MODEL_NAME = "mistral-large-latest" # Using for all steps for robustness

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(limit: int = 3) -> List[Dict[str, Any]]:
    """Load first N samples from HumanEval."""
    samples = []
    if not HUMAN_EVAL_FILE.exists():
        raise FileNotFoundError(f"Data file not found at {HUMAN_EVAL_FILE}")
    
    with open(HUMAN_EVAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if len(samples) >= limit:
                break
            if line.strip():
                samples.append(json.loads(line))
    return samples

class MegaJudge:
    """Validates if distorted prompt maintains original meaning."""
    def __init__(self, api_key: str, model: str):
        self.client = Mistral(api_key=api_key)
        self.model = model

    def validate(self, original: str, distorted: str) -> bool:
        """Returns True if meanings are equivalent."""
        prompt = f"""
You are an expert impartial judge of semantic equivalence in coding tasks.
Compare the two following problem descriptions.

[ORIGINAL PROMPT]
{original}

[DISTORTED PROMPT]
{distorted}

Task:
Determine if the [DISTORTED PROMPT] asks for the EXACT SAME logical code solution as the [ORIGINAL PROMPT].
Ignore stylistic changes, story wrapper, or tone differences.
Focus ONLY on input/output requirements, constraints, and algorithmic logic.

Respond with valid JSON only:
{{
  "equivalent": boolean,
  "reason": "short explanation"
}}
"""
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return result.get("equivalent", False)
        except Exception as e:
            logger.error(f"Judge error: {e}")
            return False

class CodeGenerator:
    """Generates code using Mistral."""
    def __init__(self, api_key: str, model: str):
        self.client = Mistral(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        # Simple prompt wrapping
        messages = [
            {"role": "system", "content": "You are a Python coding assistant. Return only the function implementation. Do not include markdown fences or explanation."},
            {"role": "user", "content": prompt}
        ]
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=0.2 
            )
            code = response.choices[0].message.content
            # Cleanup markdown if present
            code = code.replace("```python", "").replace("```", "").strip()
            return code
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return ""

def main():
    logger.info("Starting Custom Pipeline...")
    
    # 0. Setup
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("MISTRAL_API_KEY not found in .env")
        return

    # 1. Load Data
    data = load_data(3)
    logger.info(f"Loaded {len(data)} samples.")

    # Components
    # We reuse DistortionEngine but configure it manually here for simplicity or via config dict
    distorter = DistortionEngine.create({
        "engine_type": "api",
        "vendor": "mistral",
        "model_name": MODEL_NAME,
        "api_key_env_var": "MISTRAL_API_KEY"
    }, project_path=PROJECT_ROOT)
    
    judge = MegaJudge(api_key, MODEL_NAME)
    coder = CodeGenerator(api_key, MODEL_NAME)

    results = []

    for task in data:
        task_id = task["task_id"]
        original_prompt = task["prompt"]
        logger.info(f"\n--- Processing {task_id} ---")

        # A. Distortion
        logger.info("Distorting...")
        dist_res = distorter.distort_question(
            question_id=task_id, 
            question=original_prompt, 
            miu=0.6 # Medium distortion
        )
        if not dist_res.success:
            logger.error("Distortion failed")
            continue
            
        distorted_prompt = dist_res.distorted_question

        # B. Mega Validation
        logger.info("Validating...")
        is_valid = judge.validate(original_prompt, distorted_prompt)
        logger.info(f"Judge Verdict: {'PASS' if is_valid else 'FAIL'}")

        if not is_valid:
            logger.warning("Skipping execution due to validation failure.")
            results.append({
                "task_id": task_id,
                "status": "validation_failed",
                "original": original_prompt,
                "distorted": distorted_prompt
            })
            continue

        # C. Code Generation & Eval
        logger.info("Generating Code...")
        # 1. Original
        code_orig = coder.generate(original_prompt)
        # 2. Distorted
        code_dist = coder.generate(distorted_prompt)

        # D. Execution (Evaluation)
        logger.info("Executing Tests...")
        # We need a 'problem' dict that matches HumanEval format for check_correctness
        # The 'task' object loaded from JSONL is already in that format.
        
        # Test Original
        res_orig = check_correctness(task, code_orig, timeout=3.0)
        # Test Distorted
        res_dist = check_correctness(task, code_dist, timeout=3.0)

        logger.info(f"Original: {'PASS' if res_orig['passed'] else 'FAIL'}")
        logger.info(f"Distorted: {'PASS' if res_dist['passed'] else 'FAIL'}")

        results.append({
            "task_id": task_id,
            "status": "completed",
            "validation": "passed",
            "original_result": res_orig,
            "distorted_result": res_dist,
            "distorted_prompt": distorted_prompt,
            "code_original": code_orig,
            "code_distorted": code_dist
        })

    # Save
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nPipeline finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
