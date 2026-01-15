"""
HumanEval Benchmark - Consolidated Implementation

A single-file implementation of the HumanEval robustness evaluation pipeline.
This version consolidates all steps into one class for simplicity.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import os
import ast
import numpy as np
import logging

from chameleon.core.base import BaseBenchmark
from benchmarks.human_eval.engine.data import read_problems
from benchmarks.human_eval.engine.execution import check_correctness
from mistralai import Mistral
from chameleon.distortion.engine import DistortionEngine
from benchmarks.human_eval.prompts import (
    get_distortion_prompt,
    get_validation_prompt,
    get_generation_prompt
)
from chameleon.distortion.constants import MIU_RULES


# ============================================================================
# PROMPTS
# ============================================================================


def get_code_generation_messages(prompt: str) -> List[Dict[str, str]]:
    """Generate messages for code generation API call."""
    return [
        {"role": "system", "content": GENERATE_CODE_SYSTEM},
        {"role": "user", "content": prompt}
    ]


# ============================================================================
# MAIN BENCHMARK CLASS
# ============================================================================

class HumanEvalBenchmark(BaseBenchmark):
    """
    Consolidated HumanEval benchmark for robustness evaluation.
    
    Pipeline steps:
    1. Load tasks from JSONL
    2. Distort prompts using LLM
    3. Validate distortions preserve semantic meaning
    4. Generate code for both original and distorted prompts
    5. Validate Python syntax
    6. Run unit tests and calculate metrics
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the benchmark.
        
        Args:
            config: Configuration dict with optional keys:
                - distortion_model: Model for distortion (default: mistral-large-latest)
                - validation_model: Model for validation (default: mistral-large-latest)
                - generation_model: Model for code generation (default: mistral-large-latest)
                - timeout: Timeout for code execution (default: 3.0)
                - k_values: List of k values for pass@k (default: [1])
                - miu: Distortion level 0.0-1.0 (default: 0.6)
                - project_path: Path to project root
        """
        self.config = config or {}
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the pipeline."""
        logger = logging.getLogger("HumanEvalBenchmark")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    # ========================================================================
    # STEP 1: LOAD TASKS
    # ========================================================================
    
    def load_tasks(self, data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Step 1: Load tasks from JSONL file.
        
        Args:
            data_path: Path to JSONL file (can be .gz)
            limit: Maximum number of tasks to load (None = all)
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        
        # Handle gzipped or regular JSONL
        if data_path.endswith('.gz'):
            all_tasks = list(read_problems(data_path).values())
            tasks = all_tasks[:limit] if limit else all_tasks
        else:
            # Read directly from JSONL
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if limit and len(tasks) >= limit:
                        break
                    if line.strip():
                        tasks.append(json.loads(line))
        
        self.logger.info(f"✓ Loaded {len(tasks)} tasks from {data_path}")
        return tasks

    # ========================================================================
    # STEP 2: DISTORT PROMPTS
    # ========================================================================
    
    def distort_prompts(self, tasks: List[Dict[str, Any]], miu: float = 0.6) -> List[Dict[str, Any]]:
        """
        Step 2: Distort prompts using Mistral.
        
        Args:
            tasks: List of original tasks
            miu: Distortion level (0.0 = no distortion, 1.0 = maximum distortion)
            
        Returns:
            Same tasks list with added 'distorted_prompt' field
        """
        # Initialize DistortionEngine
        distorter = DistortionEngine.create({
            "engine_type": "api",
            "vendor": "mistral",
            "model_name": self.config.get("distortion_model", "mistral-large-latest"),
            "api_key_env_var": "MISTRAL_API_KEY"
        }, project_path=self.config.get("project_path"))
        
        self.logger.info(f"🔄 Distorting {len(tasks)} prompts with miu={miu}...")
        
        success_count = 0
        for task in tasks:
            task_id = task["task_id"]
            original_prompt = task["prompt"]
            
            try:
                # Get centralized prompts
                user_prompt = get_distortion_prompt(original_prompt, miu, n_distortions=1)
                
                # Run distortion with custom prompts
                dist_result = distorter.distort_question(
                    question_id=task_id,
                    question=original_prompt,
                    miu=miu,
                    custom_prompt=user_prompt,
                )
                
                if dist_result.success:
                    task["distorted_prompt"] = dist_result.distorted_question
                    task["distortion_success"] = True
                    task["distortion_miu"] = miu
                    success_count += 1
                    self.logger.debug(f"  ✓ {task_id} - Distortion completed")
                else:
                    task["distorted_prompt"] = None
                    task["distortion_success"] = False
                    self.logger.warning(f"  ⚠️ {task_id} - Distortion failed")
                    
            except Exception as e:
                task["distorted_prompt"] = None
                task["distortion_success"] = False
                self.logger.error(f"  ⚠️ {task_id} - Error: {e}")
        
        self.logger.info(f"✅ Distorted {success_count}/{len(tasks)} tasks\n")
        return tasks

    # ========================================================================
    # STEP 3: VALIDATE DISTORTIONS
    # ========================================================================
    
    def validate_distortions(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Validate that logical meaning is preserved.
        
        Uses MegaJudge (Mistral) to verify that the distorted prompt
        requires the same logical function as the original.
        
        Args:
            tasks: List of tasks with distorted_prompt field
            
        Returns:
            Same tasks list with added 'validation_passed' field
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            self.logger.warning("⚠️ MISTRAL_API_KEY not found - skipping validation")
            for task in tasks:
                task["validation_passed"] = None
            return tasks
        
        client = Mistral(api_key=api_key)
        model = self.config.get("validation_model", "mistral-large-latest")
        
        self.logger.info(f"🔍 Validating {len(tasks)} distortions...")
        
        success_count = 0
        for task in tasks:
            # Skip if distortion failed
            if not task.get("distortion_success", False):
                task["validation_passed"] = False
                task["validation_reason"] = "Distortion failed"
                task["equivalence_level"] = "Not Equivalent"
                continue
                
            original = task["prompt"]
            distorted = task.get("distorted_prompt", "")
            task_id = task["task_id"]
            
            try:
                # Build judge prompt
                user_prompt = get_validation_prompt(original, distorted)
                
                # Call API
                response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"}
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Check equivalence
                equivalence = result.get("equivalence_level", "")
                is_safe = result.get("is_safe_to_test", False)
                is_valid = (equivalence == "Identical") and is_safe
                
                task["validation_passed"] = is_valid
                task["validation_reason"] = result.get("analysis", "")
                task["equivalence_level"] = equivalence
                
                if is_valid:
                    success_count += 1
                    self.logger.debug(f"  ✓ {task_id} - Passed validation")
                else:
                    self.logger.debug(f"  ✗ {task_id} - Failed ({equivalence})")
                    
            except Exception as e:
                task["validation_passed"] = None
                task["validation_reason"] = f"Error: {str(e)}"
                task["equivalence_level"] = ""
                self.logger.error(f"  ⚠️ {task_id} - Validation error: {e}")
        
        self.logger.info(f"✅ {success_count}/{len(tasks)} tasks passed validation\n")
        return tasks

    # ========================================================================
    # STEP 4: GENERATE CODE
    # ========================================================================
    
    def generate_code(self, tasks: List[Dict[str, Any]], for_distorted: bool = False) -> List[Dict[str, Any]]:
        """
        Step 4: Generate Python code using Mistral.
        
        Args:
            tasks: List of tasks
            for_distorted: If True, generate for distorted_prompt; else for original prompt
            
        Returns:
            Same tasks list with added code field
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            self.logger.error("⚠️ MISTRAL_API_KEY not found - cannot generate code")
            return tasks
        
        client = Mistral(api_key=api_key)
        model = self.config.get("generation_model", "mistral-large-latest")
        
        code_field = "distorted_code" if for_distorted else "original_code"
        prompt_type = "distorted" if for_distorted else "original"
        
        self.logger.info(f"💻 Generating {prompt_type} code for {len(tasks)} tasks...")
        
        success_count = 0
        for task in tasks:
            task_id = task["task_id"]
            
            # Choose which prompt to use
            if for_distorted:
                # Skip if no distorted prompt or validation failed
                if not task.get("distortion_success") or not task.get("validation_passed"):
                    task[code_field] = None
                    continue
                prompt = task["distorted_prompt"]
            else:
                prompt = task["prompt"]
            
            try:
                # Get full prompt from helper
                full_prompt = get_generation_prompt(prompt)
                
                # Send request to Mistral
                response = client.chat.complete(
                    model=model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.2
                )
                
                code = response.choices[0].message.content
                # Remove markdown if present
                code = code.replace("```python", "").replace("```", "").strip()
                
                task[code_field] = code
                success_count += 1
                self.logger.info(f"  ✓ {task_id} - {prompt_type} code generated")
                self.logger.debug(f"RAW GENERATED CODE:\n{code}")
                
            except Exception as e:
                task[code_field] = None
                self.logger.error(f"  ⚠️ {task_id} - Error: {e}")
        
        self.logger.info(f"✅ Generated {success_count}/{len(tasks)} {prompt_type} code\n")
        return tasks

    # ========================================================================
    # STEP 5: VALIDATE SYNTAX
    # ========================================================================
    
    def validate_syntax(self, tasks: List[Dict[str, Any]], for_distorted: bool = False) -> List[Dict[str, Any]]:
        """
        Step 5: Validate Python syntax.
        
        Args:
            tasks: List of tasks with generated code
            for_distorted: If True, validate distorted_code; else validate original_code
            
        Returns:
            Same tasks list with added syntax validation field
        """
        code_field = "distorted_code" if for_distorted else "original_code"
        status_field = "distorted_syntax_valid" if for_distorted else "original_syntax_valid"
        code_type = "distorted" if for_distorted else "original"
        
        self.logger.info(f"✅ Validating {code_type} syntax for {len(tasks)} tasks...")
        
        valid_count = 0
        for task in tasks:
            task_id = task["task_id"]
            code = task.get(code_field, "")
            
            if not code:
                task[status_field] = False
                continue
            
            try:
                # Try to parse the code
                # The generated code should be a complete, standalone function
                ast.parse(code)
                task[status_field] = True
                valid_count += 1
                self.logger.debug(f"  ✓ {task_id} - {code_type} syntax valid")
            except SyntaxError as e:
                task[status_field] = False
                self.logger.warning(f"  ✗ {task_id} - {code_type} syntax error: {e}")
                self.logger.debug(f"INVALID CODE:\n{code}")
        
        self.logger.info(f"✅ {valid_count}/{len(tasks)} {code_type} tasks have valid syntax\n")
        return tasks

    # ========================================================================
    # STEP 6: EVALUATE CODE
    # ========================================================================
    
    def evaluate_completion(self, task: Dict[str, Any], completion: str) -> Dict[str, Any]:
        """
        Evaluate a single code completion against unit tests.
        
        Args:
            task: Task with test cases
            completion: Generated code to test
            
        Returns:
            Result dictionary with task_id, passed, and result fields
        """
        timeout = self.config.get("timeout", 1.5)
        
        try:
            result = check_correctness(task, completion, timeout)
            return result
        except Exception as e:
            return {
                "task_id": task["task_id"],
                "passed": False,
                "result": f"Error: {str(e)}"
            }
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate pass@k metrics from evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with pass@k values
        """
        # Group results by task_id
        task_results = {}
        for res in results:
            task_id = res["task_id"]
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(res["passed"])

        total = []
        correct = []
        for task_id, passed_list in task_results.items():
            total.append(len(passed_list))
            correct.append(sum(passed_list))
        
        total = np.array(total)
        correct = np.array(correct)
        
        # Calculate pass@k
        k_values = self.config.get("k_values", [1])
        metrics = {}
        for k in k_values:
            if (total >= k).all():
                pass_at_k = self._estimate_pass_at_k(total, correct, k).mean()
                metrics[f"pass@{k}"] = float(pass_at_k)
        
        return metrics

    def _estimate_pass_at_k(self, n, c, k):
        """
        Estimate pass@k using the formula from the HumanEval paper.
        
        Args:
            n: Total number of samples per task
            c: Number of correct samples per task
            k: Number of samples to consider
            
        Returns:
            Array of pass@k estimates
        """
        def estimator(n, c, k):
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
        
        return np.array([estimator(int(ni), int(ci), k) for ni, ci in zip(n, c)])

    # ========================================================================
    # FULL PIPELINE
    # ========================================================================
    
    def run_full_pipeline(
        self, 
        data_path: str, 
        output_dir: str, 
        miu: float = 0.6, 
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to evaluation.
        
        Args:
            data_path: Path to HumanEval JSONL file
            output_dir: Directory to save results
            miu: Distortion level (0.0-1.0)
            limit: Optional limit on number of tasks to process
            
        Returns:
            Summary dictionary with all results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("=" * 80)
        self.logger.info("⭐  HumanEval - Robustness Evaluation Pipeline  ⭐")
        self.logger.info("=" * 80 + "\n")
        
        # Step 1: Loading
        self.logger.info("📚 [STEP 1/6] Loading data...")
        tasks = self.load_tasks(data_path, limit=limit)
        
        # Step 2: Distorting
        self.logger.info("\n🔀 [STEP 2/6] Distorting prompts (Mistral)...")
        tasks = self.distort_prompts(tasks, miu=miu)
        
        # Step 3: Validating
        self.logger.info("\n🔍 [STEP 3/6] Validating distortions (MegaJudge)...")
        tasks = self.validate_distortions(tasks)
        
        # Step 4a: Original Generation
        self.logger.info("\n💻 [STEP 4a/6] Generating code for ORIGINAL prompts...")
        tasks = self.generate_code(tasks, for_distorted=False)
        
        # Step 4b: Distorted Generation
        self.logger.info("\n💻 [STEP 4b/6] Generating code for DISTORTED prompts...")
        tasks = self.generate_code(tasks, for_distorted=True)
        
        # Step 5a: Original Syntax
        self.logger.info("\n✅ [STEP 5a/6] Validating ORIGINAL code syntax...")
        tasks = self.validate_syntax(tasks, for_distorted=False)
        
        # Step 5b: Distorted Syntax
        self.logger.info("\n✅ [STEP 5b/6] Validating DISTORTED code syntax...")
        tasks = self.validate_syntax(tasks, for_distorted=True)
        
        # Step 6: Testing
        self.logger.info("\n🧪 [STEP 6/6] Running unit tests and calculating metrics...")
        
        
        original_results = []
        distorted_results = []
        
        for task in tasks:
            task_id = task["task_id"]
            
            # Evaluate original code if syntax is valid
            if task.get("original_syntax_valid", False):
                result = self.evaluate_completion(task, task["original_code"])
                original_results.append(result)
                if result["passed"]:
                    self.logger.info(f"  ✓ {task_id} - ORIGINAL tests PASSED")
                else:
                    self.logger.warning(f"  ✗ {task_id} - ORIGINAL tests FAILED: {result.get('result', 'Unknown error')}")
            else:
                self.logger.debug(f"  ⊗ {task_id} - ORIGINAL code skipped (syntax invalid)")
            
            # Evaluate distorted code if syntax is valid
            if task.get("distorted_syntax_valid", False):
                result_dist = self.evaluate_completion(task, task["distorted_code"])
                distorted_results.append(result_dist)
                if result_dist["passed"]:
                    self.logger.info(f"  ✓ {task_id} - DISTORTED tests PASSED")
                else:
                    self.logger.warning(f"  ✗ {task_id} - DISTORTED tests FAILED: {result_dist.get('result', 'Unknown error')}")
            else:
                self.logger.debug(f"  ⊗ {task_id} - DISTORTED code skipped (syntax invalid)")
        
        # Calculate metrics
        original_metrics = self.calculate_metrics(original_results) if original_results else {"pass@1": 0.0}
        distorted_metrics = self.calculate_metrics(distorted_results) if distorted_results else {"pass@1": 0.0}
        
        # Create summary
        summary = {
            "total_tasks": len(tasks),
            "distortion_successful": sum(1 for t in tasks if t.get("distortion_success", False)),
            "validation_passed": sum(1 for t in tasks if t.get("validation_passed") == True),
            "original_code_generated": sum(1 for t in tasks if t.get("original_code")),
            "distorted_code_generated": sum(1 for t in tasks if t.get("distorted_code")),
            "original_syntax_valid": sum(1 for t in tasks if t.get("original_syntax_valid", False)),
            "distorted_syntax_valid": sum(1 for t in tasks if t.get("distorted_syntax_valid", False)),
            "original_metrics": original_metrics,
            "distorted_metrics": distorted_metrics,
            "miu": miu
        }
        
        # Save results
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        with open(output_path / "tasks_complete.jsonl", "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        
        # Print final results
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 Final Results:")
        self.logger.info("=" * 80)
        self.logger.info(f"✓ Total tasks: {summary['total_tasks']}")
        self.logger.info(f"✓ Distortions successful: {summary['distortion_successful']}")
        self.logger.info(f"✓ Semantic validations passed: {summary['validation_passed']}")
        self.logger.info(f"✓ Original code generated: {summary['original_code_generated']}")
        self.logger.info(f"✓ Distorted code generated: {summary['distorted_code_generated']}")
        self.logger.info("-" * 40)
        self.logger.info(f"🏆 Original pass@1: {original_metrics.get('pass@1', 0):.2%}")
        self.logger.info(f"🏆 Distorted pass@1: {distorted_metrics.get('pass@1', 0):.2%}")
        self.logger.info("=" * 80 + "\n")
        
        return summary

    def format_prompt(self, task: Dict[str, Any]) -> str:
        """HumanEval tasks already have a 'prompt' field."""
        return task.get("prompt", "")


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_humaneval_pipeline(
    data_path: str,
    output_dir: str,
    miu: float = 0.6,
    limit: Optional[int] = 5,
    distortion_model: str = "mistral-large-latest",
    generation_model: str = "mistral-large-latest",
    validation_model: str = "mistral-large-latest",
    project_path: Optional[str] = None,
    timeout: float = 12.0,
    k_values: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run the full HumanEval pipeline.
    
    Args:
        data_path: Path to HumanEval JSONL file
        output_dir: Directory to save results
        miu: Distortion level (0.0-1.0)
        limit: Optional limit on number of tasks
        distortion_model: Model for distortion
        generation_model: Model for code generation
        validation_model: Model for validation
        project_path: Path to project root
        timeout: Timeout for code execution
        k_values: List of k values for pass@k
        
    Returns:
        Summary dictionary with results
    """
    config = {
        "distortion_model": distortion_model,
        "generation_model": generation_model,
        "validation_model": validation_model,
        "project_path": project_path,
        "timeout": timeout,
        "k_values": k_values or [1],
        "miu": miu
    }
    
    benchmark = HumanEvalBenchmark(config)
    return benchmark.run_full_pipeline(data_path, output_dir, miu, limit)
