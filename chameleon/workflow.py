"""
Chameleon Workflow Orchestrator (HumanEval Compatible)

Complete end-to-end workflow for:
1. Distortion generation (via Mistral)
2. Target Model Generation (via any ModelBackend)
3. Functional Correctness Evaluation (via human-eval)
4. Analysis and reporting
"""

import os
import sys
import time
import yaml
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import tqdm

from chameleon.distortion.constants import DEFAULT_MIU_VALUES, DEFAULT_DISTORTIONS_PER_QUESTION
from chameleon.distortion.runner import DistortionRunner
from chameleon.evaluation.batch_processor import BatchProcessor, EvaluationConfig
from chameleon.models.registry import get_backend, BackendType
from chameleon.core.schemas import BackendType as EnumBackendType
from benchmarks import get_benchmark
from benchmarks.base.types import Task

@dataclass
class WorkflowConfig:
    """Configuration for the complete workflow."""
    project_name: str
    project_dir: Path
    miu_values: List[float]
    distortions_per_question: int
    distortion_model: str
    distortion_api_key: str
    target_model: str
    target_vendor: str
    target_api_key: Optional[str]
    skip_distortion: bool = False
    skip_generation: bool = False
    skip_evaluation: bool = False
    skip_analysis: bool = False
    benchmark_type: str = "human_eval"
    benchmark_config: Dict[str, Any] = None  # populated from config.yaml benchmark: section

    def __post_init__(self):
        if self.benchmark_config is None:
            self.benchmark_config = {}
    
    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "WorkflowConfig":
        """Load configuration from a project."""
        # Resolve projects_dir relative to repo root if it's a relative path
        if not os.path.isabs(projects_dir):
            base_dir = Path(__file__).parent.parent
            possible_dir = base_dir / projects_dir
            if possible_dir.exists():
                projects_dir = str(possible_dir)
        
        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "config.yaml"
        env_path = project_dir / ".env"
        
        if not config_path.exists():
            # Try to find it if current working directory is different
            if (Path("..") / projects_dir / project_name / "config.yaml").exists():
                 project_dir = Path("..") / projects_dir / project_name
                 config_path = project_dir / "config.yaml"
                 env_path = project_dir / ".env"
            else:
                 raise FileNotFoundError(f"Project config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
        # 1. Load global .env (from chameleon package dir)
        global_env = Path(__file__).parent / ".env"
        load_dotenv(global_env)
        
        # 2. Load project .env
        load_dotenv(env_path)
        
        # Distortion config
        dist_cfg = cfg.get("distortion", {})
        engine_cfg = dist_cfg.get("engine", {})
        dist_vendor = engine_cfg.get("vendor", "mistral")
        
        # Target model config
        target_cfg = cfg.get("target_model", {})
        target_vendor = target_cfg.get("vendor", "openai")
        
        # Benchmark config (new section; falls back to sensible defaults)
        benchmark_cfg = cfg.get("benchmark", {})
        benchmark_type = benchmark_cfg.get("type", cfg.get("benchmark_type", "human_eval"))

        return cls(
            project_name=project_name,
            project_dir=project_dir,
            miu_values=dist_cfg.get("miu_values", DEFAULT_MIU_VALUES),
            distortions_per_question=dist_cfg.get("distortions_per_question", DEFAULT_DISTORTIONS_PER_QUESTION),
            distortion_model=engine_cfg.get("model_name", "mistral-large-latest"),
            distortion_api_key=os.getenv("MISTRAL_API_KEY", ""),
            target_model=target_cfg.get("name", "gpt-4"),
            target_vendor=target_vendor,
            target_api_key=os.getenv("MISTRAL_API_KEY", ""), # Override: Use Mistral Key for target as well
            benchmark_type=benchmark_type,
            benchmark_config=benchmark_cfg,
        )


class ChameleonWorkflow:
    """
    Orchestrates the complete Chameleon evaluation workflow.
    """
    
    def __init__(self, config: WorkflowConfig, benchmark=None):
        self.config = config
        self.original_data_dir = config.project_dir / "original_data"
        self.distorted_data_dir = config.project_dir / "distorted_data"
        self.results_dir = config.project_dir / "results"
        self.analysis_dir = config.project_dir / "analysis"

        # Ensure directories exist
        for d in [self.distorted_data_dir, self.results_dir, self.analysis_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.distorted_jsonl = self.distorted_data_dir / "distortions_complete.jsonl"
        self.samples_jsonl = self.distorted_data_dir / "samples.jsonl"

        # Benchmark instance — drives prompts, evaluation, and metrics
        self.benchmark = benchmark or get_benchmark(
            config.benchmark_type, config.benchmark_config
        )
    
    def run(self) -> Dict[str, Any]:
        """Run the complete workflow."""
        start_time = time.time()
        results = {
            "project": self.config.project_name,
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }
        
        print("=" * 70)
        print("🦎 CHAMELEON WORKFLOW (HumanEval Edition)")
        print("=" * 70)
        
        try:
            # Stage 1: Distort
            if not self.config.skip_distortion and self.benchmark.supports_stage("distort"):
                print("\n" + "=" * 70)
                print("🔄 STAGE 1: DISTORTION GENERATION")
                print("=" * 70)
                results["stages"]["distort"] = self._stage_distort()
            else:
                print("\n⏭️ Skipping distortion stage")
                results["stages"]["distort"] = {"skipped": True}

            # Stage 2: Generate (Target Model Inference)
            if not self.config.skip_generation and self.benchmark.supports_stage("generate"):
                print("\n" + "=" * 70)
                print("⚡ STAGE 2: TARGET MODEL GENERATION")
                print("=" * 70)
                results["stages"]["generate"] = self._stage_generate()
            else:
                print("\n⏭️ Skipping generation stage")
                results["stages"]["generate"] = {"skipped": True}

            # Stage 3: Evaluate (Functional Correctness)
            if not self.config.skip_evaluation and self.benchmark.supports_stage("evaluate"):
                print("\n" + "=" * 70)
                print("🎯 STAGE 3: FUNCTIONAL EVALUATION")
                print("=" * 70)
                results["stages"]["evaluate"] = self._stage_evaluate()
            else:
                print("\n⏭️ Skipping evaluation stage")
                results["stages"]["evaluate"] = {"skipped": True}

            # Stage 4: Analyze
            if not self.config.skip_analysis and self.benchmark.supports_stage("analyze"):
                print("\n" + "=" * 70)
                print("📊 STAGE 4: ANALYSIS")
                print("=" * 70)
                results["stages"]["analyze"] = self._stage_analyze()
            else:
                print("\n⏭️ Skipping analysis stage")
                results["stages"]["analyze"] = {"skipped": True}
            
            results["status"] = "success"
            
        except Exception as e:
            print(f"\n❌ Workflow failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            import traceback
            traceback.print_exc()
        
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        results["end_time"] = datetime.now().isoformat()
        
        print("\n" + "=" * 70)
        print("🏁 WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"Status: {results['status']}")
        print(f"Time: {elapsed/60:.1f} minutes")
        print("=" * 70)
        
        return results
    
    def _stage_distort(self) -> Dict[str, Any]:
        """Run distortion runner."""
        if not self.config.distortion_api_key:
            raise ValueError(f"Distortion API key ({self.config.distortion_model}) not configured")
        
        # DistortionConfig expects bare config arguments, mostly from project
        # DistortionRunner.from_project handles loading config.yaml itself.
        # We can just reuse it to ensure consistent behavior with the CLI.
        runner = DistortionRunner.from_project(self.config.project_name, str(self.config.project_dir.parent))
        return runner.run(display_progress=True)

    def _stage_generate(self) -> Dict[str, Any]:
        """
        Generate answers from the target model for every distorted question.
        Saves result to samples.jsonl for evaluation.
        """
        if not self.distorted_jsonl.exists():
            raise FileNotFoundError(f"Distortions file not found: {self.distorted_jsonl}. Run distortion stage first.")
        
        if not self.config.target_api_key and self.config.target_vendor != "dummy":
            print(f"⚠️ Target API key for {self.config.target_vendor} not found. Ensure it is set in .env.")
        
        # Load distortions
        df = pd.read_json(self.distorted_jsonl, orient='records', lines=True)
        print(f"Loaded {len(df)} distortion rows.")
        
        # Prepare backend
        try:
            # Map mistral to openai backend for compatibility
            if self.config.target_vendor.lower() == 'mistral':
                backend_type = getattr(EnumBackendType, 'OPENAI')
            else:
                backend_type = getattr(EnumBackendType, self.config.target_vendor.upper())
        except AttributeError:
            # Fallback for known vendors if not in Enum
            if self.config.target_vendor.lower() == "openai":
                # Assuming Enum usually has OPENAI
                # If imports failed, we might need a direct string map or try/except
                pass
            raise ValueError(f"Unknown target vendor: {self.config.target_vendor}")
            
        print(f"Initializing backend: {self.config.target_vendor} ({self.config.target_model})...")
        backend = get_backend(
            backend_type=backend_type,
            model_name=self.config.target_model,
            api_key=self.config.target_api_key,
            vendor=self.config.target_vendor
        )
        
        if not backend.is_available():
             print(f"⚠️ Backend {self.config.target_vendor} reports not available (check API key).")
        
        # Identify what needs generation
        # We need to map distortions to samples.
        # Sample format: {"task_id": "...", "completion": "..."}
        # We'll use "task_id" to store the unique composite ID (distortion_id) so we can map back.
        # Note: HumanEval usually uses task_id like 'HumanEval/0'. We will use our 'distortion_id'.
        
        tasks_to_run = []
        
        for idx, row in df.iterrows():
            q_text = row.get('distorted_question')
            if pd.isna(q_text) or str(q_text).strip() == "":
                # Fallback to original if miu=0 or missing
                q_text = row.get('question_text', '')
            
            d_id = row.get('distortion_id')
            
            if not q_text or not d_id:
                continue

            # Build a Task and let the benchmark determine the final prompt.
            original_prompt = row.get('question_text', '')
            task = Task(
                task_id=str(d_id),
                data={self.benchmark.get_field_to_distort(): str(original_prompt)},
                distorted_text=str(q_text),
                miu=float(row.get('miu', 0.0)),
            )
            gen_system_prompt, final_prompt = self.benchmark.get_generation_prompt(task)

            tasks_to_run.append({
                "distortion_id": str(d_id),
                "prompt": final_prompt,
                "system_prompt": gen_system_prompt,
                "row_idx": idx,
            })
            
        print(f"Found {len(tasks_to_run)} tasks to generate.")
        
        # Check if samples already exist to resume
        completed_ids = set()
        if self.samples_jsonl.exists():
            with open(self.samples_jsonl, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        completed_ids.add(d['task_id'])
                    except: 
                        pass
            print(f"Resuming: {len(completed_ids)} already completed.")
            
        tasks_to_run = [t for t in tasks_to_run if t['distortion_id'] not in completed_ids]
        print(f"Remaining tasks: {len(tasks_to_run)}")
        
        if not tasks_to_run:
            return {"status": "complete", "generated": 0}
            
        # Run generation (Parallel)
        results = []
        max_workers = 10 if self.config.target_vendor == "openai" else 2
        
        print(f"Starting generation with {max_workers} threads...")
        
        with open(self.samples_jsonl, 'a', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(
                        backend.complete,
                        prompt=t["prompt"],
                        system_prompt=t["system_prompt"],
                        max_tokens=1024,
                        temperature=0.2
                    ): t
                    for t in tasks_to_run
                }
                
                for future in tqdm.tqdm(as_completed(future_to_task), total=len(tasks_to_run)):
                    task = future_to_task[future]
                    try:
                        completion = future.result()
                        
                        # Save result
                        result_entry = {
                            "task_id": task["distortion_id"],
                            "completion": completion,
                            # Store extra metadata if needed
                        }
                        f_out.write(json.dumps(result_entry) + "\n")
                        f_out.flush()
                        
                    except Exception as e:
                        print(f"Error on {task['distortion_id']}: {e}")
        
        return {"status": "complete", "generated": len(tasks_to_run)}

    def _stage_evaluate(self) -> Dict[str, Any]:
        """
        Run functional evaluation.

        Joins samples.jsonl (task_id=distortion_id, completion) with
        distortions_complete.jsonl (distortion_id -> question_id + original data)
        so the benchmark receives a fully-populated Task including test cases.
        """
        if not self.samples_jsonl.exists():
            return {"skipped": True, "reason": "no_samples_file"}

        # Load distortions metadata: distortion_id -> row
        dist_map: Dict[str, Any] = {}
        if self.distorted_jsonl.exists():
            dist_df = pd.read_json(self.distorted_jsonl, orient='records', lines=True)
            for _, row in dist_df.iterrows():
                dist_map[str(row.get('distortion_id', ''))] = row.to_dict()

        # Load original benchmark tasks (for test cases / canonical answers)
        data_path = self.config.benchmark_config.get("data_path", "")
        original_tasks: Dict[str, Task] = {}
        if data_path:
            resolved = Path(data_path)
            if not resolved.is_absolute():
                resolved = Path(__file__).parent.parent / data_path
            if resolved.exists():
                for t in self.benchmark.load_data(str(resolved)):
                    original_tasks[t.task_id] = t

        # Evaluate each sample
        result_path = Path(str(self.samples_jsonl) + "_results.jsonl")
        eval_results = []

        with open(self.samples_jsonl, 'r', encoding='utf-8') as f_in, \
             open(result_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                sample = json.loads(line)
                distortion_id = sample['task_id']
                completion = sample.get('completion', '')

                dist_row = dist_map.get(distortion_id, {})
                q_id = str(dist_row.get('question_id', distortion_id))
                miu = float(dist_row.get('miu', 0.0))

                orig_task = original_tasks.get(q_id)
                if orig_task is not None:
                    task = Task(
                        task_id=distortion_id,
                        data=dict(orig_task.data),
                        distorted_text=dist_row.get('distorted_question'),
                        miu=miu,
                    )
                else:
                    # Fallback: build minimal task from distortion row
                    task = Task(
                        task_id=distortion_id,
                        data={'task_id': q_id, 'prompt': dist_row.get('question_text', '')},
                        distorted_text=dist_row.get('distorted_question'),
                        miu=miu,
                    )

                eval_result = self.benchmark.evaluate(task, completion)
                # Attach miu for per-μ metric computation
                eval_result.metadata['miu'] = miu

                result_entry = {
                    'task_id': distortion_id,
                    'passed': eval_result.is_correct,
                    **eval_result.metadata,
                }
                f_out.write(json.dumps(result_entry) + '\n')
                eval_results.append(eval_result)

        passed = sum(1 for r in eval_results if r.is_correct)
        print(f"Evaluated {len(eval_results)} samples — {passed} passed ({passed/max(len(eval_results),1)*100:.1f}%)")
        return {'status': 'complete', 'evaluated': len(eval_results), 'passed': passed}

    def _stage_analyze(self) -> Dict[str, Any]:
        """Analyze results."""
        results_file = Path(str(self.samples_jsonl) + "_results.jsonl")
        if not results_file.exists():
            print("No evaluation results found.")
            return {"status": "no_results"}
            
        # Load results
        print(f"Loading results from {results_file.name}...")
        results_map = {} # task_id -> passed (bool)
        
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # "passed" is typically boolean in local eval results from batch_processor
                    # but batch_processor combines: sample["passed"] = res[1]["passed"]
                    results_map[d['task_id']] = d.get('passed', False)
                except:
                    pass
        
        # Join with metadata from distortions jsonl
        if not self.distorted_jsonl.exists():
            print("Distortions JSONL missing, cannot correlate metadata.")
            return {"status": "missing_metadata"}
            
        df = pd.read_json(self.distorted_jsonl, orient='records', lines=True)
        
        # Add is_correct column
        df['is_correct'] = df['distortion_id'].map(results_map)
        
        # Calculate stats
        valid_df = df[df['distortion_id'].isin(results_map.keys())]
        
        analysis = {}
        if len(valid_df) > 0:
            overall = valid_df['is_correct'].mean() * 100
            print(f"\nOverall Pass@1: {overall:.1f}% ({len(valid_df)} samples)")
            analysis['overall'] = overall
            
            # By Miu
            print("\nPass@1 by μ:")
            for miu in sorted(valid_df['miu'].unique()):
                subset = valid_df[valid_df['miu'] == miu]
                score = subset['is_correct'].mean() * 100
                print(f"  μ={miu}: {score:.1f}% (n={len(subset)})")
                analysis[f'miu_{miu}'] = score
                
        # Save analysis dataframe
        analysis_path = self.analysis_dir / "analysis_summary.csv"
        valid_df.to_csv(analysis_path, index=False)
        print(f"Saved detailed analysis to {analysis_path}")
        
        return analysis


def run_workflow(project_name: str, projects_dir: str = "Projects", **kwargs) -> Dict[str, Any]:
    """Convenience function."""
    config = WorkflowConfig.from_project(project_name, projects_dir)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
            
    workflow = ChameleonWorkflow(config)
    return workflow.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Chameleon HumanEval Workflow")
    parser.add_argument("project", help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    parser.add_argument("--skip-distortion", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    
    args = parser.parse_args()
    
    result = run_workflow(
        args.project,
        args.projects_dir,
        skip_distortion=args.skip_distortion,
        skip_generation=args.skip_generation,
        skip_evaluation=args.skip_evaluation,
        skip_analysis=args.skip_analysis,
    )
    
    sys.exit(0 if result.get("status") == "success" else 1)
