"""
Unified Distortion Runner

Strategies (in order of preference):
1. Mistral Batch API (fastest, cheapest for paid users)
2. Parallel API calls (fallback for free tier)

This replaces all previous distortion scripts and now supports
HumanEval-style coding prompts as a primary use case.
"""

import os
import sys
import time
import json
import threading
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from datetime import datetime

from chameleon.distortion.constants import (
    MIU_RULES,
    DEFAULT_MIU_VALUES,
    DEFAULT_DISTORTIONS_PER_QUESTION,
    calculate_temperature,
    calculate_max_tokens,
    get_batch_distortion_prompt,
    get_retry_distortion_prompt,
    BATCH_DEFAULTS,
    API_DEFAULTS,
)
from chameleon.distortion.validator import (
    validate_distortion,
    parse_llm_response,
    clean_distortion,
)


def validate_exact_count(variants: List[str], original: str, miu: float, required_n: int) -> List[str]:
    """
    Validate distortions and enforce EXACT count requirement.
    
    Designed to work for both multiple-choice questions and
    HumanEval-style coding prompts (where "original" is the prompt
    description + code stub).
    
    Args:
        variants: List of distortion variants from LLM
        original: Original prompt text
        miu: Distortion intensity
        required_n: EXACT number of unique distortions required
    
    Returns:
        List of exactly N valid unique distortions, or empty list if impossible
    """
    valid = []
    seen = set()
    original_lower = original.strip().lower()
    
    # For final slots, be more lenient - just check exact duplicates
    # Near-duplicate threshold scales with miu:
    # - Low miu (0.1-0.3): Only 1-2 word changes expected, so we just check for EXACT duplicates
    # - Medium miu (0.4-0.5): Allow variants that differ by at least 1 word
    # - High miu (0.6+): Require more variation (at least 2 words different)
    if miu <= 0.3:
        min_word_diff = 0  # Only check exact duplicates for low/medium miu
    elif miu <= 0.5:
        min_word_diff = 1
    else:
        min_word_diff = 2
    
    for v in variants:
        if not v or not isinstance(v, str):
            continue
            
        # Validate the distortion
        validation = validate_distortion(original, v, miu)
        if not validation.is_valid:
            continue
        
        cleaned = validation.distorted.strip()
        cleaned_lower = cleaned.lower()
        
        # Skip if identical to original (for miu > 0)
        if miu > 0 and cleaned_lower == original_lower:
            continue
        
        # Skip if EXACT duplicate
        if cleaned_lower in seen:
            continue
        
        # Check minimum difference from existing variants (anti-near-duplicate)
        # Only apply for higher miu values where we expect more variation
        is_near_dup = False
        if min_word_diff > 0:
            for existing in valid:
                # Count word differences
                v_words = set(cleaned_lower.split())
                e_words = set(existing.lower().split())
                diff = len(v_words.symmetric_difference(e_words))
                # Check if below threshold (only for prompts with enough words)
                if diff < min_word_diff and len(v_words) > 5:
                    is_near_dup = True
                    break
        
        if is_near_dup:
            continue
        
        valid.append(cleaned)
        seen.add(cleaned_lower)
        
        # Stop once we have exactly N
        if len(valid) >= required_n:
            break
    
    return valid


@dataclass
class DistortionConfig:
    """Configuration for distortion generation."""
    project_dir: Path
    miu_values: List[float]
    distortions_per_question: int
    model: str
    api_key: str
    questions_per_batch: int = BATCH_DEFAULTS["questions_per_batch"]
    workers_per_miu: int = BATCH_DEFAULTS["workers_per_miu"]
    max_retries_per_question: int = 100  # Keep retrying until ALL filled (practically unlimited)
    max_retries: int = BATCH_DEFAULTS["max_retries"]
    save_interval: int = BATCH_DEFAULTS["save_interval"]
    timeout: int = API_DEFAULTS["mistral"]["timeout"]

    # Optional benchmark-provided prompt factory.
    # Signature: (question_text: str, miu: float, n_distortions: int) -> DistortionPrompt
    # When set, the runner uses this to build distortion prompts instead of the
    # generic get_batch_distortion_prompt fallback.  When None, existing behaviour
    # is preserved exactly.
    prompt_factory: Optional[Callable] = None


@dataclass
class DistortionProgress:
    """Progress tracking for distortion generation."""
    miu: float
    total_batches: int
    completed_batches: int
    errors: int
    invalid: int


class DistortionRunner:
    """
    Unified distortion runner with batch API support.
    
    Supports:
    - Legacy multiple-choice question datasets (CSV)
    - HumanEval-style coding prompts stored as JSONL under original_data/
    
    Tries Mistral Batch API first (faster, cheaper),
    falls back to parallel API calls if batch fails.
    """
    
    def __init__(self, config: DistortionConfig):
        self.config = config
        self.api_url = API_DEFAULTS["mistral"]["base_url"]
        
        # Thread-safe state
        self._lock = threading.Lock()
        self._progress: Dict[float, DistortionProgress] = {}
        self._df: Optional[pd.DataFrame] = None
        self._stop_requested = False
        
        # Batch API directories
        self.batch_dir = config.project_dir / "distorted_data" / "batch_files"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_project(cls, project_name: str, projects_dir: str = "Projects") -> "DistortionRunner":
        """Create a DistortionRunner from a project configuration."""
        import yaml
        
        project_dir = Path(projects_dir) / project_name
        config_path = project_dir / "config.yaml"
        env_path = project_dir / ".env"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Project config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        
        load_dotenv(env_path)
        
        distortion_cfg = cfg.get("distortion", {})
        engine_cfg = distortion_cfg.get("engine", {})
        
        vendor = engine_cfg.get("vendor", "mistral")
        api_key = os.getenv(f"{vendor.upper()}_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not found for {vendor}. Set {vendor.upper()}_API_KEY in .env")
        
        return cls(DistortionConfig(
            project_dir=project_dir,
            miu_values=distortion_cfg.get("miu_values", DEFAULT_MIU_VALUES),
            distortions_per_question=distortion_cfg.get("distortions_per_question", DEFAULT_DISTORTIONS_PER_QUESTION),
            model=engine_cfg.get("model_name", API_DEFAULTS["mistral"]["default_model"]),
            api_key=api_key,
        ))
    
    def _load_data(self) -> pd.DataFrame:
        """
        Load the distortions CSV (uses distortions_complete.csv as the single working file).
        
        If distortions_complete.csv does not exist yet, it is created from:
        - Legacy CSV files under original_data/ (multiple-choice format)
        - OR HumanEval-style JSONL files under original_data/ (task_id, prompt, etc.)
        """
        data_dir = self.config.project_dir / "distorted_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        complete_path = data_dir / "distortions_complete.jsonl"
        preliminary_path = data_dir / "preliminary_distortions.jsonl"
        
        # If complete exists, use it
        if complete_path.exists():
            return pd.read_json(complete_path, orient='records', lines=True)
        
        # Otherwise, initialize from preliminary
        if preliminary_path.exists():
            df = pd.read_json(preliminary_path, orient='records', lines=True)
            df.to_json(complete_path, orient='records', lines=True)
            return df
        
        # Neither exists - create from original data
        print("   Creating preliminary distortions JSONL from original data...")
        df = self._create_preliminary_data()
        df.to_json(complete_path, orient='records', lines=True)
        print(f"   ✓ Created {len(df)} rows for distortion")
        return df
    
    def _create_preliminary_data(self) -> pd.DataFrame:
        """
        Create the preliminary distortions dataframe from original data files.
        
        For HumanEval-style datasets, each row corresponds to a single task,
        where 'question_text' contains the prompt (code stub + description/docstring)
        that will be paraphrased. For legacy multiple-choice datasets, the
        original semantics are preserved.
        
        Output columns (in order):
        - subject: Subject/category of the question/task
        - question_id: Unique identifier for the original question/task
        - question_text: Original prompt text (HumanEval or MCQ)
        - options_json: Answer options as JSON string {"A": "...", "B": "...", ...}
                        (usually empty '{}' for HumanEval)
        - distorted_question: Distorted version (filled later, or same as original for miu=0)
        - distortion_id: Unique distortion index (0 to N-1 per miu, combined with question_id for uniqueness)
        - miu: Distortion intensity level (0.0 to 1.0)
        - answer: Correct answer letter(s) (empty for HumanEval)
        - target_model_name: Name of the model that answered (filled during evaluation)
        - target_model_answer: Model's answer (filled during evaluation)
        - is_correct: Whether model's answer was correct (filled during evaluation)
        """
        
        original_dir = self.config.project_dir / "original_data"
        
        if not original_dir.exists():
            raise FileNotFoundError(f"No original_data folder found at {original_dir}")
        
        # Prefer legacy CSVs if present (for backward compatibility)
        csv_files = list(original_dir.glob("*.csv"))
        jsonl_files = list(original_dir.glob("*.jsonl"))
        
        if not csv_files and not jsonl_files:
            # Fallback: Check global data folder
            global_data_dir = self.config.project_dir.parent.parent / "data"
            if global_data_dir.exists():
                print(f"   ⚠️ No local data found. Checking global folder: {global_data_dir}")
                csv_files = list(global_data_dir.glob("*.csv"))
                jsonl_files = list(global_data_dir.glob("*.jsonl"))

        if not csv_files and not jsonl_files:
            raise FileNotFoundError(f"No CSV or JSONL files found in {original_dir} or {global_data_dir}")
        
        # --- LEGACY CSV PATH (multiple-choice style) ---
        if csv_files:
            all_data = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, encoding='utf-8')
                all_data.append(df)
            
            combined = pd.concat(all_data, ignore_index=True)
            print(f"   Loaded {len(combined)} questions from {len(csv_files)} CSV file(s)")
            
            # Standardize column names from various formats
            col_map = {
                # Question text variations
                'question': 'question_text',
                'Question': 'question_text',
                'Question_Text': 'question_text',
                'text': 'question_text',
                # Options variations
                'answer_options': 'options_json',
                'Options_JSON': 'options_json',
                'options': 'options_json',
                # Answer variations
                'correct_answer': 'answer',
                'Answer': 'answer',
                'correct': 'answer',
                # Subject variations
                'Subject': 'subject',
                'category': 'subject',
            }
            combined = combined.rename(columns={k: v for k, v in col_map.items() if k in combined.columns})
        
        # --- HUMANEVAL JSONL PATH ---
        else:
            # Build a DataFrame from all JSONL HumanEval tasks
            records: List[Dict[str, Any]] = []
            for jsonl_file in jsonl_files:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        records.append(data)
            
            if not records:
                raise ValueError(f"No valid JSON lines found in {len(jsonl_files)} JSONL file(s) under {original_dir}")
            
            combined = pd.DataFrame(records)
            print(f"   Loaded {len(combined)} HumanEval tasks from {len(jsonl_files)} JSONL file(s)")
            
            # For HumanEval, standardize into the expected columns:
            # - question_text: prompt (Python stub + description/docstring if present)
            # - question_id: task_id
            # - subject: 'humaneval' (or derived from task_id prefix)
            if 'task_id' not in combined.columns or 'prompt' not in combined.columns:
                raise ValueError(
                    "HumanEval JSONL format expected to contain 'task_id' and 'prompt' keys. "
                    "If your dataset differs, the loader logic must be adapted."
                )
            
            subject_values: List[str] = []
            question_ids: List[str] = []
            question_texts: List[str] = []
            
            for idx, row in combined.iterrows():
                task_id = row.get('task_id', f'HE_{idx:04d}')
                prompt = row.get('prompt', '')
                
                # Subject: either prefix of task_id or generic tag
                if isinstance(task_id, str) and '/' in task_id:
                    subj = task_id.split('/', 1)[0]
                else:
                    subj = 'humaneval'
                
                subject_values.append(subj)
                question_ids.append(task_id)
                question_texts.append(prompt)
            
            combined = combined.copy()
            combined['subject'] = subject_values
            combined['question_id'] = question_ids
            combined['question_text'] = question_texts
            # For HumanEval we don't have MCQ options or a single "answer" letter
            if 'answer' not in combined.columns:
                combined['answer'] = ''
        
        # At this point, `combined` has at least:
        # subject, question_id, question_text, answer (for both paths)
        
        # Ensure required columns exist with defaults if missing
        if 'subject' not in combined.columns:
            combined['subject'] = ''
        if 'question_id' not in combined.columns:
            combined['question_id'] = [f'Q_{i:04d}' for i in range(len(combined))]
        if 'question_text' not in combined.columns:
            # Fallback: use any "text" column that may exist
            if 'text' in combined.columns:
                combined['question_text'] = combined['text']
            else:
                combined['question_text'] = ''
        if 'answer' not in combined.columns:
            combined['answer'] = ''
        
        print(f"   Finalized {len(combined)} base prompts for distortion")
        
        # Create preliminary data structure with exact column order
        preliminary_data = []
        N = self.config.distortions_per_question
        miu_values = sorted(self.config.miu_values)
        
        for idx, row in combined.iterrows():
            q_id = row.get('question_id', f'Q_{idx:04d}')
            q_text = row.get('question_text', row.get('question', ''))
            answer = row.get('answer', '')
            subject = row.get('subject', '')
            
            for miu in miu_values:
                if miu == 0.0:
                    # miu=0 means no distortion - just 1 copy of original
                    preliminary_data.append({
                        'subject': subject,
                        'question_id': q_id,
                        'question_text': q_text,
                        'distorted_question': q_text,  # Same as original for miu=0
                        'distortion_id': f"{q_id}_d0_m0.0",  # Unique: question_id + distortion + miu
                        'miu': miu,
                        'answer': answer,
                        'target_model_name': '',
                        'target_model_answer': '',
                        'is_correct': '',
                    })
                else:
                    # miu>0 means create N distortion slots
                    for i in range(N):
                        preliminary_data.append({
                            'subject': subject,
                            'question_id': q_id,
                            'question_text': q_text,
                            'distorted_question': '',  # To be filled by distortion
                            'distortion_id': f"{q_id}_d{i}_m{miu}",  # Unique: question_id + distortion + miu
                            'miu': miu,
                            'answer': answer,
                            'target_model_name': '',
                            'target_model_answer': '',
                            'is_correct': '',
                        })
        
        # Create DataFrame with exact column order
        df = pd.DataFrame(preliminary_data)
        column_order = [
            'subject', 'question_id', 'question_text',
            'distorted_question', 'distortion_id', 'miu', 'answer',
            'target_model_name', 'target_model_answer', 'is_correct'
        ]
        df = df[column_order]
        
        print(f"   Created {len(df)} rows ({len(combined)} base prompts × {len(miu_values)} miu levels)")
        print(f"   Columns: {list(df.columns)}")
        
        return df
    
    def _save_data(self):
        """Save current state directly to distortions_complete.jsonl."""
        with self._lock:
            jsonl_path = self.config.project_dir / "distorted_data" / "distortions_complete.jsonl"
            self._df.to_json(jsonl_path, orient='records', lines=True)
    
    def _cleanup(self):
        """Clean up temporary files after completion."""
        import shutil
        
        data_dir = self.config.project_dir / "distorted_data"
        
        # Remove batch files directory
        if self.batch_dir.exists():
            shutil.rmtree(self.batch_dir, ignore_errors=True)
        
        # Remove batch API log
        log_path = data_dir / "batch_api.log"
        if log_path.exists():
            log_path.unlink()
        
        # Remove preliminary file (already incorporated into complete)
        preliminary_path = data_dir / "preliminary_distortions.jsonl"
        if preliminary_path.exists():
            preliminary_path.unlink()
    
    def _get_pending_by_miu(self) -> Dict[float, List[Dict]]:
        """Get pending questions grouped by miu (questions with ANY empty slots)."""
        miu_questions = {}
        N = self.config.distortions_per_question
        
        for miu in sorted([m for m in self._df['miu'].unique() if m > 0]):
            # Find questions that don't have EXACTLY N distortions filled
            miu_df = self._df[self._df['miu'] == miu]
            
            questions = []
            for q_id, group in miu_df.groupby('question_id'):
                filled_mask = group['distorted_question'].notna() & (group['distorted_question'] != '')
                filled_count = filled_mask.sum()
                
                # Need more distortions if we don't have exactly N
                if filled_count < N:
                    q_text = group['question_text'].iloc[0]
                    q_ans = group.get('answer', pd.Series([''])).iloc[0] if 'answer' in group.columns else ''
                    
                    # Get already-filled distortions to avoid generating duplicates
                    existing_distortions = group[filled_mask]['distorted_question'].tolist()
                    needed = N - filled_count
                    
                    questions.append({
                        "id": q_id,
                        "text": q_text,
                        "ans": q_ans,
                        "existing": existing_distortions,  # What we already have
                        "needed": needed  # How many more we need
                    })
            
            if questions:
                miu_questions[miu] = questions
        
        return miu_questions
    
    def _count_incomplete(self) -> int:
        """Count questions that don't have exactly N distortions."""
        N = self.config.distortions_per_question
        incomplete = 0
        
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            for q_id, group in miu_df.groupby('question_id'):
                filled = group['distorted_question'].notna() & (group['distorted_question'] != '')
                if filled.sum() < N:
                    incomplete += 1
        
        return incomplete
    
    def _find_duplicate_distortions(self) -> Dict[float, List[Dict]]:
        """
        Find questions where distortions are duplicates or too similar.
        Returns dict of miu -> list of questions with duplicates.
        """
        duplicates = {}
        
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            questions_with_dups = []
            
            for q_id, group in miu_df.groupby('question_id'):
                distortions = group['distorted_question'].tolist()
                
                # Check for exact duplicates
                seen = {}  # text_lower -> list of indices
                for i, d in enumerate(distortions):
                    if pd.notna(d) and d != '':
                        d_lower = str(d).strip().lower()
                        if d_lower not in seen:
                            seen[d_lower] = []
                        seen[d_lower].append(i)
                
                # Find duplicates (texts that appear more than once)
                dup_indices = []
                for text, indices in seen.items():
                    if len(indices) > 1:
                        # Keep the first, mark others as duplicates
                        dup_indices.extend(indices[1:])
                
                if dup_indices:
                    questions_with_dups.append({
                        "id": q_id,
                        "text": group['question_text'].iloc[0],
                        "duplicate_indices": dup_indices,
                        "existing": [d for d in distortions if pd.notna(d) and d != ''],
                        "needed": len(dup_indices)
                    })
            
            if questions_with_dups:
                duplicates[miu] = questions_with_dups
        
        return duplicates
    
    def _clear_duplicate_slots(self, duplicates: Dict[float, List[Dict]]) -> int:
        """
        Clear the duplicate slots so they can be regenerated.
        Also resets validation flag for the entire question at that miu.
        Returns total number of slots cleared.
        """
        cleared = 0
        
        for miu, questions in duplicates.items():
            for q in questions:
                q_id = q["id"]
                dup_indices = q["duplicate_indices"]
                
                # Get the row indices in the dataframe
                q_rows = self._df[(self._df['question_id'] == q_id) & (self._df['miu'] == miu)]
                row_indices = q_rows.index.tolist()
                
                # Reset validation flag for ALL distortions of this question at this miu
                # (since we need to re-validate after regeneration)
                if 'llm_validated' in self._df.columns:
                    mask = (self._df['question_id'] == q_id) & (self._df['miu'] == miu)
                    self._df.loc[mask, 'llm_validated'] = False
                
                # Clear the duplicate slots
                for dup_idx in dup_indices:
                    if dup_idx < len(row_indices):
                        df_idx = row_indices[dup_idx]
                        self._df.at[df_idx, 'distorted_question'] = ''
                        cleared += 1
        
        return cleared
    
    def _run_post_validation(self):
        """
        Post-distortion validation:
        0. Static validator (template + quality rules via validate_distortion)
        1. Check for encoding errors (garbage characters)
        2. Check for exact duplicates
        3. Check for near-duplicates (>95% word overlap between distortions)
        4. LLM Judge (Mistral) validates according to miu rules
        
        Works for both HumanEval prompts and legacy MCQ questions.
        """
        print(f"\n{'='*60}")
        print("🔍 POST-DISTORTION VALIDATION")
        print(f"{'='*60}")
        
        validation_round = 0
        last_unvalidated = None
        stuck_counter = 0
        max_stuck_rounds = 5  # Stop if no progress for 5 rounds
        
        while True:
            validation_round += 1

            # Step 0: Static validator (syntax/template/quality)
            print(f"\n   📋 Step 0: Static validator checks (template + quality rules)...")
            static_invalid = self._find_validator_invalid_distortions()
            if static_invalid:
                total_static = sum(len(qs) for qs in static_invalid.values())
                print(f"      ⚠️ Found {total_static} questions with invalid distortions")
                cleared_static = self._clear_duplicate_slots(static_invalid)
                print(f"      🗑️ Cleared {cleared_static} invalid slots (will be regenerated)")
            else:
                print(f"      ✅ All current distortions pass static validator")
            
            # Step 1: Check for encoding errors (garbage characters)
            print(f"\n   📋 Step 1: Checking for encoding errors...")
            encoding_errors = self._find_encoding_errors()
            
            if encoding_errors:
                total_enc = sum(len(qs) for qs in encoding_errors.values())
                print(f"      ⚠️ Found {total_enc} questions with encoding errors")
                cleared = self._clear_duplicate_slots(encoding_errors)
                print(f"      🗑️ Cleared {cleared} encoding error slots")
            else:
                print(f"      ✅ No encoding errors found")
            
            # Step 2: Check for exact duplicates
            print(f"\n   📋 Step 2: Checking for exact duplicates...")
            duplicates = self._find_duplicate_distortions()
            
            if duplicates:
                total_dups = sum(len(qs) for qs in duplicates.values())
                print(f"      ⚠️ Found {total_dups} questions with duplicate distortions")
                cleared = self._clear_duplicate_slots(duplicates)
                print(f"      🗑️ Cleared {cleared} duplicate slots")
            else:
                print(f"      ✅ No exact duplicates found")
            
            # Step 3: Check for near-duplicates (>95% similar)
            print(f"\n   📋 Step 3: Checking for near-duplicates (>95% similar)...")
            near_dups = self._find_near_duplicate_distortions()
            
            if near_dups:
                total_near = sum(len(qs) for qs in near_dups.values())
                print(f"      ⚠️ Found {total_near} questions with near-duplicate distortions")
                cleared = self._clear_duplicate_slots(near_dups)
                print(f"      🗑️ Cleared {cleared} near-duplicate slots")
            else:
                print(f"      ✅ No near-duplicates found")
            
            # Step 4: LLM Judge validation with Mistral (ALL questions)
            print(f"\n   📋 Step 4: Mistral LLM Judge (full validation - ALL questions)...")
            bad_from_llm = self._mistral_judge_validation(sample_per_miu=0)  # 0 = validate ALL
            
            if bad_from_llm:
                total_bad = sum(len(qs) for qs in bad_from_llm.values())
                print(f"      ⚠️ Mistral flagged {total_bad} distortions as invalid")
                cleared = self._clear_llm_flagged(bad_from_llm)
                print(f"      🗑️ Cleared {cleared} LLM-flagged slots")
            else:
                print(f"      ✅ All distortions passed Mistral validation")
            
            # Check if we need to regenerate
            miu_questions = self._get_pending_by_miu()
            
            if not miu_questions:
                # Check if all questions are validated
                unvalidated = self._count_unvalidated()
                if unvalidated == 0:
                    print(f"\n   ✅ All {self._count_total_questions()} questions validated!")
                    break
                else:
                    print(f"\n   ⚠️ {unvalidated} questions still unvalidated but no pending regeneration")
                    break
            
            # Count unvalidated for progress tracking
            current_unvalidated = self._count_unvalidated()
            
            # Check if we're making progress
            if last_unvalidated is not None:
                if current_unvalidated >= last_unvalidated:
                    stuck_counter += 1
                    print(f"\n   ⚠️ No progress this round. Stuck counter: {stuck_counter}/{max_stuck_rounds}")
                else:
                    stuck_counter = 0
                    progress = last_unvalidated - current_unvalidated
                    print(f"\n   📈 Progress: {progress} more questions validated ({current_unvalidated} remaining)")
            
            last_unvalidated = current_unvalidated
            
            # Stop if we're stuck
            if stuck_counter >= max_stuck_rounds:
                print(f"\n   ⚠️ No progress for {max_stuck_rounds} rounds. {current_unvalidated} questions still unvalidated.")
                print(f"      Force-accepting remaining questions (they passed duplicate/encoding checks)...")
                
                # Force-validate remaining questions since they passed basic checks
                self._force_validate_remaining()
                print(f"   ✅ Force-validated remaining questions")
                break
            
            # Regenerate bad ones
            total_pending = sum(len(qs) for qs in miu_questions.values())
            print(f"\n   🔄 Regenerating {total_pending} flagged distortions (Round {validation_round})...")
            
            batch_success = self._try_batch_api(miu_questions)
            if not batch_success:
                miu_questions = self._get_pending_by_miu()
                if miu_questions:
                    self._run_parallel_api(miu_questions)
            self._save_data()
    
    def _count_unvalidated(self) -> int:
        """Count questions that are not yet LLM validated."""
        if 'llm_validated' not in self._df.columns:
            return len(self._df[self._df['miu'] > 0]['question_id'].unique())
        
        count = 0
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            for q_id, group in miu_df.groupby('question_id'):
                if not group['llm_validated'].all():
                    count += 1
        return count
    
    def _count_total_questions(self) -> int:
        """Count total questions (excluding miu=0)."""
        return len(self._df[self._df['miu'] > 0]['question_id'].unique()) * len([m for m in self._df['miu'].unique() if m > 0])
    
    def _force_validate_remaining(self):
        """
        Force-validate remaining questions that are stuck.
        These questions have passed duplicate/encoding checks but keep failing LLM validation.
        Since they're not duplicates and don't have encoding issues, we accept them.
        """
        if 'llm_validated' not in self._df.columns:
            return
        
        # Find unvalidated rows with non-empty distortions
        mask = (
            (self._df['miu'] > 0) & 
            (self._df['llm_validated'] == False) & 
            (self._df['distorted_question'].notna()) & 
            (self._df['distorted_question'] != '')
        )
        
        self._df.loc[mask, 'llm_validated'] = True
        
        count = mask.sum()
        if count > 0:
            print(f"      Force-validated {count} distortion slots")
    
    def _find_encoding_errors(self) -> Dict[float, List[Dict]]:
        """
        Find distortions with encoding errors (garbage characters like â€™ â€" etc.)
        These are UTF-8 encoding issues that need to be regenerated.
        """
        # Common encoding error patterns
        encoding_garbage = [
            'â€™', 'â€"', 'â€œ', 'â€', 'Ã©', 'Ã¨', 'Ã ', 'Ã¢', 'Ã®', 'Ã´', 'Ã»',
            'Â°', 'Â©', 'Â®', 'Â´', 'â€¦', 'â€˜', 'â€¢', 'Ã¼', 'Ã¶', 'Ã¤',
            '�', '\ufffd', 'Â ', 'â€'
        ]
        
        encoding_errors = {}
        
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            questions_with_errors = []
            
            for q_id, group in miu_df.groupby('question_id'):
                distortions = group['distorted_question'].tolist()
                error_indices = []
                
                for i, d in enumerate(distortions):
                    if pd.notna(d) and d:
                        d_str = str(d)
                        # Check for encoding garbage
                        for pattern in encoding_garbage:
                            if pattern in d_str:
                                error_indices.append(i)
                                break
                
                if error_indices:
                    questions_with_errors.append({
                        "id": q_id,
                        "text": group['question_text'].iloc[0],
                        "duplicate_indices": error_indices,
                        "existing": [d for i, d in enumerate(distortions) if pd.notna(d) and d and i not in error_indices],
                        "needed": len(error_indices)
                    })
            
            if questions_with_errors:
                encoding_errors[miu] = questions_with_errors
        
        return encoding_errors
    
    def _find_near_duplicate_distortions(self) -> Dict[float, List[Dict]]:
        """
        Find questions where distortions are near-duplicates (>95% word overlap).
        This is different from exact duplicates - catches cases where only 1-2 words differ.
        
        Returns dict of miu -> list of questions with near-duplicates.
        """
        near_dups = {}
        
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            questions_with_near_dups = []
            
            for q_id, group in miu_df.groupby('question_id'):
                distortions = group['distorted_question'].tolist()
                
                # Build word sets for each distortion
                word_sets = []
                for d in distortions:
                    if pd.notna(d) and d != '':
                        word_sets.append((str(d).strip().lower(), set(str(d).strip().lower().split())))
                    else:
                        word_sets.append(('', set()))
                
                # Find near-duplicates (>95% word overlap)
                dup_indices = set()
                for i in range(len(word_sets)):
                    if i in dup_indices:
                        continue
                    text_i, words_i = word_sets[i]
                    if not words_i:
                        continue
                    
                    for j in range(i + 1, len(word_sets)):
                        if j in dup_indices:
                            continue
                        text_j, words_j = word_sets[j]
                        if not words_j:
                            continue
                        
                        # Calculate overlap
                        overlap = len(words_i & words_j) / max(len(words_i), len(words_j))
                        if overlap > 0.95:  # >95% similar = near-duplicate
                            # Mark the second one as duplicate (keep first)
                            dup_indices.add(j)
                
                if dup_indices:
                    questions_with_near_dups.append({
                        "id": q_id,
                        "text": group['question_text'].iloc[0],
                        "duplicate_indices": list(dup_indices),
                        "existing": [d for i, d in enumerate(distortions) if pd.notna(d) and d != '' and i not in dup_indices],
                        "needed": len(dup_indices)
                    })
            
            if questions_with_near_dups:
                near_dups[miu] = questions_with_near_dups
        
        return near_dups

    def _find_validator_invalid_distortions(self) -> Dict[float, List[Dict]]:
        """
        Use validate_distortion() to find slots whose text violates our
        structural/quality rules (including HumanEval template preservation).

        Returns dict of miu -> list of questions, in the same shape used by
        _find_duplicate_distortions / _find_near_duplicate_distortions, so that
        _clear_duplicate_slots can clear and reset them for regeneration.
        """
        invalid: Dict[float, List[Dict]] = {}

        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            questions_with_invalid: List[Dict] = []

            for q_id, group in miu_df.groupby('question_id'):
                original = group['question_text'].iloc[0]
                distortions = group['distorted_question'].tolist()

                bad_indices: List[int] = []
                for i, d in enumerate(distortions):
                    if pd.isna(d) or not d:
                        continue
                    res = validate_distortion(str(original), str(d), miu)
                    if not res.is_valid:
                        bad_indices.append(i)

                if bad_indices:
                    questions_with_invalid.append({
                        "id": q_id,
                        "text": original,
                        "duplicate_indices": bad_indices,
                        "existing": [
                            d for i, d in enumerate(distortions)
                            if pd.notna(d) and d and i not in bad_indices
                        ],
                        "needed": len(bad_indices),
                    })

            if questions_with_invalid:
                invalid[miu] = questions_with_invalid

        return invalid
    
    def _mistral_judge_validation(self, sample_per_miu: int = 0) -> Dict[float, List[Dict]]:
        """
        Use Mistral Batch API as LLM judge to validate distortions according to miu rules.
        Only validates questions that haven't been validated yet (or were flagged as bad).
        
        For HumanEval prompts, the judge is expected to:
        - Ensure the task description stays semantically equivalent.
        - Ensure that Python code snippets, function signatures, and tests
          are preserved exactly (no renaming, no literal changes).
        
        Args:
            sample_per_miu: If 0, validates ALL unvalidated questions. If >0, samples that many per miu.
        
        Returns dict of miu -> list of {q_id, bad_indices} for distortions that failed.
        """
        try:
            from mistralai import Mistral
            client = Mistral(api_key=self.config.api_key)
        except Exception as e:
            print(f"      ⚠️ Could not initialize Mistral client: {e}")
            return {}
        
        # Ensure we have a validated column
        if 'llm_validated' not in self._df.columns:
            self._df['llm_validated'] = False
        
        # Collect validation requests ONLY for unvalidated questions
        validation_requests = []
        request_map = {}  # custom_id -> (miu, q_id, n_distortions)
        
        for miu in [m for m in self._df['miu'].unique() if m > 0]:
            miu_df = self._df[self._df['miu'] == miu]
            
            # Get questions that are NOT yet validated
            unvalidated_q_ids = []
            for q_id, group in miu_df.groupby('question_id'):
                # A question needs validation if ANY of its distortions is not validated
                if not group['llm_validated'].all():
                    unvalidated_q_ids.append(q_id)
            
            if not unvalidated_q_ids:
                continue  # All questions at this miu are validated
            
            # If sample_per_miu is 0, validate ALL unvalidated questions
            if sample_per_miu > 0:
                sample_size = min(sample_per_miu, len(unvalidated_q_ids))
                sample_ids = np.random.choice(unvalidated_q_ids, sample_size, replace=False)
            else:
                sample_ids = unvalidated_q_ids  # ALL unvalidated questions
            
            rule = MIU_RULES.get(miu, MIU_RULES[0.5])
            
            for q_id in sample_ids:
                group = miu_df[miu_df['question_id'] == q_id]
                original = group['question_text'].iloc[0]
                distortions = group['distorted_question'].tolist()
                
                # Build validation prompt
                prompt = self._build_judge_prompt(original, distortions, miu, rule)
                
                custom_id = f"validate__{q_id}__miu{miu}"
                request_map[custom_id] = (miu, q_id, len(distortions))
                
                validation_requests.append({
                    "custom_id": custom_id,
                    "body": {
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 200
                    }
                })
        
        if not validation_requests:
            # Count how many are already validated
            total_validated = 0
            for miu in [m for m in self._df['miu'].unique() if m > 0]:
                miu_df = self._df[self._df['miu'] == miu]
                for q_id, group in miu_df.groupby('question_id'):
                    if group['llm_validated'].all():
                        total_validated += 1
            print(f"      ✅ All {total_validated} questions already validated - nothing to do")
            return {}
        
        # Count stats
        total_questions = sum(len(self._df[self._df['miu'] == m]['question_id'].unique()) for m in self._df['miu'].unique() if m > 0)
        already_validated = total_questions - len(validation_requests)
        
        print(f"      📊 {already_validated} questions already validated, {len(validation_requests)} need validation")
        print(f"      📤 Sending {len(validation_requests)} validation requests to Mistral Batch API...")
        
        # Create JSONL file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.batch_dir / f"validation_batch_{timestamp}.jsonl"
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for req in validation_requests:
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        
        try:
            # Upload and submit batch
            from mistralai import Mistral  # re-import to satisfy type checkers in some environments
            client = Mistral(api_key=self.config.api_key)
            with open(jsonl_path, "rb") as f:
                uploaded = client.files.upload(
                    file={"file_name": jsonl_path.name, "content": f.read()},
                    purpose="batch"
                )
            
            job = client.batch.jobs.create(
                input_files=[uploaded.id],
                model=self.config.model,
                endpoint="/v1/chat/completions",
                metadata={"type": "validation"}
            )
            
            print(f"      ✅ Validation batch job created: {job.id}")
            
            # Monitor job
            while True:
                status = client.batch.jobs.get(job_id=job.id)
                if status.status in ["SUCCESS", "FAILED", "CANCELLED"]:
                    break
                print(f"      ⏳ Validation batch: {status.status}...", end='\r')
                time.sleep(5)
            
            print(f"      ")  # Clear line
            
            if status.status != "SUCCESS":
                print(f"      ⚠️ Validation batch failed: {status.status}")
                return {}
            
            # Check if output file exists
            if not status.output_file:
                print(f"      ⚠️ No output file from validation batch")
                return {}
            
            # Small delay to ensure output file is ready
            time.sleep(2)
            
            # Download and parse results with retry
            self.batch_dir.mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            result_path = self.batch_dir / f"validation_results_{timestamp}.jsonl"
            download_success = False
            
            for attempt in range(3):
                try:
                    result_stream = client.files.download(file_id=status.output_file)
                    
                    with open(result_path, 'wb') as f:
                        f.write(result_stream.read())
                    
                    print(f"      📥 Downloaded results to {result_path.name}")
                    download_success = True
                    break
                except Exception as download_err:
                    if attempt < 2:
                        print(f"      ⚠️ Download attempt {attempt+1} failed, retrying...")
                        time.sleep(3)
                    else:
                        print(f"      ⚠️ Failed to download results after 3 attempts: {download_err}")
            
            if not download_success:
                return {}
            
            # Parse results
            bad_distortions = {}
            validated_count = 0
            
            with open(result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        custom_id = data.get('custom_id', '')
                        
                        if custom_id not in request_map:
                            continue
                        
                        miu, q_id, n_distortions = request_map[custom_id]
                        
                        content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                        bad_indices = self._parse_judge_response(content, n_distortions)
                        
                        if bad_indices:
                            # Question has bad distortions - don't mark as validated
                            if miu not in bad_distortions:
                                bad_distortions[miu] = []
                            bad_distortions[miu].append({
                                "id": q_id,
                                "bad_indices": bad_indices
                            })
                        else:
                            # Question PASSED validation - mark ALL its distortions as validated
                            mask = (self._df['question_id'] == q_id) & (self._df['miu'] == miu)
                            self._df.loc[mask, 'llm_validated'] = True
                            validated_count += 1
                    except Exception:
                        continue
            
            # Report findings
            print(f"      ✅ {validated_count} questions passed validation (marked as validated)")
            for miu in sorted(bad_distortions.keys()):
                print(f"      μ={miu}: {len(bad_distortions[miu])} questions have invalid distortions")
            
            return bad_distortions
            
        except Exception as e:
            print(f"      ⚠️ Batch validation error: {e}")
            return {}
    
    def _build_judge_prompt(self, original: str, distortions: List[str], miu: float, rule: str) -> str:
        """
        Build RLHF-style quality control prompt for Mistral judge.
        For each distortion, compares against the other N-1 distortions.
        
        The judge must:
        - Ensure meaning preservation of the HumanEval prompt.
        - Ensure all Python code (function signatures, stubs, tests) remains
          syntactically valid and unchanged in behavior.
        """
        N = len([d for d in distortions if pd.notna(d) and d])
        
        # Build numbered distortion list
        distortion_entries = []
        for i, d in enumerate(distortions):
            if pd.notna(d) and d:
                distortion_entries.append(f"[D{i+1}]: {d}")
        
        distortion_block = "\n".join(distortion_entries)

        lines = [
            "DISTORTION QUALITY CONTROL",
            "==========================",
            "",
            "ORIGINAL HUMANEVAL PROMPT:",
            original,
            "",
            f"MIU LEVEL: μ = {miu}",
            f"MIU RULE: {rule}",
            "",
            "IMPORTANT:",
            "- The underlying HumanEval task (input/output behavior) must remain identical.",
            "- Python code (function signatures, argument names, literals, tests) must NOT be changed.",
            "- Only natural-language description, comments, or docstrings may be rephrased.",
            "",
            f"ALL {N} DISTORTIONS (for this prompt at μ={miu}):",
            distortion_block,
            "",
            "TASK: Evaluate EACH distortion [D1] to [D{N}] by comparing it against:",
            "1. The ORIGINAL prompt",
            f"2. The OTHER {N-1} distortions in this set",
            "",
            "VALIDATION CHECKLIST FOR EACH DISTORTION:",
            "",
            "□ ENCODING/CHARACTER ERRORS?",
            "  ⚠️ CRITICAL: Look for garbage like: â€™ â€\" â€œ Ã© Ã¨ Â° etc.",
            "  - These are encoding errors and MUST be flagged as BAD",
            "  - Normal apostrophes (') and quotes (\") are OK",
            "",
            "□ MEANING PRESERVED?",
            "  - Does this distortion still describe the same task as the original?",
            "  - Would the same tests and reference solution still be valid here?",
            "",
            "□ PYTHON CODE PRESERVED?",
            "  - Function name, parameters, and any provided code snippets are unchanged",
            "  - No new code is added that changes behavior",
            "  - No literal values or tests are modified",
            "",
            f"□ FOLLOWS μ={miu} RULE?",
            f"  - Rule: \"{rule}\"",
            "  - Is the distortion intensity appropriate for this miu level?",
            "",
            f"□ DIFFERENT FROM OTHER {N-1} DISTORTIONS?",
            "  - Compare this distortion to all others in the list",
            "  - Is it meaningfully different? (not just 1–2 words changed)",
            "  - Would someone consider these truly distinct variations?",
            "",
            "□ QUALITY OK?",
            "  - No truncation, nonsense, or broken text?",
            "  - Grammatically correct and readable?",
            "",
            "RESPOND WITH EXACTLY ONE OF:",
            "• \"ALL_VALID\" if every distortion passes ALL checks",
            "• \"BAD: X, Y\" listing ONLY the numbers that FAIL (e.g., \"BAD: 2, 5, 8\")",
            "",
            "Verdict:",
        ]

        return "\n".join(lines)
    
    def _parse_judge_response(self, response: str, n_distortions: int) -> List[int]:
        """Parse the judge response to extract bad distortion indices."""
        response = response.strip().upper()
        
        if "ALL_VALID" in response or "ALL VALID" in response:
            return []
        
        # Extract numbers from "BAD: 3, 7, 9" format
        bad_indices = []
        import re
        numbers = re.findall(r'\d+', response)
        
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= n_distortions:
                    bad_indices.append(num - 1)  # Convert to 0-indexed
            except Exception:
                continue
        
        return bad_indices
    
    def _clear_llm_flagged(self, bad_distortions: Dict[float, List[Dict]]) -> int:
        """Clear distortion slots flagged by LLM judge and reset validation flag."""
        cleared = 0
        
        for miu, questions in bad_distortions.items():
            for q in questions:
                q_id = q["id"]
                bad_indices = q["bad_indices"]
                
                q_rows = self._df[(self._df['question_id'] == q_id) & (self._df['miu'] == miu)]
                row_indices = q_rows.index.tolist()
                
                for bad_idx in bad_indices:
                    if bad_idx < len(row_indices):
                        df_idx = row_indices[bad_idx]
                        self._df.at[df_idx, 'distorted_question'] = ''
                        # Reset validation flag so it gets re-validated after regeneration
                        if 'llm_validated' in self._df.columns:
                            self._df.at[df_idx, 'llm_validated'] = False
                        cleared += 1
        
        return cleared
    
    # ==================== BATCH API ====================
    
    def _try_batch_api(self, miu_questions: Dict[float, List[Dict]]) -> bool:
        """
        Try to use Mistral Batch API.
        Returns True if successful, False if should fallback.
        """
        log_path = self.config.project_dir / "distorted_data" / "batch_api.log"
        
        def log(msg: str):
            print(msg)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - {msg}\n")
        
        log("\n🚀 Attempting Mistral Batch API (faster, cheaper)...")
        
        try:
            from mistralai import Mistral
            client = Mistral(api_key=self.config.api_key)
            log("   ✓ mistralai SDK loaded")
        except ImportError:
            log("   ⚠️ mistralai package not installed. Falling back...")
            time.sleep(3)  # Let user see the message
            return False
        
        # FIRST: Check for existing completed batch jobs and apply their results
        try:
            existing_jobs = client.batch.jobs.list()
            project_name = self.config.project_dir.name
            
            for job in existing_jobs.data:
                if job.status == 'SUCCESS' and job.output_file:
                    job_miu = job.metadata.get('miu')
                    job_project = job.metadata.get('project', '')
                    
                    if job_miu and (job_project == project_name or project_name in str(job.input_files)):
                        log(f"   📥 Found existing batch for μ={job_miu} ({job.succeeded_requests} requests)")
                        self._apply_batch_results(client, {
                            "miu": float(job_miu),
                            "output_file": job.output_file
                        })
            
            # Reload pending after applying existing results
            miu_questions = self._get_pending_by_miu()
            
            if not miu_questions:
                log("   ✅ All distortions applied from existing batch jobs!")
                return True
                
            log(f"   Still pending: {sum(len(q) for q in miu_questions.values())} questions")
            
        except Exception as e:
            log(f"   ⚠️ Could not check existing jobs: {e}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        N = self.config.distortions_per_question
        
        # Create JSONL files per miu for remaining questions
        batch_jobs = []
        
        for miu, questions in miu_questions.items():
            log(f"\n   Creating batch for μ={miu} ({len(questions)} questions)...")
            
            jsonl_path = self.batch_dir / f"batch_miu{miu}_{timestamp}.jsonl"
            
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for q in questions:
                    # Check if this question has existing distortions (retry case)
                    existing = q.get('existing', [])
                    needed = q.get('needed', N)
                    
                    if existing and needed < N:
                        # Use retry prompt that includes existing distortions to avoid
                        prompt = get_retry_distortion_prompt(q, miu, needed)
                        # Fewer tokens needed since we're generating fewer distortions
                        max_tokens = calculate_max_tokens([q], needed, miu)
                    else:
                        # Fresh question - use standard prompt
                        prompt = get_batch_distortion_prompt([q], miu, N)
                        max_tokens = calculate_max_tokens([q], N, miu)
                    
                    request = {
                        "custom_id": f"{q['id']}__miu{miu}__needed{needed}",  # Include needed count
                        "body": {
                            "model": self.config.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": calculate_temperature(miu),
                            "max_tokens": max_tokens
                        }
                    }
                    f.write(json.dumps(request, ensure_ascii=False) + '\n')
            
            # Upload and submit
            try:
                log(f"   Uploading {jsonl_path.name}...")
                with open(jsonl_path, "rb") as f:
                    # mistralai SDK requires File dict format
                    uploaded = client.files.upload(
                        file={"file_name": jsonl_path.name, "content": f.read()},
                        purpose="batch"
                    )
                
                log(f"   Submitting batch job...")
                job = client.batch.jobs.create(
                    input_files=[uploaded.id],
                    model=self.config.model,
                    endpoint="/v1/chat/completions",
                    metadata={"miu": str(miu)}
                )
                
                batch_jobs.append({
                    "miu": miu,
                    "job_id": job.id,
                    "questions": len(questions),
                    "status": job.status
                })
                log(f"   ✅ Job created: {job.id}")
                
            except Exception as e:
                error_str = str(e)
                log(f"\n   ❌ Batch API Error: {error_str}")
                
                if "free" in error_str.lower() or "tier" in error_str.lower() or "limit" in error_str.lower() or "402" in str(e):
                    log("   ⚠️ Batch API not available (free tier limit)")
                    log("   Falling back to parallel API calls...")
                else:
                    log(f"   ⚠️ Batch API failed, falling back to parallel API...")
                
                time.sleep(3)  # Let user see the message
                return False
        
        if not batch_jobs:
            log("   ⚠️ No batch jobs created, falling back...")
            time.sleep(2)
            return False
        
        # Monitor batch jobs
        log(f"\n📊 Monitoring {len(batch_jobs)} batch jobs...")
        
        while True:
            all_done = True
            
            for job_info in batch_jobs:
                job = client.batch.jobs.get(job_id=job_info["job_id"])
                job_info["status"] = job.status
                job_info["completed"] = getattr(job, "completed_requests", 0)
                job_info["failed"] = getattr(job, "failed_requests", 0)
                job_info["output_file"] = getattr(job, "output_file", None)
                
                if job.status not in ["SUCCESS", "FAILED", "CANCELLED", "TIMEOUT_EXCEEDED"]:
                    all_done = False
            
            # Display status
            print("\r   ", end="")
            for job_info in batch_jobs:
                miu = job_info["miu"]
                status = job_info["status"][:4]
                print(f"μ={miu}:{status} ", end="")
            print("", flush=True)
            
            if all_done:
                break
            
            time.sleep(10)
        
        # Download and apply results
        log("\n\n📥 Downloading results...")
        
        for job_info in batch_jobs:
            if job_info["status"] == "SUCCESS" and job_info["output_file"]:
                self._apply_batch_results(client, job_info)
        
        self._save_data()
        return True
    
    def _apply_batch_results(self, client, job_info: Dict):
        """Apply results from a batch job to the dataframe."""
        miu = job_info["miu"]
        N = self.config.distortions_per_question
        
        try:
            result_stream = client.files.download(file_id=job_info["output_file"])
            result_path = self.batch_dir / f"results_miu{miu}.jsonl"
            
            with open(result_path, 'wb') as f:
                f.write(result_stream.read())
            
            # Parse results
            with open(result_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        custom_id = data.get('custom_id', '')
                        
                        # Parse custom_id: {question_id}__miu{miu}__needed{n}
                        parts = custom_id.split('__')
                        q_id = parts[0]
                        needed = N  # Default to full N
                        for part in parts:
                            if part.startswith('needed'):
                                try:
                                    needed = int(part.replace('needed', ''))
                                except Exception:
                                    pass
                        
                        content = data.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get('content', '')
                        parsed = parse_llm_response(content, 1)
                        
                        variants = parsed.get(1, [])
                        
                        # Get existing distortions for this question to avoid duplicates
                        q_rows = self._df[(self._df['question_id'] == q_id) & (self._df['miu'] == miu)]
                        existing = q_rows[q_rows['distorted_question'].notna() & (q_rows['distorted_question'] != '')]['distorted_question'].tolist()
                        
                        # Validate variants - need at least 'needed' new unique ones
                        orig_text = self._df[self._df['question_id'] == q_id]['question_text'].iloc[0]
                        
                        # Filter out variants that are too similar to existing ones
                        new_valid = []
                        existing_lower = {e.lower() for e in existing}
                        for v in variants:
                            validation = validate_distortion(orig_text, v, miu)
                            if validation.is_valid:
                                v_lower = validation.distorted.lower()
                                # Skip if duplicate of existing
                                if v_lower not in existing_lower:
                                    new_valid.append(validation.distorted)
                                    existing_lower.add(v_lower)
                        
                        # Fill empty slots with new valid variants
                        if new_valid:
                            idxs = self._df[(self._df['question_id'] == q_id) & (self._df['miu'] == miu)].index.tolist()
                            
                            # Find empty slots
                            empty_idxs = []
                            for idx in idxs:
                                val = self._df.at[idx, 'distorted_question']
                                if pd.isna(val) or val == '':
                                    empty_idxs.append(idx)
                            
                            # Fill empty slots with new variants
                            for i, idx in enumerate(empty_idxs):
                                if i < len(new_valid):
                                    self._df.at[idx, 'distorted_question'] = new_valid[i]
                    except Exception:
                        continue
            
            log_path = self.config.project_dir / "distorted_data" / "batch_api.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - ✅ Applied results for μ={miu}\n")
            print(f"   ✅ Applied results for μ={miu}")
            
        except Exception as e:
            log_path = self.config.project_dir / "distorted_data" / "batch_api.log"
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - ❌ Error downloading μ={miu}: {e}\n")
            print(f"   ❌ Error downloading μ={miu}: {e}")
    
    # ==================== PARALLEL API (FALLBACK) ====================
    
    def _run_parallel_api(self, miu_questions: Dict[float, List[Dict]]):
        """Fallback: Run distortions using parallel API calls."""
        print("\n🔄 Running parallel API calls...")
        
        # Start workers for each miu level
        threads = []
        for miu, questions in miu_questions.items():
            t = threading.Thread(target=self._process_miu_level, args=(miu, questions))
            t.start()
            threads.append(t)
        
        # Monitor and save loop
        last_save = time.time()
        
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
                self._display_progress()
                
                if time.time() - last_save > self.config.save_interval:
                    self._save_data()
                    last_save = time.time()
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted! Saving...")
            self._stop_requested = True
        
        for t in threads:
            t.join()
        
        self._save_data()
    
    def _call_api(self, questions: List[Dict], miu: float, attempt: int = 1) -> tuple:
        """Make API call to distortion model."""
        try:
            N = self.config.distortions_per_question
            max_tokens = calculate_max_tokens(questions, N, miu)

            if self.config.prompt_factory is not None:
                # Use benchmark-provided prompt factory (one question at a time).
                # The factory returns a DistortionPrompt with both system and user parts.
                assert len(questions) == 1, (
                    "prompt_factory mode requires questions_per_batch=1; "
                    f"got {len(questions)} questions in a single _call_api call"
                )
                dp = self.config.prompt_factory(questions[0]['text'], miu, N)
                messages = [
                    {"role": "system", "content": dp.system_prompt},
                    {"role": "user", "content": dp.user_prompt},
                ]
            else:
                prompt = get_batch_distortion_prompt(questions, miu, N)
                messages = [{"role": "user", "content": prompt}]

            response = requests.post(
                self.api_url,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": calculate_temperature(miu)
                },
                timeout=self.config.timeout
            )
            
            data = response.json()
            
            if "error" in data:
                if attempt < self.config.max_retries:
                    time.sleep(3 * attempt)
                    return self._call_api(questions, miu, attempt + 1)
                return None, data["error"].get("message", "API error")
            
            content = data["choices"][0]["message"]["content"]
            parsed = parse_llm_response(content, len(questions))
            return parsed, None
            
        except Exception as e:
            if attempt < self.config.max_retries:
                time.sleep(2 * attempt)
                return self._call_api(questions, miu, attempt + 1)
            return None, str(e)[:100]
    
    def _process_batch(self, questions: List[Dict], miu: float) -> tuple:
        """Process a single batch of questions."""
        results, error = self._call_api(questions, miu)
        
        if error:
            return 0, 1, 0
        
        success_count = 0
        invalid_count = 0
        
        with self._lock:
            for i, q in enumerate(questions, 1):
                variants = results.get(i, [])
                
                # STRICT: Require EXACTLY N unique distortions
                N = self.config.distortions_per_question
                valid_variants = validate_exact_count(variants, q['text'], miu, N)
                
                if len(valid_variants) < N:
                    invalid_count += (N - len(valid_variants))
                
                if not valid_variants:
                    continue
                
                idxs = self._df[
                    (self._df['question_id'] == q['id']) & 
                    (self._df['miu'] == miu)
                ].index.tolist()
                
                for j, idx in enumerate(idxs):
                    if j < len(valid_variants):
                        self._df.at[idx, 'distorted_question'] = valid_variants[j]
                
                success_count += 1
        
        return success_count, 0, invalid_count
    
    def _process_miu_level(self, miu: float, questions: List[Dict]):
        """Process all questions for a single miu level."""
        batches = []
        for i in range(0, len(questions), self.config.questions_per_batch):
            batches.append(questions[i:i+self.config.questions_per_batch])
        
        with self._lock:
            self._progress[miu] = DistortionProgress(
                miu=miu, total_batches=len(batches), completed_batches=0, errors=0, invalid=0
            )
        
        with ThreadPoolExecutor(max_workers=self.config.workers_per_miu) as executor:
            futures = {executor.submit(self._process_batch, batch, miu): batch for batch in batches}
            
            for future in as_completed(futures):
                if self._stop_requested:
                    break
                
                success, err, invalid = future.result()
                
                with self._lock:
                    self._progress[miu].completed_batches += 1
                    self._progress[miu].errors += err
                    self._progress[miu].invalid += invalid
    
    def _display_progress(self):
        """Display progress for all miu levels."""
        if sys.platform == 'win32':
            os.system('cls')
        else:
            print("\033[H\033[J", end="")
        
        print("=" * 60)
        print("CHAMELEON DISTORTION GENERATOR (Parallel API Fallback)")
        print("=" * 60)
        print(f"Project: {self.config.project_dir.name} | Model: {self.config.model}")
        print("=" * 60)
        
        total_done = total_batches = total_errors = 0
        
        with self._lock:
            for miu in sorted(self._progress.keys()):
                p = self._progress[miu]
                total_done += p.completed_batches
                total_batches += p.total_batches
                total_errors += p.errors
                
                if p.total_batches > 0:
                    pct = p.completed_batches * 100 // p.total_batches
                    bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                    print(f"  μ={miu}: [{bar}] {p.completed_batches:3}/{p.total_batches:3} | err:{p.errors}")
        
        print()
        if total_batches > 0:
            print(f"OVERALL: {total_done}/{total_batches} ({total_done*100//total_batches}%) | Errors: {total_errors}")
        print("=" * 60)
    
    # ==================== MAIN RUN ====================
    
    def run(self, display_progress: bool = True) -> Dict[str, Any]:
        """Run the distortion generation process."""
        start_time = time.time()
        
        print("=" * 60)
        print("🦎 CHAMELEON DISTORTION GENERATOR")
        print("=" * 60)
        print(f"Project: {self.config.project_dir.name}")
        print(f"Model: {self.config.model}")
        print(f"Distortions/question: {self.config.distortions_per_question}")
        print("=" * 60)
        
        # Load data
        print("\nLoading data...")
        self._df = self._load_data()
        
        # Get pending questions by miu
        miu_questions = self._get_pending_by_miu()
        
        if not miu_questions:
            print("\n✓ All distortions complete!")
            # Still run post-validation to check for duplicates
            self._run_post_validation()
            self._save_data()
            self._cleanup()
            elapsed = time.time() - start_time
            complete_path = self.config.project_dir / "distorted_data" / "distortions_complete.csv"
            print(f"\n✓ COMPLETE in {elapsed/60:.1f} minutes")
            print(f"Output: {complete_path}")
            return {"status": "complete", "elapsed": elapsed}
        
        total_questions = sum(len(q) for q in miu_questions.values())
        print(f"\nPending: {total_questions} questions across {len(miu_questions)} miu levels")
        for miu, qs in sorted(miu_questions.items()):
            print(f"  μ={miu}: {len(qs)} questions")
        
        # STRICT ENFORCEMENT: Keep retrying until ALL questions have EXACTLY N distortions
        N = self.config.distortions_per_question
        round_num = 0
        last_incomplete = None
        stuck_counter = 0
        max_stuck_rounds = 5  # If no progress for 5 rounds, stop
        
        while True:
            round_num += 1
            
            # Get remaining questions
            miu_questions = self._get_pending_by_miu()
            
            if not miu_questions:
                print(f"\n✅ All distortion slots filled!")
                break
            
            total_remaining = sum(len(q) for q in miu_questions.values())
            
            print(f"\n{'='*60}")
            print(f"📋 ROUND {round_num} - Enforcing exactly {N} distortions per question")
            print(f"   Remaining: {total_remaining} questions need distortions")
            print(f"{'='*60}")
            
            # Strategy 1: Try Batch API first
            batch_success = self._try_batch_api(miu_questions)
            
            if not batch_success:
                # Reload pending (some might have been done by batch)
                miu_questions = self._get_pending_by_miu()
                
                if miu_questions:
                    # Strategy 2: Fallback to parallel API
                    self._run_parallel_api(miu_questions)
            
            # Save progress after each round
            self._save_data()
            
            # Check if we're done (all questions have exactly N distortions)
            incomplete = self._count_incomplete()
            
            if incomplete == 0:
                print(f"\n✅ All questions have exactly {N} unique distortions!")
                break
            
            # Check if we're making progress
            if last_incomplete is not None:
                if incomplete >= last_incomplete:
                    stuck_counter += 1
                    print(f"\n⚠️ No progress this round. Stuck counter: {stuck_counter}/{max_stuck_rounds}")
                else:
                    stuck_counter = 0  # Reset if we made progress
                    progress = last_incomplete - incomplete
                    print(f"\n📈 Progress: Fixed {progress} distortions ({incomplete} remaining)")
            
            last_incomplete = incomplete
            
            # Give up if we're stuck
            if stuck_counter >= max_stuck_rounds:
                print(f"\n❌ No progress for {max_stuck_rounds} rounds. {incomplete} distortions still missing.")
                print(f"   These questions may have persistent issues. Review manually.")
                break
            
            # Small pause before next round
            print(f"\n⏳ {incomplete} distortions still need filling. Starting next round...")
            time.sleep(2)
        
        # Final check
        incomplete = self._count_incomplete()
        
        # Run post-validation to check for duplicates / template validity
        self._run_post_validation()
        
        # Final save and cleanup
        self._save_data()
        self._cleanup()
        
        elapsed = time.time() - start_time
        complete_path = self.config.project_dir / "distorted_data" / "distortions_complete.csv"
        print(f"\n✓ COMPLETE in {elapsed/60:.1f} minutes")
        print(f"Output: {complete_path}")
        
        return {"status": "complete", "elapsed_seconds": elapsed, "output_file": str(complete_path), "incomplete": incomplete}


def run_distortions(project_name: str, projects_dir: str = "Projects") -> Dict[str, Any]:
    """Convenience function to run distortions for a project."""
    runner = DistortionRunner.from_project(project_name, projects_dir)
    return runner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run distortion generation")
    parser.add_argument("project", nargs="?", default=None, help="Project name")
    parser.add_argument("--projects-dir", default="Projects", help="Projects directory")
    
    args = parser.parse_args()
    
    if not args.project:
        projects_dir = Path(args.projects_dir)
        if projects_dir.exists():
            projects = [p.name for p in projects_dir.iterdir() if p.is_dir()]
            print("Available projects:")
            for p in projects:
                print(f"  - {p}")
        sys.exit(0)
    
    result = run_distortions(args.project, args.projects_dir)
    sys.exit(0 if result.get("status") == "complete" else 1)
