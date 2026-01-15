def run_full_pipeline(self, data_path: str, output_dir: str, miu: float = 0.6) -> Dict[str, Any]:
        """
        הרצת הצינור המלא מקצה לקצה.
        
        מריץ את כל 6 השלבים:
        1. טעינת דאטה
        2. עיוות prompts
        3. ולידציה שהמשמעות נשמרה
        4. ג'נרציה של קוד (original + distorted)
        5. בדיקת סינטקסודא
        6. הרצת טסטים וחישוב מדדים
        
        Args:
            data_path: נתיב לדאטה המקורי
            output_dir: תיקייה לשמירת תוצאות
            miu: רמת העיוות
            
        Returns:
            תוצאות הצינור (מדדים, קבצים שנוצרו, וכו')
        """
        from pathlib import Path
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("🌟  HumanEvalV3 - Robustness Evaluation Pipeline  🌟")
        print("=" * 80)
        print(f"💾 דאטה: {data_path}")
        print(f"📁 תוצאות: {output_dir}")
        print(f"🎯 Miu: {miu}")
        print("=" * 80 + "\n")
        
        # שלב 1: טעינת דאטה
        print("📚 שלב 1/6: טעינת דאטה מקורי...")
        tasks = self.load_tasks(data_path)
        
        # שמירת הדאטה המקורי לקובץ
        with open(output_path / "original_tasks.jsonl", "w", encoding="utf-8") as f:
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        
        # שלב 2: עיוות
        print(f"\n🔀 שלב 2/6: עיוות prompts...")
        distorted_tasks = self.distort_prompts(tasks.copy(), miu=miu)
        
        # שמירת הדאטה המעוות
        with open(output_path / "distorted_tasks.jsonl", "w", encoding="utf-8") as f:
            for task in distorted_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
        
        # שלב 3: ולידציה
        print(f"\n🔍 שלב 3/6: ולידציה של עיוותים...")
        validated_tasks = self.validate_distortions(distorted_tasks)
        
        # שלב 4: ג'נרציה של קוד עבור המקורי
        print(f"\n💻 שלב 4a/6: ג'נרציה של קוד עבור prompts מקוריים...")
        original_with_code = self.generate_code(tasks.copy())
        
        print(f"\n💻 שלב 4b/6: ג'נרציה של קוד עבור prompts מעוותים...")
        validated_with_code = self.generate_code(validated_tasks)
        
        # שלב 5: בדיקת סינטקס
        print(f"\n✅ שלב 5a/6: בדיקת סינטקס לקוד מקורי...")
        original_valid = self.validate_syntax(original_with_code)
        
        print(f"\n✅ שלב 5b/6: בדיקת סינטקס לקוד מעוות...")
        distorted_valid = self.validate_syntax(validated_with_code)
        
        # שלב 6: הרצת טסטים
        print(f"\n🧪 שלב 6/6: הרצת טסטים והערכה...")
        original_results = []
        distorted_results = []
        
        # הערכה של הקוד המקורי
        for task in original_valid:
            result = self.evaluate_completion(task, task["generated_code"])
            original_results.append(result)
        
        # הערכה של הקוד המעוות
        for task in distorted_valid:
            result = self.evaluate_completion(task, task["generated_code"])
            distorted_results.append(result)
        
        # חישוב מדדים
        original_metrics = self.calculate_metrics(original_results)
        distorted_metrics = self.calculate_metrics(distorted_results)
        
        # סיכום תוצאות
        summary = {
            "original_tasks_count": len(tasks),
            "distorted_tasks_count": len(distorted_tasks),
            "validated_tasks_count": len(validated_tasks),
            "original_valid_code_count": len(original_valid),
            "distorted_valid_code_count": len(distorted_valid),
            "original_metrics": original_metrics,
            "distorted_metrics": distorted_metrics,
            "miu": miu
        }
        
        # שמירת סיכום
        with open(output_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print("📊 תוצאות סופיות:")
        print("=" * 80)
        print(f"🔸 Original pass@1: {original_metrics.get('pass@1', 0):.2%}")
        print(f"🔸 Distorted pass@1: {distorted_metrics.get('pass@1', 0):.2%}")
        print(f"🔸 Valid tasks: {len(validated_tasks)}/{len(tasks)}")
        print("=" * 80 + "\n")
        
        return summary
