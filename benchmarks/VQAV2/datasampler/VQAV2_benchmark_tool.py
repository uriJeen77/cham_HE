import json
import random
import os
import requests
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

# ================= CONFIGURATION =================
CONFIG = {
    # כתובות להורדת המטא-דאטה (קבצים קטנים יחסית - עשרות MB)
    "URL_QUESTIONS": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "URL_ANNOTATIONS": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    
    # שמות הקבצים שחולצו (חייבים להתאים למה שיש בתוך ה-ZIP)
    "FILENAME_QUESTIONS": "v2_OpenEnded_mscoco_val2014_questions.json",
    "FILENAME_ANNOTATIONS": "v2_mscoco_val2014_annotations.json",
    
    # פלט
    "OUTPUT_JSON": "vqav2_sampled_optimized.json", # שיניתי שם כדי לשקף שזה עבר אופטימיזציה
    "IMG_DIR": os.path.abspath("./sampled_images_300"),
    
    # הגדרות
    "SAMPLE_SIZE": 300,
    "DOWNLOAD_THREADS": 20
}
# =================================================

def download_and_extract_metadata(url, target_filename):
    """
    מוריד את ה-ZIP לזיכרון, מחלץ את ה-JSON ושומר אותו בדיסק.
    חוסך התעסקות ידנית.
    """
    if os.path.exists(target_filename):
        print(f">> Found existing metadata file: {target_filename}")
        return

    print(f">> Downloading metadata from {url}...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to download {url}")

    print(f">> Extracting {target_filename}...")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        # מחפשים את הקובץ הספציפי בתוך ה-zip
        for filename in z.namelist():
            if target_filename in filename: # התאמה חלקית למקרה של תיקיות פנימיות
                with z.open(filename) as source, open(target_filename, "wb") as target:
                    target.write(source.read())
                return
    
    print(f"WARNING: Could not find {target_filename} inside the downloaded zip!")

def download_image_worker(args):
    """ מוריד תמונה בודדת (Threaded) """
    img_id, save_dir, split_name = args
    filename = f"COCO_{split_name}_{img_id:012d}.jpg"
    url = f"http://images.cocodataset.org/{split_name}/{filename}"
    save_path = os.path.join(save_dir, filename)

    if os.path.exists(save_path):
        return

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(resp.content)
    except Exception:
        pass

def prepare_data():
    # 1. הורדת המטא-דאטה (החלק הקל)
    try:
        download_and_extract_metadata(CONFIG["URL_QUESTIONS"], CONFIG["FILENAME_QUESTIONS"])
        download_and_extract_metadata(CONFIG["URL_ANNOTATIONS"], CONFIG["FILENAME_ANNOTATIONS"])
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # 2. טעינת הנתונים לזיכרון
    print(">> Loading JSONs into memory...")
    with open(CONFIG["FILENAME_QUESTIONS"], 'r') as f:
        q_data = json.load(f)
    with open(CONFIG["FILENAME_ANNOTATIONS"], 'r') as f:
        a_data = json.load(f)

    # 3. מיפוי אנוטציות
    print(">> Indexing annotations...")
    ann_map = {ann['question_id']: ann for ann in a_data['annotations']}

    # 4. קיבוץ שאלות לפי תמונה
    print(">> Grouping questions by Image ID...")
    image_to_questions_map = defaultdict(list)
    for q in q_data['questions']:
        image_to_questions_map[q['image_id']].append(q)

    # 5. דגימה רנדומלית
    all_image_ids = list(image_to_questions_map.keys())
    print(f">> Sampling {CONFIG['SAMPLE_SIZE']} images from {len(all_image_ids)} total...")
    sampled_image_ids = random.sample(all_image_ids, CONFIG['SAMPLE_SIZE'])

    # 6. בניית הדאטה-סט הסופי
    final_dataset = []
    download_queue = []
    split_name = "val2014"

    print(">> Constructing OPTIMIZED dataset records (Flat List of Strings)...")
    for img_id in sampled_image_ids:
        filename = f"COCO_{split_name}_{img_id:012d}.jpg"
        full_image_path = os.path.join(CONFIG["IMG_DIR"], filename)
        
        # הוספה לתור ההורדה
        download_queue.append((img_id, CONFIG["IMG_DIR"], split_name))

        questions = image_to_questions_map[img_id]
        
        for q in questions:
            ann = ann_map.get(q['question_id'])
            if not ann: continue

            # === OPTIMIZATION STEP ===
            # במקום לשמור רשימה של מילונים, אנו שומרים רשימה של מחרוזות בלבד.
            # הערה: זה דורש התאמה קלה ב-vqaEval.py (להתייחס ל-answer כ-String ישיר)
            flat_answers = [ans["answer"] for ans in ann["answers"]]

            record = {
                "question_type": ann["question_type"],
                "multiple_choice_answer": ann["multiple_choice_answer"],
                
                "answers": flat_answers, # <--- הרשימה השטוחה החדשה
                
                "image_id": q["image_id"],
                "answer_type": ann["answer_type"],
                "question_id": q["question_id"],
                "question": q["question"],
                "image": full_image_path
            }
            final_dataset.append(record)

    # 7. הורדת התמונות בפועל
    print(f">> Downloading {len(sampled_image_ids)} images to {CONFIG['IMG_DIR']}...")
    if not os.path.exists(CONFIG["IMG_DIR"]):
        os.makedirs(CONFIG["IMG_DIR"])
        
    with ThreadPoolExecutor(max_workers=CONFIG["DOWNLOAD_THREADS"]) as executor:
        list(executor.map(download_image_worker, download_queue))

    # 8. שמירת הקובץ הסופי
    print(f">> Saving final dataset to {CONFIG['OUTPUT_JSON']}...")
    with open(CONFIG["OUTPUT_JSON"], 'w') as f:
        json.dump(final_dataset, f, indent=4)
        
    print("\n=================================================")
    print("SUCCESS!")
    print(f"Dataset created at: {CONFIG['OUTPUT_JSON']}")
    print(f"Images folder: {CONFIG['IMG_DIR']}")
    print("NOTE: 'answers' field is now a list of strings.")
    print("Ensure vqaEval.py logic is updated to handle strings directly.")
    print("=================================================")

if __name__ == "__main__":
    prepare_data()