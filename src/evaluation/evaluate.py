import json
import pandas as pd
import numpy as np
import os
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def evaluate_single(csv_path, gt_path, caption_col, label):
    print(f"\n=============================================")
    print(f"Evaluating: {label} (File: {csv_path}, Column: {caption_col})")
    print(f"=============================================")
    
    df = pd.read_csv(csv_path)
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
        
    
    if isinstance(gt_data, dict) and 'annotations' in gt_data:
        annotations = gt_data['annotations'] # COCO format
    else:
        annotations = gt_data 

    gts = {}
    for item in annotations:
        img_id = item['image_id']
        
        caption_text = item.get('caption', item.get('caption_original'))
        gts.setdefault(img_id, []).append(caption_text)
        
    res = {int(row['image_id']): [row[caption_col]] for _, row in df.iterrows()}
    
    common_ids = set(gts.keys()).intersection(set(res.keys()))
    gts = {k: gts[k] for k in common_ids}
    res = {k: res[k] for k in common_ids}
    
    print(f"Evaluating on {len(common_ids)} images...")

    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Rouge(), "ROUGE-L"),
        (Cider(), "CIDEr")
    ]

    final_scores = {}
    per_image_cider = None

    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if method == "CIDEr":
            per_image_cider = scores # Save per-image scores for Wilcoxon test
        if isinstance(method, list):
            for sc, m in zip(score, method):
                final_scores[m] = sc
        else:
            final_scores[method] = score

    for metric, score in final_scores.items():
        print(f"{metric}: {score:.4f}")
        
    return final_scores, per_image_cider

if __name__ == "__main__":
    # File Paths
    csv_100 = "results/generated_captions/rag_captions_comparison.csv"
    csv_1000 = "results/generated_captions/rag_captions_comparison_1000.csv"
    gt_path = "outputs/captions_val2017.json"
    
    # 1. Evaluate 100 images
    scores_b_100, _ = evaluate_single(csv_100, gt_path, "baseline_caption", "BASELINE - 100 Images")
    scores_r_100, _ = evaluate_single(csv_100, gt_path, "rag_caption", "RAG - 100 Images")
    
    # 2. Evaluate 1000 images
    scores_b_1000, _ = evaluate_single(csv_1000, gt_path, "baseline_caption", "BASELINE - 1,000 Images")
    scores_r_1000, _ = evaluate_single(csv_1000, gt_path, "rag_caption", "RAG - 1,000 Images")
    
    # 3. Save to CSV (This is what was missing!)
    print("\nSaving metrics to results/tables/metrics.csv...")
    os.makedirs("results/tables", exist_ok=True)
    
    table_data = [
        {"Method": "Baseline (100)", "CIDEr": scores_b_100['CIDEr'], "BLEU-4": scores_b_100['BLEU-4'], "ROUGE-L": scores_b_100['ROUGE-L']},
        {"Method": "RAG (100)", "CIDEr": scores_r_100['CIDEr'], "BLEU-4": scores_r_100['BLEU-4'], "ROUGE-L": scores_r_100['ROUGE-L']},
        {"Method": "Baseline (1000)", "CIDEr": scores_b_1000['CIDEr'], "BLEU-4": scores_b_1000['BLEU-4'], "ROUGE-L": scores_b_1000['ROUGE-L']},
        {"Method": "RAG (1000)", "CIDEr": scores_r_1000['CIDEr'], "BLEU-4": scores_r_1000['BLEU-4'], "ROUGE-L": scores_r_1000['ROUGE-L']}
    ]
    
    df_metrics = pd.DataFrame(table_data)
    df_metrics.to_csv("results/tables/metrics.csv", index=False)
    print("✅ Saved successfully to results/tables/metrics.csv")