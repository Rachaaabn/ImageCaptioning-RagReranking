import json
import pandas as pd
from pycocoevalcap.cider.cider import Cider
from scipy import stats
import os

def get_scores(csv_path, gt_data):
    df = pd.read_csv(csv_path)
    annotations = gt_data['annotations'] if 'annotations' in gt_data else gt_data
    
    gts = {}
    for item in annotations:
        caption = item.get('caption', item.get('caption_original'))
        gts.setdefault(item['image_id'], []).append(caption)
        
    res_b = {int(row['image_id']): [row['baseline_caption']] for _, row in df.iterrows()}
    res_r = {int(row['image_id']): [row['rag_caption']] for _, row in df.iterrows()}
    
    common = set(gts.keys()).intersection(set(res_b.keys())).intersection(set(res_r.keys()))
    gts_c = {k: gts[k] for k in common}
    
    _, scores_b = Cider().compute_score(gts_c, {k: res_b[k] for k in common})
    _, scores_r = Cider().compute_score(gts_c, {k: res_r[k] for k in common})
    
    return scores_b, scores_r

def run_tests(scores_b, scores_r, label):
    # Shapiro-Wilk
    _, p_b = stats.shapiro(scores_b)
    _, p_r = stats.shapiro(scores_r)
    
    # Wilcoxon
    w_stat, p_val = stats.wilcoxon(scores_b, scores_r)
    
    # Effect size
    n = len(scores_b)
    z_stat = stats.norm.ppf(p_val / 2)
    effect_size = abs(z_stat) / (n ** 0.5)
    
    return {
        "Dataset": label,
        "Images_Count": n,
        "Shapiro_Baseline_p": p_b,
        "Shapiro_RAG_p": p_r,
        "Wilcoxon_Statistic": w_stat,
        "Wilcoxon_p_value": p_val,
        "Effect_Size_Rosenthal_r": effect_size
    }

if __name__ == "__main__":
    print("Loading Ground Truth...")
    with open("outputs/captions_val2017.json", 'r') as f:
        gt_data = json.load(f)
        
    print("Running tests for 100 images...")
    b_100, r_100 = get_scores("results/generated_captions/rag_captions_comparison.csv", gt_data)
    res_100 = run_tests(b_100, r_100, "100 Images")
    
    print("Running tests for 1,000 images...")
    b_1000, r_1000 = get_scores("results/generated_captions/rag_captions_comparison_1000.csv", gt_data)
    res_1000 = run_tests(b_1000, r_1000, "1,000 Images")
    
    # Save to CSV
    os.makedirs("results/tables", exist_ok=True)
    df_results = pd.DataFrame([res_100, res_1000])
    df_results.to_csv("results/tables/statistical_tests.csv", index=False)
    
    print("\n✅ Results saved to results/tables/statistical_tests.csv")
    print(df_results.to_string())