import json
import argparse
import pandas as pd
from sklearn.metrics import r2_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import pearsonr

def analyze(file_path):
    print(f"Analyzing {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['records'])
    
    # Regression Metrics
    y_true = df['actual_damage_pct']
    y_pred = df['pred_damage_pct']
    
    mae = (y_true - y_pred).abs().mean()
    rmse = ((y_true - y_pred) ** 2).mean() ** 0.5
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    
    print("\n--- Regression Metrics ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Pearson Correlation: {corr:.4f}")
    
    # Classification Metrics (Threshold > 0.0)
    # Treat any damage > 0.0 as "Positive" (Damaged)
    threshold = 0.0
    y_true_cls = (y_true > threshold).astype(int)
    y_pred_cls = (y_pred > threshold).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_true_cls, y_pred_cls).ravel()
    
    print(f"\n--- Classification Metrics (Threshold > {threshold}%) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # Generation Metrics
    if 'faithfulness' in df.columns:
        print("\n--- Generation Metrics ---")
        print(f"Avg Faithfulness: {df['faithfulness'].mean():.4f}")
        print(f"Avg Relevance: {df['relevance'].mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to evaluation results JSON")
    args = parser.parse_args()
    analyze(args.file_path)
