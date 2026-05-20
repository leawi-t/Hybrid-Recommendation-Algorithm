from operator import truth
import os
import numpy as np
import pandas as pd
from src.preprocessing import (
    load_and_clean_data, 
    calculate_sparsity, 
    isolate_cold_start, 
    train_test_split_interactions,
    load_and_merge_metadata
)
from src.evaluation import calculate_precision_at_k, calculate_intra_list_diversity, calculate_recall_at_k, calculate_coverage
from src.content_based import build_item_features_real_meta
from src.collaborative import train_hybrid_cf
from src.hybrid import CascadeOrchestrator
from src.evaluation import calculate_precision_at_k, calculate_intra_list_diversity
import src.config as cfg

def main():
    if not os.path.exists(cfg.DATA_PATH) or not os.path.exists(cfg.META_PATH):
        print("Missing dataset files in data/raw/ directory!")
        return
    
    # 1. Clean logs using string parsing rules
    cleaned_df = load_and_clean_data(cfg.DATA_PATH, cfg.MIN_USER_INTERACTIONS, cfg.MIN_ITEM_INTERACTIONS)
    calculate_sparsity(cleaned_df)
    
    # 2. Extract and preserve text data profile records
    meta_path = "data/processed/item_metadata.csv"
    if not os.path.exists(meta_path):
        meta_df = load_and_merge_metadata(cleaned_df, cfg.META_PATH)
        os.makedirs("data/processed", exist_ok=True)
        meta_df.to_csv(meta_path, index=False)
    else:
        print("Loading cached metadata...")
        meta_df = pd.read_csv(meta_path, dtype={'item_id': str})
    
    # Filter ratings dataset to ensure we only retain rows containing valid metadata mappings
    cleaned_df = cleaned_df[cleaned_df['item_id'].isin(meta_df['item_id'])]
    
    # 3. Handle splits
    train_df, cold_test_df = isolate_cold_start(cleaned_df, cold_ratio=cfg.COLD_START_RATIO)
    train_df, test_df = train_test_split_interactions(train_df, test_ratio=cfg.TEST_RATIO)
    
    train_path = "data/processed/train_interactions.csv"
    train_df.to_csv(train_path, index=False)
    
    # 4. Generate TF-IDF representations using real item titles/descriptions
    tfidf_mat, feature_item_ids, meta_lookup = build_item_features_real_meta(meta_path)
    
    # 5. Model training
    model, dataset, item_features = train_hybrid_cf(train_path, tfidf_mat, feature_item_ids)
    
    orchestrator = CascadeOrchestrator(
        model=model, 
        dataset=dataset, 
        item_features=item_features, 
        tfidf_matrix=tfidf_mat, 
        feature_item_ids=feature_item_ids, # Pass the feature_item_ids list here
        train_path=train_path
    )

    # --- DAY 3 COMPARATIVE EVALUATION ---
    print("\n--- Evaluation Loop (Evaluating the Filter Bubble) ---")

    std_recalls, mmr_recalls = [], []
    
    # Pull sample active test users to evaluate
    test_grouped = test_df.groupby('user_id')['item_id'].apply(list).to_dict()

    # Only keep users with enough ground truth to evaluate fairly
    test_grouped = {uid: items for uid, items in test_grouped.items() if len(items) >= 3}   
    sample_users = list(test_grouped.keys())[:200]
    print(f"Evaluating on {len(sample_users)} users with >=3 ground truth items")
    
    std_precisions, std_diversities = [], []
    mmr_precisions, mmr_diversities = [], []
    
    user_to_idx, _, item_to_idx, _ = dataset.mapping()
    
    test_item_pool = set(test_df['item_id'].unique())

    all_std_recs = []
    all_mmr_recs = []

    for uid in sample_users:
        ground_truth = test_grouped[uid]
    
        recs_std = orchestrator.get_top_n(uid, n=10, use_mmr=False)
        recs_mmr = orchestrator.get_top_n(uid, n=10, use_mmr=True, lambda_param=0.5)
    
        all_std_recs.append(recs_std)
        all_mmr_recs.append(recs_mmr)
    
        std_precisions.append(calculate_precision_at_k(recs_std, ground_truth))
        std_diversities.append(calculate_intra_list_diversity(recs_std, tfidf_mat, orchestrator.item_id_to_tfidf_row))
        std_recalls.append(calculate_recall_at_k(recs_std, ground_truth))
    
        mmr_precisions.append(calculate_precision_at_k(recs_mmr, ground_truth))
        mmr_diversities.append(calculate_intra_list_diversity(recs_mmr, tfidf_mat, orchestrator.item_id_to_tfidf_row))
        mmr_recalls.append(calculate_recall_at_k(recs_mmr, ground_truth))

    total_items = len(feature_item_ids)
    std_coverage = calculate_coverage(all_std_recs, total_items)
    mmr_coverage = calculate_coverage(all_mmr_recs, total_items)

    print("\n==============================================")
    print(f"STANDARD RECS  -> Precision@10: {np.mean(std_precisions):.4f} | Recall@10: {np.mean(std_recalls):.4f} | ILD: {np.mean(std_diversities):.4f} | Coverage: {std_coverage:.4f}")
    print(f"MMR DIVERSIFIED -> Precision@10: {np.mean(mmr_precisions):.4f} | Recall@10: {np.mean(mmr_recalls):.4f} | ILD: {np.mean(mmr_diversities):.4f} | Coverage: {mmr_coverage:.4f}")
    print("==============================================")

if __name__ == "__main__":
    main()



