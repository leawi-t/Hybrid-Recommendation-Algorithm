import numpy as np
import pandas as pd
from src.mmr import maximal_marginal_relevance

class CascadeOrchestrator:
    def __init__(self, model, dataset, item_features, tfidf_matrix, feature_item_ids, train_path):
        self.model = model
        self.dataset = dataset
        self.item_features = item_features
        self.tfidf_matrix = tfidf_matrix
        
        # 1. Unpack mappings
        self.user_to_idx, _, self.item_to_idx, _ = dataset.mapping()
        self.idx_to_item = {idx: item_id for item_id, idx in self.item_to_idx.items()}
        
        # 2. Map item IDs directly to their row in the TF-IDF matrix
        self.item_id_to_tfidf_row = {item_id: idx for idx, item_id in enumerate(feature_item_ids)}
        
        # 3. Store the raw surprise algo object for direct unmapped predictions
        self.algo = model.algo

    def get_top_n(self, raw_user_id, n=10, use_mmr=False, lambda_param=0.5):
        # Fallback if user is completely unknown to training set
        if raw_user_id not in self.user_to_idx:
            return []
            
        # Candidate pool: string item IDs that exist in BOTH SVD training vocabulary AND TF-IDF matrix
        candidate_item_ids = [
            item_id for item_id in self.item_to_idx.keys() 
            if item_id in self.item_id_to_tfidf_row
        ]
        
        # Call the native surprise predict method directly using raw string IDs 
        # This completely bypasses any internal numpy array index alignment bugs
        scored_candidates = []
        for item_id in candidate_item_ids:
            est_score = self.algo.predict(str(raw_user_id), str(item_id)).est
            scored_candidates.append((item_id, est_score))
            
        # Sort candidates strictly by their true estimated score value
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not use_mmr:
            # Return top N raw string IDs directly
            return [item_id for item_id, score in scored_candidates[:n]]
            
        # For MMR, extract the top 100 scored string items to diversify
        top_candidates = scored_candidates[:500]  # Grab the top 500 items instead of 100
        top_scores = np.array([score for item_id, score in top_candidates])

        # Re-construct the temporary indexing system that your specific mmr.py script expects
        # mmr.py uses: idx_to_item[idx] and item_to_idx[item_id]
        tmp_item_to_idx = {item_id: i for i, (item_id, _) in enumerate(top_candidates)}
        tmp_idx_to_item = {i: item_id for i, (item_id, _) in enumerate(top_candidates)}
        
        diversified_items = maximal_marginal_relevance(
            item_scores=top_scores,
            tfidf_matrix=self.tfidf_matrix,
            item_to_idx=tmp_item_to_idx,
            idx_to_item=tmp_idx_to_item,
            item_id_to_tfidf_row=self.item_id_to_tfidf_row,
            top_k=n,
            lambda_param=lambda_param
        )
        
        return diversified_items
