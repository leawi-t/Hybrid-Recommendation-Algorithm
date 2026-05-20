import numpy as np
import pandas as pd
from src.mmr import maximal_marginal_relevance

class CascadeOrchestrator:
    def __init__(self, model, dataset, item_features, tfidf_matrix, feature_item_ids, train_path):
        self.model = model
        self.dataset = dataset
        self.item_features = item_features
        self.tfidf_matrix = tfidf_matrix
        
        # Unpack indices mappings
        self.user_to_idx, _, self.item_to_idx, _ = dataset.mapping()
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        # CRITICAL RE-ALIGNMENT FIX: Map Item String ID -> Row position in TF-IDF matrix
        self.item_id_to_tfidf_row = {item_id: idx for idx, item_id in enumerate(feature_item_ids)}
        
        # Global popularity fallback metrics
        train_df = pd.read_csv(train_path, dtype={'item_id': str})
        self.popularity_fallback = train_df['item_id'].value_counts().index.tolist()

    def get_top_n(self, user_id, n=10, use_mmr=False, lambda_param=0.5, candidate_pool=None):
        if user_id not in self.user_to_idx:
            fallback = self.popularity_fallback
            if candidate_pool:
                fallback = [i for i in fallback if i in candidate_pool]
            return fallback[:n]

        internal_user_idx = self.user_to_idx[user_id]
        all_item_indices = np.arange(len(self.item_to_idx))

        # Filter to candidate pool if provided
        if candidate_pool:
            all_item_indices = np.array([
                idx for idx, item_id in self.idx_to_item.items()
                if item_id in candidate_pool
            ])

        scores = self.model.predict(
            user_ids=internal_user_idx,
            item_ids=all_item_indices,
            item_features=self.item_features
        )

        if use_mmr:
            # Rebuild scores array indexed from 0 for MMR compatibility
            local_idx_to_item = {i: self.idx_to_item[all_item_indices[i]] for i in range(len(all_item_indices))}
            local_item_to_idx = {v: k for k, v in local_idx_to_item.items()}
            return maximal_marginal_relevance(
                item_scores=scores,
                tfidf_matrix=self.tfidf_matrix,
                item_to_idx=local_item_to_idx,
                idx_to_item=local_idx_to_item,
                item_id_to_tfidf_row=self.item_id_to_tfidf_row,
                top_k=n,
                lambda_param=lambda_param
            )

        top_indices = np.argsort(-scores)[:n]
        return [self.idx_to_item[all_item_indices[i]] for i in top_indices]

