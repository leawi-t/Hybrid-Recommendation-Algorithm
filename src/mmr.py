import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def maximal_marginal_relevance(item_scores, tfidf_matrix, item_to_idx, idx_to_item, item_id_to_tfidf_row, top_k=10, lambda_param=0.5):
    """
    Applies MMR by securely mapping internal model indices back to true TF-IDF rows.
    """
    # Filter candidates to only those that exist in our TF-IDF metadata lookup
    valid_candidates = [
        idx for idx in np.argsort(-item_scores) 
        if idx_to_item[idx] in item_id_to_tfidf_row
    ]
    
    # Take a candidate generation slice
    candidate_indices = valid_candidates[:top_k * 3]
    
    if not candidate_indices:
        return [idx_to_item[idx] for idx in np.argsort(-item_scores)[:top_k]]
        
    selected_indices = []
    unselected_indices = list(candidate_indices)
    
    # Start with the highest scoring item
    first_choice = unselected_indices.pop(0)
    selected_indices.append(first_choice)
    
    while len(selected_indices) < top_k and unselected_indices:
        best_mmr_score = -np.inf
        best_candidate = None
        
        # Pull the correct mapped TF-IDF rows for selected items
        selected_rows = [item_id_to_tfidf_row[idx_to_item[idx]] for idx in selected_indices]
        selected_vectors = tfidf_matrix[selected_rows]
        
        for candidate in unselected_indices:
            relevance = item_scores[candidate]
            
            # Map candidate index to its actual TF-IDF row index
            candidate_row = item_id_to_tfidf_row[idx_to_item[candidate]]
            candidate_vector = tfidf_matrix[candidate_row].reshape(1, -1)
            
            sim_matrix = cosine_similarity(candidate_vector, selected_vectors)
            max_similarity = np.max(sim_matrix)
            
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_candidate = candidate
                
        if best_candidate is None:
            break
            
        unselected_indices.remove(best_candidate)
        selected_indices.append(best_candidate)
        
    return [idx_to_item[idx] for idx in selected_indices]
