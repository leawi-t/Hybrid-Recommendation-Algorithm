import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def calculate_precision_at_k(recommended, ground_truth):
    """
    Measures evaluation precision accuracy.
    """
    if not ground_truth:
        return 0.0
    common = set(recommended).intersection(set(ground_truth))
    return len(common) / len(recommended)

# evaluation.py
def calculate_intra_list_diversity(recommended, tfidf_matrix, item_id_to_tfidf_row):
    indices = [item_id_to_tfidf_row[item] for item in recommended if item in item_id_to_tfidf_row]
    if len(indices) <= 1:
        return 0.0
    vectors = tfidf_matrix[indices]
    dist_matrix = cosine_distances(vectors)
    triu_indices = np.triu_indices_from(dist_matrix, k=1)
    return np.mean(dist_matrix[triu_indices])

def calculate_recall_at_k(recommended, ground_truth):
    """
    Recall: what fraction of ground truth items were recommended.
    More meaningful than precision when ground truth is small.
    """
    if not ground_truth:
        return 0.0
    return len(set(recommended) & set(ground_truth)) / len(ground_truth)

def calculate_coverage(all_recommendations, total_items):
    """
    Measures what fraction of the total item catalog the system recommends.
    Low coverage = popularity bias (same items recommended to everyone).
    High coverage = system explores the catalog broadly.
    
    all_recommendations: list of recommendation lists, one per user
    total_items: total number of unique items in the catalog
    """
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    
    coverage = len(recommended_items) / total_items
    print(f"Catalog Coverage: {coverage:.4f} ({len(recommended_items)} unique items recommended out of {total_items})")
    return coverage