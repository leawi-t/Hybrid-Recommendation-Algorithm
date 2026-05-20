import pandas as pd
import numpy as np
import gzip
import json
from src.config import RANDOM_SEED

def load_and_clean_data(file_path, min_user_interactions=5, min_item_interactions=5):
    """
    Loads raw interaction data as STRINGS to ensure zero-padding matches metadata.
    """
    df = pd.read_csv(file_path, 
                     names=['user_id', 'item_id', 'rating', 'timestamp'], 
                     dtype={'user_id': str, 'item_id': str, 'rating': float})
    
    df = df.dropna(subset=['user_id', 'item_id', 'rating'])
    print(f"Initial structural shape: {df.shape[0]} rows")
    
    while True:
        initial_shape = df.shape[0]
        user_counts = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= min_user_interactions].index)]
        
        item_counts = df['item_id'].value_counts()
        df = df[df['item_id'].isin(item_counts[item_counts >= min_item_interactions].index)]
        
        if df.shape[0] == initial_shape:
            break
            
    print(f"Cleaned structural shape: {df.shape[0]} rows")
    return df

def load_and_merge_metadata(ratings_df, meta_path):
    """
    Streams metadata efficiently, grabbing text records for active items.
    """
    valid_items = set(ratings_df['item_id'].unique())
    records = {}

    print("Streaming compressed metadata file...")
    with gzip.open(meta_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                asin = str(data.get("asin", "")).strip()

                if asin in valid_items:
                    title = data.get("title", "")
                    categories = " ".join([
                        str(c)
                        for sub in data.get("categories", [])
                        for c in (sub if isinstance(sub, list) else [sub])
                    ])
                    desc = " ".join(data.get("description", [])) if isinstance(data.get("description", []), list) else data.get("description", "")
                    combined_text = f"{title} {categories} {desc}".strip()

                    if asin in records:
                        if combined_text and combined_text not in records[asin]:
                            records[asin] = f"{records[asin]} {combined_text}".strip()
                    else:
                        records[asin] = combined_text
            except Exception:
                continue  # Skip corrupt lines if any

    meta_df = pd.DataFrame(
        [{"item_id": item_id, "combined_text": text.strip()} for item_id, text in records.items()]
    )
    print(f"Successfully matched metadata for {len(meta_df)} / {len(valid_items)} items")
    return meta_df

def calculate_sparsity(df):
    """
    Computes the matrix sparsity percentage to quantify the 'Sparse Matrix' problem.
    """
    unique_users = df['user_id'].nunique()
    unique_items = df['item_id'].nunique()
    total_interactions = len(df)
    
    possible_interactions = unique_users * unique_items
    
    if possible_interactions == 0:
        return 100.0
        
    sparsity = (1 - (total_interactions / possible_interactions)) * 100
    
    print(f"Unique Users: {unique_users} | Unique Items: {unique_items}")
    print(f"Matrix Sparsity: {sparsity:.2f}%")
    return sparsity

def isolate_cold_start(df, cold_ratio=0.1):
    """
    Simulates cold-start by isolating both cold users and cold items.
    """
    np.random.seed(42)

    # Cold ITEMS — items with no training history
    unique_items = df['item_id'].unique()
    cold_items = np.random.choice(unique_items, size=int(len(unique_items) * cold_ratio), replace=False)

    # Cold USERS — users held out entirely from training
    unique_users = df['user_id'].unique()
    cold_users = np.random.choice(unique_users, size=int(len(unique_users) * cold_ratio), replace=False)

    # Train: remove all cold users and cold items
    train_df = df[~df['item_id'].isin(cold_items) & ~df['user_id'].isin(cold_users)]

    # Test cold: the hidden interactions for evaluation
    cold_test_df = df[df['item_id'].isin(cold_items) | df['user_id'].isin(cold_users)]

    print(f"Cold-start items: {len(cold_items)} | Cold-start users: {len(cold_users)}")
    print(f"Train size: {len(train_df)} | Cold test size: {len(cold_test_df)}")
    return train_df, cold_test_df

def train_test_split_interactions(df, test_ratio=0.2):
    """
    For each user, holds out a fraction of their interactions as the test set.
    This is better than a random row split because every user appears in both sets.
    """
    train_list = []
    test_list = []

    for user_id, group in df.groupby('user_id'):
        group = group.sample(frac=1, random_state=42)  # shuffle
        split = max(1, int(len(group) * test_ratio))
        test_list.append(group.iloc[:split])
        train_list.append(group.iloc[split:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    print(f"Train interactions: {len(train_df)} | Test interactions: {len(test_df)}")
    return train_df, test_df