import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_item_features_real_meta(meta_path):
    """
    Vectorizes raw metadata textual assets to protect against Cold Starts.
    """
    meta_df = pd.read_csv(meta_path, dtype={'item_id': str})

    if meta_df['item_id'].duplicated().any():
        print(f"Found {meta_df['item_id'].duplicated().sum()} duplicate metadata rows. Deduplicating by item_id.")
        meta_df = (
            meta_df.groupby('item_id', as_index=False)
            .agg({'combined_text': ' '.join})
        )

    # Fill empty text strings to avoid vectorizer crashes
    meta_df['combined_text'] = meta_df['combined_text'].fillna('electronics product product')
    
    # Restrict dictionary vocabulary size to 100 top words for speed
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(meta_df['combined_text'])
    
    print(f"Generated Vocabulary TF-IDF shape: {tfidf_matrix.shape}")
    
    meta_lookup = dict(zip(meta_df['item_id'], meta_df['combined_text']))
    return tfidf_matrix, meta_df['item_id'].tolist(), meta_lookup

