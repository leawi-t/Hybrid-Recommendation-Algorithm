import os
import pickle
import numpy as np
import pandas as pd
from surprise import Dataset as SurpriseDataset
from surprise import Reader
from surprise import SVD


class SurpriseSVDModel:
    def __init__(self, algo, idx_to_user, idx_to_item):
        self.algo = algo
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item

    def predict(self, user_ids, item_ids, item_features=None):
        user_idxs = np.atleast_1d(user_ids)
        item_idxs = np.atleast_1d(item_ids)

        if user_idxs.size != 1:
            raise ValueError('Only a single user is supported per prediction call.')

        raw_user_id = self.idx_to_user[int(user_idxs[0])]
        scores = np.array([
            self.algo.predict(raw_user_id, self.idx_to_item[int(item_idx)]).est
            for item_idx in item_idxs
        ], dtype=float)

        return scores


class MappingDataset:
    def __init__(self, user_to_idx, item_to_idx):
        self._user_to_idx = user_to_idx
        self._item_to_idx = item_to_idx

    def mapping(self):
        return self._user_to_idx, None, self._item_to_idx, None


def train_hybrid_cf(train_path, tfidf_matrix, feature_item_ids):
    """
    Trains a collaborative filtering recommender using Surprise SVD;
    avoids LightFM and TruncatedSVD instability on Windows.
    """

    train_df = pd.read_csv(train_path, dtype={'user_id': str, 'item_id': str, 'rating': float})

    users = train_df['user_id'].unique().tolist()
    items = train_df['item_id'].unique().tolist()
    user_to_idx = {user_id: idx for idx, user_id in enumerate(users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
    idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
    idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}

    print(f"Training users: {len(users)} | Training items: {len(items)}")
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    reader = Reader(rating_scale=(train_df['rating'].min(), train_df['rating'].max()))
    surprise_data = SurpriseDataset.load_from_df(
        train_df[['user_id', 'item_id', 'rating']], reader
    )
    trainset = surprise_data.build_full_trainset()

    algo = SVD(
        n_factors=50,          # Give the model more capacity to learn detailed nuances
        reg_all=0.1,           # Increase baseline regularization across the board
        biased=True,           
        random_state=42, 
        verbose=False
    )
    
    print('Training Surprise SVD collaborative model...')
    algo.fit(trainset)
    print('Surprise SVD model training complete!')

    model = SurpriseSVDModel(algo=algo, idx_to_user=idx_to_user, idx_to_item=idx_to_item)
    dataset = MappingDataset(user_to_idx=user_to_idx, item_to_idx=item_to_idx)

    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/hybrid_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('data/processed/dataset_mapping.pkl', 'wb') as f:
        pickle.dump(dataset, f)

    return model, dataset, None
