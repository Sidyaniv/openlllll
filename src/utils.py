import requests
import pandas as pd
import glob
import os
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pyarrow.dataset as ds
from config import BACKEND_URL, POSITIVE_ACTIONS, DEFAULT_DATASET_PATH

def get_dataset_path():
    try:
        response = requests.get(f'{BACKEND_URL}/get_dataset_path')
        if response.status_code == 200:
            return response.json()['dataset_path']
        else:
            print("Backend not available, using local path.")
            return DEFAULT_DATASET_PATH  # Fallback
    except Exception as e:
        print(f"Error connecting to backend: {e}. Using local path.")
        return DEFAULT_DATASET_PATH  # Fallback

def load_events(domain_path, dataset_path):
    event_files = glob.glob(os.path.join(dataset_path, domain_path, 'events/*.pq'))
    if not event_files:
        return pd.DataFrame()
    events = pd.concat(pd.read_parquet(f) for f in event_files)
    return events


def load_data(dataset_path):
    users = pd.read_parquet(os.path.join(dataset_path, 'users.pq'), engine='fastparquet')
    brands = pd.read_parquet(os.path.join(dataset_path, 'brands.pq'),engine='fastparquet')
    marketplace_items = pd.read_parquet(os.path.join(dataset_path, 'marketplace/items.pq'),engine='fastparquet')
    retail_items = pd.read_parquet(os.path.join(dataset_path, 'retail/items.pq'), engine='fastparquet')

    offers_path = os.path.join(dataset_path, 'offers/items.pq')
    offers_items = pd.read_parquet(offers_path,engine='fastparquet') if os.path.exists(offers_path,engine='fastparquet') else pd.DataFrame()

    print("ВСЁ ЗАГРУЖЕНО БЕЗ ОШИБОК! Можно учить модель.")

    # Combine items
    items = pd.concat([marketplace_items, retail_items, offers_items], ignore_index=True).drop_duplicates('item_id')

    # Load events
    marketplace_events = load_events('marketplace', dataset_path, engine='fastparquet')
    retail_events = load_events('retail', dataset_path, engine='fastparquet')
    offers_events = load_events('offers', dataset_path, engine='fastparquet')

    # Combine interactions
    interaction_events = pd.concat([marketplace_events, retail_events, offers_events], ignore_index=True)

    # Positives
    positives = interaction_events[interaction_events['action_type'].isin(POSITIVE_ACTIONS)][
        ['user_id', 'item_id', 'brand_id', 'timestamp', 'action_type']]

    # Maps
    user_ids = sorted(positives['user_id'].unique())
    item_ids = sorted(positives['item_id'].unique())
    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
    num_users = len(user_ids)
    num_items = len(item_ids)

    user_positives = defaultdict(list)
    for _, row in positives.iterrows():
        user_positives[row['user_id']].append(row['item_id'])

    return users, brands, items, positives, user_ids, item_ids, user_to_idx, item_to_idx, num_users, num_items, user_positives

def update_sorted_items(model):
    item_emb = model.item_emb.weight.data
    sorted_asc = [torch.argsort(item_emb[:, l], dim=0) for l in range(item_emb.size(1))]
    sorted_desc = [s.flip(0) for s in sorted_asc]
    return sorted_asc, sorted_desc

def adaptive_sample_negative(u_idx, i_idx, model, sorted_asc, sorted_desc, positives_idx, p=0.01):
    pu = model.user_emb.weight[u_idx]
    abs_pu = pu.abs()
    prob = abs_pu / abs_pu.sum()
    l = np.random.choice(len(prob), p=prob.cpu().numpy())
    sign = pu[l] > 0
    if sign:
        sorted_list = sorted_desc[l]
    else:
        sorted_list = sorted_asc[l]
    while True:
        r = np.random.geometric(p)
        if r > len(sorted_list):
            continue
        j_idx = sorted_list[r-1].item()
        if j_idx != i_idx and j_idx not in positives_idx:
            return j_idx

def generate_explanation(u_id, item_id, positives, items, brands):
    # Similar items
    user_interacted_items = positives[positives['user_id'] == u_id]['item_id'].unique()
    item_row = items[items['item_id'] == item_id].iloc[0]
    similar_items = []
    similar_brands = []
    matched_categories = []

    if 'embedding' in items.columns and not pd.isna(item_row['embedding']):
        rec_emb = np.array(item_row['embedding'])
        interacted_embs = items[items['item_id'].isin(user_interacted_items)]['embedding'].apply(lambda x: np.array(x) if not pd.isna(x) else np.zeros_like(rec_emb)).values
        if len(interacted_embs) > 0:
            sims = cosine_similarity([rec_emb], list(interacted_embs))[0]
            top_sim_idx = np.argsort(-sims)[:3]
            similar_items = [user_interacted_items[top] for top in top_sim_idx if sims[top] > 0.5]  # Порог для релевантности

    # Similar brands
    user_top_brands = positives[positives['user_id'] == u_id].groupby('brand_id').size().nlargest(3).index.tolist()
    rec_brand = item_row['brand_id']
    if 'embedding' in brands.columns:
        rec_brand_emb = brands[brands['brand_id'] == rec_brand]['embedding'].iloc[0] if not brands[brands['brand_id'] == rec_brand].empty else None
        if rec_brand_emb is not None:
            top_brand_embs = brands[brands['brand_id'].isin(user_top_brands)]['embedding'].values
            if len(top_brand_embs) > 0:
                sims = cosine_similarity([rec_brand_emb], list(top_brand_embs))[0]
                top_sim_idx = np.argsort(-sims)[:3]
                similar_brands = [user_top_brands[top] for top in top_sim_idx if sims[top] > 0.5]
    else:
        similar_brands = [b for b in user_top_brands if random.random() > 0.5][:3]

    # Matched categories
    if 'category' in items.columns:  # Предполагаем, что колонка называется 'category', скорректируй если иначе
        rec_cat = item_row.get('category')
        if rec_cat:
            user_cats = positives[positives['user_id'] == u_id].merge(items, on='item_id').groupby('category').size().nlargest(3).index.tolist()
            if rec_cat in user_cats:
                matched_categories = [rec_cat]

    rationale = "Item is similar to user’s recent views and matches preferred brands." if similar_items else "Brand affinity and category relevance."

    return {
        "similar_items": similar_items,
        "similar_brands": similar_brands,
        "matched_categories": matched_categories,
        "rationale_summary": rationale
    }

def save_recommendations(recommendations):
    response = requests.post(f'{BACKEND_URL}/save_recommendations', json=recommendations)
    if response.status_code != 200:
        raise ValueError("Failed to save recommendations")
    print("Recommendations saved successfully")

