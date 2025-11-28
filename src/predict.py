import torch
import json

from config import K_RECS, DEVICE
from utils import get_dataset_path, load_data, generate_explanation, save_recommendations
from model import BPR

if __name__ == '__main__':
    dataset_path = get_dataset_path()
    _, brands, items, positives, user_ids, item_ids, user_to_idx, item_to_idx, num_users, num_items, user_positives = load_data(
        dataset_path)

    model = BPR(num_users, num_items)
    model.load_state_dict(torch.load('models/bpr_model.pth'))
    model.to(DEVICE)
    model.eval()

    recommendations = {}
    with torch.no_grad():
        user_embs = model.user_emb.weight
        item_embs = model.item_emb.weight
        for u_id, u_idx in user_to_idx.items():
            scores = user_embs[u_idx] @ item_embs.T
            user_pos = [item_to_idx[it] for it in user_positives[u_id]]
            scores[user_pos] = -float('inf')
            top_scores, top_idx = torch.topk(scores, K_RECS)
            rec_list = []
            for rank, (score, idx) in enumerate(zip(top_scores, top_idx), 1):
                item_id = item_ids[idx.item()]
                item_row = items[items['item_id'] == item_id].iloc[0]
                brand_id = item_row['brand_id']

                explanation = generate_explanation(u_id, item_id, positives, items, brands)

                rec = {
                    "item_id": item_id,
                    "brand_id": brand_id,
                    "score": score.item(),
                    "rank": rank,
                    "explanation": explanation
                }
                rec_list.append(rec)
            recommendations[str(u_id)] = rec_list  # user_id как str, если нужно

    save_recommendations(recommendations)