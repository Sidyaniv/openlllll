import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, LAMBDA_U, LAMBDA_I, LAMBDA_J, DEVICE
from utils import get_dataset_path, load_data, update_sorted_items, adaptive_sample_negative
from model import BPR


class BPRDataset(Dataset):
    def __init__(self, user_positives, user_ids):
        self.user_items = list(user_positives.items())
        self.user_ids = user_ids

    def __len__(self):
        return len(self.user_items)

    def __getitem__(self, idx):
        user, items = self.user_items[idx]
        i = random.choice(items)
        u_idx = user_to_idx[user]  # user_to_idx из load_data
        i_idx = item_to_idx[i]  # item_to_idx из load_data
        return u_idx, i_idx


if __name__ == '__main__':
    dataset_path = get_dataset_path()
    _, _, _, _, user_ids, _, user_to_idx, item_to_idx, num_users, num_items, user_positives = load_data(dataset_path)

    model = BPR(num_users, num_items)
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    dataset = BPRDataset(user_positives, user_ids)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        sorted_asc, sorted_desc = update_sorted_items(model)
        for u_batch, i_batch in loader:
            u = u_batch.to(DEVICE)
            i = i_batch.to(DEVICE)
            j = torch.zeros_like(u)
            for k in range(len(u)):
                u_k = u[k].item()
                i_k = i[k].item()
                positives_idx = [item_to_idx[it] for it in user_positives[user_ids[u_k]]]
                j[k] = adaptive_sample_negative(u_k, i_k, model, sorted_asc, sorted_desc, positives_idx)
            j = j.to(DEVICE)

            x = model(u, i, j)
            loss = -torch.log(torch.sigmoid(x)).mean()

            pu = model.user_emb(u)
            qi = model.item_emb(i)
            qj = model.item_emb(j)
            reg = LAMBDA_U * (pu ** 2).sum(1).mean() + LAMBDA_I * (qi ** 2).sum(1).mean() + LAMBDA_J * (qj ** 2).sum(
                1).mean()
            loss += reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS} completed")

    # Сохраняем модель
    torch.save(model.state_dict(), 'models/bpr_model.pth')
    print("Model trained and saved")