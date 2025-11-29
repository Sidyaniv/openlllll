import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pyarrow.dataset as ds
import random
import os

from config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE,
    LAMBDA_U, LAMBDA_I, LAMBDA_J, DEVICE
)
from model import BPR


def stream_events(dataset_root, domains=("retail", "marketplace", "offers")):

    for domain in domains:
        domain_path = os.path.join(dataset_root, domain, "events")
        if not os.path.exists(domain_path):
            continue

        dataset = ds.dataset(domain_path, format="parquet", partitioning="hive")
        scanner = dataset.scan()

        for batch in scanner.to_batches():
            # берём только нужные поля
            if "user_id" not in batch.column_names or "item_id" not in batch.column_names:
                continue
            users = batch["user_id"].to_pylist()
            items = batch["item_id"].to_pylist()

            for u, i in zip(users, items):
                yield u, i


def build_user_positives(dataset_root):
    user_pos = {}
    for u, i in stream_events(dataset_root):
        if u not in user_pos:
            user_pos[u] = set()
        user_pos[u].add(i)

    return user_pos

class BPRStreamDataset(Dataset):
    def __init__(self, user_positives, user_to_idx, item_to_idx):
        self.user_pos = user_positives
        self.users = list(user_positives.keys())
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos_items = list(self.user_pos[u])
        i = random.choice(pos_items)

        return (
            self.user_to_idx[u],
            self.item_to_idx[i]
        )



def sample_negative(user_idx, positives, num_items):
    while True:
        j = random.randint(0, num_items - 1)
        if j not in positives:
            return j



def train_bpr(dataset_root):
    print(">>> STREAMING dataset...")
    user_pos = build_user_positives(dataset_root)

    all_users = list(user_pos.keys())
    all_items = sorted({i for items in user_pos.values() for i in items})

    user_to_idx = {u: idx for idx, u in enumerate(all_users)}
    item_to_idx = {i: idx for idx, i in enumerate(all_items)}

    num_users = len(user_to_idx)
    num_items = len(item_to_idx)

    print(f"Users: {num_users}, Items: {num_items}")

    dataset = BPRStreamDataset(user_pos, user_to_idx, item_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BPR(num_users, num_items).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        for u_batch, i_batch in loader:
            u = u_batch.to(DEVICE)
            i = i_batch.to(DEVICE)

            j = torch.zeros_like(i)
            for k in range(len(u)):
                user = u[k].item()
                item_pos = {
                    item_to_idx[it] for it in user_pos[all_users[user]]
                }
                j[k] = sample_negative(user, item_pos, num_items)

            j = j.to(DEVICE)

            # === BPR loss ===
            x = model(u, i, j)
            bpr_loss = -torch.log(torch.sigmoid(x)).mean()

            # === regularization ===
            pu = model.user_emb(u)
            qi = model.item_emb(i)
            qj = model.item_emb(j)

            reg = (
                LAMBDA_U * (pu ** 2).sum(1).mean() +
                LAMBDA_I * (qi ** 2).sum(1).mean() +
                LAMBDA_J * (qj ** 2).sum(1).mean()
            )

            loss = bpr_loss + reg

            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"[Epoch {epoch+1}] loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "models/bpr_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    train_bpr("dataset")
