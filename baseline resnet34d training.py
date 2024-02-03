"""

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import random
import warnings


warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device:{device}")


model = timm.create_model(
    'resnet34d',
    pretrained=True,
    num_classes=6,
    in_chans=1
)


# config
class CFG:
    SEED = 2024
    IMAGE_TRANSFORM = transforms.Resize((512, 512))
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LABELS = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    FEATS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']


def seed_everything(seed):
    """

    :param seed:
    :return:
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def KL_loss(p, q):
    """
    loss function
    :param p:
    :param q:
    :return:
    """
    epsilon = 10 ** (-15)
    p = torch.clip(p, epsilon, 1 - epsilon)
    q = nn.functional.log_softmax(q, dim=1)
    return torch.mean(torch.sum(p * (torch.log(p) - q), dim=1))


def get_batch(paths, batch_size=CFG.BATCH_SIZE):
    """
    get batch data
    :param paths:
    :param batch_size:
    :return:
    """
    eps = 1e-6
    batch_data = []
    for path in paths:
        data = pd.read_parquet(path[0])
        data = data.fillna(-1).values[:, 1:].T
        data = data[:, 0:300]
        data = np.clip(data, np.exp(-6), np.exp(10))
        data = np.log(data)
        data_mean = data.mean(axis=(0, 1))
        data_std = data.std(axis=(0, 1))
        data = (data-data_mean)/(data_std+eps)
        data_tensor = torch.unsqueeze(torch.Tensor(data), dim=0)
        data = CFG.IMAGE_TRANSFORM(data_tensor)
        batch_data.append(data)
    batch_data = torch.stack(batch_data)
    return batch_data


seed_everything(CFG.SEED)

# labels
train_df = pd.read_csv("data/train.csv")
train_df = train_df[train_df['eeg_sub_id'] == 0]

for label in CFG.LABELS:
    group = train_df[f'{label}_vote'].groupby(train_df['spectrogram_id']).sum()
    label_vote_sum = pd.DataFrame(
        {'spectrogram_id': group.index, f'{label}_vote_sum': group.values})
    if label == 'seizure':
        train_feats = label_vote_sum
    else:
        train_feats = train_feats.merge(
            label_vote_sum,
            on='spectrogram_id',
            how='left'
        )

train_feats['total_vote'] = 0

for label in CFG.LABELS:
    train_feats['total_vote'] += train_feats[f'{label}_vote_sum']

for label in CFG.LABELS:
    train_feats[f'{label}_vote'] = train_feats[f'{label}_vote_sum'] / train_feats['total_vote']

choose_cols = ['spectrogram_id']

for label in CFG.LABELS:
    choose_cols += [f'{label}_vote']

train_feats = train_feats[choose_cols]

train_feats['path'] = train_feats['spectrogram_id'].apply(
    lambda x: "data/train_spectrograms/" + str(x) + ".parquet")

train_feats.head()

# test train street
split = 0.8
random_num = np.arange(len(train_feats))
np.random.shuffle(random_num)
split_num = int(len(train_feats) * split)
train_idx = random_num[:split_num]
test_idx = random_num[split_num:]

print(f"train_idx")

for label in CFG.LABELS:
    print(f"{label}:{train_feats.iloc[train_idx][label+'_vote'].sum()}")

print(f"test_idx")

for label in CFG.LABELS:
    print(f"{label}:{train_feats.iloc[test_idx][label+'_vote'].sum()}")

# model training
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.5, 0.999),
    weight_decay=0.01
)

model.to(device)
train_losses = []
test_losses = []


if __name__ == "__main__":
    print(f"start")
    for epoch in range(CFG.NUM_EPOCHS):
        print(f"epoch {epoch}:")
        model.train()
        train_loss = []
        random_num = np.arange(len(train_idx))
        np.random.shuffle(random_num)
        train_idx = train_idx[random_num]

        for idx in range(0, len(train_idx), CFG.BATCH_SIZE):
            optimizer.zero_grad()
            train_idx1 = train_idx[idx: idx + CFG.BATCH_SIZE]
            train_X1_path = train_feats[['path']].iloc[train_idx1].values
            train_X1 = get_batch(train_X1_path, batch_size=CFG.BATCH_SIZE)
            train_y1 = train_feats[CFG.FEATS].iloc[train_idx1].values
            train_y1 = torch.Tensor(train_y1)
            train_pred = model(train_X1.to(device))
            loss = KL_loss(train_y1.to(device), train_pred.to(device)).to(
                device)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

        train_loss = np.mean(np.array(train_loss))
        print(f"train_loss: {train_loss}")
        test_loss = []
        model.eval()
        with torch.no_grad():
            for idx in range(0, len(test_idx), CFG.BATCH_SIZE):
                test_idx1 = test_idx[idx:idx + CFG.BATCH_SIZE]
                test_X1_path = train_feats[['path']].iloc[test_idx1].values
                test_X1 = get_batch(test_X1_path, batch_size=CFG.BATCH_SIZE)
                test_y1 = train_feats[CFG.FEATS].iloc[test_idx1].values
                test_y1 = torch.Tensor(test_y1)
                test_pred = model(test_X1.to(device))
                loss = KL_loss(test_y1.to(device), test_pred.to(device)).to(
                    device)
                test_loss.append(loss.detach().cpu().numpy())
        test_loss = np.mean(np.array(test_loss))
        print(f"test_loss:{test_loss}")
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        print("-" * 50)

        plt.title("train_losses VS test_losses")
        epochs = [i for i in range(len(train_losses))]

        plt.plot(
            epochs,
            train_losses,
            marker="o",
            markersize=1,
            label="train_losses"
        )

        plt.plot(
            epochs,
            test_losses,
            marker="x",
            markersize=1,
            label="test_losses"
        )

        plt.legend()
        plt.show()

    torch.save(model.to('cpu'), "resnet_models/HMS_resnet.pth")
