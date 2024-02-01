"""
asdf
"""
import albumentations as A
import gc
import math
import numpy as np
import os
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')


class CFG:
    AMP = True
    BATCH_SIZE_TRAIN = 32  # fix_me
    BATCH_SIZE_VALID = 32  # fix_me
    EPOCHS = 4  # fix_me
    FOLDS = 5  # fix_me
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1  # fix_me
    MAX_GRAD_NORM = 1e7  # fix_me
    MODEL = "tf_efficientnet_b0"  # fix_me
    NUM_FROZEN_LAYERS = 39  # fix_me
    NUM_WORKERS = 0   # multiprocessing.cpu_count()
    PRINT_FREQ = 20  # fix_me
    READ_EEG_SPEC_FILES = False
    READ_SPEC_FILES = False
    SEED = 20  # fix_me
    TRAIN_FULL_DATA = False
    VISUALIZE = False
    WEIGHT_DECAY = 0.01  # fix_me


# configuration
class PATHS:
    OUTPUT_DIR = "torch_efficientnet_b0_checkpoints/"
    PRE_LOADED_EEGS = 'brain_eeg_spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = 'brain_spectrograms/specs.npy'
    TRAIN_CSV = "data/train.csv"
    TRAIN_EEGS = "brain_eeg_spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "data/train_spectrograms/"


# utility function
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))


def get_logger(filename=PATHS.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    print("-" * 100)


target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]

label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}

num_to_label = {v: k for k, v in label_to_num.items()}
LOGGER = get_logger()
seed_everything(CFG.SEED)

# load data
df = pd.read_csv(PATHS.TRAIN_CSV)
label_cols = df.columns[-6:]
print(f"Train cataframe shape is: {df.shape}")
print(f"Labels: {list(label_cols)}")
print(df.head())

# data preprocessing
train_df = df.groupby('eeg_id')[
    ['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})

train_df.columns = ['spectogram_id', 'min']

aux = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({'spectrogram_label_offset_seconds': 'max'})

train_df['max'] = aux

aux = df.groupby('eeg_id')[['patient_id']].agg('first')
train_df['patient_id'] = aux

aux = df.groupby('eeg_id')[label_cols].agg('sum')
for label in label_cols:
    train_df[label] = aux[label].values

y_data = train_df[label_cols].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train_df[label_cols] = y_data

aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train_df['target'] = aux

train_df = train_df.reset_index()
print('Train non-overlapp eeg_id shape:', train_df.shape)
print(train_df.head())

# read train spectrograms
paths_spectograms = glob(PATHS.TRAIN_SPECTOGRAMS + "*.parquet")
print(f'There are {len(paths_spectograms)} spectrogram parquets')

if CFG.READ_SPEC_FILES:
    all_spectrograms = {}
    for file_path in tqdm(paths_spectograms):
        aux = pd.read_parquet(file_path)
        name = int(file_path.split("/")[-1].split('.')[0])
        all_spectrograms[name] = aux.iloc[:, 1:].values
        del aux
else:
    all_spectrograms = np.load(PATHS.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()


# read eeg spectrograms
paths_eegs = glob(PATHS.TRAIN_EEGS + "*.npy")
print(f'There are {len(paths_eegs)} EEG spectograms')

if CFG.READ_EEG_SPEC_FILES:
    all_eegs = {}
    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectogram = np.load(file_path)
        all_eegs[eeg_id] = eeg_spectogram
else:
    all_eegs = np.load(PATHS.PRE_LOADED_EEGS, allow_pickle=True).item()


# validation
gkf = GroupKFold(n_splits=CFG.FOLDS)  # fix_me
for fold, (train_index, valid_index) in enumerate(
        gkf.split(train_df, train_df.target, train_df.patient_id)):
    train_df.loc[valid_index, "fold"] = int(fold)

print(train_df.groupby('fold').size()), sep()
print(train_df.head())


# dataset
class CustomDataset(Dataset):
    def __init__(
            self, df: pd.DataFrame, config,
            augment: bool = False, mode: str = 'train',
            specs: Dict[int, np.ndarray] = all_spectrograms,
            eeg_specs: Dict[int, np.ndarray] = all_eegs
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.spectograms = all_spectrograms
        self.eeg_spectograms = eeg_specs

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __data_generation(self, index):
        """
        Generates data containing batch_size samples.
        """
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128, 256), dtype='float32')
        row = self.df.iloc[index]
        if self.mode == 'test':
            r = 0
        else:
            r = int((row['min'] + row['max']) // 4)

        for region in range(4):
            img = self.spectograms[row.spectogram_id][r:r + 300, region * 100:(region + 1) * 100].T

            # log transformation on spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # standarize per image
            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img - mu) / (std + ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg_spectograms[row.eeg_id]
            X[:, :, 4:] = img

            if self.mode != 'test':
                y = row[label_cols].values.astype(np.float32)

        return X, y

    def __transform(self, img):
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])
        return transforms(image=img)['image']


# train data set
train_dataset = CustomDataset(
    train_df,
    CFG,
    mode="train"
)

# train data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE_TRAIN,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

X, y = train_dataset[0]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# model
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate=0.1,  # fix_me
            drop_path_rate=0.2,  # fix_me
        )

        if config.FREEZE:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """
        # Get spectograms
        spectograms = [x[:, :, :, i:i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)

        # Get EEG spectograms
        eegs = [x[:, :, :, i:i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # Reshape (512, 512, 3)
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms

        x = torch.cat([x, x, x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x


# scheduler
BATCHES = len(train_loader)
steps = []
lrs = []
optim_lrs = []
model = CustomModel(CFG)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4  # fix_me
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,  # fix_me
    epochs=CFG.EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.05,  # fix_me
    anneal_strategy="cos",
    final_div_factor=100,  # fix_me
)

for epoch in range(CFG.EPOCHS):
    for batch in range(BATCHES):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
        steps.append(epoch * BATCHES + batch)

max_lr = max(lrs)
min_lr = min(lrs)
print(f"Maximum LR: {max_lr} | Minimum LR: {min_lr}")

# loss function, reduction = "batchmean"
criterion = nn.KLDivLoss(reduction="batchmean")
y_pred = F.log_softmax(torch.randn(2, 6, requires_grad=True), dim=1)
y_true = F.softmax(torch.rand(2, 6), dim=1)
print(f"Predictions: {y_pred}")
print(f"Targets: {y_true}")
output = criterion(y_pred, y_true)
print(f"Output: {output}")


# train and validation functions
def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0

    # iterate over train batches
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.AMP):
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if CFG.GRADIENT_ACCUMULATION_STEPS > 1:
                loss /= CFG.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.MAX_GRAD_NORM)

            if (step + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
                scheduler.step()
            end = time.time()

            # log info
            if step % CFG.PRINT_FREQ == 0 or step == (
                    len(train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch + 1, step, len(train_loader),
                              remain=time_since(start, float(step + 1) / len(
                                  train_loader)),
                              loss=losses,
                              grad_norm=grad_norm,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    prediction_dict = {}
    preds = []
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch",
              desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.no_grad():
                y_preds = model(X)
                loss = criterion(F.log_softmax(y_preds, dim=1), y)
            if CFG.GRADIENT_ACCUMULATION_STEPS > 1:
                loss /= CFG.GRADIENT_ACCUMULATION_STEPS
            losses.update(loss.item(), batch_size)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy())
            end = time.time()

            # LOG INFO
            if step % CFG.PRINT_FREQ == 0 or step == (
                    len(valid_loader) - 1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=time_since(start, float(step + 1) / len(
                                  valid_loader)),
                              loss=losses))

    prediction_dict["predictions"] = np.concatenate(preds)
    return losses.avg, prediction_dict


# train loop
def train_loop(df, fold):
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # split
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)

    # datasets
    train_dataset = CustomDataset(
        train_folds,
        CFG,
        mode="train",
        augment=True
    )

    valid_dataset = CustomDataset(
        valid_folds,
        CFG,
        mode="train",
        augment=False
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE_TRAIN,
        shuffle=False,  # fix_me
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.BATCH_SIZE_VALID,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    # instantiate model
    model = CustomModel(CFG)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.1,  # fix_me
        weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # fix_me
        epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # fix_me
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # loss
    criterion = nn.KLDivLoss(reduction="batchmean")

    best_loss = np.inf

    # iterate epochs
    for epoch in range(CFG.EPOCHS):
        start_time = time.time()

        # train
        avg_train_loss = train_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device
        )

        # evaluation
        avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)

        predictions = prediction_dict["predictions"]

        # scoring
        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_train_loss:.4f} avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(
                f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(
                {'model': model.state_dict(),
                'predictions': predictions},
                PATHS.OUTPUT_DIR + f"/{CFG.MODEL.replace('/', '_')}_fold_{fold}_best.pth"
            )

    predictions = torch.load(
        PATHS.OUTPUT_DIR + f"{CFG.MODEL.replace('/', '_')}_fold_{fold}_best.pth",
        map_location=torch.device('cpu'))['predictions']
    valid_folds[target_preds] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


# train full data
def train_loop_full_data(df):
    train_dataset = CustomDataset(
        df,
        CFG,
        mode="train",
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE_TRAIN,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    model = CustomModel(CFG)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.1,  # fix_me
        weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,  # fix_me
        epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # fix_me
        anneal_strategy="cos",
        final_div_factor=100,  # fix_me
    )

    criterion = nn.KLDivLoss(reduction="batchmean")
    best_loss = np.inf
    for epoch in range(CFG.EPOCHS):
        start_time = time.time()
        avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        elapsed = time.time() - start_time
        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f} time: {elapsed:.0f}s')
        torch.save(
            {'model': model.state_dict()},
            PATHS.OUTPUT_DIR + f"/{CFG.MODEL.replace('/', '_')}_epoch_{epoch}.pth")
    torch.cuda.empty_cache()
    gc.collect()
    # return _


# get result
def get_result(oof_df):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    preds = F.log_softmax(preds, dim=1)
    result = kl_loss(preds, labels)
    return result


if __name__ == "__main__":
    if not CFG.TRAIN_FULL_DATA:  # ie if False, train 5 folds and 5 epochs, show CV
        oof_df = pd.DataFrame()
        for fold in range(CFG.FOLDS):
            if fold in [0, 1, 2, 3, 4]:
                _oof_df = train_loop(train_df, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
                print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV: {get_result(oof_df)} ==========")
        oof_df.to_csv(PATHS.OUTPUT_DIR + '/oof_df.csv', index=False)
    else:  # ie if True
        train_loop_full_data(train_df)  # one fold & 5 epoch, no CV
