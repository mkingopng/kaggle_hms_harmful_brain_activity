"""
asdf
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc
from glob import glob
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import GroupKFold
import time
import timm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List

# set device to gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')


# configuration
class CFG:
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    DROP_RATE = 0.1
    DROP_PATH_RATE = 0.2
    EPOCHS = 4
    FINAL_DIV_FACTOR = 100
    FOLDS = 5
    FREEZE = False
    GRADIENT_ACCUMULATION_STEPS = 1
    MAX_GRAD_NORM = 1e-7
    MODEL = "tf_efficientnet_b0"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = multiprocessing.cpu_count()
    OPTIMIZER_LR = 0.1
    PCT_START = 0.05
    PRINT_FREQ = 20
    SCHEDULER_MAX_LR = 1e-3
    SEED = 20
    VERSION = 1
    WEIGHT_DECAY = 0.01


class BOOLS:
    AMP = True
    READ_EEG_SPEC_FILES = False
    READ_SPEC_FILES = False  #
    TRAIN_FULL_DATA = True  #
    VISUALIZE = False


class PATHS:
    OUTPUT_DIR = "resnet34d_1fold/"
    PRE_LOADED_EEGS = 'brain_eeg_spectrograms/eeg_specs.npy'
    PRE_LOADED_SPECTOGRAMS = 'brain_spectrograms/specs.npy'
    TRAIN_CSV = "data/train.csv"
    TRAIN_EEGS = "brain_eeg_spectrograms/EEG_Spectrograms/"
    TRAIN_SPECTOGRAMS = "data/train_spectrograms/"


# utilities
class AverageMeter(object):
    """
    computes and stores the average and current value of metrics over time
    :param val (float): The current value.
    :param avg (float): The average of all updates.
    :param sum (float): The sum of all values updated.
    :param count (int): The number of updates.
    """

    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        """reset all values to 0"""
        self.val: float = 0
        self.avg: float = 0
        self.sum: float = 0
        self.count: int = 0

    def update(self, val, n=1):
        """
        update the metrics with a new value.

        :param val (float): The value to add.
        :param n (int): The weight of update, the batch size

        Returns:
            float: The updated average.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s: float):
    """
    convert to minutes
    :param s:
    :return:
    """
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since: float, percent: float):
    """
    time delta
    :param since:
    :param percent:
    :return: time delta
    """
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))  # return time delta


def get_logger(filename=PATHS.OUTPUT_DIR):
    """
    logging function
    :param filename:
    :return:
    """
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename} v{CFG.VERSION}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed: int):
    """
    seed everything
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def sep():
    """prints a separator line"""
    print("-" * 100)


target_preds = [x + "_pred" for x in [
    'seizure_vote',
    'lpd_vote',
    'gpd_vote',
    'lrda_vote',
    'grda_vote',
    'other_vote'
]]

label_to_num = {
    'Seizure': 0,
    'LPD': 1,
    'GPD': 2,
    'LRDA': 3,
    'GRDA': 4,
    'Other': 5
}

num_to_label = {v: k for k, v in label_to_num.items()}
LOGGER = get_logger()
seed_everything(CFG.SEED)

# load data
df = pd.read_csv(PATHS.TRAIN_CSV)
label_cols = df.columns[-6:]
print(f"Train dataframe shape is: {df.shape}")
print(f"Labels: {list(label_cols)}")
print(df.head())

# data preprocessing
train_df = df.groupby('eeg_id')[
    ['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})

train_df.columns = ['spectogram_id', 'min']

aux = df.groupby('eeg_id')[
    ['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
    'spectrogram_label_offset_seconds': 'max'}
)

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

if BOOLS.READ_SPEC_FILES:
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

if BOOLS.READ_EEG_SPEC_FILES:
    all_eegs = {}
    for file_path in tqdm(paths_eegs):
        eeg_id = file_path.split("/")[-1].split(".")[0]
        eeg_spectogram = np.load(file_path)
        all_eegs[eeg_id] = eeg_spectogram
else:
    all_eegs = np.load(PATHS.PRE_LOADED_EEGS, allow_pickle=True).item()


# validation
gkf = GroupKFold(n_splits=CFG.FOLDS)
for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
    train_df.loc[valid_index, "fold"] = int(fold)

print(train_df.groupby('fold').size()), sep()
print(train_df.head())


# dataset
class CustomDataset(Dataset):
    """
    Custom dataset
    """
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
        denotes the number of batches per epoch.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        generates one sample of data
        :param index:
        :return:
        """
        X, y = self.__data_generation(index)
        if self.augment:
            X = self.__transform(X)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def __data_generation(self, index):
        """
        generates data containing batch_size samples
        :param index:
        :return:
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
            img = self.spectograms[row.spectogram_id][r: r + 300, region * 100:(region + 1) * 100].T

            # log transformation on spectogram
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            # standarize per image
            ep = 1e-6  # to avoid division by zero
            mu = np.nanmean(img.flatten())  # mean
            std = np.nanstd(img.flatten())  # std dev
            img = (img - mu) / (std + ep)  # standarize image
            img = np.nan_to_num(img, nan=0.0)  # set nan to 0
            X[14:-14, :, region] = img[:, 22:-22] / 2.0  # crop and downsample
            img = self.eeg_spectograms[row.eeg_id]  #
            X[:, :, 4:] = img  #

            if self.mode != 'test':
                y = row[label_cols].values.astype(np.float32)  # get labels

        return X, y  # return spectogram and labels

    def __transform(self, img):
        """
        apply data augmentation
        :param img:
        :return:
        """
        transforms = A.Compose([A.HorizontalFlip(p=0.5)])  # img horizontal flip
        # todo: add more transforms
        return transforms(image=img)['image']  # return transformed image


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

X, y = train_dataset[0]  # get first sample

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# model
class CustomModel(nn.Module):
    def __init__(self, config, num_classes: int = 6, pretrained: bool = True):
        """
        custom model using pretrained spine and 4 custom layers
        :param config:
        :param num_classes:
        :param pretrained:
        """
        super(CustomModel, self).__init__()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            config.MODEL,
            pretrained=pretrained,
            drop_rate=CFG.DROP_RATE,
            drop_path_rate=CFG.DROP_PATH_RATE,
        )

        if config.FREEZE:
            for i, (name, param) in enumerate(list(self.model.named_parameters())[0:config.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes),
            # todo: add more custom layers or improve this
        )

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image
        :param x:
        :return:
        """
        # get spectograms
        spectograms = [x[:, :, :, i:i + 1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)

        # get EEG spectograms
        eegs = [x[:, :, :, i:i + 1] for i in range(4, 8)]
        eegs = torch.cat(eegs, dim=1)

        # reshape (512, 512, 3)
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
        """
        forward pass
        :param x:
        :return:
        """
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
    lr=CFG.OPTIMIZER_LR
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=CFG.SCHEDULER_MAX_LR,
    epochs=CFG.EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=CFG.PCT_START,
    anneal_strategy="cos",
    final_div_factor=CFG.FINAL_DIV_FACTOR,
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
    """
    train epoch
    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param scheduler:
    :param device:
    :return:
    """
    model.train()
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=BOOLS.AMP)
    losses = AverageMeter()
    start = time.time()
    global_step = 0

    # iterate over train batches
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = X.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            with torch.cuda.amp.autocast(enabled=BOOLS.AMP):
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
                # scheduler.step()
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
    scheduler.step()  # Call scheduler.step() here, after completing all batches for the epoch
    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    """
    validation epoch
    :param valid_loader:
    :param model: efficientnet b0
    :param criterion: KL divergence
    :param device: gpu if available else cpu
    :return: return average loss and predictions
    """
    model.eval()  # switch to evaluation mode
    softmax = nn.Softmax(dim=1)  # softmax for predictions
    losses = AverageMeter()  # average loss
    prediction_dict = {}  # empty dictionary to store predictions
    preds = []  # empty list to store predictions
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch",
              desc='Validation') as tqdm_valid_loader:  #
        for step, (X, y) in enumerate(tqdm_valid_loader):  # iterate over valid batches
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

            # log info
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
    return losses.avg, prediction_dict  # return average loss and predictions


# train loop
def train_loop(df, fold):
    """
    train loop
    :param df:
    :param fold:
    :return:
    """
    LOGGER.info(f"========== Fold: {fold} training ==========")

    # split
    train_folds = df[df['fold'] != fold].reset_index(drop=True)  # train folds
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)  # validation folds

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
        shuffle=True,
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
        lr=CFG.OPTIMIZER_LR,
        weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=CFG.SCHEDULER_MAX_LR,
        epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=CFG.PCT_START,
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
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(
                {'model': model.state_dict(),
                'predictions': predictions},
                PATHS.OUTPUT_DIR + f"/ver_{CFG.VERSION}_{CFG.MODEL.replace('/', '_')}_fold_{fold}_best.pth"
            )

    predictions = torch.load(
        PATHS.OUTPUT_DIR + f"ver_{CFG.VERSION}_{CFG.MODEL.replace('/', '_')}_fold_{fold}_best.pth",
        map_location=torch.device('cpu'))['predictions']
    valid_folds[target_preds] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds


# train full data
def train_loop_full_data(df):
    """
    train loop full data
    :param df:
    :return:
    """
    train_dataset = CustomDataset(
        df,
        CFG,
        mode="train",
        augment=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    model = CustomModel(CFG)

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.OPTIMIZER_LR,
        weight_decay=CFG.WEIGHT_DECAY
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=CFG.OPTIMIZER_LR,
        epochs=CFG.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=CFG.PCT_START,
        anneal_strategy="cos",
        final_div_factor=CFG.FINAL_DIV_FACTOR,
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
            PATHS.OUTPUT_DIR + f"/ver_{CFG.VERSION}_{CFG.MODEL.replace('/', '_')}_epoch_{epoch}.pth")
    torch.cuda.empty_cache()
    gc.collect()
    # return _


# get result
def get_result(oof_df):
    """

    :param oof_df:
    :return:
    """
    kl_loss = nn.KLDivLoss(reduction="batchmean")  # KL divergence loss
    labels = torch.tensor(oof_df[label_cols].values)  # covert label_cols to tensor
    preds = torch.tensor(oof_df[target_preds].values)  # convert target preds to tensor
    preds = F.log_softmax(preds, dim=1)  # apply log softmax to predictions
    result = kl_loss(preds, labels)  #
    return result   #


if __name__ == "__main__":
    if not BOOLS.TRAIN_FULL_DATA:
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
    else:
        train_loop_full_data(train_df)
