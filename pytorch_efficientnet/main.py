"""

"""
from functions import *

df, label_cols = load_data()  # load data

print(df.columns)

train_df, _ = preprocess_data(df, label_cols)  # preprocess data

gkf = GroupKFold(n_splits=CFG.FOLDS)  # validation

for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
    train_df.loc[valid_index, "fold"] = int(fold)

print(train_df.groupby('fold').size()), sep()
print(train_df.head())


# train data set
train_dataset = CustomDataset(
    df=train_df,
    config=CFG,
    label_cols=label_cols,
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
