"""

"""
from config import *
from model import *
from dataset import *
from utils import *


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
    valid_folds[CFG.TARGET_PREDS] = predictions

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
    preds = torch.tensor(oof_df[CFG.TARGET_PREDS].values)  # convert target preds to tensor
    preds = F.log_softmax(preds, dim=1)  # apply log softmax to predictions
    result = kl_loss(preds, labels)  #
    return result   #