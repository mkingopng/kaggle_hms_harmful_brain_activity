"""
asdf
"""
import os
import gc
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as albu
from sklearn.model_selection import GroupKFold
import efficientnet.tfkeras as efn
import cv2

print('TensorFlow version =', tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CFG:
    """
    asdf
    """
    MIX = True  # use mixed precision
    VER = 5
    USE_KAGGLE_SPECTROGRAMS = True
    USE_EEG_SPECTROGRAMS = True
    LOAD_MODELS_FROM = 'brain_efficientnet_models_v3_v4_v5/'  # if this equals none, then we train new models, if this equals disk path, then we load previously trained models
    BATCH = 128
    PATH = 'data/train_spectrograms/'
    READ_EEG_SPEC_FILES = False
    TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
    TARS2 = {x: y for y, x in TARS.items()}
    NSPLITS = 5
    READ_SPEC_FILES = False


# use multiple gpus
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

# mixed or full precision
if CFG.MIX:
    tf.config.optimizer.set_experimental_options(
        {"auto_mixed_precision": True}
    )
    print('Mixed precision enabled')
else:
    print('Using full precision')

df = pd.read_csv('data/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape)
print('Targets', list(TARGETS))


train = df.groupby('eeg_id')[[
    'spectrogram_id',
    'spectrogram_label_offset_seconds'
]].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})

train.columns = ['spec_id', 'min']

tmp = df.groupby('eeg_id')[[
    'spectrogram_id',
    'spectrogram_label_offset_seconds'
]].agg({
    'spectrogram_label_offset_seconds': 'max'
})

train['max'] = tmp

tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp

tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape)
train.head()

# read all spectrograms
files = os.listdir(CFG.PATH)
print(f'There are {len(files)} spectrogram parquets')

if CFG.READ_SPEC_FILES:
    spectrograms = {}
    for i, f in enumerate(files):
        if i % 100 == 0:
            print(i, ', ', end='')
        tmp = pd.read_parquet(f'{CFG.PATH}{f}')
        name = int(f.split('.')[0])
        spectrograms[name] = tmp.iloc[:, 1:].values
else:
    spectrograms = np.load(
        'brain_spectrograms/specs.npy',
        allow_pickle=True
    ).item()

###########
if CFG.READ_EEG_SPEC_FILES:
    all_eegs = {}
    for i, e in enumerate(train.eeg_id.values):
        if i % 100 == 0:
            print(i, ', ', end='')
        x = np.load(f'brain_eeg_spectrograms/EEG_Spectrograms/{e}.npy')
        all_eegs[e] = x
else:
    all_eegs = np.load(
        'brain_eeg_spectrograms/eeg_specs.npy',
        allow_pickle=True
    ).item()

print(f'There are {len(all_eegs)} eeg parquets')


# data loader
class DataGenerator(tf.keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(
            self, data, batch_size=32, shuffle=False, augment=False,
            mode='train',
            specs=spectrograms, eeg_specs=all_eegs
            ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.mode = mode
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return:
        """
        ct = int(np.ceil(len(self.data) / self.batch_size))
        return ct

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        if self.augment:
            X = self.__augment_batch(X)
        return X, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        :return:
        """
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        :param indexes:
        :return:
        """

        X = np.zeros((len(indexes), 128, 256, 8), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')
        img = np.ones((128, 256), dtype='float32')

        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            if self.mode == 'test':
                r = 0
            else:
                r = int((row['min'] + row['max']) // 4)

            for k in range(4):
                # extract 300 rows of spectrogram
                img = self.specs[row.spec_id][r: r + 300,
                      k * 100: (k + 1) * 100].T

                # log transform spectrogram
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)

                # standardize per image
                ep = 1e-6
                m = np.nanmean(img.flatten())
                s = np.nanstd(img.flatten())
                img = (img - m) / (s + ep)
                img = np.nan_to_num(img, nan=0.0)

                # crop to 256 time steps
                X[j, 14: -14, :, k] = img[:, 22: -22] / 2.0

            # eeg spectrograms
            img = self.eeg_specs[row.eeg_id]
            X[j, :, :, 4:] = img

            if self.mode != 'test':
                y[j,] = row[TARGETS]

        return X, y

    def __random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
            # albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,fill_value=0,p=0.5),
        ])
        return composition(image=img)['image']

    def __augment_batch(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self.__random_transform(img_batch[i,])
        return img_batch


# build grad cam
def build_cam_model(pretrain=None):
    """

    :param pretrain:
    :return:
    """
    inp = tf.keras.Input(shape=(128, 256, 8))
    base_model = efn.EfficientNetB0(include_top=False, weights=None,
                                    input_shape=None)
    if pretrain:
        base_model.load_weights('tf_efficientnet_imagenet_weights/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

    # reshape input 128 x 256 x 8 => 512 x 512 x 3 monotone image
    # kaggle spectrograms
    x1 = [inp[:, :, :, i: i + 1] for i in range(4)]
    x1 = tf.keras.layers.Concatenate(axis=1)(x1)

    # eeg spectrograms
    x2 = [inp[:, :, :, i + 4:i + 5] for i in range(4)]
    x2 = tf.keras.layers.Concatenate(axis=1)(x2)

    # make 512 x 512 x 3
    if CFG.USE_KAGGLE_SPECTROGRAMS & CFG.USE_EEG_SPECTROGRAMS:
        x = tf.keras.layers.Concatenate(axis=2)([x1, x2])
    elif CFG.USE_EEG_SPECTROGRAMS:
        x = x2
    else:
        x = x1
    x = tf.keras.layers.Concatenate(axis=3)([x, x, x])

    # output
    x0 = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x0)
    x = tf.keras.layers.Dense(6, activation='softmax', dtype='float32')(x)

    # compile model
    model = tf.keras.Model(inputs=inp, outputs=[x, x0])
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)
    return model


gkf = GroupKFold(n_splits=CFG.NSPLITS)
for fold, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)):
    # load weights into grad cam model
    with strategy.scope():
        model = build_cam_model()
    model.load_weights(f'{CFG.LOAD_MODELS_FROM}EffNet_v{CFG.VER}_f{fold}.h5')
    layer_weights = model.layers[-1].get_weights()[0][:, 0]
    break

print('Using fold 0 model and inferring fold 0 OOF (out of fold) samples...')


# display grad cam
# helper function
def mask2contour(mask, width=5):
    """

    :param mask:
    :param width:
    :return:
    """
    w = mask.shape[1]
    h = mask.shape[0]

    mask2 = np.concatenate(
        [mask[:, width:],
        np.zeros((h, width))],
        axis=1
    )
    mask2 = np.logical_xor(mask, mask2)

    mask3 = np.concatenate(
        [mask[width:, :],
        np.zeros((width, w))],
        axis=0
    )
    mask3 = np.logical_xor(mask, mask3)

    return np.logical_or(mask2, mask3)


clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(8, 8))

for ii, tt in enumerate(TARGETS):
    ttt = tt.split('_')[0].upper()

    print()
    print('#' * 25)
    print('###', tt.upper())
    print('#' * 25)

    # find train samples in oof (out of fold) with target >= 0.5
    IDX = train.loc[train.index.isin(valid_index) & (
                train[tt] >= 0.5), TARGETS].index.values
    print(f'Found {len(IDX)} samples in fold zero OOF for {tt} with true>0.5')

    # infer train samples with model (save preds and activations)
    valid_gen = DataGenerator(
        train.iloc[IDX[:128]],
        shuffle=False,
        batch_size=CFG.BATCH,
        mode='valid'
    )

    p, xx = model.predict(valid_gen, verbose=0)
    # print(xx.shape)

    # display grad cam
    for x, y in valid_gen:
        ct = 0
        for i in range(CFG.BATCH):

            # find samples with pred >= 0.5 for target
            if i >= len(p):
                continue
            pred = p[i]
            if pred[ii] < 0.5:
                continue

            # format predictions as string
            pred2 = ''
            true2 = ''
            true = train.loc[IDX[i]][TARGETS].values
            for j, t in enumerate(TARGETS):
                n = t.split('_')[0]
                pred2 += f' {n}={pred[j]:0.3f}'
                true2 += f' {n}={true[j]:0.3f}'
            print()
            print('==> TRUE:', true2)
            print('==> PRED:', pred2)

            # plot grad cam results
            plt.figure(figsize=(20, 8))

            # plot grad cam image (plot 1 of 3)
            plt.subplot(1, 3, 1)
            img = np.sum(xx[i,] * layer_weights, axis=-1)
            img = cv2.resize(img, (512, 512))
            plt.imshow(img[::-1, ])
            plt.title(f'{ttt} - Grad Cam', size=14)

            # find grad cam contours for areas of interest
            cut = np.percentile(img.flatten(), [90])[0]
            cntr = img.copy()
            cntr[cntr >= cut] = 100
            cntr[cntr < cut] = 0
            cntr = mask2contour(cntr)

            # plot embossed spectrograms with gradcam contours (plot 3 of 3)
            plt.subplot(1, 3, 3)
            x1 = [x[i, :, :, k: k + 1] for k in range(4)]  # kaggle LL RL LP RP
            x1 = np.concatenate(x1, axis=0)
            # fix order LL RL LP RP
            x2 = [x[i, :, :, k + 4: k + 5] for k in [0, 2, 1, 3]]
            x2 = np.concatenate(x2, axis=0)
            x3 = np.concatenate([x1, x2], axis=1)
            img = cv2.resize(x3, (512, 512))
            img0 = img.copy()

            # emboss image for image feature visibility
            img = img[1:, 1:] - img[:-1, :-1]  # emboss
            img -= np.min(img)
            img /= np.max(img)
            img = (img * 255).astype('uint8')
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = clahe.apply(img)
            mx = np.max(img)

            cntr2 = cntr[1:, 1:]
            img[cntr2 > 0] = mx
            plt.imshow(img[::-1, ])
            plt.title(
                f'{ttt} - Embossed Spectrogram with Grad Cam Contours',
                size=14
            )

            # plot spectrograms with gradcam contours (plot 2 of 3)
            plt.subplot(1, 3, 2)
            mx = np.max(img0)
            img0[cntr > 0] = mx
            plt.imshow(img0[::-1, ])
            plt.title(
                f'{ttt} - Spectrogram with Grad Cam Contours',
                size=14
            )
            plt.show()
            ct += 1
            if ct == 8:
                break
        break
