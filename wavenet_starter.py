"""
asdf
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display
from kaggle_kl_div.kaggle_kl_div import score
from scipy.signal import butter, lfilter
import tensorflow as tf
from tensorflow.keras.layers import Multiply, Add, Conv1D
from sklearn.model_selection import GroupKFold
import tensorflow.keras.backend as K
import gc


# initialise GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
print('TensorFlow version =', tf.__version__)


class CFG:
    """
    Configuration class
    """
    MIX = True
    CREATE_EEGS = False  # choice to create or load eegs from notebook v1
    TRAIN_MODEL = False
    DISPLAY = 4
    FEATS = ['Fp1', 'T3', 'C3', 'O1', 'Fp2', 'C4', 'T4', 'O2']
    FEAT2IDX = {x: y for x, y in zip(FEATS, range(len(FEATS)))}
    PATH = 'data/train_eegs/'
    TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
    TARS2 = {x: y for y, x in TARS.items()}
    EPOCHS = 5
    VERBOSE = 1
    FOLDS_TO_TRAIN = 5
    DISPLAY2 = 1
    PATH2 = 'data/test_eegs/'


# use multiple gpus
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) <= 1:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    print(f'Using {len(gpus)} GPU')
else:
    strategy = tf.distribute.MirroredStrategy()
    print(f'Using {len(gpus)} GPUs')

# use mixed precision
if CFG.MIX:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print('Mixed precision enabled')
else:
    print('Using full precision')

train = pd.read_csv('data/train.csv')
print(train.shape)
print(train.head())

df = pd.read_parquet('data/train_eegs/1000913311.parquet')
FEATS = df.columns
print(f'There are {len(FEATS)} raw eeg features')
print(list(FEATS))

print('We will use the following subset of raw EEG features:')
print(list(FEATS))


def eeg_from_parquet(parquet_path, display=False):
    """

    :param parquet_path:
    :param display:
    :return:
    """
    # extract middle 50 seconds
    eeg = pd.read_parquet(parquet_path, columns=FEATS)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset:offset + 10_000]

    if display:
        plt.figure(figsize=(10, 5))
        offset = 0

    # convert to numpy
    data = np.zeros((10_000, len(FEATS)))
    for j, col in enumerate(FEATS):

        # fill nan
        x = eeg[col].values.astype('float32')
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0

        data[:, j] = x

        if display:
            if j != 0:
                offset += x.max()
            plt.plot(range(10_000), x - offset, label=col)
            offset -= x.min()

    if display:
        plt.legend()
        name = parquet_path.split('/')[-1]
        name = name.split('.')[0]
        plt.title(f'EEG {name}', size=16)
        plt.show()

    return data


all_eegs = {}

EEG_IDS = train.eeg_id.unique()

for i, eeg_id in enumerate(EEG_IDS):
    if (i % 100 == 0) & (i != 0):
        print(i, ', ', end='')

    # save eeg to python dictionary of numpy arrays
    data = eeg_from_parquet(
        f'{CFG.PATH}{eeg_id}.parquet',
        display=i < CFG.DISPLAY
    )
    all_eegs[eeg_id] = data
    if i == CFG.DISPLAY:
        if CFG.CREATE_EEGS:
            print(f'Processing {train.eeg_id.nunique()} eeg parquets... ',
                  end='')
        else:
            print(f'Reading {len(EEG_IDS)} eeg NumPys from disk.')
            break

if CFG.CREATE_EEGS:
    np.save('eegs', all_eegs)
else:
    all_eegs = np.load('brain_eegs/eegs.npy', allow_pickle=True).item()

# deduplicate train eef id
# load train
df = pd.read_csv('data/train.csv')
TARGETS = df.columns[-6:]
train = df.groupby('eeg_id')[['patient_id']].agg('first')
tmp = df.groupby('eeg_id')[TARGETS].agg('sum')
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

train = train.reset_index()
train = train.loc[train.eeg_id.isin(EEG_IDS)]
print('Train Data with unique eeg_id shape:', train.shape)
train.head()


# butter low-pass filter
def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    """

    :param data:
    :param cutoff_freq:
    :param sampling_rate:
    :param order:
    :return:
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


FREQS = [1, 2, 4, 8, 16][::-1]
x = [all_eegs[EEG_IDS[0]][:, 0]]
for k in FREQS:
    x.append(butter_lowpass_filter(x[0], cutoff_freq=k))

plt.figure(figsize=(20, 20))
plt.plot(range(10_000), x[0], label='without filter')
for k in range(1, len(x)):
    plt.plot(
        range(10_000),
        x[k]-k*(x[0].max()-x[0].min()),
        label=f'with filter {FREQS[k-1]}Hz'
    )
plt.legend()
plt.title('Butter Low-Pass Filter Examples', size=18)
plt.show()


# data loader with butter low-pass filter
class DataGenerator(tf.keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(
            self, data, batch_size=32, shuffle=False, eegs=all_eegs,
            mode='train',
            downsample=5
            ):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.eegs = eegs
        self.mode = mode
        self.downsample = downsample
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
        return X[:, ::self.downsample, :], y

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
        X = np.zeros((len(indexes), 10_000, 8), dtype='float32')
        y = np.zeros((len(indexes), 6), dtype='float32')

        sample = np.zeros((10_000, X.shape[-1]))
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
            data = self.eegs[row.eeg_id]

            # feature engineer
            sample[:, 0] = data[:, CFG.FEAT2IDX['Fp1']] - data[:, CFG.FEAT2IDX['T3']]
            sample[:, 1] = data[:, CFG.FEAT2IDX['T3']] - data[:, CFG.FEAT2IDX['O1']]

            sample[:, 2] = data[:, CFG.FEAT2IDX['Fp1']] - data[:, CFG.FEAT2IDX['C3']]
            sample[:, 3] = data[:, CFG.FEAT2IDX['C3']] - data[:, CFG.FEAT2IDX['O1']]

            sample[:, 4] = data[:, CFG.FEAT2IDX['Fp2']] - data[:, CFG.FEAT2IDX['C4']]
            sample[:, 5] = data[:, CFG.FEAT2IDX['C4']] - data[:, CFG.FEAT2IDX['O2']]

            sample[:, 6] = data[:, CFG.FEAT2IDX['Fp2']] - data[:, CFG.FEAT2IDX['T4']]
            sample[:, 7] = data[:, CFG.FEAT2IDX['T4']] - data[:, CFG.FEAT2IDX['O2']]

            # standardize
            sample = np.clip(sample, -1024, 1024)
            sample = np.nan_to_num(sample, nan=0) / 32.0

            # butter low-pass filter
            sample = butter_lowpass_filter(sample)

            X[j,] = sample
            if self.mode != 'test':
                y[j] = row[TARGETS]

        return X, y


# display data loader
gen = DataGenerator(train, shuffle=False)

for x, y in gen:
    for k in range(4):
        plt.figure(figsize=(20, 4))
        offset = 0
        for j in range(x.shape[-1]):
            if j != 0:
                offset -= x[k, :, j].min()
            plt.plot(
                range(2_000), x[k, :, j] + offset,
                label=f'feature {j + 1}'
            )
            offset += x[k, :, j].max()
        tt = f'{y[k][0]:0.1f}'
        for t in y[k][1:]:
            tt += f', {t:0.1f}'
        plt.title(f'EEG_Id = {EEG_IDS[k]}\nTarget = {tt}', size=14)
        plt.legend()
        plt.show()
    break


# build wavenet model
# train schedule
def lrfn(epoch):
    """

    :param epoch:
    :return:
    """
    return [1e-3, 1e-3, 1e-4, 1e-4, 1e-5][epoch]


LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


def wave_block(x, filters, kernel_size, n):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param n:
    :return:
    """
    dilation_rates = [2 ** i for i in range(n)]
    x = Conv1D(
        filters=filters,
        kernel_size=1,
        padding='same'
    )(x)

    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='tanh',
            dilation_rate=dilation_rate
        )(x)

        sigm_out = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation='sigmoid',
            dilation_rate=dilation_rate
        )(x)

        x = Multiply()([tanh_out, sigm_out])

        x = Conv1D(
            filters=filters,
            kernel_size=1,
            padding='same'
        )(x)

        res_x = Add()([res_x, x])

    return res_x


def build_model():
    """

    :return:
    """
    # input
    inp = tf.keras.Input(shape=(2_000, 8))

    # feature extraction sub model
    inp2 = tf.keras.Input(shape=(2_000, 1))
    x = wave_block(inp2, 8, 3, 12)
    x = wave_block(x, 16, 3, 8)
    x = wave_block(x, 32, 3, 4)
    x = wave_block(x, 64, 3, 1)
    model2 = tf.keras.Model(inputs=inp2, outputs=x)

    # left temporal chain
    x1 = model2(inp[:, :, 0:1])
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x2 = model2(inp[:, :, 1:2])
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    z1 = tf.keras.layers.Average()([x1, x2])

    # left parasagittal chain
    x1 = model2(inp[:, :, 2:3])
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x2 = model2(inp[:, :, 3:4])
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    z2 = tf.keras.layers.Average()([x1, x2])

    # right parasagittal chain
    x1 = model2(inp[:, :, 4:5])
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x2 = model2(inp[:, :, 5:6])
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    z3 = tf.keras.layers.Average()([x1, x2])

    # right temporal chain
    x1 = model2(inp[:, :, 6:7])
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)
    x2 = model2(inp[:, :, 7:8])
    x2 = tf.keras.layers.GlobalAveragePooling1D()(x2)
    z4 = tf.keras.layers.Average()([x1, x2])

    # combine chains
    y = tf.keras.layers.Concatenate()([z1, z2, z3, z4])
    y = tf.keras.layers.Dense(64, activation='relu')(y)
    y = tf.keras.layers.Dense(6, activation='softmax', dtype='float32')(y)

    # compile model
    model = tf.keras.Model(inputs=inp, outputs=y)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.KLDivergence()
    model.compile(loss=loss, optimizer=opt)

    return model


# train group kfold
if not os.path.exists('WaveNet_Model'):
    os.makedirs('WaveNet_Model')

all_oof = []
all_oof2 = []
all_true = []
gkf = GroupKFold(n_splits=5)
for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)):

    print('#' * 25)
    print(f'### Fold {i + 1}')

    train_gen = DataGenerator(
        train.iloc[train_index],
        shuffle=True,
        batch_size=32
    )

    valid_gen = DataGenerator(
        train.iloc[valid_index],
        shuffle=False,
        batch_size=64,
        mode='valid'
    )

    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#' * 25)

    # train model
    K.clear_session()
    with strategy.scope():
        model = build_model()
    if CFG.TRAIN_MODEL:
        model.fit(
            train_gen,
            verbose=CFG.VERBOSE,
            validation_data=valid_gen,
            epochs=CFG.EPOCHS,
            callbacks=[LR]
        )
        model.save_weights(f'WaveNet_Model/WaveNet_fold{i}.h5')
    else:
        model.load_weights(f'brain_eegs/WaveNet_Model/WaveNet_fold{i}.h5')

    # wavenet oof
    oof = model.predict(valid_gen, verbose=CFG.VERBOSE)
    all_oof.append(oof)
    all_true.append(train.iloc[valid_index][TARGETS].values)

    # train mean oof
    y_train = train.iloc[train_index][TARGETS].values
    y_valid = train.iloc[valid_index][TARGETS].values
    oof = y_valid.copy()
    for j in range(6):
        oof[:, j] = y_train[:, j].mean()
    oof = oof / oof.sum(axis=1, keepdims=True)
    all_oof2.append(oof)

    del model, oof, y_train, y_valid
    gc.collect()

    if i == CFG.FOLDS_TO_TRAIN - 1:
        break

all_oof = np.concatenate(all_oof)
all_oof2 = np.concatenate(all_oof2)
all_true = np.concatenate(all_true)

# cv score for wavenet
oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score with WaveNet Raw EEG =', cv)

# cv score using train means
oof = pd.DataFrame(all_oof2.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score with Train Means =', cv)

# submit to kaggle LB
del all_eegs, train
gc.collect()

test = pd.read_csv('data/test.csv')
print('Test shape:', test.shape)
print(test.head())

all_eegs2 = {}
EEG_IDS2 = test.eeg_id.unique()

print('Processing Test EEG parquets...')
print()
for i, eeg_id in enumerate(EEG_IDS2):
    # save eeg to python dictionary of numpy arrays
    data = eeg_from_parquet(
        f'{CFG.PATH2}{eeg_id}.parquet',
        i < CFG.DISPLAY2
    )
    all_eegs2[eeg_id] = data

# infer mlp on test
preds = []
model = build_model()

test_gen = DataGenerator(
    test,
    shuffle=False,
    batch_size=64,
    eegs=all_eegs2,
    mode='test'
)

print('Inferring test... ', end='')
for i in range(CFG.FOLDS_TO_TRAIN):
    print(f'fold {i+1}, ', end='')
    if CFG.TRAIN_MODEL:
        model.load_weights(f'WaveNet_Model/WaveNet_fold{i}.h5')
    else:
        model.load_weights(f'brain_eegs/WaveNet_Model/WaveNet_fold{i}.h5')
    pred = model.predict(test_gen, verbose=0)
    preds.append(pred)

pred = np.mean(preds, axis=0)
print()
print('Test preds shape', pred.shape)

# create submission.csv
sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv', index=False)
print('Submission shape', sub.shape)
print(sub.head())

# sanity check to confirm predictions sum to one
print('Sub row 0 sums to:', sub.iloc[0, -6:].sum())
