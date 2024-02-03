"""
asdf
"""
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import catboost as cat
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupKFold
import warnings
from kaggle_kl_div.kaggle_kl_div import score
import pywt
import librosa

warnings.filterwarnings('ignore')
print('CatBoost version', cat.__version__)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class CFG:
    """
    Configuration class
    """
    VER = 3
    READ_SPEC_FILES = False
    READ_EEG_SPEC_FILES = False
    PATH = 'data/train_spectrograms/'
    TOP = 25
    TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
    PATH3 = 'data/test_eegs/'
    PATH4 = 'data/test_spectrograms/'
    DISPLAY = False
    DISPLAY2 = 0
    USE_WAVELET = None
    NAMES = ['LL', 'LP', 'RP', 'RR']
    NSPLITS = 5
    FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
             ['Fp1', 'F3', 'C3', 'P3', 'O1'],
             ['Fp2', 'F8', 'T4', 'T6', 'O2'],
             ['Fp2', 'F4', 'C4', 'P4', 'O2']]


df = pd.read_csv('data/train.csv')

# create non-overlapping eeg id train data
TARGETS = df.columns[-6:]
print('Train shape:', df.shape)
print('Targets', list(TARGETS))
df.head()

#
train = df.groupby('eeg_id')[[
    'spectrogram_id',
    'spectrogram_label_offset_seconds'
]].agg({
    'spectrogram_id': 'first',
    'spectrogram_label_offset_seconds': 'min'
})
train.columns = ['spec_id', 'min']

#
tmp = df.groupby('eeg_id')[[
    'spectrogram_id',
    'spectrogram_label_offset_seconds'
]].agg({
    'spectrogram_label_offset_seconds': 'max'
})
train['max'] = tmp

#
tmp = df.groupby('eeg_id')[['patient_id']].agg('first')
train['patient_id'] = tmp
tmp = df.groupby('eeg_id')[TARGETS].agg('sum')

for t in TARGETS:
    train[t] = tmp[t].values

#
y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

#
tmp = df.groupby('eeg_id')[['expert_consensus']].agg('first')
train['target'] = tmp

#
train = train.reset_index()
print('Train non-overlapp eeg_id shape:', train.shape)
train.head()

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


# read all eeg spectrograms
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

# engineer features

# feature names
SPEC_COLS = pd.read_parquet(f'{CFG.PATH}1000086677.parquet').columns[1:]
FEATURES = [f'{c}_mean_10m' for c in SPEC_COLS]
FEATURES += [f'{c}_min_10m' for c in SPEC_COLS]
FEATURES += [f'{c}_mean_20s' for c in SPEC_COLS]
FEATURES += [f'{c}_min_20s' for c in SPEC_COLS]
FEATURES += [f'eeg_mean_f{x}_10s' for x in range(512)]
FEATURES += [f'eeg_min_f{x}_10s' for x in range(512)]
FEATURES += [f'eeg_max_f{x}_10s' for x in range(512)]
FEATURES += [f'eeg_std_f{x}_10s' for x in range(512)]
print(
    f'We are creating {len(FEATURES)} features for {len(train)} rows... ',
    end=''
)

data = np.zeros((len(train), len(FEATURES)))
for k in range(len(train)):
    if k % 100 == 0:
        print(k, ', ', end='')
    row = train.iloc[k]
    r = int((row['min'] + row['max'])//4)

    # 10-minute window features (means and mins)
    x = np.nanmean(spectrograms[row.spec_id][r: r + 300, :], axis=0)
    data[k, :400] = x

    x = np.nanmin(spectrograms[row.spec_id][r: r + 300, :], axis=0)
    data[k, 400: 800] = x

    # 20-second window features (means and mins)
    x = np.nanmean(spectrograms[row.spec_id][r + 145: r + 155, :], axis=0)
    data[k, 800: 1200] = x

    x = np.nanmin(spectrograms[row.spec_id][r + 145: r + 155, :], axis=0)
    data[k, 1200: 1600] = x

    # reshape eeg spectrograms 128x256x4 => 512x256
    eeg_spec = np.zeros((512, 256), dtype='float32')
    xx = all_eegs[row.eeg_id]
    for j in range(4):
        eeg_spec[128 * j: 128 * (j + 1), ] = xx[:, :, j]

    # 10-second windows from eeg spectrograms
    x = np.nanmean(eeg_spec.T[100: -100, :], axis=0)
    data[k, 1600: 2112] = x
    x = np.nanmin(eeg_spec.T[100: -100, :], axis=0)
    data[k, 2112: 2624] = x
    x = np.nanmax(eeg_spec.T[100: -100, :], axis=0)
    data[k, 2624: 3136] = x
    x = np.nanstd(eeg_spec.T[100: -100, :], axis=0)
    data[k, 3136: 3648] = x

train[FEATURES] = data
print()
print('New train shape:', train.shape)

# free memory
del all_eegs, spectrograms, data
gc.collect()

all_oof = []
all_true = []

gkf = GroupKFold(n_splits=CFG.NSPLITS)
for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)):
    print('#' * 25)
    print(f'### Fold {i + 1}')
    print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
    print('#' * 25)

    model = CatBoostClassifier(
        task_type='GPU',
        loss_function='MultiClass'
    )

    train_pool = Pool(
        data=train.loc[train_index, FEATURES],
        label=train.loc[train_index, 'target'].map(CFG.TARS)
    )

    valid_pool = Pool(
        data=train.loc[valid_index, FEATURES],
        label=train.loc[valid_index, 'target'].map(CFG.TARS)
    )

    model.fit(
        train_pool,
        verbose=100,
        eval_set=valid_pool
    )

    model.save_model(f'CAT_v{CFG.VER}_f{i}.cat')

    oof = model.predict_proba(valid_pool)
    all_oof.append(oof)
    all_true.append(train.loc[valid_index, TARGETS].values)

    del train_pool, valid_pool, oof  # model
    gc.collect()
    # break


all_oof = np.concatenate(all_oof)
all_true = np.concatenate(all_true)

# feature importance
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

fig = plt.figure(figsize=(10, 8))
plt.barh(
    np.arange(len(sorted_idx))[-CFG.TOP:],
    feature_importance[sorted_idx][-CFG.TOP:],
    align='center'
)
plt.yticks(
    np.arange(len(sorted_idx))[-CFG.TOP:],
    np.array(FEATURES)[sorted_idx][-CFG.TOP:]
)
plt.title(f'Feature Importance - Top {CFG.TOP}')
plt.show()

# cv score for catboost KL-Div
oof = pd.DataFrame(all_oof.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(
    solution=true,
    submission=oof,
    row_id_column_name='id'
)

print('CV Score KL-Div for CatBoost = ', cv)

# cv score for preds 1/6
oof = pd.DataFrame(all_oof.copy())
for c in oof.columns:
    oof[c] = 1/6.
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(
    solution=true,
    submission=oof,
    row_id_column_name='id'
)

print('CV Score for "Use Equal Preds 1/6" =', cv)

# cv score for eeg id means
all_oof2 = []
gkf = GroupKFold(n_splits=CFG.NSPLITS)

for i, (train_index, valid_index) in enumerate(
        gkf.split(train, train.target, train.patient_id)):

    y_train = train.iloc[train_index][TARGETS].values
    y_valid = train.iloc[valid_index][TARGETS].values

    oof = y_valid.copy()
    for j in range(6):
        oof[:, j] = y_train[:, j].mean()

    oof = oof / oof.sum(axis=1, keepdims=True)
    all_oof2.append(oof)

all_oof2 = np.concatenate(all_oof2)

oof = pd.DataFrame(all_oof2.copy())
oof['id'] = np.arange(len(oof))

true = pd.DataFrame(all_true.copy())
true['id'] = np.arange(len(true))

cv = score(
    solution=true,
    submission=oof,
    row_id_column_name='id'
)

print('CV Score for "Use Train Means" = ', cv)

# infer test and create submission csv
del train
gc.collect()

test = pd.read_csv('data/test.csv')
print('Test shape', test.shape)
test.head()


# denoise function
def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    :param d:
    :param axis:
    :return:
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
    """
    Wavelet denoising
    :param x:
    :param wavelet:
    :param level:
    :return:
    """
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode='per')
    return ret


def spectrogram_from_eeg(parquet_path, display=CFG.DISPLAY):
    """
    Create spectrogram from EEG parquet file
    :param parquet_path:
    :param display:
    :return:
    """
    # load middle 50 seconds of eeg series
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    # variable to hold spectrogram
    img = np.zeros((128, 256, 4), dtype='float32')

    if display:
        plt.figure(figsize=(10, 7))
    signals = []
    for k in range(4):
        cols = CFG.FEATS[k]
        for kk in range(4):
            # compute pair differences
            x = eeg[cols[kk]].values - eeg[cols[kk + 1]].values

            # fill nans
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0

            # denoise
            if CFG.USE_WAVELET:
                x = denoise(x, wavelet=CFG.USE_WAVELET)
            signals.append(x)

            # raw spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 256,
                n_fft=1024,
                n_mels=128,
                fmin=0,
                fmax=20,
                win_length=128
            )

            # log transform
            width = (mel_spec.shape[1] // 32) * 32

            #
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(
                np.float32)[:, :width]

            # standardize to -1 to 1
            mel_spec_db = (mel_spec_db + 40) / 40
            img[:, :, k] += mel_spec_db

        # average the 4 montage differences
        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect='auto', origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {CFG.NAMES[k]}')

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=CFG.NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print()
        print('#' * 25)
        print()

    return img


# create all eeg spectrograms
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print('Converting Test EEG to Spectrograms...')
print()

for i, eeg_id in enumerate(EEG_IDS2):
    # create spectrogram from eeg parquet
    img = spectrogram_from_eeg(
        f'{CFG.PATH3}{eeg_id}.parquet', i < CFG.DISPLAY2
    )
    all_eegs2[eeg_id] = img

# feature engineer test
data = np.zeros((len(test), len(FEATURES)))

for k in range(len(test)):
    row = test.iloc[k]
    s = int(row.spectrogram_id)
    spec = pd.read_parquet(f'{CFG.PATH4}{s}.parquet')

    # 10-minute window features
    x = np.nanmean(spec.iloc[:, 1:].values, axis=0)
    data[k, :400] = x
    x = np.nanmin(spec.iloc[:, 1:].values, axis=0)
    data[k, 400:800] = x

    # 20-second window features
    x = np.nanmean(spec.iloc[145:155, 1:].values, axis=0)
    data[k, 800:1200] = x

    x = np.nanmin(spec.iloc[145:155, 1:].values, axis=0)
    data[k, 1200:1600] = x

    # reshape eeg spectrograms 128 x 256 x 4 => 512 x 256
    eeg_spec = np.zeros((512, 256), dtype='float32')
    xx = all_eegs2[row.eeg_id]
    for j in range(4):
        eeg_spec[128 * j:128 * (j + 1), ] = xx[:, :, j]

    # 10-second window from eeg spectrograms
    x = np.nanmean(eeg_spec.T[100: -100, :], axis=0)
    data[k, 1600: 2112] = x
    x = np.nanmin(eeg_spec.T[100: -100, :], axis=0)
    data[k, 2112: 2624] = x
    x = np.nanmax(eeg_spec.T[100: -100, :], axis=0)
    data[k, 2624: 3136] = x
    x = np.nanstd(eeg_spec.T[100: -100, :], axis=0)
    data[k, 3136: 3648] = x

test[FEATURES] = data
print('New test shape', test.shape)

# infer catboost on test
preds = []

for i in range(5):
    print(i, ', ', end='')
    model = CatBoostClassifier(task_type='GPU')
    model.load_model(f'CAT_v{CFG.VER}_f{i}.cat')
    test_pool = Pool(data=test[FEATURES])
    pred = model.predict_proba(test_pool)
    preds.append(pred)

pred = np.mean(preds, axis=0)
print()
print('Test preds shape', pred.shape)

sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv', index=False)
print('Submissionn shape', sub.shape)
sub.head()

# sanity check to confirm that predictions sum to one
sub.iloc[:, -6:].sum(axis=1)
