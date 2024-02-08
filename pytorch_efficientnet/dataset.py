"""

"""
from utils import *


num_to_label = {v: k for k, v in CFG.LABEL_TO_NUM.items()}
LOGGER = get_logger()
seed_everything(CFG.SEED)

# load data
# df = pd.read_csv(PATHS.TRAIN_CSV)
# label_cols = df.columns[-6:]
# print(f"Train dataframe shape is: {df.shape}")
# print(f"Labels: {list(label_cols)}")
# print(df.head())


def load_data():
    df = pd.read_csv(PATHS.TRAIN_CSV)
    label_cols = df.columns[-6:]
    LOGGER.info(f"Train dataframe shape is: {df.shape}")
    LOGGER.info(f"Labels: {list(label_cols)}")
    LOGGER.debug(df.head())
    return df, label_cols


# data preprocessing
# train_df = df.groupby('eeg_id')[
#     ['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
#     'spectrogram_id': 'first',
#     'spectrogram_label_offset_seconds': 'min'
# })
#
# train_df.columns = ['spectogram_id', 'min']
#
# aux = df.groupby('eeg_id')[
#     ['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({
#     'spectrogram_label_offset_seconds': 'max'}
# )
#
# train_df['max'] = aux
#
# aux = df.groupby('eeg_id')[['patient_id']].agg('first')
# train_df['patient_id'] = aux
#
# aux = df.groupby('eeg_id')[label_cols].agg('sum')
# for label in label_cols:
#     train_df[label] = aux[label].values
#
# y_data = train_df[label_cols].values
# y_data = y_data / y_data.sum(axis=1, keepdims=True)
# train_df[label_cols] = y_data
#
# aux = df.groupby('eeg_id')[['expert_consensus']].agg('first')
# train_df['target'] = aux
#
# train_df = train_df.reset_index()
# print('Train non-overlapp eeg_id shape:', train_df.shape)
# print(train_df.head())


# new preprocess data
def preprocess_data(df: pd.DataFrame, label_cols):
    train_df = df.groupby('eeg_id').agg({
        'spectrogram_id': 'first',
        'spectrogram_label_offset_seconds': ['min', 'max'],
        'patient_id': 'first',
        **{label: 'sum' for label in label_cols},
        'expert_consensus': 'first'
    })
    train_df.columns = ['spectogram_id', 'min', 'max',
                        'patient_id'] + list(label_cols) + ['expert_consensus']
    y_data = train_df[label_cols].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train_df[label_cols] = y_data
    train_df['target'] = train_df[label_cols].sum(axis=1)
    train_df.reset_index(inplace=True)
    LOGGER.info('Train non-overlapping eeg_id shape: %s', train_df.shape)
    LOGGER.debug(train_df.head())
    return train_df, y_data


# read train spectrograms
# paths_spectograms = glob(PATHS.TRAIN_SPECTOGRAMS + "*.parquet")
# print(f'There are {len(paths_spectograms)} spectrogram parquets')
#
# if BOOLS.READ_SPEC_FILES:
#     all_spectrograms = {}
#     for file_path in tqdm(paths_spectograms):
#         aux = pd.read_parquet(file_path)
#         name = int(file_path.split("/")[-1].split('.')[0])
#         all_spectrograms[name] = aux.iloc[:, 1:].values
#         del aux
# else:
#     all_spectrograms = np.load(PATHS.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()


# new read spectrograms
def read_spectrograms():
    paths_spectograms = glob(PATHS.TRAIN_SPECTOGRAMS + "*.parquet")
    LOGGER.info(f'There are {len(paths_spectograms)} spectrogram parquets')
    if BOOLS.READ_SPEC_FILES:
        all_spectrograms = {}
        for file_path in tqdm(paths_spectograms):
            aux = pd.read_parquet(file_path)
            name = int(os.path.basename(file_path).split('.')[0])
            all_spectrograms[name] = aux.iloc[:, 1:].values
            del aux
    else:
        all_spectrograms = np.load(PATHS.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    return all_spectrograms


# read eeg spectrograms
# paths_eegs = glob(PATHS.TRAIN_EEGS + "*.npy")
# print(f'There are {len(paths_eegs)} EEG spectograms')
#
# if BOOLS.READ_EEG_SPEC_FILES:
#     all_eegs = {}
#     for file_path in tqdm(paths_eegs):
#         eeg_id = file_path.split("/")[-1].split(".")[0]
#         eeg_spectogram = np.load(file_path)
#         all_eegs[eeg_id] = eeg_spectogram
# else:
#     all_eegs = np.load(PATHS.PRE_LOADED_EEGS, allow_pickle=True).item()


# new read eeg spectrograms
def read_eeg_spectrograms():
    paths_eegs = glob(PATHS.TRAIN_EEGS + "*.npy")
    LOGGER.info(f'There are {len(paths_eegs)} EEG spectrograms')
    if BOOLS.READ_EEG_SPEC_FILES:
        all_eegs = {}
        for file_path in tqdm(paths_eegs):
            eeg_id = os.path.basename(file_path).split(".")[0]
            eeg_spectogram = np.load(file_path)
            all_eegs[eeg_id] = eeg_spectogram
    else:
        all_eegs = np.load(PATHS.PRE_LOADED_EEGS, allow_pickle=True).item()
    return all_eegs


# dataset
class CustomDataset(Dataset):
    """
    Custom dataset for training/validation
    """
    def __init__(
            self, df: pd.DataFrame,
            config,
            label_cols: List[str],  # Add label_cols as an argument
            augment: bool = False,
            mode: str = 'train',
            specs: Dict[int, np.ndarray] = None,
            eeg_specs: Dict[int, np.ndarray] = None
    ):
        self.df = df
        self.config = config
        self.batch_size = self.config.BATCH_SIZE_TRAIN
        self.augment = augment
        self.mode = mode
        self.label_cols = label_cols  # Store label_cols as an attribute
        self.spectograms = specs or read_eeg_spectrograms()
        self.eeg_spectograms = eeg_specs or read_eeg_spectrograms()

    def __len__(self):
        """
        denotes the number of samples in the dataset
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
                y = row[self.label_cols].values.astype(np.float32)  # get labels

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
