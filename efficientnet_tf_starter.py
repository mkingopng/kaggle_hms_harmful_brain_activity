"""

"""
import os
import gc
import wandb
import warnings
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import math
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold, GroupKFold
import tensorflow.keras.backend as K
from kaggle_kl_div.kaggle_kl_div import score
import pywt
import librosa
import albumentations as albu

# environmental variables
load_dotenv()
warnings.filterwarnings('ignore')

# tensorflow and gpu configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress most TensorFlow logs
tf.get_logger().setLevel('ERROR')  # suppress TensorFlow INFO and WARNING logs


class CFG:
	LR_START = 1e-4
	LR_MAX = 1e-3
	LR_MIN = 1e-6
	LR_RAMPUP_EPOCHS = 0
	LR_SUSTAIN_EPOCHS = 0
	LR_STEP_DECAY = 0.1
	EVERY = 1
	EPOCHS = 4
	EPOCHS2 = 10
	VER = 5
	LOAD_MODELS_FROM = 'brain_efficientnet_models_v3_v4_v5/'  # load pretrained models
	USE_KAGGLE_SPECTROGRAMS = True
	USE_EEG_SPECTROGRAMS = True
	MIX = True  # use mixed precision
	READ_EEG_SPEC_FILES = False
	TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
	TARS2 = {x: y for y, x in TARS.items()}
	USE_WAVELET = None
	NAMES = ['LL', 'LP', 'RP', 'RR']
	FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
			 ['Fp1', 'F3', 'C3', 'P3', 'O1'],
			 ['Fp2', 'F8', 'T4', 'T6', 'O2'],
			 ['Fp2', 'F4', 'C4', 'P4', 'O2']]


# use multiple gpus if available
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) <= 1:
	strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
	print(f'Using {len(gpus)} GPU')
else:
	strategy = tf.distribute.MirroredStrategy()
	print(f'Using {len(gpus)} GPUs')

if CFG.MIX:
	tf.config.optimizer.set_experimental_options(
		{"auto_mixed_precision": True})
	print('Mixed precision enabled')
else:
	print('Using full precision')

# load train data
df = pd.read_csv('data/train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape)
print('Targets', list(TARGETS))
df.head()

# create non-overlapping eeg id train data
train = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({'spectrogram_id': 'first', 'spectrogram_label_offset_seconds': 'min'})
train.columns = ['spec_id', 'min']

tmp = df.groupby('eeg_id')[['spectrogram_id', 'spectrogram_label_offset_seconds']].agg({'spectrogram_label_offset_seconds': 'max'})
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
print(train.head())

# read train spectrograms
READ_SPEC_FILES = False

# read all spectrograms
PATH = 'data/train_spectrograms'
files = os.listdir(PATH)
print(f'There are {len(files)} spectrogram parquets')

if READ_SPEC_FILES:
	spectrograms = {}
	for i, f in enumerate(files):
		if i % 100 == 0:
			print(i, ', ', end='')
		tmp = pd.read_parquet(f'{PATH}{f}')
		name = int(f.split('.')[0])
		spectrograms[name] = tmp.iloc[:, 1:].values
else:
	spectrograms = np.load(
		'brain_spectrograms/specs.npy',
		allow_pickle=True
	).item()


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


# train data loader
class DataGenerator(tf.keras.utils.Sequence):
	"""Generates data for Keras"""
	def __init__(self, data, batch_size=32, shuffle=False, augment=False, mode='train', specs=spectrograms, eeg_specs=all_eegs):
		self.data = data
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.augment = augment
		self.mode = mode
		self.specs = specs
		self.eeg_specs = eeg_specs
		self.on_epoch_end()

	def __len__(self):
		"""Denotes the number of batches per epoch"""
		ct = int(np.ceil(len(self.data) / self.batch_size))
		return ct

	def __getitem__(self, index):
		"""Generate one batch of data"""
		indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
		X, y = self.__data_generation(indexes)
		if self.augment:
			X = self.__augment_batch(X)
		return X, y

	def on_epoch_end(self):
		"""Updates indexes after each epoch"""
		self.indexes = np.arange(len(self.data))
		if self.shuffle:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes):
		"""Generates data containing batch_size samples"""

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
				X[j, 14:-14, :, k] = img[:, 22:-22] / 2.0

			# eeg spectrograms
			img = self.eeg_specs[row.eeg_id]
			X[j, :, :, 4:] = img

			if self.mode != 'test':
				y[j,] = row[TARGETS]

		return X, y

	def __random_transform(self, img):
		composition = albu.Compose([
			albu.HorizontalFlip(p=0.5),
			# albu.CoarseDropout(max_holes=8,max_height=32,max_width=32,
			# fill_value=0,p=0.5),
		])
		return composition(image=img)['image']

	def __augment_batch(self, img_batch):
		for i in range(img_batch.shape[0]):
			img_batch[i,] = self.__random_transform(img_batch[i,])
		return img_batch


# display data loader
gen = DataGenerator(train, batch_size=32, shuffle=False)

ROWS = 2
COLS = 3
BATCHES = 2

for i, (x, y) in enumerate(gen):
	plt.figure(figsize=(20, 8))
	for j in range(ROWS):
		for k in range(COLS):
			plt.subplot(ROWS, COLS, j * COLS + k + 1)
			t = y[j * COLS + k]
			img = x[j * COLS + k, :, :, 0][::-1, ]
			mn = img.flatten().min()
			mx = img.flatten().max()
			img = (img - mn) / (mx - mn)
			plt.imshow(img)
			tars = f'[{t[0]:0.2f}'
			for s in t[1:]: tars += f', {s:0.2f}'
			eeg = train.eeg_id.values[i * 32 + j * COLS + k]
			plt.title(f'EEG = {eeg}\nTarget = {tars}', size=12)
			plt.yticks([])
			plt.ylabel('Frequencies (Hz)', size=14)
			plt.xlabel('Time (sec)', size=16)
	plt.show()
	if i == BATCHES - 1:
		break


def lrfn(epoch):
	if epoch < CFG.LR_RAMPUP_EPOCHS:
		lr = ((CFG.LR_MAX - CFG.LR_START) / CFG.LR_RAMPUP_EPOCHS * epoch + CFG.LR_START)
	elif epoch < CFG.LR_RAMPUP_EPOCHS + CFG.LR_SUSTAIN_EPOCHS:
		lr = CFG.LR_MAX
	else:
		decay_total_epochs = (CFG.EPOCHS2 - CFG.LR_RAMPUP_EPOCHS - CFG.LR_SUSTAIN_EPOCHS - 1)
		decay_epoch_index = (epoch - CFG.LR_RAMPUP_EPOCHS - CFG.LR_SUSTAIN_EPOCHS)
		phase = math.pi * decay_epoch_index / decay_total_epochs
		cosine_decay = 0.5 * (1 + math.cos(phase))
		lr = (CFG.LR_MAX - CFG.LR_MIN) * cosine_decay + CFG.LR_MIN
	return lr


rng = [i for i in range(CFG.EPOCHS2)]
lr_y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, lr_y, '-o')
plt.xlabel('epoch', size=14)
plt.ylabel('learning rate', size=14)
plt.title('Cosine Training Schedule', size=16)
plt.show()

LR2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


def lrfn(epoch):
	"""

	:param epoch:
	:return:
	"""
	if epoch < CFG.LR_RAMPUP_EPOCHS:
		lr = ((CFG.LR_MAX - CFG.LR_START) / CFG.LR_RAMPUP_EPOCHS * epoch + CFG.LR_START)
	elif epoch < CFG.LR_RAMPUP_EPOCHS + CFG.LR_SUSTAIN_EPOCHS:
		lr = CFG.LR_MAX
	else:
		lr = CFG.LR_MAX * CFG.LR_STEP_DECAY ** ((epoch - CFG.LR_RAMPUP_EPOCHS - CFG.LR_SUSTAIN_EPOCHS) // CFG.EVERY)
	return lr


rng = [i for i in range(CFG.EPOCHS)]
y = [lrfn(x) for x in rng]
plt.figure(figsize=(10, 4))
plt.plot(rng, y, 'o-')
plt.xlabel('epoch', size=14)
plt.ylabel('learning rate', size=14)
plt.title('Step Training Schedule', size=16)
plt.show()

LR = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)


# build efficientnet model
def build_model():
	"""

	:return:
	"""
	inp = tf.keras.Input(shape=(128, 256, 8))

	base_model = efn.EfficientNetB0(
		include_top=False,
		weights=None,
		input_shape=None
	)
	base_model.load_weights(
		'tf_efficientnet_imagenet_weights/efficientnet'
		'-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

	# reshape input 128x256x8 => 512x512x3 monotone image
	# kaggle spectrograms
	x1 = [inp[:, :, :, i:i + 1] for i in range(4)]
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
	x = base_model(x)
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(6, activation='softmax', dtype='float32')(x)

	# compile model
	model = tf.keras.Model(inputs=inp, outputs=x)
	opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
	loss = tf.keras.losses.KLDivergence()  # KL divergence

	model.compile(loss=loss, optimizer=opt)
	return model


# denoise function
def maddest(d, axis=None):
	"""

	:param d:
	:param axis:
	:return:
	"""
	return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet='haar', level=1):
	"""

	:param x:
	:param wavelet:
	:param level:
	:return:
	"""
	coeff = pywt.wavedec(x, wavelet, mode="per")
	sigma = (1 / 0.6745) * maddest(coeff[-level])

	uthresh = sigma * np.sqrt(2 * np.log(len(x)))
	coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in
				 coeff[1:])

	ret = pywt.waverec(coeff, wavelet, mode='per')

	return ret


def spectrogram_from_eeg(parquet_path, display=False):
	"""

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

	if display: plt.figure(figsize=(10, 7))
	signals = []
	for k in range(4):
		COLS = CFG.FEATS[k]

		for kk in range(4):

			# compute pair differences
			x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

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
			if k > 0: offset -= signals[3 - k].min()
			plt.plot(range(10_000), signals[k] + offset, label=CFG.NAMES[3 - k])
			offset += signals[3 - k].max()
		plt.legend()
		plt.title(f'EEG {eeg_id} Signals')
		plt.show()
		print()
		print('#' * 25)
		print()

	return img


if __name__ == "__main__":
	# train model
	all_oof = []
	all_true = []

	# initialize W&B run for each fold
	wandb.init(
		project="harmful_brain_activity",
		entity="michael_kingston",
		tags=["EfficientNet", "Kaggle", f"fold_{i}"],
		reinit=True
	)

	# update or add any additional hyperparameters
	wandb.config.update({
		"epochs": CFG.EPOCHS,
		"learning_rate": LR,
		"batch_size": 32
	})

	# cross validation
	gkf = GroupKFold(n_splits=5)

	for i, (train_index, valid_index) in enumerate(
			gkf.split(train, train.target, train.patient_id)):

		print('#' * 25)
		print(f'### Fold {i + 1}')

		train_gen = DataGenerator(
			train.iloc[train_index],
			shuffle=True,
			batch_size=32,
			augment=False
		)

		valid_gen = DataGenerator(
			train.iloc[valid_index],
			shuffle=False,
			batch_size=64,
			mode='valid'
		)

		print(f'### train size {len(train_index)}, valid size {len(valid_index)}')
		print('#' * 25)

		K.clear_session()
		with strategy.scope():
			model = build_model()

		if CFG.LOAD_MODELS_FROM is None:
			model.fit(
				train_gen,
				verbose=1,
				validation_data=valid_gen,
				epochs=CFG.EPOCHS,
				callbacks=[LR, wandb.keras.WandbCallback()]
			)
			model.save_weights(f'EffNet_v{CFG.VER}_f{i}.h5')
		else:
			model.load_weights(f'{CFG.LOAD_MODELS_FROM}EffNet_v{CFG.VER}_f{i}.h5')

		oof = model.predict(valid_gen, verbose=1)
		all_oof.append(oof)
		all_true.append(train.iloc[valid_index][TARGETS].values)

		del model, oof
		gc.collect()

	all_oof = np.concatenate(all_oof)
	all_true = np.concatenate(all_true)

	# CV score for EfficientNet
	oof = pd.DataFrame(all_oof.copy())
	oof['id'] = np.arange(len(oof))

	true = pd.DataFrame(all_true.copy())
	true['id'] = np.arange(len(true))

	cv = score(solution=true, submission=oof, row_id_column_name='id')
	print(f'CV Score KL-Div for EfficientNetB2 = {cv}')

	# infer test and create submission CSV
	del all_eegs, spectrograms

	gc.collect()

	test = pd.read_csv('data/test.csv')
	print('Test shape', test.shape)
	test.head()

	# read all spectrograms
	PATH2 = 'data/test_spectrograms/'  # path to test spectrograms
	files2 = os.listdir(PATH2)  #
	print(f'There are {len(files2)} test spectrogram parquets')  #

	#
	spectrograms2 = {}
	for i, f in enumerate(files2):
		if i % 100 == 0:
			print(i, ', ', end='')
		tmp = pd.read_parquet(f'{PATH2}{f}')
		name = int(f.split('.')[0])
		spectrograms2[name] = tmp.iloc[:, 1:].values

	# rename for dataloader
	test = test.rename({'spectrogram_id': 'spec_id'}, axis=1)

	# read all eeg spectrograms
	PATH2 = 'data/test_eegs/'
	DISPLAY = 1
	EEG_IDS2 = test.eeg_id.unique()
	all_eegs2 = {}

	print('Converting Test EEG to Spectrograms...')
	print()

	#
	for i, eeg_id in enumerate(EEG_IDS2):
		# create spectrogram from eeg parquet
		img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
		all_eegs2[eeg_id] = img

	# infer efficientnet on test
	preds = []
	model = build_model()
	test_gen = DataGenerator(
		test,
		shuffle=False,
		batch_size=64,
		mode='test',
		specs=spectrograms2,
		eeg_specs=all_eegs2
	)

	for i in range(5):
		print(f'Fold {i + 1}')
		if CFG.LOAD_MODELS_FROM:
			model.load_weights(f'{CFG.LOAD_MODELS_FROM}EffNet_v{CFG.VER}_f{i}.h5')
		else:
			model.load_weights(f'EffNet_v{CFG.VER}_f{i}.h5')

		pred = model.predict(test_gen, verbose=1)
		preds.append(pred)

	pred = np.mean(preds, axis=0)
	print()
	print('Test preds shape', pred.shape)

	sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
	sub[TARGETS] = pred
	sub.to_csv('submission.csv', index=False)
	print('Submissionn shape', sub.shape)
	sub.head()

	wandb.finish()  # finish the W&B run for this fold

	# sanity-check to confirm predictions sum to one
	sub.iloc[:, -6:].sum(axis=1)
