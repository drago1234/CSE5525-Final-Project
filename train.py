from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as kr
import time

from utils import readEmbeddings, readInfo

def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
						conflict_handler='resolve')
	parser.add_argument('--dir', required=False, default='./processed_data/',
						help='directory of preprocessed data')
	parser.add_argument('--graphemb', required=False, default='embeddings.txt',
						help='filename of graph embeddings')
	parser.add_argument('--textemb', required=False, default='text_embeddings.txt',
						help='filename of graph embeddings')
	parser.add_argument('--setting', required=False, type=int, default=0,
						help='input for classifier. 0 for graph embeddings only, 1 for text embedding only, 2 for both.')
	parser.add_argument('--epoch', required=False, type=int, default=5,
						help='number of epochs for training')
	parser.add_argument('--batch-size', required=False, type=int, default=5000,
						help='batch_size for training')
	parser.add_argument('--lr', required=False, type=float, default=1e-3,
						help='learning rate')
	parser.add_argument('--easy', action='store_true',
						help='use binary rating instead of decimal')
	parser.add_argument('--finalemb', required=False, type=int, default=128,
						help='final dimensionality of movie embeddings')
	args = parser.parse_args()
	return args

def getIndInfo(args):
	num_movies, num_genres, num_cast, num_users, total = readInfo(args)
	movie_base = 0
	user_base = num_movies + num_genres + num_cast
	return num_users, user_base, num_movies, movie_base

class DataLoader():
	def __init__(self, args, user_emb, movie_emb, user_base, movie_base):
		filename = "/".join([args.dir, "rating_train.csv"])
		train_df = pd.read_csv(filename)
		train_uid = np.array(train_df["uId"], dtype=np.int32)
		train_mid = np.array(train_df["mId"], dtype=np.int32)
		train_binary = np.array(train_df["binary"], dtype=np.int32)
		train_rating = np.array(train_df["rating"] * 2 - 1, dtype=np.int32)

		X_train_user = user_emb[train_uid - user_base]
		X_train_movie = movie_emb[train_mid - movie_base]
		self.train_data = np.concatenate([X_train_user, X_train_movie], axis=1)
		self.train_label = train_binary if args.easy else train_rating

		filename = "/".join([args.dir, "rating_test.csv"])
		test_df = pd.read_csv(filename)
		test_uid = np.array(test_df["uId"], dtype=np.int32)
		test_mid = np.array(test_df["mId"], dtype=np.int32)
		test_binary = np.array(test_df["binary"], dtype=np.int32)
		test_rating = np.array(test_df["rating"] * 2 - 1, dtype=np.int32)

		X_test_user = user_emb[test_uid - user_base]
		X_test_movie = movie_emb[test_mid - movie_base]
		self.test_data = np.concatenate([X_test_user, X_test_movie], axis=1)
		self.test_label = test_binary if args.easy else test_rating

		self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

		print("**********\nDataLoader")
		print("num_train_data: %d, num_test_data: %d"%(self.num_train_data, self.num_test_data))
		print("data_dim: %d"%(self.train_data.shape[1]))

	def get_batch(self, batch_size):
		index = np.random.randint(0, self.num_train_data, batch_size)
		return self.train_data[index, :], self.train_label[index]

class MLP(kr.Model):
	""" Multi-layer perceptrons """
	def __init__(self, args, data_loader, epoch=5, batch_size=5000, learning_rate=1e-3):
		super().__init__()
		self.flatten = kr.layers.Flatten()
		self.dense1 = kr.layers.Dense(
											units=100,
											activation=tf.nn.relu
											)
		self.dense2 = kr.layers.Dense(
											units=2 if args.easy else 10,
											)
		self.optimizer = kr.optimizers.Adam(learning_rate=learning_rate)
		self.epoch = epoch
		self.batch_size = batch_size
		self.data_loader = data_loader

	def call(self, input_data):
		flattened = self.flatten(input_data)
		l1 = self.dense1(flattened)
		l2 = self.dense2(l1)
		output = tf.nn.softmax(l2)
		return output

	def train(self):
		start = time.time()
		for i in range(int(self.data_loader.num_train_data / self.batch_size * self.epoch)):
			st = time.time()
			X, y = self.data_loader.get_batch(self.batch_size)
			with tf.GradientTape() as tape:
				pred = self.call(X)
				loss = kr.losses.sparse_categorical_crossentropy(y_true=y, y_pred=pred)
				loss = tf.reduce_mean(loss)
			grads = tape.gradient(loss, self.variables)
			self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))
			ed = time.time()
			if i % 50 == 0:
				print("Batch %d: loss = %f, time = %.3f"%(i, loss.numpy(), ed - st))
		endt = time.time()
		print("Total time for training: %.3f"%(endt - start))

	def eval(self):
		sca = kr.metrics.SparseCategoricalAccuracy()
		for i in range(int(self.data_loader.num_test_data / self.batch_size * self.epoch)):
			start_idx = i * self.batch_size
			end_idx = (i + 1) * self.batch_size
			X = self.data_loader.test_data[start_idx : end_idx]
			y = self.data_loader.test_label[start_idx : end_idx]
			pred = self.call(X)
			sca.update_state(y_true = y, y_pred = pred)

		print("Accuracy = %f"%(sca.result()))

if __name__ == "__main__":

	# parse arguments
	args = args_parser()

	# get index information of users and movies
	print(f"args: {args}")
	num_users, user_base, num_movies, movie_base = getIndInfo(args)
	print(f"num_users: {num_users}, user_base: {user_base}, num_movies: {num_movies}, movie_base: {movie_base}")
	
	
	# read embeddings
	user_emb, movie_emb = readEmbeddings(args.dir, args.graphemb, num_users, user_base, num_movies, movie_base)
	if args.setting == 1:
		# TODO: read text embedding
		movie_emb = np.empty(shape=(num_movies, 128))
	elif args.setting == 2:
		# TODO: read text embedding then concatenate
		text_emb = np.empty(shape=(num_movies, 128))
		movie_emb = np.concatenate([movie_emb, text_emb], axis=1)

	# build X_train, y_train
	dataloader = DataLoader(args, user_emb, movie_emb, user_base, movie_base)
	model = MLP(args, dataloader, args, epoch=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
	model.train()
	model.eval()

