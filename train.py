from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import pandas as pd

from utils import readEmbeddings, readInfo

def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
	parser.add_argument('--dir', required=False, default='./processed_data/',
                        help='directory of preprocessed data')
	parser.add_argument('--graphemb', required=False, default='embeddings.txt.txt',
						help='filename of graph embeddings')
	parser.add_argument('--textemb', required=False, default='text_embeddings.txt',
						help='filename of graph embeddings')
	parser.add_argument('--setting', required=False, type=int, default=0,
						help='input for classifier. 0 for graph embeddings only, 1 for text embedding only, 2 for both.')
	parser.add_argument('--finalemb', required=False, type=int, default=128,
						help='final dimensionality of movie embeddings')
	args = parser.parse_args()
	return args

def getIndInfo(args):
	num_movies, num_genres, num_cast, num_users, total = readInfo(args)
	movie_base = 0
	user_base = num_movies + num_genres + num_cast
	return num_users, user_base, num_movies, movie_base

def getXandY(args, name, user_emb, movie_emb, user_base, movie_base):
	filename = "/".join([args.dir, name])

	df = pd.read_csv(filename)
	uid = np.array(df["uId"], dtype=np.int32)
	mid = np.array(df["mId"], dtype=np.int32)
	binary = np.array(df["binary"], dtype=np.int32)
	rating = np.array(df["rating"], dtype=np.float32)

	X_user = user_emb[uid - user_base]
	X_movie = movie_emb[mid - movie_base]
	return X_user, X_movie, rating

if __name__ == "__main__":

	# parse arguments
	args = args_parser()

	# get index information of users and movies
	num_users, user_base, num_movies, movie_base = getIndInfo(args)
	
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
	X_train_user, X_train_movie, y_train = getXandY(args, "rating_test.csv", user_emb, movie_emb, user_base, movie_base)
	
	

