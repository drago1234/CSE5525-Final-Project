from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import pymp

from collections import defaultdict

class dataset(object):

	def __init__(self, num_movies, num_genres, num_cast, num_users, total, m2g, g2m, m2c, c2m, u2m, u2mr, m2u, m2ur, alpha):
		self.num_movies = num_movies
		self.num_genres = num_genres
		self.num_cast = num_cast
		self.num_users = num_users
		self.total = total
		self.m2g = m2g
		self.g2m = g2m
		self.m2c = m2c
		self.c2m = c2m
		self.u2m = u2m
		self.u2mr = u2mr
		self.m2u = m2u
		self.m2ur = m2ur
		self.alpha = alpha
		self.dist = [0.3, 0.3, 0.4]

def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
	parser.add_argument('--dir', required=False, default='./processed_data/',
                        help='movieId mapping')
	parser.add_argument('--output', required=False, default='walks.txt',
						help='output file name for walks')
	parser.add_argument('--num_walks', required=False, type=int, default=100,
						help='number of walks generated for each user')
	parser.add_argument('--length', required=False, type=int, default=80,
						help='length of each walk')
	parser.add_argument('--alpha', required=False, type=float, default=1.0,
						help='adjust softmax')
	parser.add_argument('--num-threads', default=pymp.config.num_threads[0], type=int,
                        help='Number of threads used for coarsening')
	args = parser.parse_args()
	return args

def readInfo(args):
	"""
	Read statistics of the dataset, and create id2type.txt
	"""
	f = open("/".join([args.dir, 'datainfo.md']), "r")
	nums = {}
	print("**********")
	for line in f:
		eles = line.strip().split()
		if len(eles) > 1 and eles[1].isnumeric():
			nums[eles[0]] = int(eles[1])
			print("%s %d"%(eles[0], nums[eles[0]]))
	print("**********")
	num_movies, num_genres, num_cast, num_users, total = nums['num_movies'], nums['num_genres'], nums['num_cast'], nums['num_users'], nums['total']
	f.close()

	# create id2type
	f = open("/".join([args.dir, 'id2type.txt']), "w")
	id_base = 0
	for i in range(num_movies):
		f.write("%d movie\n"%(i))
	id_base += num_movies
	for i in range(num_genres):
		f.write("%d genre\n"%(i + id_base))
	id_base += num_genres
	for i in range(num_cast):
		f.write("%d cast\n"%(i + id_base))
	id_base += num_cast
	for i in range(num_users):
		f.write("%d user\n"%(i + id_base))
	id_base += num_users
	assert id_base == total

	f.close()
	return num_movies, num_genres, num_cast, num_users, total

def readMap(args, filename):
	"""
	Read txt data file of relationship between movies and genre/cast
	"""
	print("read %s\n**********"%(filename))
	f = open("/".join([args.dir, filename]), "r")
	a2b = defaultdict(list)
	b2a = defaultdict(list)

	for line in f:
		eles = line.strip().split()
		lid, num_attr = int(eles[0]), int(eles[1])
		if len(eles) > 2:
			a2b[lid] = [int(ele) for ele in eles[2:]]
			for ele in eles[2:]:
				b2a[int(ele)].append(lid)
	f.close()
	return a2b, b2a

def readRating(args):
	print("read rating_train.csv\n**********")
	f = open("/".join([args.dir, "rating_train.csv"]), "r")
	u2m = defaultdict(list)
	u2mr = defaultdict(list)
	m2u = defaultdict(list)
	m2ur = defaultdict(list)

	for line in f:
		eles = line.strip().split(",")
		if not eles[0].isnumeric():  # ignore header line
			continue
		uId, mId, binary = [int(ele) for ele in eles[:-1]]
		rating = float(eles[-1])
		u2m[uId].append(mId)
		u2mr[uId].append(rating)
		m2u[mId].append(uId)
		m2ur[mId].append(rating)

	f.close()
	return u2m, u2mr, m2u, m2ur

def randomWalk(uId, data, length):
	walk = [uId]
	last_rating = -1
	while (len(walk) < length):
		''' randomly pick a metapath
		0: U-M-U
		1: U-M-G-M-U
		2: U-M-C-M-U
		'''
		if length - len(walk) < 4:
			walkType = 0
		else:
			walkType = np.random.choice(3, size=1, p=data.dist)

		current = walk[-1]
		nextstep = None
		# pick a M
		while True:
			if last_rating == -1:
				nextstep = np.random.choice(len(data.u2m[current]))
			else:
				nextstep = np.random.choice(len(data.u2m[current]), p=softmax(-data.alpha * np.abs(last_rating - np.array(data.u2mr[current]))))
			mId = data.u2m[current][nextstep]

			if (walkType == 0) \
				or (walkType == 1 and len(data.m2g[mId]) > 0) \
				or (walkType == 2 and len(data.m2c[mId]) > 0) :
				# ensure that the movie has at least one cast/genre attribute if walkType > 0
				break
		last_rating = data.u2mr[current][nextstep]
		nextstep = data.u2m[current][nextstep]
		walk.append(nextstep)
		current = nextstep  # M

		if walkType == 1:
			nextstep = np.random.choice(data.m2g[current])
			walk.append(nextstep)
			current = nextstep  # G

			while True:
				nextstep = np.random.choice(data.g2m[current])
				if len(data.m2u[nextstep]) > 0:
					break
			walk.append(nextstep)
			current = nextstep  # M (s.t. len(m2u) > 0)
		elif walkType == 2:
			nextstep = np.random.choice(data.m2c[current])
			walk.append(nextstep)
			current = nextstep  # C

			while True:
				nextstep = np.random.choice(data.c2m[current])
				if len(data.m2u[nextstep]) > 0:
					break
			walk.append(nextstep)
			current = nextstep  # M (s.t. len(m2u) > 0)

		# the last user
		nextstep = np.random.choice(len(data.m2u[current]), p=softmax(-data.alpha * np.abs(last_rating - np.array(data.m2ur[current]))))
		last_rating = data.m2ur[current][nextstep]
		nextstep = data.m2u[current][nextstep]
		walk.append(nextstep)
		current = nextstep  # U
	return walk


if __name__ == "__main__":
	args = args_parser()

	num_movies, num_genres, num_cast, num_users, total = readInfo(args)

	m2g, g2m = readMap(args, "mId2Genre.txt")
	m2c, c2m = readMap(args, "mId2CC.txt")
	u2m, u2mr, m2u, m2ur = readRating(args)

	data = dataset(num_movies, num_genres, num_cast, num_users, total, m2g, g2m, m2c, c2m, u2m, u2mr, m2u, m2ur, args.alpha)

	f = open("/".join([args.dir, args.output]),"w")
	for uId in tqdm(u2m.keys()):
		if len(u2m[uId]) > 0:
			walks = [None] * args.num_walks
			with pymp.Parallel(args.num_threads) as p:
				for i in p.range(args.num_walks):
					walks[i] = randomWalk(uId, data, args.length)
			for i in range(num_walks):
				f.write("%s\n"%(walks[i]))
	f.close()

