from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np

def args_parser():
	parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
	parser.add_argument('--links', required=False, default='data/links.csv',
                        help='movieId mapping')
	parser.add_argument('--metadata', required=False, default='data/movies_metadata.csv',
                        help='movie metadata file')
	args = parser.parse_args()
	return args

def movieIdIndexing(args):
	links = pd.read_csv(args.links)
	movieIds = links['movieId']
	mvid2mid = dict(zip(movieIds, range(movieIds.size)))
	links['newId'] = links['movieId'].map(mvid2mid)
	tmdbIds = links['tmdbId']
	tmid2mid = dict(zip(tmdbIds, links['newId']))
	return mvid2mid, tmid2mid

def readMovieMetadata(args):
	movies = pd.read_csv(args.metadata, usecols=['genres', 'id', 'overview', 'title'])
	

if __name__ == "__main__":
	args = parse_args()

	''' get movie id mappings from links.csv
	mvid2mid: mapping from 'movieId' to range(45843)
	tmid2mid: mapping from 'tmdbId' to range(45843)
	Note that in links.csv, there are missing values of tmdbId
	'''
	mvid2mid, tmid2mid = movieIdIndexing(args)