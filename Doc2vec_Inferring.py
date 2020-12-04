from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

file_name = 'Doc2vec_overview.model'
model = Doc2Vec.load(file_name)
# Infer a vector for a new doc
stoplist = set('for a of the and to in'.split(' '))
test = ["Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstances separate Buzz and Woody from their owner, the duo eventually learns to put aside their differences"]
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in test]
print(f"test sample: {texts}")
vector = model.infer_vector(texts[0])
print(f"vector: {vector}")
print(f"Size of corpus: {len(model.wv.vocab)}")   # Now the vocab contains 28322 uinque words
print(f"Dimension of the the entire corpus: {model.wv.vectors.shape}")
print(f"Similarity word for 'beautiful': {model.wv.most_similar('beautiful')}")
print(f"Check the dim of single word: {model.wv['right']}")