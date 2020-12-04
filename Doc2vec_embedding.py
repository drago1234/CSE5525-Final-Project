''' Reference
1) Tokenizer: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#sequences_to_texts
2) gensim.Doc2vec: https://radimrehurek.com/gensim/models/doc2vec.html
3) Medium tutorial for Doc2vec: https://medium.com/@klintcho/doc2vec-tutorial-using-gensim-ab3ac03d3a1
4) Kaggle tutorial: https://www.kaggle.com/farsanas/are-you-ready-to-build-your-own-word-embedding
'''
# Tokenize the sentences
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np

# Ref: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
def tokenize_overview(sentences, vocab_size=10000, oov_tok='<ooV>', max_length=32, padding_type='post', trunc_type='post'):
    ''' 
        Args: 
            - num_words: int, Size of word corpus
            - oov_token: str, out of word token
            - maxlen: int, maimum length of sequences(if not provided it will be the longest individual sequences)
            - padding：str, 'pre' or 'post' (optional, defaults to 'pre'): pad either before or after each sequence.
            - truncating: str, 'pre' or 'post' (optional, defaults to 'pre'): remove values from sequences larger than maxlen, either at the beginning or at the end of the sequences.
        Return:
            - overview_padded: Numpy array with shape (len(sequences), maxlen), embedding sequence
            - word_index: 
    '''
    
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    # Ref: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer#texts_to_sequences
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded_sequences, word_index


overview = 'processed_data/overviews.csv'
data = pd.read_csv(overview)
overview_sentences = data['overview'].astype(str)
mId = data['mId'].astype(int)
print("Number of sentence: %d"%(len(overview_sentences)) )

import pprint
# Some preprocessing
stoplist = set('for a of the and to in'.split(' '))
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in overview_sentences]

print(texts[:1])

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
# pprint.pprint(processed_corpus)

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
documents = [TaggedDocument(texts[i], [mId[i]]) for i in range(len(texts))] # You need the [] for mId[i], because both input need to be iterable, e.g., str, list, dict
print(type(documents))
print(f"Shape of document: {len(documents)}")
print(f"Sample of documents: {documents[:2]}")
model = Doc2Vec(documents, epochs=20, vector_size=128, window=10, min_count=5, workers=4)
''' 
    - documents: iterable of list of TaggedDocument
    - vector_size=128: int, Dimensionality of the feature vectors.
    - window=5: int, The number of neighbors on the either side of a word
    - alpha: float, The initial learning rate.
    - min_count=10: int, we want words appearing atleast 10 times in the vocab otherwise ignore 
    - workers=4: int, Use these many worker threads to train the model (=faster training with multicore machines).
'''

print(f"Model: {model}")
file_name = 'Doc2vec_overview.model'
model.save(file_name)
# If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

