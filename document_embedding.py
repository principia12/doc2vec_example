from gensim.models.doc2vec import Doc2Vec

import os
import gensim

from preprocess import KoalaTokenizer

def read_corpus(fname):
    with open(fname, 'r') as f, KoalaTokenizer(api_name = 'KKMA') as t:
        for i, line in enumerate(f):
            line = line.strip()
            doc = []
            for sent in t.tagger(line):
                print(sent)
                print(type(sent))
                for word in sent.words:
                    doc.append(word.surface)
            yield gensim.models.doc2vec.TaggedDocument(doc, [i])

import pickle
# train_corpus = list(read_corpus('example.txt'))
# pickle.dump(train_corpus, open('corpus.pickle', 'wb+'))

if 'doc2vec.model' in os.listdir():
    model = pickle.load(open('doc2vec.model', 'rb'))
    train_corpus = pickle.load(open('corpus.pickle', 'rb'))
else:
    train_corpus = list(read_corpus('example.txt'))
    pickle.dump(train_corpus, open('corpus.pickle', 'wb+'))
    embedding_config = {\
        'vector_size' : 50,
        'min_count' : 2,
        'epochs' : 40, }

    model = Doc2Vec(**embedding_config)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    pickle.dump(model, open('doc2vec.model', 'wb+'))

# Pick a random document from the corpus and infer a vector from the model
vectors = []
from collections import defaultdict
sim = defaultdict(list)

if 'sim.pickle' not in os.listdir():

    for doc_id in range(len(train_corpus)):
        inferred_vector = model.infer_vector(train_corpus[doc_id].words)
        vectors.append((inferred_vector, train_corpus[doc_id].words, doc_id))

    def f(a, b):
        from math import sqrt
        return sum([x*y for x, y in zip(a,b)]) / (sqrt(sum([x**2 for x in a])*sum([x**2 for x in b])))

    for v, v_text, v_id in vectors:
        for u, u_text, u_id in vectors:
            sim[v_id].append((u_id, f(u, v)))

    for k, v in sim.items():
        new_v = sorted(v, key = lambda x:x[1], reverse = True)
        sim[k] = new_v

    pickle.dump(sim, open('sim.pickle', 'wb+'))
else:
    sim = pickle.load(open('sim.pickle', 'rb'))

# from pprint import pprint
# pprint(sim)


import random
doc_id = random.randint(0, len(train_corpus) - 1)

# Compare and print the second-most-similar document
print('Train Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))

sim_id = sim[doc_id][1]
print('Similar Document {}: «{}»\n'.format(sim_id, ' '.join(train_corpus[sim_id[0]].words)))