from sklearn import svm
import os
import gensim
import numpy as np
import random
import geopandas as gpd
import csv


def theta_(x, y) :
    return np.arccos(cosine_sim(x,y)) + np.radians(10)

def euclidean(x, y) :
    return np.linalg.norm(x-y)

def triangle(x, y) :
    theta = np.radians(theta_(x,y))
    return (np.linalg.norm(x) * np.linalg.norm(y) * np.sin(theta)) / 2

def magnitude_difference(x, y) :
    return abs(np.linalg.norm(x) - np.linalg.norm(y))

def sector(x, y) :
    ED = euclidean(x, y)
    MD = magnitude_difference(x, y)
    theta = theta_(x, y)

    return np.pi * np.power((ED+MD),2) * theta/360

def ts_ss(x, y) :
    return -(triangle(x, y) * sector(x, y))

def cosine_sim(x, y) :
    result = np.inner(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    return result


def rank_countries(term, topn=10, field='name'):
    if not term in model:
        return []
    vec = model[term]
    # dists = np.dot(country_vecs, vec)
    # dists = cosine_sim(country_vecs, vec)
    dists = ts_ss(country_vecs, vec)

    return [(countries[idx][field], float(dists[idx]))
            for idx in reversed(np.argsort(dists)[-topn:])]

def map_term(world, term):
    d = {k.upper(): v for k, v in rank_countries(term, topn=0, field='cc3')}
    world[term] = world['iso_a3'].map(d)
    world[term] /= world[term].max()
    world.dropna().plot(term, cmap='OrRd')


bin_file = os.path.join('pretrain_vector', 'GoogleNews-vectors-negative300.bin')
model = gensim.models.KeyedVectors.load_word2vec_format(bin_file, binary=True) # long time
model.most_similar(positive=['Germany'])

countries = list(csv.DictReader(open('data/countries.csv')))

# make positive, negative words
positive = [x['name'] for x in random.sample(countries, 40)]
negative = random.sample(model.vocab.keys(), 5000)

# labelling
labelled = [(p, 1) for p in positive] + [(n, 0) for n in negative]
random.shuffle(labelled)
x = np.asarray([model[w] for w, l in labelled])
y = np.asarray([l for w, l in labelled])

TRAINING_FRACTION = 0.7
cut_off = int(TRAINING_FRACTION * len(labelled))
clf = svm.SVC(kernel='linear')
clf.fit(x[:cut_off], y[:cut_off])

res = clf.predict(x[cut_off:])

missed = [country for (pred, truth, country) in zip(res, y[cut_off:], labelled[cut_off:])
          if pred != truth]

print(100 - 100 * float(len(missed)) / len(res), missed)

print(model.syn0)
all_predictions = clf.predict(model.syn0)
res = []
print()

for word, pred in zip(model.index2word, all_predictions):
    if pred:
        res.append(word)
        if len(res) == 150:
            break
print(random.sample(res, 10))

country_to_idx = {country['name']: idx for idx, country in enumerate(countries)}
country_vecs = np.asarray([model[c['name']] for c in countries])

dists = np.inner(country_vecs, country_vecs[country_to_idx['Canada']])
dists = cosine_sim(country_vecs, country_vecs[country_to_idx['Canada']])
dists = ts_ss(country_vecs, country_vecs[country_to_idx['Canada']])

for idx in reversed(np.argsort(dists)[-10:]):
    print(countries[idx]['name'], dists[idx])

print(rank_countries('cricket'))

# world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# print(map_term(world, 'coffee'))