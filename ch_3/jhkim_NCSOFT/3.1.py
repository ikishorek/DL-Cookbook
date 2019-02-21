import os
from keras.utils import get_file
import gensim
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def get_pretrain_vector() :
    # https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download
    file = 'GoogleNews-vectors-negative300.bin'
    path = get_file(file + '.gz', 'https://deeplearning4jblob.blob.core.windows.net/resources/wordvectors/%s.gz' % file)

    if not os.path.isdir('pretrain_vector'):
        os.mkdir('pretrain_vector')

    x = os.path.join('pretrain_vector', file)
    if not os.path.isfile(x):
        with open(x, 'wb') as fout:
            zcat = subprocess.Popen(['zcat'],
                                    stdin=open(path),
                                    stdout=fout
                                    )
            zcat.wait()

    return x

bin_file = os.path.join('pretrain_vector', 'GoogleNews-vectors-negative300.bin')
if not os.path.isfile(bin_file) :
    bin_file = get_pretrain_vector()

model = gensim.models.KeyedVectors.load_word2vec_format(bin_file, binary=True) # long time

print(model.most_similar(positive=['espresso']))


def A_is_to_B_as_C_is_to(a, b, c, topn=1):
    a, b, c = map(lambda x:x if type(x) == list else [x], (a, b, c))
    res = model.most_similar(positive=b + c, negative=a, topn=topn)
    if len(res):
        if topn == 1:
            return res[0][0]
        return [x[0] for x in res]
    return None

print(A_is_to_B_as_C_is_to('man', 'woman', 'king'))


for country in 'Italy', 'France', 'India', 'China':
    print('%s is the capital of %s' % (A_is_to_B_as_C_is_to('Germany', 'Berlin', country), country))

for company in 'Google', 'IBM', 'Boeing', 'Microsoft', 'Samsung':
    products = A_is_to_B_as_C_is_to(['Starbucks', 'Apple'], ['Starbucks_coffee', 'iPhone'], company, topn=3)
    print('%s -> %s' % (company, ', '.join(products)))


beverages = ['espresso', 'beer', 'vodka', 'wine', 'cola', 'tea']
countries = ['Italy', 'Germany', 'Russia', 'France', 'USA', 'India']
sports = ['soccer', 'handball', 'hockey', 'cycling', 'basketball', 'cricket']

items = beverages + countries + sports

item_vectors = [(item, model[item]) for item in items if item in model]
vectors = np.asarray([x[1] for x in item_vectors]) # [18, 300]
lengths = np.linalg.norm(vectors, axis=1) # [18, ]
norm_vectors = (vectors.T / lengths).T # [18, 300]

tsne = TSNE(n_components=2, perplexity=10, verbose=2).fit_transform(norm_vectors)

x=tsne[:,0]
y=tsne[:,1]

fig, ax = plt.subplots()
ax.scatter(x, y)

for item, x1, y1 in zip(item_vectors, x, y):
    ax.annotate(item[0], (x1, y1), size=14)

plt.show()