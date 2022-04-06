import pickle
from scipy.sparse import *
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import tqdm

bert_vocab_size = 30521
cls_token_id = 101  # [self.tokenizer.cls_token_id]
sep_token_id = 102
x_trn = pickle.load(open('data/Wiki10-31K/bert-base/train_encoded.pkl', 'rb'))
x_tst = pickle.load(open('data/Wiki10-31K/bert-base/test_encoded.pkl', 'rb'))

x = x_trn + x_tst

tfidf = lil_matrix((len(x), bert_vocab_size), dtype=np.float32)
for i, doc in enumerate(tqdm(x)):
    for word in doc:
        tfidf[i, word] += 1
    # if i == 10:
    #     break

tfidf = tfidf.tocsr()

# tf = normalize(tfidf, norm='l1', axis=1)
tf = tfidf

idf = tfidf > 0
idf = idf.tocsc()
idf = np.log(tfidf.shape[0]/(np.array(idf.sum(0)) + 1))

# import pdb; pdb.set_trace()
tfidf = tf.multiply(idf).tolil()

for i in range(tfidf.shape[0]):
    tfidf[i, cls_token_id] = 1/8
    tfidf[i, sep_token_id] = 1/8

tfidf = tfidf.tocsr()

tfidf_train = tfidf[:len(x_trn)]
tfidf_test = tfidf[len(x_trn):]

save_npz('data/Wiki10-31K/bert-base/bert_unnorm_tfidf_train', tfidf_train)
save_npz('data/Wiki10-31K/bert-base/bert_unnorm_tfidf_test', tfidf_test)