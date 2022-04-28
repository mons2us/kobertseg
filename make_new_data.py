import random
import json
from tqdm import tqdm
import numpy as np
import torch
import gc 

import konlpy
from multiprocess import Pool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    word_list = torch.load('word_list.pt')

    tfidf_vectorizer = TfidfVectorizer(min_df=1, max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(word_list)

    tot_num = 100000
    batch_size = 1000

    all_sims = []
    idx = 0
    while idx < tot_num:
        print(idx) if idx % 10000 == 0 else None
        lh, rh = tfidf_matrix[idx:idx+batch_size], tfidf_matrix
        sim = cosine_similarity(lh, rh)
        all_sims.append(sim)
        idx += batch_size

        sim = []
        gc.collect()
        
    all_sims = np.vstack([all_sims])
    all_sims[np.tril_indices(tot_num)] = 0.0
        
    print('done')
        
if __name__ == '__main__':
    main()