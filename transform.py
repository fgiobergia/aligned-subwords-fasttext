import os 
import sys
import pickle

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

# read txt file for languages 
def read_txt(file):
    with open(file, 'r') as f:
        return f.read().splitlines()

if __name__ == '__main__':
    for lang in read_txt('langs'):
        print(lang)
        try:
            with open(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}/wiki.{lang}.pkl", "rb") as f:
                lang_new = pickle.load(f)
            lang_new.save_word2vec_format(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}/wiki.{lang}.vec")
        except:
            print(f"Error with {lang}")
            continue