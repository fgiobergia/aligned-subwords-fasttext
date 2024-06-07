import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
import pickle

import os 

def create_bin(lang):
    
    print("LANG", lang)

    # not aligned
    try:
        src = load_facebook_vectors(f"/data1/malto/csavelli/aligned_subwords_fasttext/wiki/wiki.{lang}.bin") 
        print("Loaded fastText vectors")
    except:
        print("Going to 'vec'")
        src = KeyedVectors.load_word2vec_format(f"/data1/malto/csavelli/aligned_subwords_fasttext/wiki/wiki.{lang}.vec")
    
    path = "W/"
    file = f"{lang}.pkl"
    with open(os.path.join(path, file), "rb") as f:
        W_ = pickle.load(f)

    src.vectors = src.vectors @ W_
    src.vectors_ngrams = src.vectors_ngrams @ W_
    src.vectors = src.vectors / np.linalg.norm(src.vectors, axis=1).reshape(-1,1)
    src.vectors_ngrams = src.vectors_ngrams / np.linalg.norm(src.vectors_ngrams, axis=1).reshape(-1,1)

    # create a folder for the new vectors
    os.makedirs(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}", exist_ok=True)
    with open(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}/wiki.{lang}.pkl", "wb") as f:
                pickle.dump(src, f)
                
if __name__ == "__main__":
    
    with open("langs") as f:
        for lang in f:
            lang = lang.strip()  
            print("lang: ", lang)          
            create_bin(lang)