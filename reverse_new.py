import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
import pickle
import os 

def find_matrix(lang, step=1000):
    
    print("LANG", lang)
    dict = {}

    # not aligned
    try:
        src = load_facebook_vectors(f"/data1/malto/csavelli/aligned_subwords_fasttext/wiki/wiki.{lang}.bin") 
        print("Loaded fastText vectors")
    except:
        print("Going to 'vec'")
        src = KeyedVectors.load_word2vec_format(f"/data1/malto/csavelli/aligned_subwords_fasttext/wiki/wiki.{lang}.vec")
    dst = KeyedVectors.load_word2vec_format(f"/data1/malto/csavelli/aligned_subwords_fasttext/aligned/wiki.{lang}.align.vec") # aligned
    
    if src.index_to_key != dst.index_to_key:
        print("src and dst vocabularies differ. ")
        print("src", len(src))
        print("dst", len(dst))
        print("in src, not in dst", len(set(src.index_to_key) - set(dst.index_to_key)))
        print("in dst, not in src", len(set(dst.index_to_key) - set(src.index_to_key)))
    
    dict["missing_elements"] = [set(src.index_to_key) - set(dst.index_to_key), set(dst.index_to_key) - set(src.index_to_key)] # missing words in common vocabulary

    vocab = sorted(list(set(src.index_to_key) & set(dst.index_to_key)))
        
    Y = dst[vocab]
    X = src[vocab]

    W_ = np.linalg.pinv(X) @ Y

    prod = (X @ W_)
    prod = prod / np.linalg.norm(prod, axis=1).reshape(-1,1)

    dict["MSE"] = np.square(np.subtract(prod, Y)).sum(axis=1).mean() # mean squared error

    error_couples = []
    right_values = np.array([])

    for i in range(0, len(prod), step):
            M = (prod[i:i+step] @ Y.T)
            v = M.argmax(axis=1)

            # sum of the diagonal
            right_values = np.concatenate((right_values, np.diagonal(M[:,v])))
            for j in range(len(v)):
                if v[j] != i+j: # check that the most vector is the word itself
                    #print("words do not match", i+j, v[j], M[j,v[j]])
                    #print("instead the right word should be ", M[j,j])
                    error_couples.append((i+j, v[j], M[j,v[j]], M[j,j]))

            print(i, "/", len(prod), "done") if i % 50_000 == 0 else None
    
    dict["accuracy"] = (right_values.mean(), right_values.std()) # values of the diagonal of the matrix
    dict["n_errors"] = len(error_couples) # number of errors

    src.vectors = src.vectors @ W_
    src.vectors_ngrams = src.vectors_ngrams @ W_
    src.vectors = src.vectors / np.linalg.norm(src.vectors, axis=1).reshape(-1,1)
    src.vectors_ngrams = src.vectors_ngrams / np.linalg.norm(src.vectors_ngrams, axis=1).reshape(-1,1)

    return src, X, Y, W_, right_values, error_couples, dict

            
if __name__ == "__main__":
    
    with open("langs") as f:
        for lang in f:
            lang = lang.strip()  
            print("lang: ", lang)  
            src, X, Y, W_, right_values, error_couples, dict = find_matrix(lang)
            with open(f"res/{lang}.pkl", "wb") as f:
                pickle.dump((X, Y, W_, right_values, error_couples, dict), f)
            #src.save(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/wiki.{lang}.bin")
            os.makedirs(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}", exist_ok=True)
            with open(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}/wiki.{lang}.pkl", "wb") as f:
                        pickle.dump(src, f)