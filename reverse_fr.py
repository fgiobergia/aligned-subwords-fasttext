import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors
import fasttext
import pickle
import os 

def find_matrix(lang, step=1000):
    
    print("LANG", lang)
    dict = {}

    # not aligned
    src = fasttext.load_model(f"/data1/malto/csavelli/aligned_subwords_fasttext/wiki/wiki.fr.bin")
    dst = KeyedVectors.load_word2vec_format(f"/data1/malto/csavelli/aligned_subwords_fasttext/aligned/wiki.{lang}.align.vec") # aligned
    
    if src.words != dst.index_to_key:
        print("src and dst vocabularies differ. ")
        print("src", len(src.words))
        print("dst", len(dst))
        print("in src, not in dst", set(src.words) - set(dst.index_to_key))
        print("in dst, not in src", set(dst.index_to_key) - set(src.words))

    vocab = list(set(src.words) & set(dst.index_to_key))
    
    dict["missing_elements"] = [set(src.words) - set(dst.index_to_key), set(dst.index_to_key) - set(src.words)] # missing words in common vocabulary

    vocab = sorted(list(set(src.words) & set(dst.index_to_key)))
        
    Y = dst[vocab]
    X = np.zeros((len(vocab), 300))
    for i, word in enumerate(vocab):
        X[i] = src[word]

    W_ = np.linalg.pinv(X) @ Y

    vectors = src.get_input_matrix()
    matrix_in = vectors @ W_
    matrix_in = matrix_in / np.linalg.norm(matrix_in, axis=1).reshape(-1,1)
    src.set_matrices(input_matrix=matrix_in, output_matrix=src.get_output_matrix())

    return src

            
if __name__ == "__main__":
     
    lang = "fr"        
    src = find_matrix(lang)
    os.makedirs(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}", exist_ok=True)
    src.save_model(f"/data1/malto/csavelli/aligned_subwords_fasttext/res/{lang}/wiki.{lang}.pkl")