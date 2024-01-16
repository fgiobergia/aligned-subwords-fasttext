import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

def find_matrix(lang):
    
    print("LANG", lang)
    # not aligned
    try:
        src = load_facebook_vectors(f"embs/wiki.{lang}.bin") 
        print("Loaded fastText vectors")
    except:
        print("Going to 'vec'")
        src = KeyedVectors.load_word2vec_format(f"embs/wiki.{lang}.vec")
    dst = KeyedVectors.load_word2vec_format(f"embs/wiki.{lang}.align.vec") # aligned
    
    if src.index_to_key != dst.index_to_key:
        print("src and dst vocabularies differ. ")
        print("src", len(src))
        print("dst", len(dst))
        print("in src, not in dst", set(src.index_to_key) - set(dst.index_to_key))
        print("in dst, not in src", set(dst.index_to_key) - set(src.index_to_key))
    
    vocab = list(set(src.index_to_key) & set(dst.index_to_key))
        
    Y = dst[vocab]
    X = src[vocab]

    W_ = np.linalg.pinv(X) @ Y

    prod = (X @ W_)
    prod = prod / np.linalg.norm(prod, axis=1).reshape(-1,1)
    size = min(len(vocab), 10000)
    subs = np.random.choice(len(prod), size, replace=False)
    
    M = (prod[subs] @ Y[subs].T)
    v = M.argmax(axis=1)
    for i in range(size):
        if v[i] != i: # check that the most vector is the word itself
            print("words do not match", i, v[i], M[i,v[i]])
        if M[i,v[i]] < .98:
            print("small similarity" , i, v[i], M[i,v[i]])
    return src, dst, X, Y, W_

            
if __name__ == "__main__":
    
    with open("langs") as f:
        for lang in f:# ["it"]:
            lang = lang.strip()            
            src, dst, X, Y, W_ = find_matrix(lang)