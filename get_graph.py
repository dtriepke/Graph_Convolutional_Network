import pandas as pd 
import numpy as np
import keras
from sklearn.feature_extraction.text import TfidfVectorizer 
import pickle
import os
from collections import OrderedDict
import networkx as nx

from tqdm import tqdm
from itertools import combinations
import math

def nCr(n, r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))



def create_graph(docs):

    tfidf = TfidfVectorizer(input = "content")
    tfidf_vector = tfidf.fit(docs) 
    df_tfidf = tfidf_vector.transform(docs)
    df_tfidf = df_tfidf.toarray()

    # Transform to a data frame
    vocab = tfidf.get_feature_names(); vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)


    # Filter the docs by by all word including in the vocab (tfidf applies a stop word remover)
    names = vocab # vocab from tf-idf calculation
    docs_filtered = list(map(lambda x: " ".join([w for w in x.split() if w in names]), docs)) 

    # Dictionaries and lookups
    word_index = OrderedDict( (name, index) for index, name in enumerate(names) )
    index_word = OrderedDict( (idx, name )  for idx, name in enumerate(names) ) 

    word_word_index = OrderedDict()
    # index_word_word = OrderedDict()

    for w1, w2 in tqdm(combinations(names, 2), total = nCr(len(names), 2)):
        word_word = "{}_{}".format(w1,w2)
        word_word_index[word_word] = len(word_word_index)
    #     index_word_word[len(word_word_index)] = word_word
        

    # This function help later to access the word word dictionary because for w1_w2 = w2_w1 is not both represented
    def get_word_word_index(w1, w2):
        word_word = "{}_{}".format(w1, w2)
        idx = word_word_index.get(word_word, "?")
        
        if  idx == "?":
            word_word = "{}_{}".format(w2, w1)
            idx = word_word_index.get(word_word, "?")
        
        return idx

    # Find the co-occurrences:
    window_size = 10 # sliding window size to calculate point-wise mutual information between words

    # Window counter
    W = 0 # total number of sliding windows in the corpus
    W_ij = np.zeros(len(word_word_index), dtype = np.int32) # OrderedDict((ww, 0) for ww in word_word_index.items()) # co-occurencies of word i and word j
    W_i  = OrderedDict((name, 0) for name in names) # occurencies of word i


    for doc in tqdm(docs_filtered, total = len(docs_filtered)):
        
        # In each doc run a sliding window with size 10
        doc = doc.split()
        for idx in range(len(doc) - window_size):
            W += 1 # count total windows
            
            words = set(doc[idx:(idx + window_size)]) # distict list of words in the current window
        
            # count windows conaining word i
            for word in words:
                W_i[word] += 1
            
            # count windows containing co-occurrences of words i and j 
            for i in combinations(words, 2):
                w1 = i[0]
                w2 = i[1]
                idx = get_word_word_index(w1, w2)
                W_ij[idx] += 1 # Update window counter


    # Now can we calculate the probabilities and the PMI score.
                
    # Relative frequency
    p_ij = pd.Series(W_ij, index = word_word_index.keys()) / W
    p_i = pd.Series(W_i, index = W_i.keys()) / W
    pmi_ij = p_ij.copy() 

    # Calculate the PMI
    for w in tqdm(combinations(names, 2), total = nCr(len(names), 2)):
        i = w[0]
        j = w[1] 
        word_word = "{}_{}".format(i, j)
        
        try:
            frac = p_ij[word_word] / (p_i[i] * p_i[j])
            frac = frac + 1E-9 
            pmi = math.log(frac)
            pmi_ij[word_word] = round(pmi, 4)
        except:
            print("ERROR with: ", word_word, frac)
        

    import networkx as nx

    def word_word_edges(pmi_ij):
        word_word_edges_list = []
        for w in tqdm(combinations(names, 2), total = nCr(len(names), 2)):
            i = w[0]
            j = w[1] 
            word_word = "{}_{}".format(i, j)
            pmi = pmi_ij.loc[word_word]
            if (pmi > 0):
                word_word_edges_list.append((i, j,{"weight": pmi}))
                
        return word_word_edges_list

    # Build graph with document nodes and word nodes
    G = nx.Graph()
    G.add_nodes_from(df_tfidf.index) # document nodes as index
    G.add_nodes_from(vocab) # word nodes

    # Build edges between document-word pairs
    document_word = [(doc, w, {"weight" : df_tfidf.loc[doc,w]}) for doc in df_tfidf.index for w in df_tfidf.columns if df_tfidf.loc[doc,w] > 0]
    G.add_edges_from(document_word)

    # Build edges between word-word pairs
    word_word = word_word_edges(pmi_ij)
    G.add_edges_from(word_word)

    return G, word_word, pmi_ij, names




def create_graph_debug(docs):

    tfidf = TfidfVectorizer(input = "content")
    tfidf_vector = tfidf.fit(docs) 
    df_tfidf = tfidf_vector.transform(docs)
    df_tfidf = df_tfidf.toarray()

    # Transform to a data frame
    vocab = tfidf.get_feature_names(); vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns=vocab)
    
    # Filter the docs by by all word including in the vocab (tfidf applies a stop word remover)
    names = vocab # vocab from tf-idf calculation
    docs_filtered = list(map(lambda x: " ".join([w for w in x.split() if w in names]), docs)) 

    n_i  = OrderedDict((name, 0) for name in names)
    word2index = OrderedDict( (name,index) for index,name in enumerate(names) )
    occurrences = np.zeros( (len(names),len(names)) ,dtype=np.int32)
    no_windows = 0
    window = 10

    for l in tqdm(docs_filtered, total=len(docs_filtered)):
        l = l.split()
        
        for i in range(len(l)-window):
            no_windows += 1
            d = set(l[i:(i+window)])
            
            for w in d:
                n_i[w] += 1

            for w1,w2 in combinations(d,2):
                i1 = word2index[w1]
                i2 = word2index[w2]

                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1
            
            
    ### convert to PMI
    p_ij_o = pd.DataFrame(occurrences, index = names,columns=names)/no_windows
    p_i_o = pd.Series(n_i, index=n_i.keys())/no_windows

    for col in p_ij_o.columns:
        p_ij_o[col] = p_ij_o[col]/p_i_o[col]

    for row in p_ij_o.index:
        p_ij_o.loc[row,:] = p_ij_o.loc[row,:]/p_i_o[row]

    p_ij_o = p_ij_o + 1E-9

    for col in p_ij_o.columns:
        p_ij_o[col] = p_ij_o[col].apply(lambda x: math.log(x))
        
        
    def word_word_edges_o(p_ij):
        word_word = []
        cols = list(p_ij.columns); 
        cols = [str(w) for w in cols]
        for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
            if (p_ij.loc[w1, w2] > 0):
                word_word.append((w1,w2,{"weight": round(p_ij.loc[w1,w2],4)}))
        return word_word


        
    word_word_o = word_word_edges_o(p_ij_o)

    # Build graph with document nodes and word nodes
    G_o = nx.Graph()
    G_o.add_nodes_from(df_tfidf.index) # document nodes as index
    G_o.add_nodes_from(vocab) # word nodes

    # Build edges between document-word pairs
    document_word = [(doc, w, {"weight" : df_tfidf.loc[doc,w]}) for doc in df_tfidf.index for w in df_tfidf.columns if df_tfidf.loc[doc,w] > 0]
    G_o.add_edges_from(document_word)

    # Build edges between word-word pairs
    G_o.add_edges_from(word_word_o)

    return G_o, word_word_o,  list(p_ij_o.columns)