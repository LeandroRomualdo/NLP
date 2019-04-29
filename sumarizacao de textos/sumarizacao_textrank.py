# -*- coding: utf-8 -*- 

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import re
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')
nltk.download('punkt')


# Stopwords em portugues
stop_words = stopwords.words('portuguese')


# Função que remove stopwords.
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    
    return sen_new    

def sumarizacao(doc_input):
    sentences = []
    for s in doc_input:
        sentences.append(sent_tokenize(s))

    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    #clean_sentences = [s.lower() for s in clean_sentences]

    #sentences = [y for x in sentences for y in x]
    #clean_sentences = [remove_stopwords(r for r in clean_sentences)]

    # Extract word vectors
    word_embeddings = {}
    f = open('glove_s600.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])#, dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if i == i:#len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    
    # Matrix de similaridade
    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    ## Cria grapho
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    sn = 3
    for i in range(sn):
        return(print(ranked_sentences[i][1]))