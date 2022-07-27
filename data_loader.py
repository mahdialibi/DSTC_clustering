# -*- coding: utf-8 -*-

import os
from collections import Counter

import nltk
import numpy as np
import scipy.io
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import MinMaxScaler
from clip_client import Client

c = Client('grpc://0.0.0.0:51000')

def load_stackoverflow(data_path,text_file,label_file):
    sentences = []
    with open(data_path + text_file, 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:]

        for line in all_lines:
            sentences.append(line)
    
    
    with open(data_path + label_file) as label_file:
        y = np.array(list((list(map(int, label_file.readlines())))))
        
    
    r = c.encode(
        sentences,show_progress=True,batch_size=500
    )
    
    XX = r

    
    return XX, y



def load_search_snippet2(data_path='data/SearchSnippets/'):
    mat = scipy.io.loadmat(data_path + 'SearchSnippets-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat
    sentences = []
    

    with open(data_path + 'SearchSnippets.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:]
        for line in all_lines:
            sentences.append(line)
    
    r = c.encode(
        sentences,show_progress=True,batch_size=500
    )
    XX = r
    
    return XX, y


def load_biomedical(data_path='data/Biomedical/'):
    mat = scipy.io.loadmat(data_path + 'Biomedical-STC2.mat')

    emb_index = np.squeeze(mat['vocab_emb_Word2vec_48_index'])
    emb_vec = mat['vocab_emb_Word2vec_48']
    y = np.squeeze(mat['labels_All'])

    del mat

    

    sentences = []

    with open(data_path + 'Biomedical.txt', 'r') as inp_txt:
        all_lines = inp_txt.readlines()[:]
        for line in all_lines:
            sentences.append(line)
    r = c.encode(
        sentences,show_progress=True,batch_size=500
    )
    XX = r

    
    return XX, y


def load_data(dataset_name):
    print('load data')
    if dataset_name == 'stackoverflow':
        return load_stackoverflow(data_path="data/stackoverflow/",text_file="title_StackOverflow.txt",label_file="label_StackOverflow.txt")
    if dataset_name == 'big_stackoverflow':
        return load_stackoverflow(data_path="data/Big_stackoverflow/",text_file="stackoverflow_titles.csv",label_file="stackoverflow_labels.csv")
    elif dataset_name == 'biomedical':
        return load_biomedical()
    elif dataset_name == 'search_snippets':
        return load_search_snippet2()
    else:
        raise Exception('dataset not found...')
