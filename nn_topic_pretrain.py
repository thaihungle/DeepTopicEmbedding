import os
import sys
import numpy as np
import re
from collections import OrderedDict as od 
import operator
from keras.datasets import imdb

def get_topic_emb(beta_file):
    beta_mat=[]
    print('load beta file...')
    with open(beta_file) as f:
        for line in f:
            row=[]
            for num in re.compile(r"\s+").split(line.strip()):
                row.append(float(num))
            beta_mat.append(row)
    print('done load beta!')
    beta_mat=np.asarray(beta_mat,dtype='float32')
    beta_mat=np.transpose(beta_mat)
    print('beta info:')
    num_word=beta_mat.shape[0]
    num_dim=beta_mat.shape[1]
    print('num word {} vs num dim {}'.format(num_word, num_dim))
    embedding_weights = np.zeros((num_word+1,num_dim))
    embedding_weights[1:,:]=beta_mat
    return embedding_weights

def get_imdb_and_emb(top_words,glove_dir, em_dim=100):
    print('indexing word vectors.')

    embeddings_index = {}
    if em_dim==100:
        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    else:
        assert em_dim==100

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('done load glove!')
    print('start load imdb...')

    word2ind = imdb.get_word_index()
    ind2word={}
    for k,v in word2ind.items():
        ind2word[v]=k
    (X_train, y_train), (X_test, y_test) = imdb.load_data()
    n_train = len(y_train)
    n_test = len(y_test)
    global_count={}

    print('done load imdb')
    print ('start counting ...')

    for i in range(n_train):
        words=X_train[i]
        for w in words:
            if not global_count.get(w):
                global_count[w]=0
            global_count[w]+=1
    for i in range(n_test):
        words=X_train[i]
        for w in words:
            if not global_count.get(w):
                global_count[w]=0
            global_count[w]+=1

    global_count=od(sorted(global_count.items()))
    global_count_sorted=sorted(global_count.items(), key=operator.itemgetter(1), reverse=True)
    print(len(word2ind))
    print(len(global_count))
    #assert len(word2ind)==len(global_count)
    if top_words is None:
        top_words=len(global_count)
    print ('prepare mapping from full to limit index...')
    X_train2=[]
    X_test2=[]
    keep_index={}
    map_new2old={}
    c=1
    for i in range(top_words):
        k,v=global_count_sorted[i]
        keep_index[k]=c
        map_new2old[c]=k
        c+=1

    print('some examples of mapping:')
    c=0
    for k,v in keep_index.items():
        print('keep index {} : {}'.format(k,v))
        print('keep word {} : {}'.format(ind2word[k],ind2word[map_new2old[v]]))
        if c>10:
            break
        c+=1
    print('start limiting data.....')
    for i in range(n_train):
        words=X_train[i]
        words2=[]
        for w in words:
            if w in keep_index:
                words2.append(keep_index[w])
        X_train2.append(words2)

    for i in range(n_test):
        words=X_test[i]
        words2=[]
        for w in words:
            if w in keep_index:
                words2.append(keep_index[w])
        X_test2.append(words2)

    print('done limiting data!')
    print('Build pretrain weight layer...')

    n_symbols = top_words + 1 # adding 1 to account for 0th index (for masking)
    embedding_weights = 2*np.random.rand(n_symbols,em_dim)-1
    # all_index=ntp.get_all_index(X_train, X_test)
    for i in range(top_words):
        index=i+1
        ii=map_new2old[index]
        word=None
        if ind2word.get(ii):
            word=ind2word[ii]
        if embeddings_index.get(word) is not None:
            embedding_weights[index,:] = embeddings_index[word]
    print('Done pretrain weight layer')
    
    return (X_train2, y_train), (X_test2, y_test), embedding_weights


           


def gen_imdb_tf(train_filename, test_filename, top_words):
    print('Start load imdb data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    print('Done load imdb data!')
    n_train = len(y_train)
    n_test = len(y_test)
    print ('num train samples: {}'.format(n_train))
    print ('num test samples: {}'.format(n_test))
    global_count={}
    train_docs=[]
    test_docs=[]
   
    print ('start counting training...')
    for i in range(n_train):
        local_count={}
        words=X_train[i]
        for w in words:
            if not local_count.get(w):
                local_count[w]=0
            local_count[w]+=1
            if not global_count.get(w):
                global_count[w]=0
            global_count[w]+=1
        train_docs.append(local_count)

    print('done counting trainining...')
    print('start write train data')
    with open(train_filename, 'w') as f:
        c=0
        for doc in train_docs:
            doc=od(sorted(doc.items()))
            y=y_train[c]
            f.write(str(y)+' ')
            for k,v in doc.items():    
                f.write(str(k)+':'+str(v)+' ')
            f.write('\n')
            c+=1
    print('done write train data...')
    print ('start counting testing...')

    for i in range(n_test):
            local_count={}
            words=X_test[i]
            for w in words:
                if not local_count.get(w):
                    local_count[w]=0
                local_count[w]+=1
                if not global_count.get(w):
                    global_count[w]=0
                global_count[w]+=1
            test_docs.append(local_count)

    print('done counting test...')
    print('start write test data...')

    with open(test_filename, 'w') as f:
        c=0
        for doc in test_docs:
            doc=od(sorted(doc.items()))
            y=y_test[c]
            f.write(str(y)+' ')
            for k,v in doc.items():    
                f.write(str(k)+':'+str(v)+' ')
            f.write('\n')
            c+=1
    print('done write test data...')

if __name__ == '__main__':
    gen_imdb_tf('./data/count_data/imdb_raw_train.dat','./data/count_data/imdb_raw_test.dat',30000)
    #get_imdb_and_emb(5000,'./embfiles/glove/')
    #get_topic_emb('./data/count_data/fstm.beta')