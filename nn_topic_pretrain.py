import os
import sys
import numpy as np
import re
from collections import OrderedDict as od 
import operator
from keras.datasets import imdb
import random
import pickle
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import data_getter as dg

pattern = re.compile(r'\b(' + r'|'.join(nltk.corpus.stopwords.words('english')) + r')\b\s*')
keep_pattern = re.compile('([\W])')

def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def preprocess_raw_imdb_contex(MAX_NB_WORDS, MAX_SENT_LENGTH):
    raw_text=[]
    raw_label=[]

    with open('./data/imdb-train.txt.tok') as f:
        for line in f:
            raw_text.append(line)
    with open('./data/imdb-test.txt.tok') as f:
        for line in f:
            raw_text.append(line)
    with open('./data/imdb-train.cat') as f:
        for line in f:
            if line.strip()=='pos':
                raw_label.append(1)
            else:
                raw_label.append(0)

    with open('./data/imdb-test.cat') as f:
        for line in f:
            if line.strip()=='pos':
                raw_label.append(1)
            else:
                raw_label.append(0)

    print('start get train text...')

    texts=[]
    labels=[]
    for idx in range(len(raw_text)):
        text = clean_str(raw_text[idx])
        texts.append(text)
        labels.append(raw_label[idx])

    labels = to_categorical(np.asarray(labels))

    tokenizer = Tokenizer(filters='"#$%&+,-./;<=>@[\\]_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count = []
    word2ind = {}
    list_num_word = []
    for i, doc in enumerate(texts):
        avg_doc = 0
        k = 1
        wordTokens = text_to_word_sequence(doc)
        list_num_word.append(len(wordTokens))

        if k < MAX_SENT_LENGTH:
            for _, word in enumerate(reversed(wordTokens)):
                if k < MAX_SENT_LENGTH and tokenizer.word_index.get(word) \
                        and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, -k] = tokenizer.word_index[word]
                    word2ind[word] = tokenizer.word_index[word]
                    avg_doc += 1
                    k = k + 1
            test_add_word_count.append(avg_doc)

    print(data[:10])
    print(labels[:10])
    print('===========word statistic========')
    print(sum(test_add_word_count) / len(test_add_word_count))
    print(min(test_add_word_count))
    print(max(test_add_word_count))
    print(sum(list_num_word) / len(list_num_word))
    print(min(list_num_word))
    print(max(list_num_word))
    print('=======================')
    data_train=data[:25000]
    labels_train = labels[:25000]
    data_test = data[25000:]
    labels_test = labels[25000:]
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
    print(labels_test.shape)

    print('start dump....')
    pickle.dump(((data_train, labels_train),
                 (data_test, labels_test),
                 word2ind, (MAX_NB_WORDS, MAX_SENT_LENGTH)),
                open('./data/imdb_prep_context.pkl', 'wb'))
    print('done!!!')




def preprocess_raw_text_ha_chop(MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
                        train_dir='./data/labeledTrainData.tsv',
                        test_dir='./data/testData.tsv'):
    print('start read...')
    data_train = pd.read_csv(train_dir, sep='\t')
    data_test = pd.read_csv(test_dir, sep='\t')
    print('done read!!!')

    from nltk import tokenize

    reviews = []
    labels = []
    texts = []

    pattern = re.compile(r'\b(' + r'|'.join(nltk.corpus.stopwords.words('english')) + r')\b\s*')
    print('start get train text...')
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], "html.parser")
        text = clean_str(text.get_text())
#        text = pattern.sub('', text)
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

        labels.append(data_train.sentiment[idx])

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    word2ind = {}
    list_num_word = []

    for i, doc in enumerate(texts):
        avg_doc = 0
        k = 0
        j = 0
        wordTokens = text_to_word_sequence(doc)
        list_num_word.append(len(wordTokens))
        for _, word in enumerate(wordTokens):

            if j < MAX_SENTS:

                if k < MAX_SENT_LENGTH and tokenizer.word_index.get(word) \
                        and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    word2ind[word] = tokenizer.word_index[word]
                    avg_doc+=1
                    k = k + 1

                if k>=MAX_SENT_LENGTH:
                    k=0
                    j+=1


    print(sum(list_num_word)/len(list_num_word))
    print(max(list_num_word))
    print('Total %s unique tokens.' % len(tokenizer.word_index))
    print('num word in limit text {}'.format(len(word2ind)))

    print('=====some words======')
    print((sorted(word2ind.items(), key=operator.itemgetter(1)))[:10])


    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    data_train = data
    labels_train = labels

    reviews = []
    labels = []
    texts = []

    print('start get test text...')
    for idx in range(data_test.review.shape[0]):
        text = BeautifulSoup(data_test.review[idx], "html.parser")
        text = clean_str(text.get_text())
#        text = pattern.sub('', text)
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        sen = int(data_test.id[idx].split('_')[1])

        if sen > 5:
            labels.append(1)
        else:
            labels.append(0)

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count=0
    strange_words=set()
    list_num_word = []

    for i, doc in enumerate(texts):
        avg_doc = 0
        k = 0
        j = 0
        wordTokens = text_to_word_sequence(doc)
        list_num_word.append(len(wordTokens))
        for _, word in enumerate(wordTokens):

            if j < MAX_SENTS:

                if k < MAX_SENT_LENGTH and tokenizer.word_index.get(word) \
                        and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    word2ind[word] = tokenizer.word_index[word]
                    avg_doc+=1
                    k = k + 1

                if k>=MAX_SENT_LENGTH:
                    k=0
                    j+=1


    print('num word add count {}'.format(test_add_word_count))
    print('num word strange {}'.format(len(strange_words)))


    print('words statistic')
    print(min(list_num_word))
    print(max(list_num_word))
    print(sum(list_num_word) / (1.0 * len(list_num_word)))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)


    data_test = data
    labels_test = labels

    print ('=======================')
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
    print(labels_test.shape)

    print('start dump....')
    pickle.dump(((data_train,labels_train),
                 (data_test,labels_test),
                 word2ind,(MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH)),
                open('./data/imdb_prep.pkl','wb'))
    print('done!!!')



def preprocess_raw_imdb_ha(MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
                           train_dir='./data/labeledTrainData.tsv',
                           test_dir='./data/testData.tsv', is_tf=False):



    print('start read...')
    data_train = pd.read_csv(train_dir, sep='\t')
    data_test = pd.read_csv(test_dir, sep='\t')
    print('done read!!!')

    from nltk import tokenize

    reviews = []
    labels = []
    texts = []
    p_stemmer = PorterStemmer()
    print('start get train text...')
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx],"html.parser")
        text = clean_str(text.get_text())
        # text = pattern.sub('', text)
        # sentences = tokenize.sent_tokenize(text)
        tokens = re.split(keep_pattern, text)
        # stemmed_tokens = [p_stemmer.stem(i)  for i in tokens if i is not ' ' and i is not '']
        text=' '.join(tokens).strip()
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        if  idx%1000==0:
            print('done {}'.format(idx))

        labels.append(data_train.sentiment[idx])

    print(reviews[0][:10])
    print(reviews[1][:10])

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    text_full=texts

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='')
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count = 0
    word2ind={}
    list_num_sen=[]
    list_num_word=[]
    for i, sentences in enumerate(reviews):
        avg_sen = 0
        for j, sent in enumerate(sentences):
            avg_sen += 1
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 1
                list_num_word.append(len(wordTokens))
                for _, word in enumerate(reversed(wordTokens)):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index.get(word)\
                            and tokenizer.word_index[word]<MAX_NB_WORDS:
                        data[i, j, -k] = tokenizer.word_index[word]
                        word2ind[word]=tokenizer.word_index[word]
                        test_add_word_count += 1
                        k = k + 1
        list_num_sen.append(avg_sen)

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    print('num word add count {}'.format(test_add_word_count))
    print('num word in limit text {}'.format(len(word2ind)))

    print('=====some words======')
    print((sorted(word2ind.items(), key=operator.itemgetter(1)))[:10])

    print('sen statistic')
    print(min(list_num_sen))
    print(max(list_num_sen))
    print(sum(list_num_sen)/(1.0*len(list_num_sen)))

    print('words statistic')
    print(min(list_num_word))
    print(max(list_num_word))
    print(sum(list_num_word) / (1.0 * len(list_num_word)))
    y_train=labels
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    data_train = data
    labels_train = labels

    reviews = []
    labels = []
    texts = []

    print('start get test text...')
    for idx in range(data_test.review.shape[0]):
        text = BeautifulSoup(data_test.review[idx], "html.parser")
        text = clean_str(text.get_text())
        # text = pattern.sub('', text)
        # texts.append(text)
        # sentences = tokenize.sent_tokenize(text)
        tokens = re.split(keep_pattern, text)
        # stemmed_tokens = [p_stemmer.stem(i) for i in tokens if i is not ' ' and i is not '']
        text = ' '.join(tokens).strip()
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

        if idx % 1000 == 0:
            print('done {}'.format(idx))

        sen = int(data_test.id[idx].split('_')[1])

        if sen > 5:
            labels.append(1)
        else:
            labels.append(0)

    print(reviews[0][:10])
    print(reviews[1][:10])

    text_full+=texts

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count=0
    strange_words=set()
    list_num_sen = []
    list_num_word = []
    for i, sentences in enumerate(reviews):
        avg_sen = 0
        for j, sent in enumerate(sentences):
            avg_sen += 1
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 1
                list_num_word.append(len(wordTokens))
                for _, word in enumerate(reversed(wordTokens)):
                    if k < MAX_SENT_LENGTH and word2ind.get(word):
                        data[i, j, -k] = tokenizer.word_index[word]
                        test_add_word_count += 1
                        k = k + 1
                    if not word2ind.get(word):
                        strange_words.add(word)

        list_num_sen.append(avg_sen)

    print('num word add count {}'.format(test_add_word_count))
    print('num word strange {}'.format(len(strange_words)))



    print('sen statistic')
    print(min(list_num_sen))
    print(max(list_num_sen))
    print(sum(list_num_sen) / (1.0 * len(list_num_sen)))

    print('words statistic')
    print(min(list_num_word))
    print(max(list_num_word))
    print(sum(list_num_word) / (1.0 * len(list_num_word)))
    y_test=labels
    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)


    if is_tf:
        tokenizer_full = Tokenizer(filters='')
        tokenizer_full.fit_on_texts(text_full)
        X_train=tokenizer_full.texts_to_sequences(text_full[:25000])
        X_test=tokenizer_full.texts_to_sequences(text_full[25000:])
        gen_imdb_tf_from_obj(X_train,y_train,X_test,y_test,
                             './data/count_data/imdb_raw_train_full.dat',
                             './data/count_data/imdb_raw_test_full.dat')

        word_index=od(sorted(tokenizer_full.word_index.items(),key=operator.itemgetter(1)))
        with open('./data/count_data/imdb_raw_full.tok','w') as f:
            for k,v in word_index.items():
                f.write(k)
                f.write('\n')


    data_test = data
    labels_test = labels

    print ('=======================')
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
    print(labels_test.shape)

    print('start dump....')
    pickle.dump(((data_train,labels_train),
                 (data_test,labels_test),
                 word2ind,(MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH)),
                open('./data/imdb_prep_ha50200.pkl','wb'))
    print('done!!!')


def preprocess_rawbig_imdb_ha(MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH):
    import json
    objs=json.load(open('./data/data.json'))
    X=[]
    y=[]
    for docobj in objs:
        X.append(docobj['review'])
        y.append(docobj['rating'])
    print(len(X))
    print(X[:10])
    print(y[:10])

    num_sam = len(X)
    num_train = int(num_sam * 0.9)
    index_shuf = list(range(num_sam))
    print(index_shuf[:10])
    random.shuffle(index_shuf)
    print(index_shuf[:10])
    print(len(index_shuf))
    X_train=[]
    y_train=[]
    for i in index_shuf:
        X_train.append(X[i])
        y_train.append(y[i])

    X_test = X_train[num_train:]
    X_train = X_train[:num_train]
    y_test = y_train[num_train:]
    y_train = y_train[:num_train]



    from nltk import tokenize

    reviews = []
    labels = []
    texts = []
    p_stemmer = PorterStemmer()
    print('start get train text...')
    for idx in range(len(X_train)):
        text = BeautifulSoup(X_train[idx], "html.parser")
        text = clean_str(text.get_text())
        # text = pattern.sub('', text)
        # sentences = tokenize.sent_tokenize(text)
        tokens = re.split(keep_pattern, text)
        # stemmed_tokens = [p_stemmer.stem(i)  for i in tokens if i is not ' ' and i is not '']
        text = ' '.join(tokens).strip()
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        if idx % 1000 == 0:
            print('done {}'.format(idx))

        labels.append(y_train[idx])

    print(reviews[0][:10])
    print(reviews[1][:10])

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='')
    tokenizer.fit_on_texts(texts)

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count = 0
    word2ind = {}
    list_num_sen = []
    list_num_word = []
    for i, sentences in enumerate(reviews):
        avg_sen = 0
        for j, sent in enumerate(sentences):
            avg_sen += 1
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 1
                list_num_word.append(len(wordTokens))
                for _, word in enumerate(reversed(wordTokens)):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index.get(word) \
                            and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, -k] = tokenizer.word_index[word]
                        word2ind[word] = tokenizer.word_index[word]
                        test_add_word_count += 1
                        k = k + 1
        list_num_sen.append(avg_sen)

    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    print('num word add count {}'.format(test_add_word_count))
    print('num word in limit text {}'.format(len(word2ind)))

    print('=====some words======')
    print((sorted(word2ind.items(), key=operator.itemgetter(1)))[:10])

    print('sen statistic')
    print(min(list_num_sen))
    print(max(list_num_sen))
    print(sum(list_num_sen) / (1.0 * len(list_num_sen)))

    print('words statistic')
    print(min(list_num_word))
    print(max(list_num_word))
    print(sum(list_num_word) / (1.0 * len(list_num_word)))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    data_train = data
    labels_train = labels

    reviews = []
    labels = []
    texts = []

    print('start get test text...')
    for idx in range(len(X_test)):
        text = BeautifulSoup(X_test[idx], "html.parser")
        text = clean_str(text.get_text())
        # text = pattern.sub('', text)
        # texts.append(text)
        # sentences = tokenize.sent_tokenize(text)
        tokens = re.split(keep_pattern, text)
        # stemmed_tokens = [p_stemmer.stem(i) for i in tokens if i is not ' ' and i is not '']
        text = ' '.join(tokens).strip()
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)

        if idx % 1000 == 0:
            print('done {}'.format(idx))

        labels.append(y_test[idx])

    print(reviews[0][:10])
    print(reviews[1][:10])

    print('number of docs: {}'.format(len(reviews)))
    print('number of labels: {}'.format(len(labels)))

    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
    test_add_word_count = 0
    strange_words = set()
    list_num_sen = []
    list_num_word = []
    for i, sentences in enumerate(reviews):
        avg_sen = 0
        for j, sent in enumerate(sentences):
            avg_sen += 1
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                k = 1
                list_num_word.append(len(wordTokens))
                for _, word in enumerate(reversed(wordTokens)):
                    if k < MAX_SENT_LENGTH and word2ind.get(word):
                        data[i, j, -k] = tokenizer.word_index[word]
                        test_add_word_count += 1
                        k = k + 1
                    if not word2ind.get(word):
                        strange_words.add(word)

        list_num_sen.append(avg_sen)

    print('num word add count {}'.format(test_add_word_count))
    print('num word strange {}'.format(len(strange_words)))

    print('sen statistic')
    print(min(list_num_sen))
    print(max(list_num_sen))
    print(sum(list_num_sen) / (1.0 * len(list_num_sen)))

    print('words statistic')
    print(min(list_num_word))
    print(max(list_num_word))
    print(sum(list_num_word) / (1.0 * len(list_num_word)))

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    data_test = data
    labels_test = labels

    print('=======================')
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
    print(labels_test.shape)

    print('start dump....')
    dg.split_data(data_train, labels_train,data_test, labels_test,
                  word2ind,MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH,
                  './data/big_imdb_prep_ha50200.pkl',50,50)
    print('done!!!')


def preppare_word2vec():
    from gensim.models import word2vec
    model = word2vec.Word2Vec.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary=True)
    model.save("./data/GoogleNews-vectors-negative300.dat")

def get_glove_emb_100(GLOVE_DIR, word_index, MAX_NB_WORDS):
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    num_embable=0
    embedding_matrix = np.random.random((MAX_NB_WORDS, 100))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            num_embable+=1
    print('Total {} embbed words/ {} total words.'.format(num_embable, len(word_index)))

    return embedding_matrix

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
    embedding_weights = np.random.random((num_word+1,num_dim))
    embedding_weights[1:,:]=beta_mat
    return embedding_weights

def get_topic_emb2(beta_file, word2ind, MAX_NB_WORDS, wordind_file):
    word_index={}
    index_word={}
    with open(wordind_file) as f:
        ind=0
        for l in f:
            index_word[ind]=str(l.strip())
            word_index[index_word[ind]]=ind
            ind+=1

    beta_mat = []
    print('load beta file...')
    with open(beta_file) as f:
        for line in f:
            row=[]
            for num in re.compile(r"\s+").split(line.strip()):
                row.append(float(num))
            beta_mat.append(row)

    beta_mat = np.asarray(beta_mat, dtype='float32')
    beta_mat = np.transpose(beta_mat)
    print('beta info:')
    num_word = beta_mat.shape[0]
    num_dim = beta_mat.shape[1]
    print('num words beta {}'.format(num_word))
    print('num words ha {}'.format(len(word2ind)))
    print('num dims {}'.format(num_dim))

    embedding_weights = np.random.random((MAX_NB_WORDS,num_dim))
    num_embable=0
    for word, i in word2ind.items():
        embedding_vector=None
        if word_index.get(word) and word_index[word]<num_word:
            embedding_vector = beta_mat[word_index[word]]
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_weights[i] = embedding_vector
            num_embable+=1
    print('Total {} embbed words/ {} total words.'.format(num_embable, len(word_index)))

    return embedding_weights

def get_full_imdb():
    path = imdb.get_file('imdb_full.pkl',
                    origin='https://s3.amazonaws.com/text-datasets/imdb_full.pkl',
                    md5_hash='d091312047c43cf9e4e38fef92437263')
    f = open(path, 'rb')
    (training_data, training_labels), (test_data, test_labels) = pickle.load(f)
    return (training_data, training_labels), (test_data, test_labels)


def get_imdb_and_emb(top_words,glove_dir, em_dim=100):
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

    if em_dim<=0:
        return (X_train2, y_train), (X_test2, y_test)
    embeddings_index = {}

    if em_dim==100:
        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Build pretrain weight layer...')

    n_symbols = top_words + 1 # adding 1 to account for 0th index (for masking)
    embedding_weights = 2*np.random.rand(n_symbols,em_dim)-1
    # all_index=ntp.get_all_index(X_train, X_test)

    num_in_web=0
    for i in range(top_words):
        index=i+1
        ii=map_new2old[index]
        word=None
        if ind2word.get(ii):
            word=ind2word[ii]
        if embeddings_index.get(word) is not None:
            embedding_weights[index,:] = embeddings_index[word]
            num_in_web+=1
    print('{} covered by wem/ {} total words kept'.format(num_in_web,top_words))
    print('Done pretrain weight layer')
    
    return (X_train2, y_train), (X_test2, y_test), embedding_weights


def flatten_ha_data(X):
    fX=[]
    print(X.shape)
    for i in range(X.shape[0]):
        doc=[]
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                if X[i,j,k]!=0:
                    doc.append(X[i,j,k])

        fX.append(doc)
        if i%500==0:
            print('done {}'.format(i))
    return fX


def gen_imdb_tf_from_obj(X_train, y_train, X_test, y_test,
                         train_filename, test_filename):

    train_fn_word=train_filename+'.tok'
    test_fn_word = test_filename + '.tok'

    print('done load')
    print('start flatten..')

    print(y_train[:10])
    print(y_test[:10])

    if not isinstance(y_train,list):
        y_train = np.argmax(y_train, 1)
    if not isinstance(y_test, list):
        y_test = np.argmax(y_test, 1)

    print(y_train[:10])
    print(y_test[:10])

    print(X_train[0])
    print(X_train[1])
    print(X_test[0])
    print(X_test[1])

    print('Done load imdb data!')
    n_train = len(y_train)
    n_test = len(y_test)
    print('num train samples: {}'.format(n_train))
    print('num test samples: {}'.format(n_test))
    global_count = {}
    train_docs = []
    test_docs = []

    print('start counting training...')
    for i in range(n_train):
        local_count = {}
        words = X_train[i]
        for w in words:
            if w != 0:
                if not local_count.get(w):
                    local_count[w] = 0
                local_count[w] += 1
                if not global_count.get(w):
                    global_count[w] = 0
                global_count[w] += 1
        train_docs.append(local_count)

    print('done counting trainining...')
    print('start write train data')
    with open(train_filename, 'w') as f:
        c = 0
        for doc in train_docs:
            doc = od(sorted(doc.items()))
            y = y_train[c]
            f.write(str(y) + ' ')
            for k, v in doc.items():
                f.write(str(k) + ':' + str(v) + ' ')
            f.write('\n')
            c += 1
    print('done write train data...')
    print('start counting testing...')

    for i in range(n_test):
        local_count = {}
        words = X_test[i]
        for w in words:
            # print(w)
            if w != 0:
                if not local_count.get(w):
                    local_count[w] = 0
                local_count[w] += 1
                if not global_count.get(w):
                    global_count[w] = 0
                global_count[w] += 1
        test_docs.append(local_count)

    print('done counting test...')
    print('start write test data...')

    with open(test_filename, 'w') as f:
        c = 0
        for doc in test_docs:
            doc = od(sorted(doc.items()))
            y = y_test[c]
            f.write(str(y) + ' ')
            for k, v in doc.items():
                f.write(str(k) + ':' + str(v) + ' ')
            f.write('\n')
            c += 1
    print('done write test data...')

def gen_imdb_tf_form_one_pickle(train_filename, test_filename, inputfn):
    print('start read...')

    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS, MAX_SENT_LENGTH)) \
        = pickle.load(open(inputfn, 'rb'))

    gen_imdb_tf_from_obj(X_train, y_train, X_test, y_test,
                         train_filename, test_filename)

def gen_imdb_tf_form_batch(train_filename, test_filename, input_dir):
    import glob
    all_files = glob.glob(input_dir + '/train/*.pkl')
    X_train=None
    y_train=None
    print('start get obj train')
    for i,fn in enumerate(all_files):
        (Xs, ys) = pickle.load(open(fn, 'rb'))
        if not X_train:
            X_train=Xs
            y_train=ys
        else:
            X_train=np.vstack((X_train,Xs))
            y_train = np.vstack((y_train, ys))
        if i%1000==0:
            print('done {}'.format(i))

    all_files = glob.glob(input_dir + '/test/*.pkl')
    X_test = None
    y_test = None
    print('start get obj test')
    for i, fn in enumerate(all_files):
        (Xs, ys) = pickle.load(open(fn, 'rb'))
        if not X_test:
            X_test = Xs
            y_test = ys
        else:
            X_test = np.vstack((X_test, Xs))
            y_test = np.vstack((y_test, ys))
        if i%1000==0:
            print('done {}'.format(i))

    gen_imdb_tf_from_obj(X_train, y_train, X_test, y_test, train_filename, test_filename)




def gen_imdb_tf_raw_ha(train_filename, test_filename, inputfn):

    print('start read...')

    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH)) \
        = pickle.load(open(inputfn, 'rb'))

    print('done load')
    print('start flatten..')

    print(y_train[:10])
    print(y_test[:10])

    y_train=np.argmax(y_train,1)
    y_test = np.argmax(y_test, 1)

    print(y_train[:10])
    print(y_test[:10])

    #
    # X_train=flatten_ha_data(X_train)
    # X_test=flatten_ha_data(X_test)


    print(X_train[0])
    print(X_train[1])
    print(X_test[0])
    print(X_test[1])

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
        for w in words.flatten():
            if w != 0:
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
            for w in words.flatten():
                # print(w)
                if w !=0:
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


def gen_imdb_tf(train_filename, test_filename, top_words):
    print('Start load imdb data...')
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
    print('Done load imdb data!')
    n_train = len(y_train)
    n_test = len(y_test)
    print('num train samples: {}'.format(n_train))
    print('num test samples: {}'.format(n_test))
    global_count = {}
    train_docs = []
    test_docs = []

    print('start counting training...')
    for i in range(n_train):
        local_count = {}
        words = X_train[i]
        for w in words:
            if not local_count.get(w):
                local_count[w] = 0
            local_count[w] += 1
            if not global_count.get(w):
                global_count[w] = 0
            global_count[w] += 1
        train_docs.append(local_count)

    print('done counting trainining...')
    print('start write train data')
    with open(train_filename, 'w') as f:
        c = 0
        for doc in train_docs:
            doc = od(sorted(doc.items()))
            y = y_train[c]
            f.write(str(y) + ' ')
            for k, v in doc.items():
                f.write(str(k) + ':' + str(v) + ' ')
            f.write('\n')
            c += 1
    print('done write train data...')
    print('start counting testing...')

    for i in range(n_test):
        local_count = {}
        words = X_test[i]
        for w in words:
            if not local_count.get(w):
                local_count[w] = 0
            local_count[w] += 1
            if not global_count.get(w):
                global_count[w] = 0
            global_count[w] += 1
        test_docs.append(local_count)

    print('done counting test...')
    print('start write test data...')

    with open(test_filename, 'w') as f:
        c = 0
        for doc in test_docs:
            doc = od(sorted(doc.items()))
            y = y_test[c]
            f.write(str(y) + ' ')
            for k, v in doc.items():
                f.write(str(k) + ':' + str(v) + ' ')
            f.write('\n')
            c += 1
    print('done write test data...')


if __name__ == '__main__':
    # gen_imdb_tf_raw_ha('./data/count_data/imdb_raw_train3.dat',
    #              './data/count_data/imdb_raw_test3.dat','./data/imdb_prep_stem.pkl')

    #gen_imdb_tf_raw_norm('./data/count_data/imdb_raw_train1.dat',
    #            './data/count_data/imdb_raw_test1.dat','./data/imdb_prep_context.pkl')

    #gen_imdb_tf('./data/count_data/imdb_raw_train.dat','./data/count_data/imdb_raw_test.dat',30000)
    #get_imdb_and_emb(5000,'./embfiles/glove/')
    #get_topic_emb('./data/count_data/fstm.beta')
    # get_full_imdb()

    #preprocess_raw_imdb_ha(MAX_NB_WORDS=30000,
    #                        MAX_SENT_LENGTH=200,
    #                        MAX_SENTS=50, is_tf=True)

   # preprocess_raw_text_ha_chop(MAX_NB_WORDS=50000,
   #                      MAX_SENT_LENGTH=50,
   #                      MAX_SENTS=50)


    # preprocess_raw_imdb_contex(MAX_NB_WORDS=50000,
    #                        MAX_SENT_LENGTH=1000)


    # preprocess_rawbig_imdb_ha(MAX_NB_WORDS=30000,
    #                           MAX_SENT_LENGTH=200,
    #                           MAX_SENTS=20)

    gen_imdb_tf_form_batch('./data/big_imdb_count_data/big_imdb_raw_train.data',
                           './data/big_imdb_count_data/big_imdb_raw_test.data',
                           './data/big_imdb_prep_ha50200/')