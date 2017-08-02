import glob
import json
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
import shutil
from nltk.stem.porter import PorterStemmer
import nltk

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

def split_data(X_train,y_train,X_test,y_test, word_index,
               MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
                singlefn, BATCH_TRAIN_SIZE=20, BATCH_TEST_SIZE=20):

    dir, fname = os.path.split(singlefn)

    newdir=dir+'/'+fname.split('.')[0]
    print(newdir)

    if not os.path.isdir(newdir):
        os.mkdir(newdir)

    train_dir=newdir+'/train'
    print(train_dir)
    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)
    else:
        shutil.rmtree(train_dir)
        os.mkdir(train_dir)

    test_dir = newdir + '/test'
    print(test_dir)
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)
    else:
        shutil.rmtree(test_dir)
        os.mkdir(test_dir)
    num_sample_train=X_train.shape[0]
    num=0

    print('dump train data...')

    while num<num_sample_train:
        bfxname=train_dir+'/Xy_train.{}.pkl'.format(num)
        pickle.dump((X_train[num:num+BATCH_TRAIN_SIZE],y_train[num:num+BATCH_TRAIN_SIZE]), open(bfxname,'wb'))
        num+=BATCH_TRAIN_SIZE

    num=0
    num_sample_test = X_test.shape[0]

    print('dump test data')

    while num<num_sample_test:
        bfxname=test_dir+'/Xy_test.{}.pkl'.format(num)
        pickle.dump((X_test[num:num+BATCH_TEST_SIZE],y_test[num:num+BATCH_TEST_SIZE]), open(bfxname,'wb'))
        num += BATCH_TEST_SIZE

    pickle.dump((word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
                 BATCH_TRAIN_SIZE, BATCH_TEST_SIZE,
                 num_sample_train, num_sample_test),
                open(newdir+'/meta.pkl','wb'))

    print('done!')

def preprocess_rawbig_imdb_ha(MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
                              inputfn='./data/data.json',
                              output_dir='./data/big_imdb_prep_ha50200.pkl'):
    objs=json.load(open(inputfn))
    X=[]
    y=[]
    for docobj in objs:
        X.append(docobj['review'])
        y.append(docobj['rating']-1)
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
    split_data(data_train, labels_train,data_test, labels_test,
                  word2ind,MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH,
                  output_dir,20,20)
    print('done!!!')


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
        if len(words.shape)>1:
            words=words.flatten()
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
        if len(words.shape)>1:
            words=words.flatten()
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

def gen_imdb_tf_form_batch(train_filename, test_filename, input_dir):
    all_files = glob.glob(input_dir + '/train/*.pkl')
    X_train=None
    y_train=None
    print('start get obj train')
    for i,fn in enumerate(all_files):
        (Xs, ys) = pickle.load(open(fn, 'rb'))
        if X_train is None:
            X_train=Xs
            y_train=ys
        else:
            X_train=np.concatenate((X_train,Xs), axis=0)
            y_train = np.concatenate((y_train, ys), axis=0)
        if i%100==10:
            print('done {}'.format(i))
            # break

    all_files = glob.glob(input_dir + '/test/*.pkl')
    X_test = None
    y_test = None
    print('start get obj test')
    for i, fn in enumerate(all_files):
        (Xs, ys) = pickle.load(open(fn, 'rb'))
        if X_test is None:
            X_test = Xs
            y_test = ys
        else:
            X_test = np.concatenate((X_test,Xs), axis=0)
            y_test = np.concatenate((y_test,ys), axis=0)
        if i%100==10:
            print('done {}'.format(i))
            # break

    gen_imdb_tf_from_obj(X_train, y_train, X_test, y_test, train_filename, test_filename)

if __name__ == '__main__':
    mode='PREPARE_HA'

    if len(sys.argv) >= 2:
        mode=sys.argv[1]

    if mode=='PREPARE_HA':
        maxnbw = 30000
        maxsenl = 200
        maxsen = 50
        inputfn = './data/data.json'
        output_dir = './data/big_imdb_prep_ha50200.pkl'
        if len(sys.argv) >= 3:
            maxnbw = sys.argv[2]
        if len(sys.argv) >= 4:
            maxsenl = sys.argv[3]
        if len(sys.argv) >= 5:
            maxsen = sys.argv[4]
        if len(sys.argv) >= 6:
            inputfn = sys.argv[5]
        if len(sys.argv) >= 7:
            output_dir = sys.argv[6]

        preprocess_rawbig_imdb_ha(MAX_NB_WORDS=int(maxnbw),
                          MAX_SENT_LENGTH=int(maxsenl),
                          MAX_SENTS=int(maxsen),
                          inputfn=inputfn,
                          output_dir=output_dir)
    elif mode=='PREPARE_SDR':
        output_train_fn='./data/big_imdb_count_data/big_imdb_raw_train.data'
        output_test_fn='./data/big_imdb_count_data/big_imdb_raw_test.data'
        input_dir='./data/big_imdb_prep_ha50200/'
        if len(sys.argv) >= 3:
            output_train_fn = sys.argv[4]
        if len(sys.argv) >= 4:
            output_test_fn = sys.argv[5]
        if len(sys.argv) >= 5:
            input_dir = sys.argv[6]

        gen_imdb_tf_form_batch(output_train_fn, output_test_fn, input_dir)


