import glob
import os
import random
import pickle
import shutil
import numpy as np

def split_data(X_train,y_train,X_test,y_test, word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
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

def split_full_data(singlefn, BATCH_TRAIN_SIZE=20, BATCH_TEST_SIZE=20):
    print('start load data')
    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH)) \
        = pickle.load(open(singlefn, 'rb'))
    print('done load data...')

    split_data(X_train, y_train, X_test, y_test,
               word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
               singlefn, BATCH_TRAIN_SIZE, BATCH_TEST_SIZE)



class File_Generator:

    def __init__(self, data_dir , num_class):
        self.data_dir=data_dir
        self.num_class=num_class


    def get_meta(self):
        (word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH,
         BATCH_TRAIN_SIZE, BATCH_TEST_SIZE,
         num_sample_train, num_sample_test)=\
            pickle.load(open(self.data_dir + '/meta.pkl','rb'))

        print('========INFO============')
        print((MAX_NB_WORDS,MAX_SENTS,MAX_SENT_LENGTH))
        print((BATCH_TRAIN_SIZE, BATCH_TEST_SIZE))
        print((num_sample_train, num_sample_test))

        return word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH, \
               BATCH_TRAIN_SIZE, BATCH_TEST_SIZE, num_sample_train, num_sample_test

    def train_generator(self):
        all_files=glob.glob(self.data_dir+'/train/*.pkl')
        while 1:
            sall_files=all_files.copy()
            random.shuffle(sall_files)
            for fn in sall_files:
                (Xs,ys)=pickle.load(open(fn,'rb'))
                #return
                yield (Xs,ys)

    def valid_generator(self):
        all_files = glob.glob(self.data_dir+'/test/*.pkl')
        while 1:
            sall_files = all_files.copy()
            random.shuffle(sall_files)
            for fn in sall_files:
                (Xs, ys) = pickle.load(open(fn, 'rb'))
                # return
                yield (Xs, ys)

if __name__ == '__main__':
    split_full_data('./data/imdb_prep_stem.pkl',20, 20)

