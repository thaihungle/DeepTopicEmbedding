import glob
import os
import random
import pickle
import numpy as np

def split_full_data(singlefn, BATCH_TRAIN_SIZE=20, BATCH_TEST_SIZE=20):
    dir, fname = os.path.split(singlefn)
    ((X_train, y_train),
     (X_test, y_test),
     word_index,
     (MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH)) \
        = pickle.load(open(singlefn, 'rb'))
    newdir=dir+'/'+fname
    if os.path.isdir(newdir):
        os.mkdir(newdir)

    train_dir=newdir+'/train'
    if os.path.isdir(train_dir):
        os.mkdir(train_dir)

    test_dir = newdir + '/test'
    if os.path.isdir(test_dir):
        os.mkdir(test_dir)

    num_sample_train=X_train.shape[0]
    num=0

    while num<num_sample_train:
        bfxname=train_dir+'/Xy_train.{}.pkl'.sample(num)
        pickle.dump((X_train[num:num+BATCH_TRAIN_SIZE],y_train[num:num+BATCH_TRAIN_SIZE]), open(bfxname,'wb'))
        num+=BATCH_TRAIN_SIZE

    num=0
    num_sample_test = X_test.shape[0]
    while num<num_sample_test:
        bfxname=test_dir+'/Xy_test.{}.pkl'.sample(num)
        pickle.dump((X_test[num:num+BATCH_TEST_SIZE],y_test[num:num+BATCH_TEST_SIZE]), open(bfxname,'wb'))
        num += BATCH_TEST_SIZE

    pickle.dump((word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH, BATCH_TRAIN_SIZE, BATCH_TEST_SIZE),
                open(newdir+'/meta.pkl'))

    print('done!')



class File_Generator:

    def __init__(self, data_dir , num_class):
        self.data_dir=data_dir
        self.num_class=num_class
        self.train_batch=20
        self.test_batch=20

    def get_meta(self):
        (word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH, BATCH_TRAIN_SIZE, BATCH_TEST_SIZE)=\
            pickle.load(open(self.data_dir + '/meta.pkl'))
        return word_index, MAX_NB_WORDS, MAX_SENTS, MAX_SENT_LENGTH, BATCH_TRAIN_SIZE, BATCH_TEST_SIZE

    def train_generator(self):
        all_files=glob.glob(self.data_dir+'/train')
        while 1:
            sall_files=all_files.copy()
            random.shuffle(sall_files)
            for fn in sall_files:
                (Xs,ys)=pickle.load(open(fn,'rb'))
                #return
                yield (Xs,ys)

    def valid_generator(self):
        all_files = glob.glob(self.data_dir+'/test')
        while 1:
            sall_files = all_files.copy()
            random.shuffle(sall_files)
            for fn in sall_files:
                (Xs, ys) = pickle.load(open(fn, 'rb'))
                # return
                yield (Xs, ys)

if __name__ == '__main__':
    split_full_data('./data/imdb_prep_stem.pkl')

