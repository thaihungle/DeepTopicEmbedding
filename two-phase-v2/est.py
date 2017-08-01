#! /usr/bin/python

# usage: python sdr.py <model> <topics> <train-data> <test-data> <save-folder>
#
#example: python sdr.py fstm 40 data/news20.dat data/news20.t news

import sys, os
import time

def read_settings(setting_file):
    settings = file(setting_file, 'r').readlines()
    R    = int(settings[6].split()[0])
    kNN  = int(settings[8].split()[0])
    lamb = float(settings[9].split()[0])
    return R, kNN, lamb

if (__name__ == '__main__'):

    if (len(sys.argv) != 6):
       print 'usage: python sdr.py <model> <topics> <train-data> <test-data> <save-folder>\n'
       sys.exit(1)

    R, kNN, lamb = read_settings('inf-settings.txt')
    model = sys.argv[1]
    topics = sys.argv[2]
    train_data = sys.argv[3]
    test_data = sys.argv[4]
    it = sys.argv[5]
    start_time=time.time()
    print '\n==========Learning the model====='
    infer = './%s-run est %s %s %s' % (model, it, train_data, topics)
    os.system(infer)
    print '-----%s seconds------' %(time.time()-start_time)