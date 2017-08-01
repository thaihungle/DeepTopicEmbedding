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
       print(len(sys.argv))
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
    print (infer)
    os.system(infer)
    print '-----%s seconds------' %(time.time()-start_time)

    # Find nearest neighbors
    print '\n==========Find nearest neighbors====='
    infer = './knn2 %s %s.k%s %s 2' % (train_data, train_data, kNN, kNN)
    os.system(infer)
    
    # Find new space by the Two-step framework
    print '\n==========Learning the discriminative space====='
    infer = './infer %s sin %s %s %s.k%s %s ./' % (model, it, train_data, train_data, kNN, topics)
    print(infer)
    os.system(infer)
    
    # Infer
    print '\n==========Projecting data onto the discriminative space====='
    model2 = 'sin%d-k%d-ld%1.2f' % (R, kNN, lamb)
    infer = './%s-run inf-train %s %s %s %s' % (model, model2, it, train_data, topics)
    os.system(infer)
    infer = './%s-run inf-test %s %s %s %s' % (model, model2, it, test_data, topics)
    os.system(infer)
    
    #print '\n==========Projecting data onto the undiscriminative space====='
    #infer = './%s-run inf-test %s %s %s %s' % (model, model, it, test_data, topics)
    #os.system(infer)

    #Classication
    print '==========Doing classification=========='
    print '    in the discriminative space...'
    model2 = '_%s_%s%s' % (it, model, topics) 
    data = '%s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
    cmodel = '%s/zmodel-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
    infer = 'liblinear-1.8/train -s 4 %s %s' % (data, cmodel)
    os.system(infer)
    data = '%s/final-sin%d-k%d-ld%1.2f-inf-test-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
    predict = '%s/predict-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
    accuracy = '%s/Accuracy-sin%d-k%d-ld%1.2f.dat' % (model2, R, kNN, lamb)
    infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
    os.system(infer)
    
    #unsupervised learning
    print '    in the unsupervised space...'
    print ' infer new representation of train data   '
    infer = './%s-run inf-train %s %s %s %s' % (model, model, it, train_data, topics)
    os.system(infer)
    print ' infer new representation of test data   '
    infer = './%s-run inf-test %s %s %s %s' % (model, model, it, test_data, topics)
    os.system(infer)
    data = '%s/final-%s-inf-train-topics-docs-contribute.dat' % (model2, model)
    cmodel = '%s/zmodel-%s-s4.dat' % (model2, model)
    infer = 'liblinear-1.8/train -s 4 %s %s' % (data, cmodel)
    os.system(infer)
    data = '%s/final-%s-inf-test-topics-docs-contribute.dat' % (model2, model)
    predict = '%s/predict-%s-s4.dat' % (model2, model)
    accuracy = '%s/Accuracy-%s.dat' % (model2, model)
    infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
    os.system(infer)
