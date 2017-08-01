#! /usr/bin/python

# usage: python sdr_ope.py <model> <topics> <train-data> <test-data> <save-folder>
#
#example: python sdr_ope.py fstm 40 data/news20.dat data/news20.t news ./OPE-master/settings.txt ML-FW

import sys, os
sys.path.insert(0, './OPE-master/ML-FW')
sys.path.insert(0, './OPE-master/ML-OPE')
sys.path.insert(0, './OPE-master/Online-FW')
sys.path.insert(0, './OPE-master/Online-OPE')
sys.path.insert(0, './OPE-master/Streaming-FW')
sys.path.insert(0, './OPE-master/Streaming-OPE')
sys.path.insert(0, './OPE-master/common')
import time
import shutil
import utilities as ut
import run_ML_FW as mfw
import run_ML_OPE as mope
import run_Online_FW as ofw
import run_Online_OPE as oope
import run_Streaming_FW as sfw
import run_Streaming_OPE as sope
import os.path


def read_settings(setting_file):
    settings = file(setting_file, 'r').readlines()
    R    = int(settings[6].split()[0])
    kNN  = int(settings[8].split()[0])
    lamb = float(settings[9].split()[0])
    return R, kNN, lamb

if __name__ == '__main__':

    if len(sys.argv) != 9:
       print 'usage: python sdr.py <model> <topics> <train-data> <test-data> <save-folder> <ope-setting> <ope-model> <start-from-step>\n'
       sys.exit(1)

    R, kNN, lamb = read_settings('inf-settings.txt')
    c = 0.01
    model = sys.argv[1]
    topics = sys.argv[2]
    train_data = sys.argv[3]
    test_data = sys.argv[4]
    it = sys.argv[5]
    adv = int(sys.argv[8])
    start_time=time.time()

    if adv < 1:
         print '\n==========Learning the model OPE====='
         train_file = train_data
         setting_file = sys.argv[6]
         algo_name = sys.argv[7]
         test_data_folder = None
         dataname=train_file.split('/')[-1].split('.')[0]
         modelname='./_%s_%s%s' % (it, model,topics)
         train_file2=train_file+'.unsup'
         if not os.path.isfile(train_file2):
           ut.convert_multilabel_unsupervised(train_file,train_file2)
         else:
           print '...dont need to convert data....'
         train_file=train_file2
         numdoc, numterm=ut.count_data(train_file)
         topic_sparsity=0.1
         print numdoc, numterm
         rf=open(setting_file,'r')
         rf.readline()
         rf.readline()
         rf.readline()
         newstr='num_docs: %d\nnum_terms: %d\nnum_topics: %s\n'%(numdoc, numterm, topics)
         wf=open(setting_file,'w')
         wf.write(newstr)
         shutil.copyfileobj(rf,wf)
         wf.close()
         setcontent=open(setting_file,'r').read()
         print (setcontent)


         if algo_name=='ML-FW':
           mfw.run(train_file,setting_file,modelname,test_data_folder)
         if algo_name=='ML-OPE':
           mope.run(train_file,setting_file,modelname,test_data_folder)
         if algo_name=='Online-FW':
           ofw.run(train_file,setting_file,modelname,test_data_folder)
         if algo_name=='Online-OPE':
           oope.run(train_file,setting_file,modelname,test_data_folder)
         if algo_name=='Streaming-FW':
           sfw.run(train_file,setting_file,modelname,test_data_folder)
         if algo_name=='Streaming-OPE':
           sope.run(train_file,setting_file,modelname,test_data_folder)
         f=open('%s/final-fstm.other' % (modelname),'w')
         f.write('num_topics: %s\n' % (topics))
         f.write('num_terms: %s\n' % (numterm))
         f.write('topic_sparsity: %f\n' % (topic_sparsity))
         f.close()
         print ' infer new representation of train data   '
         infer = './%s-run inf-temp %s %s %s %s' % (model, model, it, train_data, topics)
         print infer
         os.system(infer)
         os.rename('%s/final-fstm-inf-temp-topics-docs-contribute.dat' % (modelname), '%s/final-fstm-topics-docs-contribute.dat' % (modelname))
         print '-----%s seconds------' %(time.time()-start_time)

    if adv < 2:
         # Find nearest neighbors
         print '\n==========Find nearest neighbors====='
         infer = './knn2 %s %s.k%s %s 2' % (train_data, train_data, kNN, kNN)
         os.system(infer)

    if adv < 3:
         # Find new space by the Two-step framework
         print '\n==========Learning the discriminative space====='
         infer = './infer %s sin %s %s %s.k%s %s .' % (model, it, train_data, train_data, kNN, topics)
         print(infer)
         os.system(infer)

    if adv < 4:

         # Infer
         print '\n==========Projecting data onto the discriminative space====='
         model2 = 'sin%d-k%d-ld%1.2f' % (R, kNN, lamb)
         infer = './%s-run inf-train %s %s %s %s' % (model, model2, it, train_data, topics)
         print(infer)
         os.system(infer)

         infer = './%s-run inf-test %s %s %s %s' % (model, model2, it, test_data, topics)
         print(infer)
         os.system(infer)

    if adv <=5:
         #Classication
         print '==========Doing classification=========='
         print '    in the discriminative space...'
         model2 = '_%s_%s%s' % (it, model, topics)
         data = '%s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
         cmodel = '%s/zmodel-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
         infer = 'liblinear-1.8/train -s 4 %s %s' % (data, cmodel)
         print (infer)
         os.system(infer)
         data = '%s/final-sin%d-k%d-ld%1.2f-inf-test-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
         predict = '%s/predict-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
         accuracy = '%s/Accuracy-sin%d-k%d-ld%1.2f.dat' % (model2, R, kNN, lamb)
         infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
         print (infer)
         os.system(infer)
         ac=open(accuracy).readlines()
         print "acc: "+''.join(ac)

    if abs(adv) == 5:
         # calc measures
         infer = 'python calc_measures.py %s %s %s/measures_sdrphi.dat' % (predict, test_data, model2)
         os.system(infer)

         #unsupervised learning
         '''
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
         '''

         print '=========start of new represent============'
         # create phi
         data = '%s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
         infer = './new-represent %s %s %s/final-sin%d-k%d-ld%1.2f.beta %s/training.phi' % (train_data, data, model2, R, kNN, lamb, model2)
         os.system(infer)
         print infer
         data = '%s/final-sin%d-k%d-ld%1.2f-inf-test-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
         infer = './new-represent %s %s %s/final-sin%d-k%d-ld%1.2f.beta %s/test.phi' % (test_data, data, model2, R, kNN, lamb, model2)
         os.system(infer)
         print infer

         # new represent
         print '...new represent...'
         data = '%s/training.phi' % (model2)
         cmodel = '%s/zmodel-phi-s1.dat' % (model2)
         infer = 'liblinear-1.8/train -s 1 -c %s %s %s' % (c, data, cmodel)
         os.system(infer)
         data = '%s/test.phi' % (model2)
         predict = '%s/predict-phi-s1.dat' % (model2)
         accuracy = '%s/Accuracy-phi.dat' % (model2)
         infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
         os.system(infer)


         # calc measures
         infer = 'python calc_measures.py %s %s %s/measures_phi.dat' % (predict, test_data, model2)
         os.system(infer)

         print '=========start of new represent 2============'


         infer = './global-representation %s %s/final-sin%s-k%s-ld%1.2f.beta %s/training.gphi %s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat' % (train_data, model2, R, kNN, lamb, model2, model2, R, kNN, lamb)
#		 infer = './global-representation %s %s/final-fstm.beta %s/training.gphi %s/final-fstm-inf-train-topics-docs-#contribute' % (train_data, model2, model2, model2)
         print infer
         os.system(infer)

         infer = './global-representation %s %s/final-sin%s-k%s-ld%1.2f.beta %s/test.gphi %s/final-sin%d-k%d-ld%1.2f-inf-test-topics-docs-contribute.dat' % (test_data, model2, R, kNN, lamb, model2, model2, R, kNN, lamb)
#		 infer = './global-representation %s %s/final-fstm.beta %s/test.gphi %s/final-fstm-inf-test-topics-docs-#contribute' % (test_data, model2, model2, model2)
         print infer
         os.system(infer)


         data = '%s/training.gphi' % (model2)
         cmodel = '%s/zmodel-gphi-s1.dat' % (model2)
         infer = 'liblinear-1.8/train -s 1 -c %s %s %s' % (c, data, cmodel)
         print infer
         os.system(infer)
         data = '%s/test.gphi' % (model2)
         predict = '%s/predict-gphi-s1.dat' % (model2)
         accuracy = '%s/Accuracy-gphi.dat' % (model2)
         infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
         print infer
         os.system(infer)


         # calc measures
         infer = 'python calc_measures.py %s %s %s/measures_gphi.dat' % (predict, test_data, model2)
         os.system(infer)

         print '=========start of new represent 2============'

         ## old new beta
         # # new beta
         print '========New beta========'
         # svm new beta
         #

         infer = './new-beta-v1 %s/final-sin%s-k%s-ld%1.2f.beta %s/final-sin%s-k%s-ld%1.2f.other %s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat %s %s %s/training2.gphi %s/test2.gphi' % (model2, R, kNN, lamb, model2, R, kNN, lamb, model2, R, kNN, lamb, train_data, test_data, model2, model2)
         print infer

         os.system(infer)
         data = '%s/training2.gphi' % (model2)
         cmodel = '%s/zmodel-gphi2-s1.dat' % (model2)
         infer = 'liblinear-1.8/train -s 1 -c %s %s %s' % (c, data, cmodel)
         print infer
         os.system(infer)
         data = '%s/test2.gphi' % (model2)
         predict = '%s/predict-gphi2-s1.dat' % (model2)
    accuracy = '%s/Accuracy-new-topics.dat' % (model2)
    infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
    print infer
    os.system(infer)

    # calc measures
    infer = 'python calc_measures.py %s %s %s/measures_newtopics.dat' % (predict, test_data, model2)
    os.system(infer)
