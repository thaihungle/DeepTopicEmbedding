import sys, os

def read_settings(setting_file):
    settings = file(setting_file, 'r').readlines()
    R    = int(settings[6].split()[0])
    kNN  = int(settings[8].split()[0])
    lamb = float(settings[9].split()[0])
    return R, kNN, lamb
  
def read_labels(file_name):
  labels=[]
  for line in open(file_name):
    labels.append(line.split(None,1)[0])
  return labels

def cal_acc(l1,l2):
  c=0.0
  l1=read_labels(l1)
  l2=read_labels(l2)
  for i in range(len(l1)):
    if l1[i]==l2[i]:
	c+=1
  c=c/len(l1)
  return c

if (__name__ == '__main__'):
    R, kNN, lamb = read_settings('inf-settings.txt')
    model = sys.argv[1]
    topics = sys.argv[2]
    train_data = sys.argv[3]
    test_data = sys.argv[4]
    it = sys.argv[5]
    
#Classication

    print '==========Doing classification=========='
    print '**********in the discriminative space...*************'
    model2 = '_%s_%s%s' % (it, model, topics) 
    data = '%s/final-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
    cmodel = '%s/zmodel-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
    infer = 'liblinear-1.8/train -s 4 %s %s' % (data, cmodel)
    print 'comand argument:'
    print data
    print cmodel
    print infer
    os.system(infer)
    data = '%s/final-sin%d-k%d-ld%1.2f-inf-test-topics-docs-contribute.dat' % (model2, R, kNN, lamb)
    predict = '%s/predict-sin%d-k%d-ld%1.2f-s4.dat' % (model2, R, kNN, lamb)
    accuracy = '%s/Accuracy-sin%d-k%d-ld%1.2f.dat' % (model2, R, kNN, lamb)
    infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
    print infer
    os.system(infer)
    print 'accuracy: %f' %(cal_acc(test_data,predict)*100)
    
    #unsupervised learning
    print '***********in the unsupervised space...**************'
    data = '%s/final-%s-inf-train-topics-docs-contribute.dat' % (model2, model)
    cmodel = '%s/zmodel-%s-s4.dat' % (model2, model)
    infer = 'liblinear-1.8/train -s 4 %s %s' % (data, cmodel)
    print 'comand argument:'
    print data
    print cmodel
    print infer
    os.system(infer)
    data = '%s/final-%s-inf-test-topics-docs-contribute.dat' % (model2, model)
    predict = '%s/predict-%s-s4.dat' % (model2, model)
    accuracy = '%s/Accuracy-%s.dat' % (model2, model)
    infer = 'liblinear-1.8/predict %s %s %s > %s' % (data, cmodel, predict, accuracy)
    print infer
    os.system(infer)
    print 'accuracy: %f' %(cal_acc(test_data,predict)*100)
