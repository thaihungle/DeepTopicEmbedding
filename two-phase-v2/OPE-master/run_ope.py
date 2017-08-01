import sys
sys.path.insert(0, './')
sys.path.insert(0, './common')
sys.path.insert(0, './ML-FW')
sys.path.insert(0, './ML-OPE')
sys.path.insert(0, './Online-FW')
sys.path.insert(0, './Online-OPE')
sys.path.insert(0, './Streaming-FW')
sys.path.insert(0, './Streaming-OPE')
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

'''
example command: python run.py ./data2/news20.dat ./settings.txt ML-FW
'''

if __name__ == '__main__':
  # Get environment variables
  train_file = sys.argv[1]
  setting_file = sys.argv[2]
  algo_name = sys.argv[3]
  test_data_folder = None
  if len(sys.argv)==5:
    test_data_folder = sys.argv[4]
  dataname=train_file.split('/')[-1].split('.')[0]
  modelname='./models/%s/%s' % (algo_name, dataname)
  train_file2=train_file+'.unsup'
  if os.path.isfile(train_file2):
    ut.convert_multilabel_unsupervised(train_file,train_file2)
  train_file=train_file2
  numdoc, numterm=ut.count_data(train_file)
  print numdoc, numterm
  rf=open(setting_file,'r')
  rf.readline()
  rf.readline()
  newstr='num_docs: %d\nnum_terms: %d\n'%(numdoc, numterm)
  wf=open(setting_file,'w')
  wf.write(newstr)
  shutil.copyfileobj(rf,wf)
  wf.close()

  start_time=time.time()
  
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
  print '-----%s seconds------' %(time.time()-start_time)
