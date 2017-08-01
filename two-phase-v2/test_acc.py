import sys, os
import time
import shutil
import re
import numpy

if (__name__ == '__main__'):
  algo_name = sys.argv[1]
  r1 = float(sys.argv[2])
  r2 = float(sys.argv[3])
  setting_file = sys.argv[4]
  param = sys.argv[5]
  print algo_name
  r=numpy.linspace(r1,r2,10)
  if param=='num_topics':
    r=[20, 40, 80, 100]
  if param=='alpha':
    r=[0.001, 0.01, 0.1]
  if param=='eta':
    r=[0.001, 0.01, 0.1]
  if param=='tau0':
    r=[1, 10, 100]
  if param=='kappa':
    r=[0.5, 0.6, 0.7, 0.8, 0.9, 1]
  if param=='topics':
    r=[20, 40, 80, 100]
  print r
  f2=open('test_results/acc_%s_%s.txt' % (param,algo_name),'w')
  for alpha in r:
    lines = open(setting_file,'r').readlines()
    if param=='num_topics':
      lines[2]='num_topics: %s\n' % alpha
    if param=='alpha':
      lines[4]='alpha: %s\n' % alpha
    if param=='eta':
      lines[5]='eta: %s\n' % alpha
    if param=='tau0':
      lines[6]='tau0: %s\n' % alpha
    if param=='kappa':
      lines[7]='kappa: %s\n' % alpha
    f=open(setting_file,'w')
    f.writelines(lines)
    f.close()
    start_time=time.time()
    command='python sdr_ope.py fstm 40 data/news20.dat data/news20.t news ./OPE-master/settings.txt %s' %(algo_name)
    os.system(command)
    accuracy='_news_fstm40/Accuracy-sin1000-k20-ld0.10.dat'
    acc=re.findall('\d+\.\d*',open(accuracy).readline().strip())[0] 
    t=time.time()-start_time
    print '-----%s seconds with acc %s------' %(t,acc)
    f2.write('%s %s\n' % (str(alpha), str(acc)))
  f2.close()