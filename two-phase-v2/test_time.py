import sys, os
import time
import shutil
import re

if (__name__ == '__main__'):
  algo_name = sys.argv[1]
  ra1 = sys.argv[2]
  ra2 = sys.argv[3]
  setting_file = sys.argv[4]
  print algo_name, ra1, ra2
  ra1=int(ra1)
  ra2=int(ra2)
  start_time=time.time()
  command='python sdr.py fstm 40 data/news20.dat data/news20.t news'
  os.system(command)
  accuracy='_news_fstm40/Accuracy-sin1000-k20-ld0.10.dat'
  acc=re.findall('\d+\.\d*',open(accuracy).readline().strip())[0] 
  t=time.time()-start_time
  print '-----%s seconds with acc %s------' %(t,acc)
  f2=open('test_results/time_base.txt','w')
  f2.write('%s %s\n' % (str(t), str(acc)))
  f2.close()
  f2=open('test_results/time_%s.txt' % (algo_name),'w')
  for train_iter in range(ra1,ra2+1):
    lines = open(setting_file,'r').readlines()
    lines[-1]='iter_train: %s\n' % train_iter
    f=open(setting_file,'w')
    f.writelines(lines)
    f.close()
    start_time=time.time()
    command='python sdr_ope.py fstm 40 data/news20.dat data/news20.t news ./OPE-master/settings.txt %s' %(algo_name)
    os.system(command)
    acc=re.findall('\d+\.\d*',open(accuracy).readline().strip())[0] 
    t=time.time()-start_time
    print '-----%s seconds with acc %s------' %(t,acc)
    f2.write('%s %s\n' % (str(t), str(acc)))
  f2.close()
    