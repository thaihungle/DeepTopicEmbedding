import re
import itertools
import sys, os
import time
import shutil
import re
import numpy
import json

'''
alpha, eta: {0.001; 0.01; 0.1}
tau0: {1; 10; 100}
kappa: {0.5; 0.6; 0.7; 0.8; 0.9; 1}
topics: 20; 40; 80; 100
'''
topics=[20,40,80,100]
alphas=[0.001,0.01, 0.1]
etas=[0.001, 0.01, 0.1]
taus=[1, 10, 100]
kappas=[0.5,0.6,0.7,0.9,0.9,1]
knns=[10,20,40,100]
lamdas=[0.1,0.3,0.5]

list_params=[]
list_params.append(topics)
list_params.append(alphas)
list_params.append(etas)
list_params.append(taus)
list_params.append(kappas)

#list_params.append(knns)
#list_params.append(lamdas)

param_combinations=list(itertools.product(*list_params))
print (param_combinations)
print (len(param_combinations))

#python test_opesdr_grid.py fstm 20 data/data3/7-train-5.libsvm data/data3/7-test-5.libsvm news ./OPE-master/#settings.txt ML-FW 

if (__name__ == '__main__'):
	ope_algo_name = sys.argv[7]#ML-OPE
	model_folder = sys.argv[5]#news
	setting_file = sys.argv[6]#./OPE-master/settings.txt
	knn = sys.argv[2]#20
	train_file = sys.argv[3]#data/news20.data
	test_file = sys.argv[4]#data/news20.t
	gen_setting_file = './gen_settings.txt'
	c=0
	hold=False
	try:
		f2=open('test_results/acc_%s.txt' % (ope_algo_name),'r')
		fromcom=json.loads(f2.readlines()[-1]).items()[0][0]			
		f2.close()
		hold=True		
	except:
		pass
	
	for com in param_combinations:
		if hold:
			if str(com)==fromcom:
				hold=False
			print ("%s pass %s"%(fromcom,str(com)))
			continue
		lines = open(setting_file,'r').readlines()
		lines[2]='num_topics: %s\n' % com[0]
		lines[4]='alpha: %s\n' % com[1]
		lines[5]='eta: %s\n' % com[2]
		lines[6]='tau0: %s\n' % com[3]
		lines[7]='kappa: %s\n' % com[4]
		f=open(gen_setting_file,'w')
		f.writelines(lines)
		f.close()
		start_time=time.time()
		command='python sdr_ope.py fstm %s %s %s %s %s %s' %		(knn,train_file,test_file,model_folder,gen_setting_file,ope_algo_name)
		os.system(command)
		accuracy1='_news_fstm40/Accuracy-sin1000-k%s-ld0.10.dat'%(knn)
		acc1=re.findall('\d+\.\d*',open(accuracy1).readline().strip())[0] 
		accuracy2='_news_fstm40/Accuracy-gphi.dat'
		acc2=re.findall('\d+\.\d*',open(accuracy2).readline().strip())[0] 
		accuracy3='_news_fstm40/Accuracy-new-topics.dat'
		acc3=re.findall('\d+\.\d*',open(accuracy3).readline().strip())[0] 
		accuracy4='_news_fstm40/Accuracy-phi.dat'
		acc4=re.findall('\d+\.\d*',open(accuracy4).readline().strip())[0] 
		t=time.time()-start_time
		print '-----%s seconds with acc %s------' %(t,str((acc1,acc2,acc3,acc4)))
		dic={}
		dic[str(com)]=(acc1,acc2,acc3,acc4)
		f2=open('test_results/acc_%s.txt' % (ope_algo_name),'a')		
		f2.write('%s\n' % (json.dumps(dic)))
		c+=1
		f2.close()

