import pylab
import numpy
import sys
import glob

model2='_news_fstm40'
mstr='%s/measures*' % (model2)
files= glob.glob(mstr)
for f in files:
	data0=pylab.loadtxt(f)
	data0=numpy.array(data0)
	print data0
	x=range(data0.shape[0])
	pylab.plot(x,data0)
pylab.xlabel("folds")
pylab.ylabel('accuracy (%)')
pylab.ylim([0,100])
names=[]
for f in files:
	n=f.split('_')[-1]
	names.append(n)
pylab.legend(names,loc="best" )
pylab.tight_layout()
pylab.show()

