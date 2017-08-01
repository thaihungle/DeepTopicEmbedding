from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import sys

# model2 = '_data_c0.01_kNN5_ld0.05_0_fstm20'
# test_data = '../data/5-test-5.libsvm'
# outfile = '_data_c0.01_kNN5_ld0.05_0_fstm20/gphi_measures.dat'

predict_file = sys.argv[1]
test_data = sys.argv[2]
outfile = sys.argv[3]

out = open(outfile, 'w')
predict = open(predict_file).read()

y_true = np.array(predict.splitlines())
y_pred_array = []
test = open(test_data).read()
for line in test.splitlines():
	words = line.split(' ')
	y_pred_array.append(words[0])

y_pred = np.array(y_pred_array)
(precision_mac, recall_mac, fscore_mac, support_mac) = precision_recall_fscore_support(y_true, y_pred, average='macro')
(precision_mic, recall_mic, fscore_mic, support_mic) = precision_recall_fscore_support(y_true, y_pred, average='micro')
acc = accuracy_score(y_true, y_pred)

out.write('%s %s %s %s %s %s %s' % (acc*100, precision_mac*100, recall_mac*100, fscore_mac*100, precision_mic*100, recall_mic*100, fscore_mic*100))