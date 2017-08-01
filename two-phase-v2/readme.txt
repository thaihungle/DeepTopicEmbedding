An effective framework for supervised dimension reduction
********************************************
Version: April 2013

********************************************
Consider topics models to be dimension reduction approaches. Label information and local 
structure are used further to guide finding of latent space (topical space). 
	- Nearest neighbors are used when inferring latent representation of a document.
	- The Frank-Wolfe algorithm is used to do inference/projection.

********************************************
Other: support reading of data in either LIBSVM or LDA format.

********************************************
Running this package consists of 2 steps:
	- Set an appropriate setting for topic models and the inference method.
	- Learn the discriminative space by using label information and local structure.

********************************************
1. SET an appropriate setting
********************************************
Modify the file "inf-setting.txt"


********************************************
2. LEARN the discriminative space
********************************************
- Type 'Make' to compile the package
- Use the following command to learn:
	python sdr.py <model-name> <topics> <train-data> <test-data> <save-folder>

<model-name>:		fstm/lda /plsa
<model-folder>:	folder to save things to be learned
<train-data>:	name of the training data
<test-data>:	name of the testing data
<topics>:		number of topics

example: 	python sdr.py fstm 40 data/news20.dat data/news20.t news


********************************************
RESULT of learning
********************************************
Various files will be produced. Some impotant files contain:
- Topics: saved in files with extension ".beta". Each row is a topic
	"final-fstm.beta" is the topics of the unsupervised model. 
	"final-fstm-**.beta" is the topics learned by the Two-steps framework.

- Projection onto the discriminative space: 
  saved in files "*topics-docs-*.dat". Each row is the projection (latent representation) of a document.

- Accuracy of classification by SVM with linear kernel
  saved in files "Accuracy-*.dat"
