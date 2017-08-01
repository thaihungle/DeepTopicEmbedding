****************************************
FULLY SPARSE TOPIC MODELS
****************************************

(C) Copyright 2012, Khoat Than (khoat [at] jaist [dot] ac [dot] jp)

This file is part of FSTM.

FSTM is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.

FSTM is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA

------------------------------------------------------------------------
This is a C implementation of Fully sparse topic models (FSTM). 

Inference of documents is done by the Frank-Wolfe algorithm. This inference 
algorithm has linear convergence rate, offers a principled way to trade off 
sparsity of solutions against quality and time. Learing topics is simply doing 
a multiplication of two sparse matrices.

------------------------------------------------------------------------
TABLE OF CONTENTS


A. COMPILING

B. LEARNING

   1. SETTINGS FILE

   2. DATA FILE FORMAT

C. INFERENCE

D. PRINTING TOPICS


------------------------------------------------------------------------
A. COMPILING

Type "make" in a shell.


------------------------------------------------------------------------
B. LEARNING

Estimate the model by executing:

     ./fstm  est  <model-folder>  <train-data>  <K>

<model-folder>:	folder will contain the learned model
<train-data>:	name of the training data
<K>:		number of topics

The model will be learned with K topics, and then saved in the given
folder <model-folder>. This folder will contain some more files. These files 
contain some statistics of how the model is at every 5 iterations. Statistics 
include topic sparsity, document sparsity, and log likelihood. Also, the new 
representations (topic proportions) of documents are saved for doing (probably) 
other tasks. Learned topics are save in the file which is ended with '.topic'. 

example: ./fstm est ap10 data/ap-train.txt 10


1. Settings file

See fstm-settings.txt for a sample. It contains convergence criteria for learning 
and inference, data type.

2. Data format

FSTM can read two data types: one is LDA for unsupervised data, and the other is 
LIBSVM for multiclass problems. Please refer to the following sites for instructions.

http://www.cs.princeton.edu/~blei/lda-c/
http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Under LDA, the words of each document are assumed exchangeable.  Thus,
each document is succinctly represented as a sparse vector of word
counts. The data is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the
[count] associated with each term is how many times that term appeared
in the document.  Note that [term_1] is an integer which indexes the
term; it is not a string.


------------------------------------------------------------------------

C. INFERENCE

     ./fstm  inf  <model-folder>  <test-data>

<model-folder>:	the folder that contains a learned model
<test-data>:	name of the testing data

Inference will result in a new representation of data, and will save it in the model folder. 
Other statistics will also be saved, including log likelihood and document sparsity.

example: ./fstm inf ap10 data/ap-test.txt


------------------------------------------------------------------------

D. PRINTING TOPICS

The Python script topics.py lets you print out the top N
words from each topic in a .topic file.  Usage is:

     python topics.py <topic file> <vocab file> <n words>
