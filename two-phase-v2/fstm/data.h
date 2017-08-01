#ifndef DATA_H
#define DATA_H

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int* words;
    double* counts;
	double entropy;		//entropy of this document
    double total;
	int length;
	int label;
} document;


typedef struct
{
    document* docs;
	char **labels;		//for multi-label data
	int *label_names;	//for multi-class single-label data
    int num_terms;
    int num_docs;
	int num_labels;
} corpus;

//Some functions
corpus* read_data(char* data_filename, int UNSUPERVISED);

int max_corpus_length(corpus* c);

void free_corpus(corpus *corp);

void L1_normalize_document(document *doc);

corpus* new_corpus(int num_docs, int num_terms);

#endif

