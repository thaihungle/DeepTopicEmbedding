#ifndef MULTICLASS_INF_H
#define MULTICLASS_INF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "plsa/plsa-est-inf.h"
#include "obj-functions.h"
#include "custom-inf.h"
#include "compute-topics.h"

int kNN;	//number of nearest neighbors
double lambda; //coefficient in combination
double reg;		//regularization constant

void compute_init_reg(double **C, int nclasses, int ntopics, char *training_rep);

void compute_reg(double **C, int nclasses, int ntopics, double **aa, corpus *corp);

void multi_discriminative_topics(double **C, int nclasses, int ntopics, double reg);

void multi_promote_topics(double **C, int nclasses, int ntopics, double reg, double bound);

void multi_exclude_common_topics(double **C, int nclasses, int ntopics, double reg);

corpus *multi_find_neighbors(corpus *corp, int knn, double lambda);

corpus* read_neighbors(char* filename, corpus *corp);

void combine_knn(corpus *corp, corpus *nbors, double lambda);

double multi_Infer_disc(char *model_root, corpus *corp, corpus *nbors, char *function_name, char *model_name);


#endif

