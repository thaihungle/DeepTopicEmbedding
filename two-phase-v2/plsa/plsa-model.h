#ifndef PLSA_MODEL_H
#define PLSA_MODEL

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "plsa.h"
#include "data.h"
#include "cokus.h"
#include "utils.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

float EM_CONVERGED;
int EM_MAX_ITER;
float T_SPARSE;  //sparse degree of topics
int NTOPICS;
float INF_CONVERGED;
int INF_MAX_ITER;
int UNSUPERVISED;	//the data is supervised or not

void plsa_model_Free(plsa_model*);

void plsa_model_Save(plsa_model*, char*);

plsa_model* plsa_model_New(int num_terms, int num_topics, float t_sparse);

plsa_model* plsa_model_Load(char* model_root);

void save_topic_docs(char* model_root, double **aa, corpus *corp, int num_topics);

void read_settings(char* filename);

#endif

