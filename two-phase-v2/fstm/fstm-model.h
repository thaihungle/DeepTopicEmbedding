#ifndef FSTM_MODEL_H
#define FSTM_MODEL

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "fstm.h"
#include "data.h"
#include "utils.h"

#define myrand() (double) (((unsigned long) randomMT()) / 4294967296.)
#define NUM_INIT 1

#define EPS 1e-10 //smooth for topics

float EM_CONVERGED;
int EM_MAX_ITER;
float T_SPARSE;  //sparse degree of topics
int NTOPICS;
float INF_CONVERGED;
int INF_MAX_ITER;
int UNSUPERVISED;
int WARM_START;

void fstm_model_Free(fstm_model* model);

void fstm_model_Save(fstm_model* model, char* model_root);

fstm_model* fstm_model_New(int num_terms, int num_topics, float t_sparse);

fstm_model* fstm_model_Load(char* model_root);

void save_topic_docs(char* model_root, corpus *aa);

void save_topic_docs_fstm(char* model_root, corpus *aa, corpus *corp, int num_topics);

void L1_normalize_sparse_topics(fstm_model* model);

void fstm_topics_Save(fstm_model* model, char* model_root);

void initialize_random_topics(fstm_model *model);

void read_settings(char* filename);

#endif

