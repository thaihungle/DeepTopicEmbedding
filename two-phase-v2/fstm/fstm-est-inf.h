#ifndef FSTM_EST_INF_H
#define FSTM_EST_INF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "fstm.h"
#include "data.h"
#include "utils.h"
#include "fstm-model.h"

void fstm_Learn(char* directory, corpus* corp);

double doc_projection(fstm_model* model, document doc, corpus *aa, int docID);

void warm_start_init(fstm_model* model, document doc, document coeff, double *opt, double *temp, double ep);

void update_sparse_topics(fstm_model* model, corpus* corp, corpus *aa);

void write_sparse_statistics(char *model_root, fstm_model* model, corpus* corp, corpus *aa);

double alpha_binary_search(document doc, double *x, double *opt);

double alpha_gradient_search(document doc, double *x, double *opt);

double fstm_Infer(char *model_root, char *save, corpus* corp);

#endif
