#ifndef PLSA_EST_INF_H
#define PLSA_EST_INF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "plsa.h"
#include "data.h"
#include "utils.h"
#include "cokus.h"
#include "plsa-model.h"

double doc_inference(plsa_model* model, document doc, double *aa);

void plsa_Learn(char* directory, corpus* corpus);

void write_sparse_statistics(char *model_root, plsa_model* model, corpus* corpus, double **aa);

double data_likelihood(plsa_model* model, corpus* corpus, double **aa);

double plsa_Infer(char *model_root, char *save, corpus* corpus);

#endif
