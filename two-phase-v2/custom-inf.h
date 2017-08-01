#ifndef CUSTOM_INF_H
#define CUSTOM_INF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "plsa/plsa-est-inf.h"
#include "obj-functions.h"

int PRIOR;

double doc_projection(plsa_model* model, document *doc, double *aa, 
	double (*compute_f)(document*, double*, double*, int, double*), 
	double (*compute_df)(document*, double*, double*, double*, int, double*), double *reg_coeffs);

double alpha_binary_search(document *doc, double *x, double *vertex, double *opt, 
	double *aa, int ind, int dim, double *reg_coeffs, 
	double (*compute_f)(document*, double*, double*, int, double*));

double alpha_gradient_search(document *doc, double *x, double *vertex, double *opt, 
	double *aa, int ind, int dim, double *reg_coeffs, 
	double (*compute_f)(document*, double*, double*, int, double*), 
	double (*compute_df)(document*, double*, double*, double*, int, double*));

plsa_model * load_model(char *topic_file, char *other_file);

void compute_entropy(corpus *corp);

double custom_Infer(char *model_root, corpus* corp, char *function_name, double reg, char *model_name);

#endif

