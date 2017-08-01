#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

double log_sum(double log_a, double log_b);
double trigamma(double x);
double digamma(double x);
double log_gamma(double x);
void make_directory(char* name);
int argmax(double* x, int n);
void L1_normalize(double *vec, int dim);

void sparsify_vector(double *vec, int dim, int nnz);

void correct_zero_terms(double **bb, int num_terms, int num_topics);

void matrix_initialize_sparse_L1(double **bb, int rows, int columns, float t_sparse);

void matrix_random_initialize_L1(double **bb, int rows, int columns);

double** matrix_initialize(int rows, int columns);

void clear_vector(double *vec, int dim);


#endif
