#ifndef OBJ_FUNCTIONS_H
#define OBJ_FUNCTIONS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "plsa/data.h"
#include "plsa/utils.h"
#include "plsa/cokus.h"

double f_lkh(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_lkh(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_tangent(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_tangent(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_cosine(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_cosine(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_sine(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_sine(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_sqr(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_sqr(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_cos_sin(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_cos_sin(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

double f_KL_cos_sin_ex(document *doc, double *x, double *theta, int dim, double *reg_coeffs);
double df_KL_cos_sin_ex(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs);

#endif
