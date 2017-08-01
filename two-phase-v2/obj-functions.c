// This package is for review purpose only. 
// Careful documentation will be freely available soon.

#include "obj-functions.h"

//============================================
//Objective function is the likehood

double f_lkh(document *doc, double *x, double *theta, int dim, double *reg_coeffs)
{//reg_coeffs: coefficients which associate with regularizations
	int j; double fx = 0;
	for (j = 0; j < doc->length; j++) 
		fx += doc->counts[j] * log(x[doc->words[j]]);
return (fx);
}

double df_lkh(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs)
{// multipication of the "derivative at x" and a vector "vertex"
	int j; double dfx=0;
	for (j = 0; j < doc->length; j++)
		dfx += doc->counts[j] * vertex[doc->words[j]] / x[doc->words[j]];
return (dfx);
}

//============================================
//Objective function is the Kullback-Leibler divergence, regularized by Sine
//	-KL + C * sine(theta)		with C > 0

double f_KL_sine(document *doc, double *x, double *theta, int dim, double *reg_coeffs)
{//reg_coeffs: coefficients which associate with regularizations
	int j; double s2, fx = 0;
	for (j = 0; j < doc->length; j++) 
		fx += doc->counts[j] * log(x[doc->words[j]]);
	s2 = 0;	
	for (j = 0; j < dim; j++)
		if (theta[j] > 0 && reg_coeffs[j] > 0)	s2 += reg_coeffs[j] *sin(theta[j]);
	fx -= doc->entropy - s2;
return (fx);
}

double df_KL_sine(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs)
{// multipication of the "derivative at x" and a vector "vertex", and then regularized
	int j; double s2, dfx = 0;
	for (j = 0; j < doc->length; j++)
		dfx += doc->counts[j] * vertex[doc->words[j]] / x[doc->words[j]];
	s2 = 0;	
	for (j = 0; j < dim; j++)
		if (theta[j] > 0 && reg_coeffs[j] > 0)	s2 += reg_coeffs[j] * cos(theta[j]);
	dfx += s2;
return (dfx);
}

//============================================
//Objective function is the Kullback-Leibler divergence, regularized by cos(x) + sin(x)
//	-KL + C * [cos(theta) + sin(theta)]		with C > 0

double f_KL_cos_sin(document *doc, double *x, double *theta, int dim, double *reg_coeffs)
{//reg_coeffs: coefficients which associate with regularizations
	int j; double s2, fx = 0;
	for (j = 0; j < doc->length; j++) 
		fx += doc->counts[j] * log(x[doc->words[j]]);
	s2 = 0;
	for (j = 0; j < dim; j++)
		if (reg_coeffs[j] > 0)		s2 += reg_coeffs[j] * cos(theta[j]);
		else if (reg_coeffs[j] < 0)	s2 -= reg_coeffs[j] * sin(theta[j]);
	fx -= doc->entropy - s2;
return (fx);
}

double df_KL_cos_sin(document *doc, double *x, double *vertex, double *theta, int dim, double *reg_coeffs)
{// multipication of the "derivative at x" and a vector "vertex", and then regularized
	int j; double s2, dfx = 0;
	for (j = 0; j < doc->length; j++)
		dfx += doc->counts[j] * vertex[doc->words[j]] / x[doc->words[j]];
	s2 = 0;
	for (j = 0; j < dim; j++)
		if (reg_coeffs[j] > 0)		s2 -= reg_coeffs[j] * sin(theta[j]);
		else if (reg_coeffs[j] < 0)	s2 -= reg_coeffs[j] * cos(theta[j]);
	dfx += s2;
return (dfx);
}

