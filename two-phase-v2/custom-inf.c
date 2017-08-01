
#include "custom-inf.h"

/*
 * perform inference on a document as a projection onto the simplex of topics
 */
double doc_projection(plsa_model* model, document *doc, double *aa, 
	double (*compute_f)(document*, double*, double*, int, double*), 
	double (*compute_df)(document*, double*, double*, double*, int, double*), double *reg_coeffs)
{	//Optimizing the function f(x) over the simplex of topics, using Frank-Wolfe algorithm [Clarkson10]
	//Alpha is approximated by binary search, directed by gradient.
    double likelihood, lkh_old, converge, fmax, alpha, *opt, *x, sum, *tmp;
    int k, j, ind, t;

    opt = malloc(model->num_terms * sizeof(double));
    x	= malloc(model->num_terms * sizeof(double));
	for (k = 0; k < model->num_topics; k++) aa[k] = 0;
	// locate the initial point to be a vertex
	for (k = 0; k < model->num_topics; k++) 
	{
		aa[k] = 1;
		sum = (*compute_f)(doc, model->bb[k], aa, model->num_topics, reg_coeffs);
		if (k == 0 || (sum > fmax))	{ fmax = sum;	ind = k; }
		aa[k] = 0;
	}
	for (j = 0; j < doc->length; j++) opt[doc->words[j]] = model->bb[ind][doc->words[j]];
	aa[ind] = 1;
	//loop to find the projection
	t = 0;  likelihood = 0;
	do{
		t++;	lkh_old = likelihood;
		//select the best direction, by selsecting the maximum derivative of f(x)
		for (k = 0; k < model->num_topics; k++) 
		{
			sum = (*compute_df)(doc, opt, model->bb[k], aa, model->num_topics, reg_coeffs);
			if (k == 0 || (sum > fmax))	{ fmax = sum;	ind = k; }	
		}
		//Search for alpha
		alpha = alpha_gradient_search(doc, x, model->bb[ind], opt, 
				aa, ind, model->num_topics, reg_coeffs, compute_f, compute_df);
		if (alpha == 0) break;
		//found a better point
		tmp = opt;	opt = x;	x = tmp;
		//compute likelihood
		likelihood = (*compute_f)(doc, opt, aa, model->num_topics, reg_coeffs);
		converge = (lkh_old - likelihood) / lkh_old;
	}while ((t < INF_MAX_ITER) && (t<2 || converge > INF_CONVERGED));
    free(opt); free(x);	tmp = 0;
return(likelihood);
}


double alpha_gradient_search(document *doc, double *x, double *vertex, double *opt, 
	double *aa, int ind, int dim, double *reg_coeffs, 
	double (*compute_f)(document*, double*, double*, int, double*), 
	double (*compute_df)(document*, double*, double*, double*, int, double*))
{//binary search with direction of Gradient
	int i, j, k;	double dfa, alpha, left, right, aa_al[dim];

	for (j = 0; j < dim; j++) aa_al[j] = aa[j];
	left =0;	right =1;
	for (j = 0; j < 20; j++) 
	{
		alpha = (left + right)/2;
		for (k = 0; k < dim; k++) if (aa_al[k] > 0)	aa_al[k] *= 1-alpha;
		aa_al[ind] += alpha;
		for (i = 0; i < doc->length; i++)
			x[doc->words[i]] = (1-alpha) * opt[doc->words[i]] + alpha * vertex[doc->words[i]];
		dfa = (*compute_df)(doc, x, vertex, aa_al, dim, reg_coeffs) 
			- (*compute_df)(doc, x, opt, aa_al, dim, reg_coeffs);
		if (abs(dfa) < 1e-10 ) break;
		if (dfa < 0)	right = alpha; 
		else			left  = alpha; 
	}
	for (j = 0; j < dim; j++) aa[j] = aa_al[j];
return alpha;
}

plsa_model * load_model(char *topic_file, char *other_file)
{
    FILE* fileptr;	plsa_model *model;
    int i, j, num_terms, num_topics;    double x;	char tmp[100];

    fileptr = fopen(other_file, "r");
	fscanf(fileptr, "%s %d", tmp, &num_topics);
	fscanf(fileptr, "\n%s %d", tmp, &num_terms);
	fclose(fileptr);

	printf("loading %s\n", topic_file);
    fileptr = fopen(topic_file, "rb");
	if (fileptr == NULL)
	{ 		printf("Cannot open %s\n", topic_file);  exit(0);	}

    model = plsa_model_New(num_terms, num_topics, 1);
	for (i = 0; i < num_topics; i++)
    {
        for (j = 0; j < num_terms; j++)
        {
            fscanf(fileptr, "%lf", &x);
            model->bb[i][j] = x;
	    }
    }
	fclose(fileptr); 
	//adding a small constant and then normalize
 	for (i = 0; i < model->num_topics; i++)
	{
		for (j = 0; j < model->num_terms; j++)	model->bb[i][j] += (1e-10);
		L1_normalize(model->bb[i], model->num_terms);
	}
	printf("Model was loaded.\n");
    return(model);
}

void compute_entropy(corpus *corp)
{//L1-normalize for each document, and then compute entropy
	int i, j;	double sum;		document *doc;
	for (i =0; i < corp->num_docs; i++)	
	{
		doc = &(corp->docs[i]);
		L1_normalize_document(doc);
		sum = 0;
		for (j =0; j < doc->length; j++) 
			sum -= doc->counts[j] * log(doc->counts[j]);
		doc->entropy = sum;
	}
	doc = NULL;
}

