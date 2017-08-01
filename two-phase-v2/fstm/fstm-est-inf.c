// (C) Copyright 2012, Khoat Than (khoat [at] jaist [dot] ac [dot] jp)

// This file is part of FSTM.

// FSTM is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// FSTM is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "fstm-est-inf.h"

void fstm_Learn(char* directory, corpus* corp)
{
	int d, i, LAG;	fstm_model *model;		corpus *aa;
	double likelihood, likelihood_old, converged;
	char filename[1000];

    aa = new_corpus(corp->num_docs, NTOPICS);	// new representation of documents
    // initialize model
	model = fstm_model_New(corp->num_terms, NTOPICS, T_SPARSE);
	initialize_random_topics(model);
	
    sprintf(filename, "%s/000-fstm", directory);
	fstm_model_Save(model, filename);
	
    // run iterative algorithm to learn the model
    sprintf(filename, "%s/likelihood-fstm.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");
	fprintf(likelihood_file, "Iteration, Likelihood, Converged \n");
	printf("Learning...\n");
	i = 0;	likelihood_old = 0;  converged = 1; 
	LAG = 1;
    while (((converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
    {
        i++; 	likelihood =0; 	printf("\n**** iteration %d ****\n", i);
		//allocate new representation
		for (d = 0; d < corp->num_docs; d++)
			likelihood += doc_projection(model, corp->docs[d], aa, d);
		//re-estimate all topics
		update_sparse_topics(model, corp, aa);
        // check for convergence
        converged = (likelihood_old - likelihood) / (likelihood_old);
        likelihood_old = likelihood;
        // output model and likelihood
		printf("\n  Likelihood: %10.10f; \t Converged %5.5e \n", likelihood, converged);
		fprintf(likelihood_file, "%d, %10.10f, %5.5e \n", i, likelihood, converged);
        fflush(likelihood_file);
		if (converged < -0.001) converged = 1; //continue to learn
    } //end learning
	fclose(likelihood_file);
	printf("\n  Final step: "); fflush(stdout);
	likelihood = 0;
    for (d = 0; d < corp->num_docs; d++)
		likelihood += doc_projection(model, corp->docs[d], aa, d);
	// output the final model
	printf("\n  saving the final model.\n");
    sprintf(filename,"%s/final-fstm", directory);
	fstm_model_Save(model, filename);
	save_topic_docs_fstm(filename, aa, corp, model->num_topics);
	write_sparse_statistics(filename, model, corp, aa);
	printf("  Model has been learned.\n");
}

/*
 * perform inference on a document as a projection onto the simplex of topics
 */
double doc_projection(fstm_model* model, document doc, corpus *aa, int docID)
{	//project a document to the simplex of topics, using Frank-Wolfe algorithm [Clarkson10]
	//it is equivalent to maximizing the likelihood of that document.
    //Alpha is approximated by binary search, directed by gradient.
    double likelihood, lkh_old, converge, fmax, *sum, *opt, alpha, *temp, *x, ep;
    int k, j, ind, t;
	opt  = (double*)malloc(sizeof(double)*doc.length);
	x    = (double*)malloc(sizeof(double)*doc.length);
	sum  = (double*)malloc(sizeof(double)*model->num_topics);
	temp = (double*)malloc(sizeof(double)*model->num_topics);
	if (opt==0 || sum==0 || temp==0) 
		{ printf("\n doc_projection: failed to allocate memory for temporary variables !\n"); exit(0); }
	if (aa->docs[docID].length > 0 && WARM_START == 1)
	{//Warm-start: take the result of the previous learning loop
		ep = 1 - EPS*model->num_terms;
		warm_start_init(model, doc, aa->docs[docID], opt, temp, ep);
	}
	else
	{// locate the initial point to be a vertex
		//compute f(x) for each vertex
		ep = EPS / (1 - EPS*model->num_terms);
		for (k = 0; k < model->num_topics; k++) sum[k] = 0;
        for (j = 0; j < doc.length; j++){
            if (model->num_terms>doc.words[j] && model->bb[doc.words[j]].length > 0 )
			  for (k = 0; k < model->bb[doc.words[j]].length; k++) 
				sum[model->bb[doc.words[j]].topicID[k]] += 
                    doc.counts[j] * log(model->bb[doc.words[j]].value[k] + ep);
        }
		for (k = 0; k < model->num_topics; k++) 
			if (k == 0 || (sum[k] > fmax))	{ fmax = sum[k];	ind = k; }

		ep = 1 - EPS*model->num_terms;
		for (j = 0; j < doc.length; j++) opt[j] = EPS;
        for (j = 0; j < doc.length; j++)
            if (model->num_terms>doc.words[j] && model->bb[doc.words[j]].length > 0 )
	  		  for (k = 0; k < model->bb[doc.words[j]].length; k++)
				if (ind == model->bb[doc.words[j]].topicID[k])	
				{	opt[j] = ep * model->bb[doc.words[j]].value[k] + EPS;		break; }
		for (k = 0; k < model->num_topics; k++) temp[k] = 0;
		temp[ind] = 1;
	}

	//loop to find the projection
	t = 0;    lkh_old = 0;
	do{
		//select the best direction, by selsecting the maximum derivative of f(x)
		for (k = 0; k < model->num_topics; k++) sum[k] = 0;
		for (j = 0; j < doc.length; j++)
		{
            if (model->num_terms>doc.words[j] && model->bb[doc.words[j]].length > 0 )
			  for (k = 0; k < model->bb[doc.words[j]].length; k++) 
				sum[model->bb[doc.words[j]].topicID[k]] += 
					doc.counts[j] * model->bb[doc.words[j]].value[k] / opt[j];
		}
		for (k = 0; k < model->num_topics; k++) 
			if (k == 0 || sum[k] > fmax)	{ fmax = sum[k];	ind = k; }
		//Take the chosen topic b, then consider x = b - opt
		for (j = 0; j < doc.length; j++) 
		{
			x[j] = -opt[j];
            if (model->num_terms>doc.words[j] && model->bb[doc.words[j]].length > 0 )
			for (k = 0; k < model->bb[doc.words[j]].length; k++)
				if (ind == model->bb[doc.words[j]].topicID[k])
					{ x[j] += ep *model->bb[doc.words[j]].value[k] + EPS;		break; }
		}
		//Search for alpha
		//alpha = alpha_binary_search(doc, x, opt);
		alpha = alpha_gradient_search(doc, x, opt);
		if (alpha == 0) break;
		//found a better point
		for (j = 0; j < doc.length; j++)   opt[j] += alpha * x[j];
		for (k = 0; k < model->num_topics; k++) 
			if (temp[k] > 0)	temp[k] *= 1-alpha;
		temp[ind] += alpha;
		//compute likelihood
		likelihood = 0;
		for (j = 0; j < doc.length; j++) likelihood += doc.counts[j] * log(opt[j]);
		converge = (lkh_old - likelihood) / lkh_old;
		lkh_old  = likelihood;
		t++;
		if (converge < 0 && t < INF_MAX_ITER) continue;
	}while ((t < INF_MAX_ITER) && (t < 2 || converge > INF_CONVERGED));
	
	//reset new representation of the document
	free(aa->docs[docID].words); 	free(aa->docs[docID].counts);
	j = 0;	
	for (k = 0; k < model->num_topics; k++) if (temp[k] > 0) j++;
	aa->docs[docID].length = j;
	if (j == 0) j = 1;
	aa->docs[docID].words = (int*)malloc(sizeof(int)*j);	
	aa->docs[docID].counts = (double*)malloc(sizeof(double)*j);	
	if (aa->docs[docID].words==0 || aa->docs[docID].counts==0)
		{ printf("\n doc_projection: failed to allocate memory for aa[%d] !\n", docID); exit(0); }
	j = 0;
	for (k = 0; k < model->num_topics; k++) 
		if (temp[k] > 0) 
			{ aa->docs[docID].words[j] = k; aa->docs[docID].counts[j] = temp[k]; j++; }
	free(opt); free(sum); free(temp); free(x);
return(likelihood);
}

void warm_start_init(fstm_model* model, document doc, document coeff, double *opt, double *temp, double ep)
{
    double sum=1;		int k, j, t;
	//remove topics with very small contributions
	for (t = 0; t < coeff.length; t++)
		if (coeff.counts[t] < 0.001) 
		{sum -= coeff.counts[t];	coeff.counts[t] = 0;}
	//L1 normalize
	for (t = 0; t < coeff.length; t++) coeff.counts[t] /= sum;
	//compute initial opt
	for (j = 0; j < doc.length; j++) opt[j] = 0;
	for (j = 0; j < doc.length; j++)
		if (model->bb[doc.words[j]].length > 0 )
			for (t = 0; t < coeff.length; t++)
				if (coeff.counts[t] > 0)
				for (k = 0; k < model->bb[doc.words[j]].length; k++)
					if (coeff.words[t] == model->bb[doc.words[j]].topicID[k])
					{ opt[j] += coeff.counts[t] * model->bb[doc.words[j]].value[k]; break; }
	for (j = 0; j < doc.length; j++) opt[j] = ep*opt[j] + EPS;
	for (k = 0; k < model->num_topics; k++) temp[k] = 0;
	for (t = 0; t < coeff.length; t++)	temp[coeff.words[t]] = coeff.counts[t];
}

void update_sparse_topics(fstm_model* model, corpus* corp, corpus *aa)
{//update the topics when all are in sparse format
	//topic[k][j] ~= sum_{d} aa[k][d] * N[d][j]
	int d, i, j, k, wid, tp, *temp;
	double counts;

	temp = (int*)malloc(sizeof(int)*model->num_terms);
	for (k = 0; k < model->num_terms; k++) temp[k] = model->bb[k].length;
	//remove old topics
	for (k = 0; k < model->num_terms; k++) model->bb[k].length = 0; 
	
	//resize topics if necessary
	for (d = 0; d < corp->num_docs; d++) 
		for (j = 0; j < corp->docs[d].length; j++)
		{
			wid		= corp->docs[d].words[j];
			counts	= corp->docs[d].counts[j];
			for (k = 0; k < aa->docs[d].length; k++)
			{	tp = -1;
				if (model->bb[wid].length > 0)
				for (i = 0; i < model->bb[wid].length; i++)
					if (model->bb[wid].topicID[i] == aa->docs[d].words[k]) 	{ tp = i; break; } 
				if (tp >= 0) model->bb[wid].value[tp] += aa->docs[d].counts[k] * counts;
				else{
					model->bb[wid].length ++;
					if (model->bb[wid].length > temp[wid])
					{
						model->bb[wid].topicID	= (int*)realloc(model->bb[wid].topicID, sizeof(int)*model->bb[wid].length);
						model->bb[wid].value	= (double*)realloc(model->bb[wid].value, sizeof(double)*model->bb[wid].length);
						if (model->bb[wid].topicID == 0 || model->bb[wid].value == 0)
						{	printf("\n update_sparse_topics: failed to allocate memory for model->bb[%d] !\n", wid); 
							exit(0); }
						temp[wid] = model->bb[wid].length;
					}
					model->bb[wid].topicID[model->bb[wid].length -1]	= aa->docs[d].words[k];
					model->bb[wid].value[model->bb[wid].length -1]		= aa->docs[d].counts[k] * counts;
				}
			}
		}
	for (k = 0; k < model->num_terms; k++) 
		if (temp[k] > model->bb[k].length) //remove redundant in memory
		{
			j = model->bb[k].length;
			if (j < 1)	j = 1;
			model->bb[k].topicID	= (int*)realloc(model->bb[k].topicID, sizeof(int)*j);
			model->bb[k].value		= (double*)realloc(model->bb[k].value, sizeof(double)*j);
			if (model->bb[k].topicID == 0 || model->bb[k].value == 0)
				{	printf("\n update_sparse_topics: failed to remove redundant memory [%d] !\n", k); 
					exit(0); }
		}
	L1_normalize_sparse_topics(model);
	free(temp);
}


void write_sparse_statistics(char *model_root, fstm_model* model, corpus* corp, corpus *aa)
{	//write some statistics about the topics and latent representations of documents
	int d, k; 	double sc, sum;
	char filename[1000];
    FILE *fileptr;

	sprintf(filename, "%s-stats.csv", model_root);
    fileptr = fopen(filename, "w");

	fprintf(fileptr, "Number of topics, =, %d\n", model->num_topics);
	fprintf(fileptr, "Number of terms, =, %d\n", model->num_terms);
	fprintf(fileptr, "Number of training documents, =, %d\n", corp->num_docs);
	//summary of sparsity
	sc =0;
	for (k=0; k < model->num_terms; k++) sc += model->bb[k].length;
	sum = model->num_topics *model->num_terms;
	fprintf(fileptr, "Topic sparsity: %d/%d, =, %f\n", (int)sc, (int)sum, sc/sum);

	sc =0;
	for (d=0; d < corp->num_docs; d++) sc += aa->docs[d].length;
	sum = model->num_topics * corp->num_docs;
	fprintf(fileptr, "Document sparsity: %d/%d, =, %f\n", (int)sc, (int)sum, sc/sum);
	sum =0;
	for (d=0; d < corp->num_docs; d++) sum += corp->docs[d].length;
	fprintf(fileptr, "Doc. compress rate: %d/%d, =, %f\n", (int)sc, (int)sum, sc/sum);
	fclose(fileptr);
}


double alpha_binary_search(document doc, double *x, double *opt)
{
	int i, j;	double fl, fr, fa, alpha, left, right;
	
	fl = 0;	fr = 0;
	for (i = 0; i < doc.length; i++) 
		{ fr += log(opt[i] + x[i]) * doc.counts[i];	fl += log(opt[i]) * doc.counts[i]; }
	left = 0; right = 1;
	for (j = 0; j < 20; j++) 
	{
		alpha = (left + right)/2;			fa  = 0;
		for (i = 0; i < doc.length; i++)	fa += log(alpha*x[i] + opt[i]) * doc.counts[i];
		if (fa >= fr)	{	fr = fa; right = alpha; }
		else			{	fl = fa; left  = alpha; }
	}
	if (fl > fr) alpha = left;
	else alpha = right;
return alpha;
}

double alpha_gradient_search(document doc, double *x, double *opt)
{//binary search with direction of Gradient
	int i, j;	double fa, alpha, left, right;
	left =0;	right =1;
	for (j = 0; j < 20; j++) 
	{
		alpha = (left + right)/2;
		fa  = 0;	//derivative
		for (i = 0; i < doc.length; i++)
			if (x[i] != 0)	fa += doc.counts[i] * x[i] / (alpha*x[i] + opt[i]);
		if (abs(fa) < 1e-10 ) break;
		if (fa < 0)	right = alpha; 
		else		left  = alpha; 
	}
return alpha;
}


/*
 * inference only
 *
 */

double fstm_Infer(char *model_root, char *save, corpus* corp)
{
    FILE* fileptr;	char filename[1000];	fstm_model *model;	
	corpus *aa;		document doc;	int d;
    double likelihood, perplexity, tnum;

	sprintf(filename, "%s-lhood.csv", save);
    fileptr = fopen(filename, "w");
	fprintf(fileptr, "docID, likelihood \n");

	model = fstm_model_Load(model_root);
	aa = new_corpus(corp->num_docs, model->num_topics);

	printf(" Number of topics: %d \n", model->num_topics);
	printf(" Number of docs: %d \n", corp->num_docs);
	printf(" Infering: ");  fflush(stdout);
	perplexity = 0;		tnum = 0;	
    for (d = 0; d < corp->num_docs; d++)
    {
		if (d % 1000 == 0) {printf("%d, ", d); fflush(stdout);}
		doc = corp->docs[d];	
		likelihood = doc_projection(model, doc, aa, d);
		fprintf(fileptr, "%d, %10.10f \n", d, likelihood);
		perplexity += likelihood;
		tnum += doc.total;
	}
	fclose(fileptr);
	save_topic_docs_fstm(save, aa, corp, model->num_topics);
	write_sparse_statistics(save, model, corp, aa);
	printf("  Completed. \n");
return (exp(-perplexity/tnum));
}

