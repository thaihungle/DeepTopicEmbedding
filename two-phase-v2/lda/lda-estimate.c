// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda-estimate.h"

/*
 * perform inference on a document and update sufficient statistics
 *
 */

double doc_e_step(document* doc, double* gamma, double** phi,
                  lda_model* model, lda_suffstats* ss)
{
    double likelihood;
    int n, k;

    // posterior inference

    likelihood = lda_inference(doc, model, gamma, phi);

    // update sufficient statistics

    double gamma_sum = 0;
    for (k = 0; k < model->num_topics; k++)
    {
        gamma_sum += gamma[k];
        ss->alpha_suffstats += digamma(gamma[k]);
    }
    ss->alpha_suffstats -= model->num_topics * digamma(gamma_sum);

    for (n = 0; n < doc->length; n++)
    {
        for (k = 0; k < model->num_topics; k++)
        {
            ss->class_word[k][doc->words[n]] += doc->counts[n]*phi[n][k];
            ss->class_total[k] += doc->counts[n]*phi[n][k];
        }
    }

    ss->num_docs = ss->num_docs + 1;

    return(likelihood);
}


/*
 * writes the word assignments line for a document to a file
 *
 */

void write_word_assignment(FILE* f, document* doc, double** phi, lda_model* model)
{
    int n;

    fprintf(f, "%03d", doc->length);
    for (n = 0; n < doc->length; n++)
    {
        fprintf(f, " %05d:%03d",
                doc->words[n], argmax(phi[n], model->num_topics));
    }
    fprintf(f, "\n");
    fflush(f);
}


/*
 * saves the gamma parameters of the current dataset
 *
 */

void save_gamma(char* filename, double** gamma, int num_docs, int num_topics)
{
    FILE* fileptr;
    int d, k;
    fileptr = fopen(filename, "w");

    for (d = 0; d < num_docs; d++)
    {
		fprintf(fileptr, "%5.10f", gamma[d][0]);
		for (k = 1; k < num_topics; k++)
		{
			fprintf(fileptr, " %5.10f", gamma[d][k]);
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}

void save_topic_docs(char* model_root, double **aa, corpus *corp, int num_topics)
{
    char filename[1000];
    FILE *fileptr;
    int i, j;

    sprintf(filename, "%s-topics-docs-contribute.dat", model_root);
    fileptr = fopen(filename, "w");

	for (i = 0; i < corp->num_docs; i++)
    {
		L1_normalize(aa[i], num_topics);
		switch (UNSUPERVISED)
		{
			case 1 :		//simply write a matrix (unsupervised)
				for (j = 0; j < num_topics; j++)
					if (aa[i][j] > 0)	fprintf(fileptr, "%1.10f, ", aa[i][j]) ;
					else fprintf(fileptr, "0, ") ;
				break;
			case 2 :		//write to a file in Libsvm format (multi-class single-label)
				fprintf(fileptr, "%d ", corp->label_names[corp->docs[i].label]);
				for (j = 0; j < num_topics; j++)
					if (aa[i][j] > 0)	fprintf(fileptr, "%d:%1.10f ", j+1, aa[i][j]);
				break;
			case 3 : 	//simply write a matrix, separated by comma (multi-label)
				for (j = 0; j < num_topics; j++)
					if (aa[i][j] > 0)	fprintf(fileptr, "%1.10f, ", aa[i][j]) ;
					else fprintf(fileptr, "0, ") ;
				break;
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}


/*
 * run_em
 *
 */

void run_em(char* start, char* directory, corpus* corpus)
{

    int d, n;
    lda_model *model = NULL;
    double **var_gamma, **phi;
	clock_t startT, end, init = clock();

    // allocate variational parameters

    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (d = 0; d < corpus->num_docs; d++)
		var_gamma[d] = malloc(sizeof(double) * NTOPICS);

    int max_length = max_corpus_length(corpus);
    phi = malloc(sizeof(double*)*max_length);
    for (n = 0; n < max_length; n++)
		phi[n] = malloc(sizeof(double) * NTOPICS);

    // initialize model
    char filename[1000];
    lda_suffstats* ss = NULL;

    if (strcmp(start, "seeded")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        corpus_initialize_ss(ss, model, corpus);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else if (strcmp(start, "random")==0)
    {
        model = new_lda_model(corpus->num_terms, NTOPICS);
        ss = new_lda_suffstats(model);
        random_initialize_ss(ss, model);
        lda_mle(model, ss, 0);
        model->alpha = INITIAL_ALPHA;
    }
    else
    {
        model = load_lda_model(start);
        ss = new_lda_suffstats(model);
    }

    sprintf(filename,"%s/000-lda",directory);
    save_lda_model(model, filename);

    // run expectation maximization

    int i = 0;
    double likelihood, likelihood_old = 0, converged = 1;
    sprintf(filename, "%s/likelihood-lda.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");
	fprintf(likelihood_file, "Iteration, Likelihood, Converged, Time (seconds) \n");

	LAG = 5;
    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i <= EM_MAX_ITER))
    {
        i++; printf("**** em iteration %d ****\n", i);
		startT = clock();
        likelihood = 0;
        zero_initialize_ss(ss, model);

        // e-step

        for (d = 0; d < corpus->num_docs; d++)
        {
            if ((d % 1000) == 0) printf("document %d\n",d);
            likelihood += doc_e_step(&(corpus->docs[d]),
                                     var_gamma[d],
                                     phi,
                                     model,
                                     ss);
        }

        // m-step

        lda_mle(model, ss, ESTIMATE_ALPHA);

        // check for convergence

        converged = (likelihood_old - likelihood) / (likelihood_old);
        if (converged < 0) VAR_MAX_ITER = VAR_MAX_ITER * 2;
        likelihood_old = likelihood;

        // output model and likelihood
		end = clock();
		printf("  Likelihood: %10.10f; \t Converged %5.5e \n\n", likelihood, converged);
        fprintf(likelihood_file, "%d, %10.10f, %5.5e, %10.10f \n", i, likelihood, converged, ((double)(end - startT)) /CLOCKS_PER_SEC);
        fflush(likelihood_file);
   //     if ((i % LAG) == 0)
   //     {
   //         sprintf(filename,"%s/%03d-lda", directory, i);
   //         save_lda_model(model, filename);
			//write_sparse_statistics(filename, model, corpus, var_gamma);
   //         /*sprintf(filename,"%s/%03d-lda.gamma", directory, i);
   //         save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);*/
			//save_topic_docs(filename, var_gamma, corpus, model->num_topics);
   //     }
    }//end EM

    // output the word assignments (for visualization)

    //sprintf(filename, "%s/word-assignments.dat", directory);
    //FILE* w_asgn_file = fopen(filename, "w");
	//likelihood = 0;		startT = clock();
 //   for (d = 0; d < corpus->num_docs; d++)
 //   {
 //       if ((d % 100) == 0) printf("final e step document %d\n",d);
 //       likelihood += lda_inference(&(corpus->docs[d]), model, var_gamma[d], phi);
 //       //write_word_assignment(w_asgn_file, &(corpus->docs[d]), phi, model);
	//	//sprintf(filename, "%s/phi-doc%d.dat", directory, d);
	//	//save_gamma(filename, phi, corpus->docs[d].length, model->num_topics);
 //   }
	//end = clock();
	//converged = (likelihood_old - likelihood) / (likelihood_old);
	//fprintf(likelihood_file, "%d, %10.10f, %5.5e, %10.10f \n", i+1, likelihood, converged, ((double)(end - startT)) /CLOCKS_PER_SEC);
	////fclose(w_asgn_file);
	// output the final model
	printf("    saving the final model.\n");
	sprintf(filename,"%s/final-lda", directory);
    save_lda_model(model, filename);
	write_sparse_statistics(filename, model, corpus, var_gamma);
    /*sprintf(filename,"%s/final-lda.gamma",directory);
    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);*/
	save_topic_docs(filename, var_gamma, corpus, model->num_topics);

	//free parameters
	max_length = max_corpus_length(corpus);
	for (d = max_length -1; d >= 0; d--)   free(phi[d]);
	for (i = corpus->num_docs-1; i >= 0; i--) free(var_gamma[i]);
	free_lda_suffstats(ss, model->num_topics);
	
	end = clock();
	fprintf(likelihood_file, "Overall time:, , , %10.10f \n", ((double)(end - init)) /CLOCKS_PER_SEC);
	fclose(likelihood_file);
	printf("    Model has been learned.\n");
}


/*
 * read settings.
 *
 */

void read_settings(char* filename)
{
    FILE* fileptr;
    char alpha_action[100];
    fileptr = fopen(filename, "r");
    fscanf(fileptr, "var max iter: %d\n", &VAR_MAX_ITER);
    fscanf(fileptr, "var convergence: %f\n", &VAR_CONVERGED);
    fscanf(fileptr, "em max iter: %d\n", &EM_MAX_ITER);
    fscanf(fileptr, "em convergence: %f\n", &EM_CONVERGED);
    fscanf(fileptr, "alpha: %s\n", alpha_action);
    if (strcmp(alpha_action, "fixed")==0)
    {
		ESTIMATE_ALPHA = 0;
    }
    else
    {
		ESTIMATE_ALPHA = 1;
    }
	fscanf(fileptr, "data type: %d", &UNSUPERVISED);
    fclose(fileptr);
}

void write_sparse_statistics(char *model_root, lda_model* model, corpus* corpus, double **aa)
{	//write some statistics about the topics and latent representations of documents
	int d, k; 
	long sc, sum;
	char filename[1000];
    FILE *fileptr;

	sprintf(filename, "%s-stats.csv", model_root);
    fileptr = fopen(filename, "w");

	fprintf(fileptr, "Number of topics, =, %d\n", model->num_topics);
	fprintf(fileptr, "Number of terms, =, %d\n", model->num_terms);
	fprintf(fileptr, "Number of training documents, =, %d\n", corpus->num_docs);
	//summary of sparsity
	sum = model->num_topics *model->num_terms;
	fprintf(fileptr, "Topic sparsity: %d/%d, =, 1\n", sum, sum);

	sc =0;
	for (d=0; d < corpus->num_docs; d++)
		for (k=0; k < model->num_topics; k++)
			if (aa[d][k] > 0) sc++;
	sum = model->num_topics * corpus->num_docs;
	fprintf(fileptr, "Document sparsity: %d/%d, =, %f\n", sc, sum, sc/(double)sum);
	sum =0;
	for (d=0; d < corpus->num_docs; d++) sum += corpus->docs[d].length;
	fprintf(fileptr, "Doc. compress rate: %d/%d, =, %f\n", sc, sum, sc/(double)sum);
	fclose(fileptr);
}

/*
 * inference only
 *
 */

//void infer(char* model_root, char* save, corpus* corpus)
//{
//    FILE* fileptr;
//    char filename[100];
//    int i, d, n;
//    lda_model *model;
//    double **var_gamma, likelihood, **phi;
//    document* doc;
//
//    model = load_lda_model(model_root);
//    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
//    for (i = 0; i < corpus->num_docs; i++)
//	var_gamma[i] = malloc(sizeof(double)*model->num_topics);
//    sprintf(filename, "%s-lda-lhood.dat", save);
//    fileptr = fopen(filename, "w");
//    for (d = 0; d < corpus->num_docs; d++)
//    {
//	if (((d % 100) == 0) && (d>0)) printf("document %d\n",d);
//
//	doc = &(corpus->docs[d]);
//	phi = (double**) malloc(sizeof(double*) * doc->length);
//	for (n = 0; n < doc->length; n++)
//	    phi[n] = (double*) malloc(sizeof(double) * model->num_topics);
//	likelihood = lda_inference(doc, model, var_gamma[d], phi);
//
//	fprintf(fileptr, "%5.5f\n", likelihood);
//    }
//    fclose(fileptr);
//    sprintf(filename, "%s-gamma.dat", save);
//    save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
//}

double infer(char* model_root, char* save, corpus* corpus)
{
    FILE* fileptr;		char filename[1000];
    int i, d, n;		lda_model *model;	document* doc;
    double **var_gamma, likelihood, **phi, perplexity, tnum;
	clock_t start, end, init;

	init	= clock();
    model = load_lda_model(model_root);
    var_gamma = malloc(sizeof(double*)*(corpus->num_docs));
    for (i = 0; i < corpus->num_docs; i++)
		var_gamma[i] = malloc(sizeof(double)*model->num_topics);
    sprintf(filename, "%s-lhood.dat", save);
    fileptr = fopen(filename, "w");
	fprintf(fileptr, "docID, likelihood, time \n");

	printf(" Number of topics: %d \n", model->num_topics);
	printf(" Number of docs: %d \n", corpus->num_docs);
	printf(" Infering...\n");
	perplexity = 0;		tnum = 0;
    for (d = 0; d < corpus->num_docs; d++)
    {
		if ((d % 100) == 0) printf("  document %d\n", d);
		start = clock();
		doc = &(corpus->docs[d]);
		phi = (double**) malloc(sizeof(double*) * doc->length);		
		for (n = 0; n < doc->length; n++)		
			phi[n] = (double*) malloc(sizeof(double) * model->num_topics);		
		likelihood = lda_inference(doc, model, var_gamma[d], phi);
		end = clock();
		fprintf(fileptr, "%d, %10.10f, %10.10f \n", d, likelihood, ((double)(end - start)) /CLOCKS_PER_SEC);
		perplexity += likelihood;
		tnum += doc->total;
		//free Phi
		for (n = doc->length -1; n >= 0; n--) free(phi[n]);
	}
 //   sprintf(filename, "%s-inf-gamma.dat", save);
	////printf("\nBefore save Gamma.");
 //   save_gamma(filename, var_gamma, corpus->num_docs, model->num_topics);
	//sprintf(filename, "%s-inf", model_root);
	write_sparse_statistics(save, model, corpus, var_gamma);
	save_topic_docs(save, var_gamma, corpus, model->num_topics);
	//free parameters	
	for (i = corpus->num_docs -1; i >= 0; i--) free(var_gamma[i]);
	free_lda_model(model); doc=NULL;

	end = clock();
	fprintf(fileptr, "Overall time:, , %10.10f \n", ((double)(end - init)) /CLOCKS_PER_SEC);
	fclose(fileptr);
	printf("  Completed.\n");
	return -perplexity/tnum;
}

/*
 * update sufficient statistics
 *
 */



/*
 * main
 *
 */

//int main(int argc, char* argv[])
//{
//    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)
//
//    corpus* corpus;
//
//    int i;	double perp;
//	FILE *fperp;	char str[1000];
//
//	if (strcmp(argv[1], "est")==0)
//	{	// Eg: ./lda est nip train-data.txt 1 10
//		//name of the corpus: argv[3]
//		read_settings("settings.txt");
//		if (UNSUPERVISED == 1 || UNSUPERVISED == 2)
//			corpus = read_data(argv[3], UNSUPERVISED);
//		else if (UNSUPERVISED == 3)
//			corpus = read_multilabel_data(argv[3], 1);
//		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
//		{
//			INITIAL_ALPHA = 0.1;
//			NTOPICS = 10*i; 
//			printf("\nNumber of topics: %d \n", NTOPICS);
//			sprintf(str, "_%s%d", argv[2], NTOPICS);
//			make_directory(str);
//			run_em("random", str, corpus);	
//		}	
//	}
//	if (strcmp(argv[1], "inf")==0)
//	{
//		//input the first 3 words of name of corpus, and the new documents needed to infer
//		// Eg: ./lda inf Nip infer-data.txt 1 10
//		sprintf(str, "%s_%s_lda.per", argv[3], argv[2]);
//		fperp =fopen(str, "w");
//		fprintf(fperp, "NTopics, perplexity \n");
//		read_settings("inf-settings.txt");
//		if (UNSUPERVISED == 1 || UNSUPERVISED == 2)
//			corpus = read_data(argv[3], UNSUPERVISED);
//		else if (UNSUPERVISED == 3)
//			corpus = read_multilabel_data(argv[3], 1);  //where to infer
//		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
//		{
//			sprintf(str, "_%s%d/final-lda", argv[2], 10*i);			
//			perp = infer(str, str, corpus);
//			fprintf(fperp, "%d, %10.10f \n", 10*i, perp);			
//		}
//		fclose(fperp);
//	}
//    return(0);
//}

