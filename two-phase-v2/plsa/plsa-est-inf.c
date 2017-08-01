
#include "plsa-est-inf.h"

int LAG = 5;

/*
 * perform inference for a document
 */

double doc_inference(plsa_model* model, document doc, double *aa)
{	//keeping the topics, infer the document representation.
    double likelihood, likelihood_old, sum, *topicD, *aa_temp, converge;
    int k, j, t;

	topicD  = malloc(sizeof(double)*model->num_topics);
	aa_temp = malloc(sizeof(double)*model->num_topics);
	for (k = 0; k < model->num_topics; k++)		aa[k] = 1/(double)model->num_topics;
	t = 0;  
	do{
		t++;
		for (k = 0; k < model->num_topics; k++)	aa_temp[k] = 0;
		for (j = 0; j < doc.length; j++)
		{
			for (k = 0; k < model->num_topics; k++)		topicD[k] = model->bb[k][doc.words[j]] * aa[k];
			L1_normalize(topicD, model->num_topics);
			for (k = 0; k < model->num_topics; k++)		aa_temp[k] += doc.counts[j] *topicD[k];
		}
		L1_normalize(aa_temp, model->num_topics);
		for (k = 0; k < model->num_topics; k++) aa[k] = aa_temp[k];
		// compute likelihood
		likelihood = doc.total * (log(doc.total) - log(doc.total + model->train_count));
		for (j = 0; j < doc.length; j++)
		{
			sum =0;
			for (k = 0; k < model->num_topics; k++)		sum += aa[k]*model->bb[k][doc.words[j]];
			if (sum <=0) { printf("\n doc_inference: probability is zero!\n"); exit(0); }
			likelihood += doc.counts[j]*log(sum);
		}
		converge = (likelihood_old - likelihood) / likelihood_old;
		if (converge < 0 && t < INF_MAX_ITER) continue;
	}while ((t < INF_MAX_ITER) && (t<2 || converge > INF_CONVERGED));
	//true log(likelihood)
	likelihood -= doc.total * (log(doc.total) - log(doc.total + model->train_count));
	free(topicD); free(aa_temp);
return(likelihood);
}

void plsa_Learn(char* directory, corpus* corpus)
{
    int d, k, i, j;		plsa_model *model;	document *doc;
    double likelihood, likelihood_old = 0, converged, **aa, **aa_temp, **bb_temp, *topicD, **tmp;
	clock_t start, end, init = clock();		char filename[1000];
	
    // new representation of documents
	aa		= matrix_initialize(corpus->num_docs, NTOPICS);
	aa_temp = matrix_initialize(corpus->num_docs, NTOPICS); //temporary doc representation
	bb_temp = matrix_initialize(NTOPICS, corpus->num_terms); //temporary topics
	topicD	= malloc(sizeof(double) * NTOPICS);				//topic propotion for each word in a doc
	
    // initialize model
	model = plsa_model_New(corpus->num_terms, NTOPICS, T_SPARSE);
	model->train_count = 0;
	for (d = 0; d < corpus->num_docs; d++) 
		model->train_count += corpus->docs[d].total;
	matrix_random_initialize_L1(aa, corpus->num_docs, model->num_topics);
	matrix_random_initialize_L1(model->bb, model->num_topics, model->num_terms);
	
    sprintf(filename,"%s/000-plsa", directory);
	plsa_model_Save(model, filename);

    // run expectation maximization
    sprintf(filename, "%s/likelihood-plsa.dat", directory);
    FILE* likelihood_file = fopen(filename, "w");
	fprintf(likelihood_file, "Iteration, Likelihood, Converged, Time (seconds) \n");
	printf("\nRun EM algorithm.....\n");
	i = 0;
    while (((converged < 0) || (converged > EM_CONVERGED) || (i <= 2)) && (i < EM_MAX_ITER))
    {
        i++;  printf("**** em iteration %d ****\n", i);
		start = clock();
		//e-step
		for (d = 0; d < corpus->num_docs;  d++) clear_vector(aa_temp[d], model->num_topics);
		for (k = 0; k < model->num_topics; k++) clear_vector(bb_temp[k], model->num_terms);
		for (d = 0; d < corpus->num_docs; d++)
		{
			doc = &(corpus->docs[d]);
			for (j = 0; j < doc->length; j++)
			{
				for (k = 0; k < model->num_topics; k++)
					topicD[k] = model->bb[k][doc->words[j]] * aa[d][k];
				L1_normalize(topicD, model->num_topics);
				for (k = 0; k < model->num_topics; k++)
				{
					bb_temp[k][doc->words[j]] += doc->counts[j] *topicD[k];
					aa_temp[d][k] += doc->counts[j] *topicD[k];
				}
			}
			L1_normalize(aa_temp[d], model->num_topics);
		}
		for (k = 0; k < model->num_topics; k++) L1_normalize(bb_temp[k], model->num_terms);

		//m-step
		tmp = aa;			aa = aa_temp;			aa_temp = tmp;
		tmp = model->bb;	model->bb = bb_temp;	bb_temp = tmp;

		//compute likelihood
		likelihood = data_likelihood(model, corpus, aa);
        // check for convergence
        converged = (likelihood_old - likelihood) / (likelihood_old);
        likelihood_old = likelihood;
        // output model and likelihood
		end = clock();
		printf("  Likelihood: %10.10f; \t Converged %5.5e \n\n", likelihood, converged);
        fprintf(likelihood_file, "%d, %10.10f, %5.5e, %10.10f \n", i, likelihood, converged, ((double)(end - start)) /CLOCKS_PER_SEC);
        fflush(likelihood_file);
        /*if ((i % LAG) == 0)
        {
            sprintf(filename,"%s/%03d-plsa", directory, i);
            plsa_model_Save(model, filename);
			save_topic_docs(filename, aa, corpus, model->num_topics);
			write_sparse_statistics(filename, model, corpus, aa);
        }*/
    } //end EM
	fprintf(likelihood_file, "%d, %10.10f, %5.5e, 0 \n", i+1, likelihood, converged);
	// output the final model
	printf("    saving the final model.\n");
    sprintf(filename,"%s/final-plsa", directory);
    plsa_model_Save(model, filename);
	save_topic_docs(filename, aa, corpus, model->num_topics);
	write_sparse_statistics(filename, model, corpus, aa);
	//free parameters
	for (d = corpus->num_docs -1;  d >=0; d--)   { free(aa[d]); free(aa_temp[d]); }
	for (d = model->num_topics -1; d >=0; d--)   free(bb_temp[d]); 
	free(topicD); tmp = NULL;
	plsa_model_Free(model);
	end = clock();
	fprintf(likelihood_file, "Overall time:, , , %10.10f \n", ((double)(end - init)) /CLOCKS_PER_SEC);
	fclose(likelihood_file);
	printf("    Model has been learned.\n");
}

void write_sparse_statistics(char *model_root, plsa_model* model, corpus* corpus, double **aa)
{	//write some statistics about the topics and latent representations of documents
	int d, k; 
	long sc, sum;
	char filename[100];
    FILE *fileptr;

	sprintf(filename, "%s-stats.csv", model_root);
    fileptr = fopen(filename, "w");

	fprintf(fileptr, "Number of topics, =, %d\n", model->num_topics);
	fprintf(fileptr, "Number of terms, =, %d\n", model->num_terms);
	fprintf(fileptr, "Number of training documents, =, %d\n", corpus->num_docs);
	//summary of sparsity
	sc =0;
	for (k=0; k < model->num_topics; k++)
		for (d=0; d < model->num_terms; d++) 
			if (model->bb[k][d] > 0) sc++;
	sum = model->num_topics *model->num_terms;
	fprintf(fileptr, "Topic sparsity: %d/%d, =, %f\n", sc, sum, sc/(double)sum);

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

double data_likelihood(plsa_model* model, corpus* corpus, double **aa)
{//compute likelihood of data when new representation is known as 'aa'.
	int i, j, k;
	double sum, lkh = 0;
	for (i = 0; i < corpus->num_docs; ++i)
	{
		lkh += corpus->docs[i].total * log(corpus->docs[i].total/model->train_count);
		for (j = 0; j < corpus->docs[i].length; j++)
		{
			sum =0;
			for (k = 0; k < model->num_topics; k++)		
				sum += aa[i][k]*model->bb[k][corpus->docs[i].words[j]];
			lkh += corpus->docs[i].counts[j]*log(sum);
		}
	}
	return (lkh);
}


/*
 * inference only
 *
 */

double plsa_Infer(char *model_root, char *save, corpus* corpus)
{
    FILE* fileptr;			char filename[1000];
    plsa_model *model;		clock_t start, end, init;
	int d;	document doc;
    double likelihood, **aa, perplexity, tnum;

    init	= clock();
    model	= plsa_model_Load(model_root);
	aa		= matrix_initialize(corpus->num_docs, model->num_topics);

	sprintf(filename, "%s-lhood.csv", save);
    fileptr = fopen(filename, "w");
	fprintf(fileptr, "docID, likelihood, time \n");

	printf("  Number of topics: %d \n", model->num_topics);
	printf("  Number of docs: %d \n", corpus->num_docs);
	printf("  Infering...\n");
	perplexity = 0;
	tnum = 0;	
    for (d = 0; d < corpus->num_docs; d++)
    {
		if (((d % 100) == 0) && (d>0)) printf("document %d\n", d);
		start = clock();
		doc = corpus->docs[d];	
		likelihood = doc_inference(model, doc, aa[d]);
		end = clock();
		fprintf(fileptr, "%d, %10.10f, %10.10f \n", d, likelihood, ((double)(end - start)) /CLOCKS_PER_SEC);
		perplexity += likelihood;
		tnum += doc.total;
	}
	//sprintf(filename, "%s-inf", model_root);
	save_topic_docs(save, aa, corpus, model->num_topics);
	write_sparse_statistics(save, model, corpus, aa);
	plsa_model_Free(model);
	for (d = corpus->num_docs -1;  d >=0; d--)   free(aa[d]);
	end = clock();
	fprintf(fileptr, "Overall time:, , %10.10f \n", ((double)(end - init)) /CLOCKS_PER_SEC);
	fclose(fileptr);
	printf("  Completed.\n");
	return -perplexity/tnum;
}



