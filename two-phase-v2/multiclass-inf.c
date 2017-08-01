
#include "multiclass-inf.h"
#include "wirth.c"

void compute_init_reg(double **C, int nclasses, int ntopics, char *training_rep)
{
	corpus *corp;	int d, j;
	corp = read_data(training_rep, UNSUPERVISED);		//read the representation by training phase
	for (j = 0; j < nclasses; j++)
		for (d = 0; d < ntopics; d++) C[j][d] = 0;
	for (d = 0; d < corp->num_docs; d++)
		for (j =0; j < corp->docs[d].length; j++) 
			C[corp->docs[d].label][corp->docs[d].words[j]] += corp->docs[d].counts[j];
	for (d = 0; d < nclasses; d++)	L1_normalize(C[d], ntopics);
//	free_corpus(corp);
}


void multi_discriminative_topics(double **C, int nclasses, int ntopics, double reg)
{//Set the regularization constants in order to exlude topics that are not
	// discriminative to a class. Such topics are derived from training.
	int d, j;	double bound, median, tmp[nclasses], min;
	bound = 1.5;	//if contribution ratio is less than 1.5.
	for (d = 0; d < ntopics; d++)
	{
		for (j = 0; j < nclasses; j++) tmp[j] = C[j][d];
		median = kth_smallest(tmp, nclasses, nclasses/2 -1);
		min = kth_smallest(tmp, nclasses, 0);
		for (j = 0; j < nclasses; j++) 
			if (C[j][d] >= median && C[j][d] > bound*min) C[j][d] = -reg;	//encourage contribution of topic d
			else C[j][d] = reg;								//discourage contribution
	}
}

void multi_promote_topics(double **C, int nclasses, int ntopics, double reg, double bound)
{//Set the regularization constants in order to promote topics that are
	// discriminative to a class. Such topics are derived from training.
	int d, j;	double min, median, tmp[nclasses];
	
	for (d = 0; d < ntopics; d++)
	{
		for (j = 0; j < nclasses; j++) tmp[j] = C[j][d];
		median	= kth_smallest(tmp, nclasses, nclasses/2 -1);
		min		= kth_smallest(tmp, nclasses, 0);
		for (j = 0; j < nclasses; j++) 
			if (C[j][d] >= median && C[j][d] > bound*min) C[j][d] = reg;	//encourage contribution of topic d
			else C[j][d] = 0;
	}
}


void combine_knn(corpus *corp, corpus *nbors, double lambda)
{//construct nearest-neighbor graph and then for each point compute the mean of its neighbors
    int n, nd, num;		double *doc;	

    printf("\t combining nearest neighbors (knn=%d, ld=%f)...", kNN, lambda);  fflush(stdout);
	doc = (double*) malloc(corp->num_terms * sizeof(double));
	for(nd = 0; nd < corp->num_docs; nd++)
	{
		for (n = 0; n < corp->num_terms; n++) doc[n] = 0;
		for (n = 0; n < corp->docs[nd].length; n++) 
			doc[corp->docs[nd].words[n]] = lambda * corp->docs[nd].counts[n];
		for (n = 0; n < nbors->docs[nd].length; n++) 
			doc[nbors->docs[nd].words[n]] += (1-lambda) * nbors->docs[nd].counts[n];
		num = 0;
		for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > 0) num++;
		//new document representation
		if (num > nbors->docs[nd].length)
		{
			nbors->docs[nd].length = num;
			free(nbors->docs[nd].words); free(nbors->docs[nd].counts);
			nbors->docs[nd].words  = malloc(sizeof(int)*num);
			nbors->docs[nd].counts = malloc(sizeof(double)*num);
			if (nbors->docs[nd].words ==0 || nbors->docs[nd].counts ==0) 
				{ printf("\n find_neighbors: cannot allocate memory..\n"); exit(0); }
		}
		num = 0;
		for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > 0) 
			{
				nbors->docs[nd].words[num]	 = n;
				nbors->docs[nd].counts[num] = doc[n];
				nbors->docs[nd].total		+= doc[n];
				num++;
			}
	}
    free(doc);	printf("\t completed.\n");
}

double multi_Infer_disc(char *model_root, corpus *corp, corpus *nbors, char *function_name, char *model_name)
{
    char filename[1000];    int d, i;
    plsa_model *model;	double **aa, *fx, *dfx, **C;
	char topic_file[1000], other_file[1000];

	if (strcmp(function_name, "none") == 0) return 0;

	sprintf(topic_file, "%s-%s.beta", model_root, model_name);
	sprintf(other_file, "%s-%s.other", model_root, model_name);
	model	= load_model(topic_file, other_file);
	aa		= matrix_initialize(corp->num_docs, model->num_topics);

	if (strcmp(function_name, "sin") == 0) { fx = f_KL_sine;	dfx = df_KL_sine; }
	else if (strcmp(function_name, "cos-sin") == 0) { fx = f_KL_cos_sin;	dfx = df_KL_cos_sin; }
	//Coefficients for regularization.
	//lower down the contribution of topics to a class if they contribute significantly to other class
	C = matrix_initialize(corp->num_labels, model->num_topics);
	sprintf(filename, "%s-%s-topics-docs-contribute.dat", model_root, model_name); //from learning
	compute_init_reg(C, corp->num_labels, model->num_topics, filename);
	if ((strcmp(function_name, "sin") == 0) )
			multi_promote_topics(C, corp->num_labels, model->num_topics, reg, 1.5);
	else if ((strcmp(function_name, "cos-sin") == 0) )
			multi_discriminative_topics(C, corp->num_labels, model->num_topics, reg);

	compute_entropy(corp);
	if (lambda == 1 || kNN == 0) nbors = corp; 
    else {
        combine_knn(corp, nbors, lambda);
        compute_entropy(nbors);
    }

	printf(" Infering: ");  fflush(stdout);
    for (d = 0; d < corp->num_docs; d++)
    {
		if (((d % 1000) == 0) && (d>0)) {printf("%d, ", d);  fflush(stdout);}
		doc_projection(model, &(nbors->docs[d]), aa[d], fx, dfx, C[corp->docs[d].label]);
	}
	//compute new topics
	sprintf(filename, "%s-%s%d-k%d-ld%1.2f", model_root, function_name, (int)reg, kNN, lambda); 
	if (strcmp(model_name, "plsa") == 0)
		save_plsa_topics(model, aa, corp, filename, other_file);
	else if (strcmp(model_name, "lda") == 0)
		save_lda_topics(model, aa, corp, filename, other_file);
	else if (strcmp(model_name, "fstm") == 0)
		save_fstm_topics(model, aa, corp, filename, other_file, C);

	sprintf(filename, "%s-%s%d-k%d-ld%1.2f", model_root, function_name, (int)reg, kNN, lambda);
//	save_topic_docs(filename, aa, corp, model->num_topics);
//	write_sparse_statistics(filename, model, corp, aa);
	plsa_model_Free(model);
	for (d = corp->num_docs -1;  d >=0; d--)   free(aa[d]);
	for (d = corp->num_labels -1;  d >=0; d--)   free(C[d]);
	printf("  Completed.\n");
return(0);
}

