
#include "plsa-model.h"

plsa_model* plsa_model_New(int num_terms, int num_topics, float t_sparse)
{
    int i;    plsa_model* model;

    model = (plsa_model*) malloc(sizeof(plsa_model));
	if (model ==0) 
		{ printf("\nNew_plsa_model: cannot allocate memory for the model.\n"); exit(0); }
	model->num_topics	= num_topics;
    model->num_terms	= num_terms;
	model->t_sparse		= t_sparse;
    model->bb = (double**)malloc(sizeof(double*)*num_topics);
	if (model->bb==0)
		{printf("\nNew_plsa_model: failed to allocate memory for model->bb!");
				exit(0); }
	for (i = 0; i < num_topics; i++)
    {
		model->bb[i] = malloc(sizeof(double)*num_terms);
		if (model->bb[i]==0)
			{printf("\nNew_plsa_model: failed to allocate memory for model->bb!"); exit(0); }
	}
    return(model);
}

void plsa_model_Free(plsa_model* model)
{
    int i;
    for (i = model->num_topics -1; i>=0; i--)
    {
		free(model->bb[i]);  
    }
	free(model);
}

void plsa_model_Save(plsa_model* model, char* model_root)
{
    char filename[1000];
    FILE *fileptr;
    int i, j;

    sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics: %d \n", model->num_topics);
	fprintf(fileptr, "num_terms: %d \n", model->num_terms);
	fprintf(fileptr, "train_count: %lf \n", model->train_count);
	fprintf(fileptr, "topic_sparsity: %f \n", model->t_sparse);
	fclose(fileptr);

	sprintf(filename, "%s.beta", model_root);
    fileptr = fopen(filename, "w");
	for (i = 0; i < model->num_topics; i++)
    {
		for (j = 0; j < model->num_terms; j++)
		{
			if (model->bb[i][j] > 0)	fprintf(fileptr, "%1.10f ", model->bb[i][j]) ;
			else fprintf(fileptr, "0 ") ;
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}


plsa_model* plsa_model_Load(char* model_root)
{
    char filename[100];    FILE* fileptr;	plsa_model *model;
    int i, j, num_terms, num_topics;    double x, count;

	sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics: %d \n", &num_topics);
	fscanf(fileptr, "num_terms: %d \n", &num_terms);
	fscanf(fileptr, "train_count: %lf \n", &count);
	fscanf(fileptr, "topic_sparsity: %lf", &x);
	fclose(fileptr);

    sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
	if (fileptr==NULL)
	{ 		printf("Cannot open %s\n", filename);  exit(0);	}

    model = plsa_model_New(num_terms, num_topics, (float)x);
	model->train_count = count;
	for (i = 0; i < num_topics; i++)
    {
        for (j = 0; j < num_terms; j++)
        {
            fscanf(fileptr, "%lf", &x);
            model->bb[i][j] = x;
	    }
    }
	correct_zero_terms(model->bb, model->num_terms, model->num_topics);
	fclose(fileptr); 
	printf("Model was loaded.\n");
    return(model);
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


void read_settings(char* filename)
{
    FILE* fileptr;
    fileptr = fopen(filename, "r");
	fscanf(fileptr, "em max iter: %d\n", &EM_MAX_ITER);
	fscanf(fileptr, "em convergence: %e\n", &EM_CONVERGED);
	fscanf(fileptr, "infer max iter: %d\n", &INF_MAX_ITER);
	fscanf(fileptr, "infer convergence: %e\n", &INF_CONVERGED);
	fscanf(fileptr, "topic sparsity: %f\n", &T_SPARSE);
	fscanf(fileptr, "data type: %d", &UNSUPERVISED);
    fclose(fileptr);
}

