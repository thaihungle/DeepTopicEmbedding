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

#include "fstm-model.h"

fstm_model* fstm_model_New(int num_terms, int num_topics, float t_sparse)
{
    fstm_model* model;

    model = (fstm_model*) malloc(sizeof(fstm_model));
	if (model ==0) 
		{ printf("\n fstm_model_New: cannot allocate memory for the model.\n"); exit(0); }
	model->num_topics	= num_topics;
    model->num_terms	= num_terms;
	model->t_sparse		= t_sparse;
    model->bb = (pairs*)malloc(sizeof(pairs)*num_terms);
	if (model->bb == 0)
		{printf("\n fstm_model_New: failed to allocate memory for model->bb!");
				exit(0); }
return(model);
}

void fstm_model_Free(fstm_model* model)
{
    int i;
    for (i = model->num_terms -1; i>=0; i--)
    {
		free(model->bb[i].topicID);  free(model->bb[i].value);  
    }
	free(model->bb);  free(model);
}


/*
 * save an fstm model
 *
 */

void fstm_model_Save(fstm_model* model, char* model_root)
{
    char filename[1000];    FILE *fileptr;
    int i, j, k;			double **tmp;

    sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "w");
	fprintf(fileptr, "num_topics: %d \n", model->num_topics);
	fprintf(fileptr, "num_terms: %d \n", model->num_terms);
	fprintf(fileptr, "topic_sparsity: %f \n", model->t_sparse);
	fclose(fileptr);
	sprintf(filename, "%s.beta", model_root);
    fileptr = fopen(filename, "w");
	tmp = malloc(sizeof(double*) * model->num_topics);
	for (k = 0; k < model->num_topics; k++)
	{
		tmp[k] = malloc(sizeof(double) * model->num_terms);
		if (tmp[k] == NULL) {printf("Cannot allocate memory.\n"); exit(0); }
		for (i = 0; i < model->num_terms; i++) tmp[k][i] = 0;
	}
	for (i = 0; i < model->num_terms; i++)
		if (model->bb[i].length > 0)
			for (j = 0; j < model->bb[i].length; j++) 
				tmp[model->bb[i].topicID[j]][i] = model->bb[i].value[j];
	//saving
	for (i = 0; i < model->num_topics; i++)
    {
		for (j = 0; j < model->num_terms; j++)
			if (tmp[i][j] > 0)	fprintf(fileptr, "%1.10f ", tmp[i][j]) ;
			else fprintf(fileptr, "0 ") ;
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
	for (k = model->num_topics -1; k>= 0; k--)	free(tmp[k]);
}


fstm_model* fstm_model_Load(char* model_root)
{
    char filename[1000];    FILE* fileptr;	fstm_model *model;
    int i, j, num_terms, num_topics, length;
    double x, **tmp;

	sprintf(filename, "%s.other", model_root);
    fileptr = fopen(filename, "r");
	fscanf(fileptr, "num_topics: %d \n", &num_topics);
	fscanf(fileptr, "num_terms: %d \n", &num_terms);
	fscanf(fileptr, "topic_sparsity: %lf", &x);
	fclose(fileptr);

    sprintf(filename, "%s.beta", model_root);
    printf("loading %s\n", filename);
    fileptr = fopen(filename, "r");
	if (fileptr == NULL)
	{ 		printf("Cannot open %s\n", filename);  exit(0);	}
	//read to array
	tmp = malloc(sizeof(double*) * num_topics);
	for (j = 0; j < num_topics; j++)
	{
		tmp[j] = malloc(sizeof(double) * num_terms);
		if (tmp[j] == NULL) {printf("Cannot allocate memory.\n"); exit(0); }
		for (i = 0; i < num_terms; i++) tmp[j][i] = 0;
	}
	for (i = 0; i < num_topics; i++)
		for (j = 0; j < num_terms; j++)
        {
			fscanf(fileptr, "%lf", &x);		tmp[i][j] = x;
	    }
	fclose(fileptr); 
	//save to model
    model = fstm_model_New(num_terms, num_topics, 0.1);
	for (i = 0; i < num_terms; i++)
    {
		length = 0;
		for (j = 0; j < num_topics; j++)
			if (tmp[j][i] > 0) length++;
		model->bb[i].length = length;
		if (length < 1)
		{
			model->bb[i].topicID	= malloc(sizeof(int));
			model->bb[i].value		= malloc(sizeof(double));
			continue;
		}
		model->bb[i].topicID	= malloc(sizeof(int)*length);
		model->bb[i].value		= malloc(sizeof(double)*length);
		if (model->bb[i].topicID == 0 || model->bb[i].value == 0)
			{printf("\nfstm_model_Load: failed to allocate memory for model->bb[%d] !", i); exit(0); }
		length = 0;
        for (j = 0; j < num_topics; j++)
			if (tmp[j][i] > 0)
			{
				model->bb[i].topicID[length] = j;
				model->bb[i].value[length] = tmp[j][i];
				length++;
			}
    }
	for (i = model->num_topics -1; i>= 0; i--)	free(tmp[i]);
	printf(" Model was loaded.\n");
return(model);
}


void save_topic_docs_fstm(char* model_root, corpus *aa, corpus *corp, int num_topics)
{
    char filename[1000];    FILE *fileptr;
    int i, j;	double tmp[num_topics];

    sprintf(filename, "%s-topics-docs-contribute.dat", model_root);
    fileptr = fopen(filename, "w");

	for (i = 0; i < corp->num_docs; i++)
    {
		for (j = 0; j < num_topics; j++) tmp[j] = 0;
		for (j = 0; j < aa->docs[i].length; j++)
			tmp[aa->docs[i].words[j]] = aa->docs[i].counts[j];
		switch (UNSUPERVISED)	
		{
			case 1 :		//simply write a matrix (unsupervised)
				for (j = 0; j < num_topics; j++)
					if (tmp[j] > 0)	fprintf(fileptr, "%1.10f ", tmp[j]) ;
					else fprintf(fileptr, "0 ") ;
				break;
			case 2 :		//write to a file in Libsvm format (multi-class single-label)
				fprintf(fileptr, "%d ", corp->label_names[corp->docs[i].label]);
				for (j = 0; j < num_topics; j++)
					if (tmp[j] > 0)	fprintf(fileptr, "%d:%1.10f ", j+1, tmp[j]);
				break;
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
}


void L1_normalize_sparse_topics(fstm_model* model)
{
	int i, j;
	double  *temp;
	temp = malloc(sizeof(double)*model->num_topics);
	for (i = 0; i < model->num_topics; i++) temp[i] = 0;
	for (i = 0; i < model->num_terms; i++)
		if (model->bb[i].length > 0)
		for (j = 0; j < model->bb[i].length; j++)
			temp[model->bb[i].topicID[j]] += model->bb[i].value[j];

	for (i = 0; i < model->num_terms; i++)
		if (model->bb[i].length > 0)
		for (j = 0; j < model->bb[i].length; j++)
			model->bb[i].value[j] /= temp[model->bb[i].topicID[j]];
	free(temp);
}


void initialize_random_topics(fstm_model *model)
{//initialize topics randomly
	int i, j, k, nnz, *temp;
	nnz  = (int)round(model->t_sparse * model->num_topics);
	temp = malloc(sizeof(int)*model->num_topics);
    for (i = 0; i < model->num_terms; i++)
    {
		model->bb[i].topicID	= malloc(sizeof(int)*nnz);
		model->bb[i].value		= malloc(sizeof(double)*nnz);
		if (model->bb[i].topicID == 0 || model->bb[i].value == 0)
			{printf("\n Initilize_random_topics: failed to allocate memory for model->bb[%d] !", i); exit(0); }
		for (j = 0; j < model->num_topics; j++) temp[j] = 0;
		for (j = 0; j < nnz; j++)
			{	k = rand() % model->num_topics; temp[k] = 1; }
		k = 0;
		for (j = 0; j < model->num_topics; j++) 
			if (temp[j] > 0)
			{
				model->bb[i].topicID[k]	= j;
				model->bb[i].value[k]	= rand(); //% 1000;
				k++;
			}
		model->bb[i].length = k;
    }
	L1_normalize_sparse_topics(model);
	free(temp);
}

void read_settings(char* filename)
{
    FILE* fileptr; char str[2000]; int n=2000;
    fileptr = fopen(filename, "r");
	fgets(str, n, fileptr);	sscanf(str, "%d", &EM_MAX_ITER);
	fgets(str, n, fileptr);	sscanf(str, "%f", &EM_CONVERGED);
	fgets(str, n, fileptr);	sscanf(str, "%d", &INF_MAX_ITER);
	fgets(str, n, fileptr);	sscanf(str, "%f", &INF_CONVERGED);
	fgets(str, n, fileptr);	sscanf(str, "%f", &T_SPARSE);
	fgets(str, n, fileptr);	sscanf(str, "%d", &UNSUPERVISED);
	fgets(str, n, fileptr);	sscanf(str, "%d", &WARM_START);
    fclose(fileptr);
}


