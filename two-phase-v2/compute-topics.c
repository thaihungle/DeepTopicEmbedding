
#include "compute-topics.h"


void save_lda_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file)
{
    FILE* fileptr;		int i, j, d, word;	
	double **tmp, phi[model->num_topics], count;
	char filename[1000];

	tmp		= matrix_initialize(model->num_topics, model->num_terms);
	printf(" Computing topics...\n");
	for (d = 0; d < model->num_topics; d++)
		for (j = 0; j < model->num_terms; j++) tmp[d][j] = 0;
    for (d = 0; d < corp->num_docs; d++)
		for (j = 0; j < corp->docs[d].length; j++)
		{
			word = corp->docs[d].words[j];		count = corp->docs[d].counts[j];
			for (i = 0; i < model->num_topics; i++) 
				phi[i] = model->bb[i][word] * exp(digamma(aa[d][i] + (1e-10)));
			L1_normalize(phi, model->num_topics);
			for (i = 0; i < model->num_topics; i++)	tmp[i][word] += count * phi[i];
		}
	//Save topics
	sprintf(filename, "%s.beta", topic_file);
	fileptr = fopen(filename, "w");
	for (i = 0; i < model->num_topics; i++)
    {
		L1_normalize(tmp[i], model->num_terms);
		for (j = 0; j < model->num_terms; j++)
		{
			if (tmp[i][j] > 0)	fprintf(fileptr, "%1.10f ", tmp[i][j]);
			else fprintf(fileptr, "0 ") ;
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
	double *tg;
	tg = tmp;	tmp = model->bb;	model->bb = tg;
	for (d = model->num_topics -1;  d >=0; d--)   free(tmp[d]);
	//Save other file
	sprintf(filename, "%s.other", topic_file);
	copy_other_file(other_file, filename);
}

void save_plsa_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file)
{
    FILE* fileptr;		int i, j, d, word;	
	double **tmp, phi[model->num_topics], count;
	char filename[1000];

	tmp		= matrix_initialize(model->num_topics, model->num_terms);
	printf(" Computing topics...\n");
	for (d = 0; d < model->num_topics; d++)
		for (j = 0; j < model->num_terms; j++) tmp[d][j] = 0;
    for (d = 0; d < corp->num_docs; d++)
		for (j = 0; j < corp->docs[d].length; j++)
		{
			word = corp->docs[d].words[j];		count = corp->docs[d].counts[j];
			for (i = 0; i < model->num_topics; i++) phi[i] = model->bb[i][word] * aa[d][i];
			L1_normalize(phi, model->num_topics);
			for (i = 0; i < model->num_topics; i++)	tmp[i][word] += count * phi[i];
		}
	//Save topics
	sprintf(filename, "%s.beta", topic_file);
	fileptr = fopen(filename, "w");
	for (i = 0; i < model->num_topics; i++)
    {
		L1_normalize(tmp[i], model->num_terms);
		for (j = 0; j < model->num_terms; j++)
		{
			if (tmp[i][j] > 0)	fprintf(fileptr, "%1.10f ", tmp[i][j]);
			else fprintf(fileptr, "0 ") ;
		}
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
	double *tg;
	tg = tmp;	tmp = model->bb;	model->bb = tg;
	for (d = model->num_topics -1;  d >=0; d--)   free(tmp[d]);
	//Save other file
	sprintf(filename, "%s.other", topic_file);
	copy_other_file(other_file, filename);
}

void save_fstm_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file, double **C)
{
    FILE* fileptr;		int i, j, d;	char filename[1000];

	printf(" Computing topics...\n");
	for (d = 0; d < model->num_topics; d++)
		for (j = 0; j < model->num_terms; j++) model->bb[d][j] = 0;

	for (d = 0; d < corp->num_docs; d++) 
		for (j = 0; j < corp->docs[d].length; j++)
			for (i = 0; i < model->num_topics; i++)
				model->bb[i][corp->docs[d].words[j]] += aa[d][i] * corp->docs[d].counts[j];
	for (i = 0; i < model->num_topics; i++)
		L1_normalize(model->bb[i], model->num_terms);
	//Save topics
	sprintf(filename, "%s.beta", topic_file);
	fileptr = fopen(filename, "w");
	for (i = 0; i < model->num_topics; i++)
    {
		for (j = 0; j < model->num_terms; j++)
			if (model->bb[i][j] > 0)	fprintf(fileptr, "%1.10f ", model->bb[i][j]);
			else fprintf(fileptr, "0 ") ;
		fprintf(fileptr, "\n");
    }
    fclose(fileptr);
	//Save other file
	sprintf(filename, "%s.other", topic_file);
	copy_other_file(other_file, filename);
}

void copy_other_file(char *source, char *dist)
{	
	char str[5000]; int n=5000;	FILE *fileptr, *filein;

	fileptr = fopen(dist, "w");
	filein = fopen(source, "r");
	while (!feof(filein))
	{
		str[0] = '\0';
		fgets(str, n, filein); 	fprintf(fileptr, "%s", str);
    }
	fclose(filein);  fclose(fileptr); 
}


