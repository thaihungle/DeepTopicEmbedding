
#include "custom-inf.h"
#include "multiclass-inf.h"

void read_settings_infer(char* filename);
corpus* read_neighbors(char* filename, corpus *corp);
void save_topic_classes(char *model_root, char *model_name);

void main(int argc, char* argv[])
{
    corpus *corp, *nbors;	char str[1000], filename[1000];
    char setting[1000];
    sprintf(setting,"%s/%s",argv[7],"inf-settings.txt");
    read_settings_infer(setting);
    sprintf(str, "%s/_%s_%s%s/final", argv[7], argv[3], argv[1], argv[6]);
    if (strcmp(argv[2], "topics-classes")==0)	save_topic_classes(str, argv[1]);
    else
        if (UNSUPERVISED == 2)	//inference for supervised data
    {	// Eg: ./infer plsa sin train train-data.txt train-neighbors.txt 100
        char datastr[1000];
        char neistr[1000];
        sprintf(datastr,"%s/%s",argv[7],argv[4]);
        sprintf(neistr,"%s/%s",argv[7],argv[5]);
        corp = read_data(datastr, UNSUPERVISED);
        nbors = read_neighbors(neistr, corp);
        if (PRIOR == 1){
                printf("start infer...: ");
                multi_Infer_disc(str, corp, nbors, argv[2], argv[1]);
        }
    }
}

void read_settings_infer(char* filename)
{
    FILE* fileptr; char str[2000]; int n=2000;
	const char delimiters[] = " \t";
    fileptr = fopen(filename, "r");
	fgets(str, n, fileptr);
	fgets(str, n, fileptr);	sscanf(str, "%d", &INF_MAX_ITER);
	fgets(str, n, fileptr);	sscanf(str, "%f", &INF_CONVERGED);
	fgets(str, n, fileptr);
	fgets(str, n, fileptr);	sscanf(str, "%d", &UNSUPERVISED);
	fgets(str, n, fileptr);	sscanf(str, "%d", &PRIOR);
	fgets(str, n, fileptr);	sscanf(str, "%lf", &reg);
	fgets(str, n, fileptr);
	fgets(str, n, fileptr);	sscanf(str, "%d", &kNN);
	fgets(str, n, fileptr);	sscanf(str, "%lf", &lambda);
    fclose(fileptr);
}

corpus* read_neighbors(char* filename, corpus *corp)
{
    FILE *fileptr;    int i, j;		corpus *c;

    fileptr = fopen(filename, "rb");
	if (fileptr == 0) 
		{ printf("\n read_neighbors: cannot open '%s'.\n", filename); exit(0); }
	c = malloc(sizeof(corpus));	
	c->docs = (document*) malloc(corp->num_docs * sizeof(document));
	if (c->docs == 0) 
		{ printf("\n read_neighbors: cannot allocate memory.\n"); exit(0); }
	printf("reading neighbors...");	fflush(stdout);
    c->num_terms	= corp->num_terms;
    c->num_docs		= corp->num_docs;	
	c->num_labels	= corp->num_labels;
	
	for (i = 0; i < corp->num_docs; i++)
    {
		fread(&(c->docs[i].length), sizeof(int), 1, fileptr);
		//printf(" %d:  %d \n", i, c->docs[i].length);
		c->docs[i].label  = corp->docs[i].label;
		c->docs[i].total  = 0;
		c->docs[i].entropy = 0;
		c->docs[i].words  = malloc(sizeof(int) * c->docs[i].length);
		c->docs[i].counts = malloc(sizeof(double) * c->docs[i].length);
		if (c->docs[i].words ==0 || c->docs[i].counts ==0) 
			{ printf("\n read_neighbors: cannot allocate memory..\n"); exit(0); }
		fread(c->docs[i].words, sizeof(int), c->docs[i].length, fileptr);
		fread(c->docs[i].counts, sizeof(double), c->docs[i].length, fileptr);
    }
	fclose(fileptr);	printf("\t completed.\n");
return (c);
}

void save_topic_classes(char *model_root, char *model_name)
{//save contribution of topics to each class
	int d, j, i; 	char filename[512];

	sprintf(filename, "%s-sin%d-k%d-ld%1.2f-inf-train-topics-docs-contribute.dat", model_root, (int)reg, kNN, lambda);
	corpus *corp = read_data(filename, UNSUPERVISED);		//read the representation
	double C[corp->num_labels][corp->num_terms];

	for (j = 0; j < corp->num_labels; j++)
		for (d = 0; d < corp->num_terms; d++) C[j][d] = 0;
	for (d = 0; d < corp->num_docs; d++)
		for (j =0; j < corp->docs[d].length; j++) 
			C[corp->docs[d].label][corp->docs[d].words[j]] += corp->docs[d].counts[j];
	for (d = 0; d < corp->num_labels; d++)	L1_normalize(C[d], corp->num_terms);

	//write to file
	sprintf(filename, "%s-sin%d-k%d-ld%1.2f-inf-train-topics-classes.csv", model_root, (int)reg, kNN, lambda);
	FILE *fileptr = fopen(filename, "w");
	fprintf(fileptr, "class");
	for (i = 0; i < corp->num_terms; i++) fprintf(fileptr, ", topic %d", i);
	for (i = 0; i < corp->num_labels; i++)
    {
		fprintf(fileptr, "\n%d", corp->label_names[i]);
		for (j = 0; j < corp->num_terms; j++)
			fprintf(fileptr, ", %1.10f", C[i][j]);
    }
    fclose(fileptr);

	////Unsupervised
	//sprintf(filename, "%s-%s-topics-docs-contribute.dat", model_root, model_name);
	//corp = read_data(filename, UNSUPERVISED);		//read the representation

	//for (j = 0; j < corp->num_labels; j++)
	//	for (d = 0; d < corp->num_terms; d++) C[j][d] = 0;
	//for (d = 0; d < corp->num_docs; d++)
	//	for (j =0; j < corp->docs[d].length; j++) 
	//		C[corp->docs[d].label][corp->docs[d].words[j]] += corp->docs[d].counts[j];
	//for (d = 0; d < corp->num_labels; d++)	L1_normalize(C[d], corp->num_terms);

	////write to file
	//sprintf(filename, "%s-%s-topics-classes.csv", model_root, model_name);
	//fileptr = fopen(filename, "w");
	//fprintf(fileptr, "class");
	//for (i = 0; i < corp->num_terms; i++) fprintf(fileptr, ", topic %d", i);
	//for (i = 0; i < corp->num_labels; i++)
 //   {
	//	fprintf(fileptr, "\n%d", corp->label_names[i]);
	//	for (j = 0; j < corp->num_terms; j++)
	//		fprintf(fileptr, ", %1.10f", C[i][j]);
 //   }
 //   fclose(fileptr);
}
