// (C) Copyright 2012, Khoat Than (khoat [at] jaist [dot] ac [dot] jp)

// This file is part of FSTM.

// FSTM-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// FSTM-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "data.h"

corpus* read_data(char* data_filename, int UNSUPERVISED)
{//this will read a file in either Libsvm format or LDA format
    FILE *fileptr;	long bytes;				corpus* c;
    int length, word, n, nd, nw, OFFSET, i;	double count;
    char *token, *str = (char*)malloc(100000);
	const char delimiters[] = " :";

    printf("reading data from %s\n", data_filename);
    c = malloc(sizeof(corpus));		c->label_names = malloc(sizeof(int));
    c->docs = 0;	    c->num_terms = 0;
    c->num_docs = 0;	c->num_labels = 0;

    fileptr = fopen(data_filename, "r");
	if (fileptr == 0) 
		{ printf("\nCannot read the file '%s'.\n", data_filename); exit(0); }
	if (UNSUPERVISED == 1)	{ OFFSET = 0; printf("unsupervised data.\n"); }
	else					{ OFFSET = 1; printf("multi-class data.\n"); }
    nd = 0; nw = 0;
    while (1)
    {
		length = 100000;	
		bytes = getline(&str, &length, fileptr);
		if (bytes < 1) break; //end of file
		if (bytes < 5) continue;
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		if (c->docs == 0) 
			{ printf("\nCannot allocate memory for Corpus.\n"); exit(0); }
		length = 0;
		for (n = 0; n < strlen(str); n++)	if (str[n] == ':') length++;
		token = strtok(str, delimiters);
		if (UNSUPERVISED != 1)	
		{			//consider to add labels
			word = atoi(token);	n = 0;
			if (c->num_labels > 0)	{	
				for (i =0; i < c->num_labels; i++)
					if (c->label_names[i] == word) {n = i+1; break;}
				if (n < 1) //do not exist
				{
					n = c->num_labels + 1;
					c->label_names = (int*) realloc(c->label_names, sizeof(int)*n);
					c->label_names[n-1] = word;
					c->num_labels = n;
				}
				n--;
			}
			else { c->num_labels = 1;	c->label_names[0] = word; }
			c->docs[nd].label = n;
		}
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].entropy = 0;
		c->docs[nd].words  = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(double)*length);
		if (c->docs[nd].words ==0 || c->docs[nd].counts ==0) 
			{ printf("\nCannot allocate memory for Corpus.\n"); exit(0); }
		for (n = 0; n < length; n++)
		{
			token = strtok(NULL, delimiters);	word  = atoi(token);
			token = strtok(NULL, delimiters);	count = atof(token);
			if (word <0 || count <0) 
				{ printf("\nInput error at line %d.\n", nd+1); exit(0); }
			word = word - OFFSET;
			c->docs[nd].words[n] = word;
			c->docs[nd].counts[n] = count;
			c->docs[nd].total += count;
			if (word >= nw) { nw = word + 1; }
		}
		nd++;		//printf("number of docs    : %d\n", nd); fflush(stdout);
    }
    fclose(fileptr);
    c->num_docs = nd;
    c->num_terms = nw;
    printf("number of docs    : %d\n", nd);
    printf("number of terms   : %d\n", nw);
	if (UNSUPERVISED != 1)
		printf("number of labels   : %d\n", c->num_labels);
    return(c);
}

corpus* read_multilabel_data(char* data_filename, char label)
{//this will read a file in matrix format, separated by comma
	//instances and labels are saved in different files: "data.dat", "data.label"
    FILE *fileptr;	long bytes;			corpus* c;
    int length, n, nd, nw, i;			double count, *tmp;
    char *token, *str = (char*)malloc(50000);
	const char delimiters[] = ", ";

	printf("multi-label data.\n"); 
	//Read data
	sprintf(str, "%s", data_filename);
    printf("reading data from %s\n", str);
    c = malloc(sizeof(corpus));
    c->docs = 0;    c->num_terms = 0;    c->num_docs = 0;
    fileptr = fopen(str, "r");
	if (fileptr == 0) 
		{ printf("\nCannot read the file '%s'.\n", str); exit(0); }
	//read initial line to know how many columns
	nw = 50000;		bytes = getline(&str, &nw, fileptr);
	for (nw = bytes-1; nw >0; nw--)
		if (str[nw] >= '0' && str[nw] <= '9') break;
	str[nw+1] = '\0';
	nw = 1;			token = strtok(str, delimiters);
	while ((token = strtok(NULL, delimiters)) != NULL) nw++;
	if (bytes < 1 || nw < 2) 
		{ printf("\nWrong format in '%s'.\n", str); exit(0); }
	tmp = malloc(sizeof(double)*nw);
    nd = 0;		rewind(fileptr);
    while (!feof(fileptr))
    {
		length = 0;
		for (n = 0; n < nw; n++){
			fscanf(fileptr, "%lf,", &count); tmp[n] = count;
			if (count > 0) length++;
		}
		if (length < 1) break;
		c->docs = (document*) realloc(c->docs, sizeof(document)*(nd+1));
		if (c->docs == 0) 
			{ printf("\nCannot allocate memory for Corpus.\n"); exit(0); }
		c->docs[nd].length = length;
		c->docs[nd].total = 0;
		c->docs[nd].entropy = 0;
		c->docs[nd].words  = malloc(sizeof(int)*length);
		c->docs[nd].counts = malloc(sizeof(double)*length);
		if (c->docs[nd].words ==0 || c->docs[nd].counts ==0) 
			{ printf("\nCannot allocate memory for Corpus.\n"); exit(0); }
		length = 0;
		for (n = 0; n < nw; n++)
			if (tmp[n] > 0)
			{
				c->docs[nd].words[length] = n;
				c->docs[nd].counts[length] = tmp[n];
				c->docs[nd].total += tmp[n];
				length++;
			}
		nd++;		//printf("number of docs    : %d\n", nd); fflush(stdout);
    }
    fclose(fileptr);
    c->num_docs = nd;
    c->num_terms = nw;
    printf("number of instances: %d\n", nd);
    printf("number of features : %d\n", nw);

	if (label != 1) goto read_multilabel_data_end;
	//Read labels
	sprintf(str, "%s.label", data_filename);
	fileptr = fopen(str, "r");
	if (fileptr == 0) 
		{ printf("\nCannot read the file '%s'.\n", str); exit(0); }
	nw = 50000;		bytes = getline(&str, &nw, fileptr);
	nw = 1;			token = strtok(str, delimiters);
	while ((token = strtok(NULL, delimiters)) != NULL) nw++;
	c->num_labels = nw;
	c->labels = malloc(sizeof(char*)*nd);
	for (n = 0; n < nd; n++)
		c->labels[n] = malloc(sizeof(char)*nw);
	rewind(fileptr);
	for (n = 0; n < nd; n++)
		for (i = 0; i < nw; i++) 
			{ fscanf(fileptr, "%d,", &length);	c->labels[n][i] = (char)length; }
	fclose(fileptr);
    printf("number of labels : %d\n", nw);

read_multilabel_data_end:
	free(tmp); free(str);
    return(c);
}

int max_corpus_length(corpus* c)
{
    int n, max = 0;
    for (n = 0; n < c->num_docs; n++)
	if (c->docs[n].length > max) max = c->docs[n].length;
    return(max);
}

void free_corpus(corpus *corp)
{
	int n;
    for (n = corp->num_docs -1; n>0;  n--) 
		{	free(corp->docs[n].words); free(corp->docs[n].counts);	}
	if (corp->labels != NULL)
		for (n = corp->num_docs -1; n>0;  n--) free(corp->labels[n]);
	if (corp->label_names != NULL) free(corp->label_names);
	free(corp->docs);  free(corp);
}

void L1_normalize_document(document *doc)
{	int i;
	for (i =0; i < doc->length; i++) doc->counts[i] /= doc->total;
	doc->total = 1;
}

corpus* new_corpus(int num_docs, int num_terms)
{
    corpus* c;	int i;
    c = malloc(sizeof(corpus));
	c->num_terms = num_terms;
    c->num_docs = num_docs;
	c->docs = (document*) malloc(sizeof(document)*num_docs);
	if (c->docs == NULL) 
	{ printf("\n new_corpus: Cannot allocate memory for new data representation !\n"); exit(0); }
	for (i = 0; i < num_docs; i++)
	{
		c->docs[i].words = (int*)malloc(sizeof(int));
		c->docs[i].counts = (double*)malloc(sizeof(double));
		c->docs[i].length = 0;
	}
    return(c);
}
