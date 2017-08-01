
#include "lda-data.h"

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
}

