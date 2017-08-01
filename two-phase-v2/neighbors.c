
#include "fstm/data.h"
#include "wirth.c"
#include <time.h>
#include <stdlib.h>

corpus *multiclass_find_neighbors(corpus *corp, int knn, double lambda);
corpus *multiclass_find_random_neighbors(corpus *corp, int knn, double lambda);
void save_neighbors(char* filename, corpus *corp);

void main(int argc, char* argv[])
{
    corpus *corp, *nbors;	int i, d;
    srand (time(NULL));
    rand();
    rand();
    rand();
	//./knn [data] [save-file] [number-of-neighbors] [data-type]
	// Eg: ./knn data.txt knn10-data.txt 10 2
	i = atol(argv[4]); 
	if (i < 3)	corp = read_data(argv[1], i); 
	else		corp = read_multilabel_data(argv[1], 1); 

//    if (i < 3)	nbors = multiclass_find_random_neighbors(corp, atol(argv[3]), 0);
    if (i < 3)	nbors = multiclass_find_neighbors(corp, atol(argv[3]), 0);
	save_neighbors(argv[2], nbors);
}

corpus *multiclass_find_neighbors(corpus *corp, int knn, double lambda)
{//construct nearest-neighbor graph and then for each point compute the mean of its neighbors
    int n, nd, nc, i, ndocs, num, index, *ind;	
	double count, reg, *tmp, *doc;	corpus *c;

    printf("\t finding nearest neighbors: ");  fflush(stdout);
    c = malloc(sizeof(corpus));	
	c->docs = (document*) malloc(corp->num_docs * sizeof(document));
	tmp = (double*) malloc(corp->num_docs * sizeof(double));
	doc = (double*) malloc(corp->num_terms * sizeof(double));
	ind = (int*) malloc(corp->num_docs * sizeof(int));
	if (c->docs == 0 || tmp == 0 || doc ==0) 
		{ printf("\n find_neighbors: cannot allocate memory.\n"); exit(0); }
    c->num_terms	= corp->num_terms;
    c->num_docs		= corp->num_docs;	
	c->num_labels	= corp->num_labels;
	c->label_names	= corp->label_names;
	c->labels		= corp->labels;
	//working on each class 
	for (nc = 0; nc < corp->num_labels; nc++)
	{
		printf("%d, ", nc);  fflush(stdout);
		ndocs = 0;
		for(nd = 0; nd < corp->num_docs; nd++)
			if (corp->docs[nd].label == nc) {ind[ndocs] = nd;	ndocs++;}
		for(nd = 0; nd < ndocs; nd++)
		{
			for(i = 0; i < ndocs; i++) tmp[i] = 0;
			for (n = 0; n < corp->num_terms; n++) doc[n] = (1e-10);	
			for (n = 0; n < corp->docs[ind[nd]].length; n++) 
				doc[corp->docs[ind[nd]].words[n]] = corp->docs[ind[nd]].counts[n];
			//compute KL divergence: KL(i | nd)
            for(i = 0; i < ndocs; i++){
				if (i != nd)
					for (n = 0; n < corp->docs[ind[i]].length; n++) 
						tmp[i] -= corp->docs[ind[i]].counts[n] * log(doc[corp->docs[ind[i]].words[n]]);
            }
			num = knn;	if (knn > ndocs || knn < 0) num = ndocs;
			reg = (1-lambda)/num;
			//find neighbors
            for (n = 0; n < corp->num_terms; n++) doc[n] = 0;
			if (lambda > 0)
				for (n = 0; n < corp->docs[ind[nd]].length; n++) 
					doc[corp->docs[ind[nd]].words[n]] = lambda * corp->docs[ind[nd]].counts[n];
			for(i = 0; i < num; i++)
			{
				index = kth_smallest_index(tmp, ndocs, i);
                printf("%d, ", index);
				for (n = 0; n < corp->docs[ind[index]].length; n++) 
					doc[corp->docs[ind[index]].words[n]] += corp->docs[ind[index]].counts[n] * reg;
			}
			num = 0;
			for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > 0) num++;
			//new document representation
			c->docs[ind[nd]].length = num;
			c->docs[ind[nd]].label  = corp->docs[ind[nd]].label;
			c->docs[ind[nd]].total  = 0;
			c->docs[ind[nd]].entropy = 0;
			c->docs[ind[nd]].words  = malloc(sizeof(int)*num);
			c->docs[ind[nd]].counts = malloc(sizeof(double)*num);
			if (c->docs[ind[nd]].words ==0 || c->docs[ind[nd]].counts ==0) 
				{ printf("\n find_neighbors: cannot allocate memory..\n"); exit(0); }
			num = 0;
			for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > 0) 
				{
					c->docs[ind[nd]].words[num]	 = n;
					c->docs[ind[nd]].counts[num] = doc[n];
					c->docs[ind[nd]].total		+= doc[n];
					num++;
				}
		}
	}
    free(tmp); free(doc); free(ind);
	printf("\t completed.\n");
return(c);
}

corpus *multiclass_find_random_neighbors(corpus *corp, int knn, double lambda)
{//construct nearest-neighbor graph and then for each point compute the mean of its neighbors
    int n, nd, nc, i, j, ndocs, num, index, *ind, besti;
    double reg, *doc, bestv, temp;
    corpus *c;

    printf("\t finding random neighbors: %f",lambda);  fflush(stdout);
    c = malloc(sizeof(corpus));
    c->docs = (document*) malloc(corp->num_docs * sizeof(document));
    doc = (double*) malloc(corp->num_terms * sizeof(double));
    ind = (int*) malloc(corp->num_docs * sizeof(int));
    if (c->docs == 0 || doc ==0)
        { printf("\n find_neighbors: cannot allocate memory.\n"); exit(0); }
    c->num_terms	= corp->num_terms;
    c->num_docs		= corp->num_docs;
    c->num_labels	= corp->num_labels;
    c->label_names	= corp->label_names;
    c->labels		= corp->labels;
    //working on each class
    for (nc = 0; nc < corp->num_labels; nc++)
    {
        printf("%d, ", nc);  fflush(stdout);
        ndocs = 0;
        for(nd = 0; nd < corp->num_docs; nd++)
            if (corp->docs[nd].label == nc) {ind[ndocs] = nd;	ndocs++;}
        //ind and ndocs refer to documents in this class nc
        for(nd = 0; nd < ndocs; nd++)
        {
            //nd is index of current document with class nc
            num = knn;	if (knn > ndocs || knn < 0) num = ndocs;
            reg = (1-lambda)/num;
            //accumulate neighbors
            for (n = 0; n < corp->num_terms; n++) doc[n] =  (1e-10);
            for (n = 0; n < corp->docs[ind[nd]].length; n++)
                doc[corp->docs[ind[nd]].words[n]] =  corp->docs[ind[nd]].counts[n];//current doc
            for(i = 0; i < num; i++)
            {
                besti=rand()%ndocs;
//                bestv=1000000000000000000000.0;
//                for (j=0;j < 10;j++){
//                    index = rand()%ndocs;
//                    temp=0;
//                    if (index != nd){
//                        for (n = 0; n < corp->docs[ind[index]].length; n++)
//                                temp -= corp->docs[ind[index]].counts[n] * log(doc[corp->docs[ind[index]].words[n]]);
////                        printf(" asdsd %f ",temp);
//                    }
//                    if (temp<bestv){
//                        besti=index;
//                        bestv=temp;
//                    }
//                }
                index=besti;
//                printf("\n %d",index);
//                printf(" %f",bestv);
                for (n = 0; n < corp->docs[ind[index]].length; n++)
                    doc[corp->docs[ind[index]].words[n]] += corp->docs[ind[index]].counts[n] * reg;//neighbor doc
            }
            for (n = 0; n < corp->docs[ind[nd]].length; n++)
                doc[corp->docs[ind[nd]].words[n]] -= corp->docs[ind[nd]].counts[n];//current doc
            num = 0;
            for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > (1e-10)) num++;
            //new document representation
            c->docs[ind[nd]].length = num;
            c->docs[ind[nd]].label  = corp->docs[ind[nd]].label;
            c->docs[ind[nd]].total  = 0;
            c->docs[ind[nd]].entropy = 0;
            c->docs[ind[nd]].words  = malloc(sizeof(int)*num);
            c->docs[ind[nd]].counts = malloc(sizeof(double)*num);
            if (c->docs[ind[nd]].words ==0 || c->docs[ind[nd]].counts ==0)
                { printf("\n find_neighbors: cannot allocate memory..\n"); exit(0); }
            num = 0;
            for (n = 0; n < corp->num_terms; n++) 	if (doc[n] > (1e-10))
                {
                    c->docs[ind[nd]].words[num]	 = n;
                    c->docs[ind[nd]].counts[num] = doc[n];
                    c->docs[ind[nd]].total		+= doc[n];
                    num++;
                }
        }
    }
    free(doc); free(ind);
    printf("\t completed.\n");
return(c);
}

void save_neighbors(char* filename, corpus *corp)
{
    FILE *fileptr;    int i;

    fileptr = fopen(filename, "wb");
	for (i = 0; i < corp->num_docs; i++)
    {
		fwrite(&(corp->docs[i].length), sizeof(int), 1, fileptr);
		fwrite(corp->docs[i].words, sizeof(int), corp->docs[i].length, fileptr);
		fwrite(corp->docs[i].counts, sizeof(double), corp->docs[i].length, fileptr);
    }
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
    c->num_terms = corp->num_terms;
    c->num_docs = corp->num_docs;	
	c->num_labels = corp->num_labels;
	
	for (i = 0; i < corp->num_docs; i++)
    {
		fread(&(corp->docs[i].length), sizeof(int), 1, fileptr);
		c->docs[i].label  = corp->docs[i].label;
		c->docs[i].total  = 0;
		c->docs[i].entropy = 0;
		c->docs[i].words  = malloc(sizeof(int) * corp->docs[i].length);
		c->docs[i].counts = malloc(sizeof(double) * corp->docs[i].length);
		if (c->docs[i].words ==0 || c->docs[i].counts ==0) 
			{ printf("\n read_neighbors: cannot allocate memory..\n"); exit(0); }
		fread(c->docs[i].words, sizeof(int), corp->docs[i].length, fileptr);
		fread(c->docs[i].counts, sizeof(double), corp->docs[i].length, fileptr);
    }
	fclose(fileptr);
return (c);
}
