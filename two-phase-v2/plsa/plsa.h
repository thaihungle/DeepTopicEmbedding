
#ifndef PLSA_H
#define PLSA_H

typedef struct
{
    float t_sparse;//percentage of non-zeros in a topic
    double **bb;		//topics
    int num_topics;
    int num_terms;
	double train_count;
} plsa_model;


#endif

