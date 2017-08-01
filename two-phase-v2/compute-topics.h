#ifndef COMPUTE_TOPICS_H
#define COMPUTE_TOPICS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#include "plsa/utils.h"
#include "custom-inf.h"

void save_lda_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file);

void save_plsa_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file);

void save_fstm_topics(plsa_model *model, double ** aa, corpus *corp, char *topic_file, char *other_file, double **C);

void copy_other_file(char *source, char *dist);

#endif

