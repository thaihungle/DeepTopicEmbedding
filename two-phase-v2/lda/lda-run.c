// (C) Copyright 2004, David M. Blei (blei [at] cs [dot] cmu [dot] edu)

// This file is part of LDA-C.

// LDA-C is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// LDA-C is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "lda-estimate.h"

int main(int argc, char* argv[])
{
    // (est / inf) alpha k settings data (random / seed/ model) (directory / out)

    corpus* corpus;

    int i;	double perp;
	FILE *fperp;	char str[1000];

	if (strcmp(argv[1], "est")==0)
	{	// Eg: ./lda est nip train-data.txt 1 10
		//name of the corpus: argv[3]
		read_settings("settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED == 2)
			corpus = read_data(argv[3], UNSUPERVISED);
		else if (UNSUPERVISED == 3)
			corpus = read_multilabel_data(argv[3], 1);
		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
		{
			INITIAL_ALPHA = 0.1;
			NTOPICS = 10*i; 
			printf("\nNumber of topics: %d \n", NTOPICS);
			sprintf(str, "_%s%d", argv[2], NTOPICS);
			make_directory(str);
			run_em("random", str, corpus);	
		}	
	}
	if (strcmp(argv[1], "inf")==0)
	{
		//input the first 3 words of name of corpus, and the new documents needed to infer
		// Eg: ./lda inf Nip infer-data.txt 1 10
		sprintf(str, "%s_%s_lda.per", argv[3], argv[2]);
		fperp =fopen(str, "w");
		fprintf(fperp, "NTopics, perplexity \n");
		read_settings("inf-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED == 2)
			corpus = read_data(argv[3], UNSUPERVISED);
		else if (UNSUPERVISED == 3)
			corpus = read_multilabel_data(argv[3], 1);  //where to infer
		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
		{
			sprintf(str, "_%s%d/final-lda", argv[2], 10*i);			
			perp = infer(str, str, corpus);
			fprintf(fperp, "%d, %10.10f \n", 10*i, perp);			
		}
		fclose(fperp);
	}
    return(0);
}

