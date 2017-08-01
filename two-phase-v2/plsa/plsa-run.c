
#include "plsa-est-inf.h"

int main(int argc, char* argv[])
{
    corpus *corpus;			FILE *fperp;
    int i;	double perp;	char str[1000];

	if (strcmp(argv[1], "est")==0)
	{	// Eg: ./plsa est Nip train-data.txt 1 10
		//name of the corpus: argv[3]	
		read_settings("settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[3], UNSUPERVISED);
		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
		{	
			NTOPICS = 10*i; 
			printf("\nNumber of topics: %d \n", NTOPICS);
			sprintf(str, "_%s%d", argv[2], NTOPICS);
			make_directory(str);
			plsa_Learn(str, corpus);	
		}
	}
	if (strcmp(argv[1], "inf")==0)
	{
		//input the first 3 words of name of corpus, and the new documents needed to infer
		// Eg: ./plsa inf Nip infer-data.txt 1 10
		sprintf(str, "%s_%s_plsa.per", argv[3], argv[2]);
		fperp =fopen(str, "w");
		fprintf(fperp, "NTopics, perplexity \n");
		read_settings("inf-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[3], UNSUPERVISED);
		for (i=atol(argv[4]); i<=atol(argv[5]); i++)
		{	
			sprintf(str, "_%s%d/final-plsa", argv[2], 10*i);	
			perp = plsa_Infer(str, str, corpus);
			fprintf(fperp, "%d, %10.10f \n", 10*i, perp);			
		}
		fclose(fperp);		
	}
    return(0);
}

