
#include "lda/lda-estimate.h"
#include <time.h>

int main(int argc, char* argv[])
{
    corpus* corpus;		char str[1000], filename[1000];

	if (strcmp(argv[1], "est")==0)
	{	// ./lda-run est <model-folder> <data> <#topics>
		// Eg: ./lda-run est news train-data.txt 100
		read_settings("lda-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED == 2)
			corpus = read_data(argv[3], UNSUPERVISED);
		INITIAL_ALPHA = 0.1;
		NTOPICS = atol(argv[4]); 
		printf("\nNumber of topics: %d \n", NTOPICS);
		sprintf(str, "_%s_lda%d", argv[2], NTOPICS);
		make_directory(str);
		run_em("random", str, corpus);	
	}
	if (strstr(argv[1], "inf") != NULL)
	{
		// Eg: ./lda-run inf-test sin-1000.0 news test-data.txt 100
		// Eg: ./lda-run inf-train lda news train-data.txt 100
		read_settings("lda-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[4], UNSUPERVISED);
		sprintf(str, "_%s_lda%s/final-%s", argv[3], argv[5], argv[2]);	
		sprintf(filename, "_%s_lda%s/final-%s-%s", argv[3], argv[5], argv[2], argv[1]);
		infer(str, filename, corpus);
	}
    return(0);
}
