
#include "plsa/plsa-est-inf.h"
#include <time.h>

int main(int argc, char* argv[])
{
    corpus *corpus;		char str[1000], filename[1000];

	if (strcmp(argv[1], "est")==0)
	{	// ./plsa-run est <model-folder> <data> <#topics>
		// Eg: ./plsa-run est news train-data.txt 100
		read_settings("plsa-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[3], UNSUPERVISED);
		NTOPICS = atol(argv[4]); 
		printf("\nNumber of topics: %d \n", NTOPICS);
		sprintf(str, "_%s_plsa%d", argv[2], NTOPICS);
		make_directory(str);
		plsa_Learn(str, corpus);
	}
	if (strstr(argv[1], "inf") != NULL)
	{
		// Eg: ./plsa-run inf-test sin-1000.0 news test-data.txt 100
		// Eg: ./plsa-run inf-train plsa news train-data.txt 100
		read_settings("plsa-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[4], UNSUPERVISED);
		sprintf(str, "_%s_plsa%s/final-%s", argv[3], argv[5], argv[2]);	
		sprintf(filename, "_%s_plsa%s/final-%s-%s", argv[3], argv[5], argv[2], argv[1]);
		plsa_Infer(str, filename, corpus);
	}
    return(0);
}
