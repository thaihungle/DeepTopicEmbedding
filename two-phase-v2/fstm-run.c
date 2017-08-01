
#include "fstm/fstm-est-inf.h"

int main(int argc, char* argv[])
{
	corpus *corpus;		char str[1000], filename[1000];

	if (strcmp(argv[1], "est")==0)
	{	// ./fstm-run est <model-folder> <data> <#topics>
		// Eg: ./fstm-run est news train-data.txt 100
		read_settings("fstm-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[3], UNSUPERVISED);
		NTOPICS = atol(argv[4]); 
		printf("\nNumber of topics: %d \n", NTOPICS);
		sprintf(str, "_%s_fstm%d", argv[2], NTOPICS);
		make_directory(str);
		fstm_Learn(str, corpus);
	}
	if (strstr(argv[1], "inf") != NULL)
	{
		// Eg: ./fstm-run inf-test sin1000-k10-ld0.50 news test-data.txt 100
		// Eg: ./fstm-run inf-train fstm news train-data.txt 100
		read_settings("fstm-settings.txt");
		if (UNSUPERVISED == 1 || UNSUPERVISED ==2)
			corpus = read_data(argv[4], UNSUPERVISED);
        printf("start fstm-infering...\n");
        sprintf(str, "_%s_fstm%s/final-%s", argv[3], argv[5], argv[2]);
		sprintf(filename, "_%s_fstm%s/final-%s-%s", argv[3], argv[5], argv[2], argv[1]);
		fstm_Infer(str, filename, corpus);
	}
    return(0);
}

