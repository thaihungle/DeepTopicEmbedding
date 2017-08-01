// (C) Copyright 2012, Khoat Than (khoat [at] jaist [dot] ac [dot] jp)

// This file is part of FSTM.

// FSTM is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free
// Software Foundation; either version 2 of the License, or (at your
// option) any later version.

// FSTM is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA

#include "fstm-est-inf.h"

void main(int argc, char* argv[])
{
    corpus *corpus;	double perp;
	char str[1000], save[1000];

	if (strcmp(argv[1], "est")==0)
        {	// ./fstm est <model-folder> <train-data> <topics>
		// Eg: ./fstm est ap10 ap-train.txt 10	
		read_settings("fstm-settings.txt");
		corpus = read_data(argv[3], UNSUPERVISED);
		NTOPICS = atol(argv[4]); 
		printf("\nNumber of topics: %d \n", NTOPICS);
		make_directory(argv[2]);
		fstm_Learn(argv[2], corpus);
	}
        if (strcmp(argv[1], "inf")==0)
        {// ./fstm inf <model-folder> <test-data>
		// Eg: ./fstm inf ap10 ap-test.txt
		read_settings("fstm-settings.txt");
		corpus = read_data(argv[3], UNSUPERVISED);  //where to infer
		sprintf(str, "%s/final-fstm", argv[2]);
		sprintf(save, "%s/final-fstm-inf", argv[2]);
		perp = fstm_Infer(str, save, corpus);
		printf("\n Perplexity = %10.10f \n", perp);
	}
}

