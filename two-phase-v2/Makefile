
.SUFFIXES: .c .u
CC= gcc
CFLAGS= -O3 -Wall -g
LDFLAGS= -lm

all: plsa-run lda-run fstm-run infer knn blas liblinear

LSOURCE= multiclass-inf.c custom-inf.c obj-functions.c infer.c compute-topics.c plsa/data.c plsa/utils.c plsa/cokus.c plsa/plsa-model.c plsa/plsa-est-inf.c lda/lda-data.c lda/utils.c lda/cokus.c  lda/lda-estimate.c lda/lda-model.c lda/lda-inference.c lda/lda-alpha.c lda-run.c plsa-run.c fstm/data.c fstm/utils.c fstm/fstm-model.c fstm/fstm-est-inf.c fstm-run.c neighbors.c

LOBJECTS1= plsa/data.o plsa/utils.o plsa/cokus.o plsa/plsa-model.o plsa/plsa-est-inf.o plsa-run.o

LOBJECTS2= lda/lda-data.o lda/utils.o lda/cokus.o lda/lda-estimate.o lda/lda-model.o lda/lda-inference.o lda/lda-alpha.o lda-run.o

LOBJECTS4= fstm/data.o fstm/utils.o fstm/fstm-model.o fstm/fstm-est-inf.o fstm-run.o

LOBJECTS3= multiclass-inf.o custom-inf.o obj-functions.o infer.o compute-topics.o plsa/data.o plsa/utils.o plsa/cokus.o plsa/plsa-model.o plsa/plsa-est-inf.o

LOBJECTS5= neighbors.o fstm/data.o

plsa-run:	$(LOBJECTS1)
	$(CC) $(CFLAGS) $(LOBJECTS1) -o plsa-run $(LDFLAGS)

lda-run:	$(LOBJECTS2)
	$(CC) $(CFLAGS) $(LOBJECTS2) -o lda-run $(LDFLAGS)

fstm-run:	$(LOBJECTS4)
	$(CC) $(CFLAGS) $(LOBJECTS4) -o fstm-run $(LDFLAGS)

infer:	$(LOBJECTS3)
	$(CC) $(CFLAGS) $(LOBJECTS3) -o infer $(LDFLAGS)

knn:	$(LOBJECTS5)
	$(CC) $(CFLAGS) $(LOBJECTS5) -o knn $(LDFLAGS)

blas: 
	cd liblinear-1.8/blas; make blas

liblinear:
	cd liblinear-1.8; make train; make predict

clean:
	cd lda; make clean
	cd plsa; make clean
	cd fstm; make clean
	cd liblinear-1.8/blas; make clean
	cd liblinear-1.8; make clean
	-rm -f *.o
	-rm infer
	-rm fstm-run
	-rm plsa-run
	-rm lda-run
	-rm knn
