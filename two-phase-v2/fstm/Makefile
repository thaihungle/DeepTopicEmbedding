
.SUFFIXES: .c .u
CC= gcc
CFLAGS= -O3 -Wall -g
LDFLAGS= -lm

LOBJECTS= data.o utils.o fstm-model.o fstm-est-inf.o fstm-run.o

LSOURCE= data.c utils.c fstm-model.c fstm-est-inf.c fstm-run.c

fstm:	$(LOBJECTS)
	$(CC) $(CFLAGS) $(LOBJECTS) -o fstm $(LDFLAGS)

clean:
	-rm -f *.o
