# (C) Copyright 2012, Khoat Than (khoat [at] jaist [dot] ac [dot] jp)

# This file is part of FSTM-C.

# FSTM-C is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your
# option) any later version.

# DLN-C is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.

# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
# USA


.SUFFIXES: .c .u
CC= gcc
CFLAGS= -O3 -Wall -g
LDFLAGS= -lm

LOBJECTS= data.o utils.o cokus.o plsa-model.o plsa-est-inf.o plsa-run.o

LSOURCE= data.c utils.c cokus.c plsa-model.c plsa-est-inf.c plsa-run.c

plsa:	$(LOBJECTS)
	$(CC) $(CFLAGS) $(LOBJECTS) -o plsa $(LDFLAGS)

clean:
	-rm -f *.o
