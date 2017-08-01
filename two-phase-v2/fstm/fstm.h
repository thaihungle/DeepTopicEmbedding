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

#ifndef FSTM_H
#define FSTM_H


typedef struct
{
    int* topicID; // topic index
    double* value;
    int length;
} pairs;

typedef struct
{
    pairs *bb;		//topics
    float t_sparse; //percentage of non-zeros in a topic
    int num_topics;
    int num_terms;
} fstm_model;



#endif

