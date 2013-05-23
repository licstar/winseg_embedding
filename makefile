CC = g++
GCC = gcc
CFLAGS = -lm -O2 -Wall -funroll-loops -ffast-math
#CFLAGS = -lm -O2 -Wall

all: embedding

embedding : sennaseg_embedding.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) sennaseg_embedding.cpp -fopenmp -DLINUX -o embedding

clean:
	rm -rf *.o embedding
