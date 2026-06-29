# 5bit C Engine — Ground Truth for Python & TypeScript
CC = gcc
CFLAGS = -O2 -Wall

all: codec encode lib

codec: fivebit_codec.c
	$(CC) $(CFLAGS) -o fivebit_codec fivebit_codec.c

encode: fivebit_encode.c
	$(CC) $(CFLAGS) -o fivebit_encode fivebit_encode.c

lib: fivebit_lib.c
	$(CC) $(CFLAGS) -shared -fPIC -o libfivebit.so fivebit_lib.c

clean:
	rm -f fivebit_codec fivebit_encode libfivebit.so
