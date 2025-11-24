CC = gcc
CFLAGS = -O2 -Wall

all: attacker ecc

attacker: attacker.c attacker.h
	$(CC) attacker.c -o attacker $(CFLAGS)

ecc: ecc.c attacker.h
	$(CC) ecc.c -o ecc $(CFLAGS) -lcrypto

clean:
	rm -f attacker ecc pipe.fifo
