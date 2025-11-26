CC = gcc
CFLAGS = -O2 -Wall

all: ecc attacker

ecc: ecc.c attacker.h
	$(CC) ecc.c -o ecc $(CFLAGS) -lcrypto

spy.o: spy.S attacker.h
	$(CC) -c spy.S -o spy.o

attacker: attacker.c attacker.h spy.o
	$(CC) attacker.c spy.o -o attacker $(CFLAGS)

clean:
	rm -f ecc attacker spy.o pipe.fifo timings.bin
	rm -rf raw_timings