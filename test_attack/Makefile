CC = gcc
CFLAGS = -O0 -Wall

all: victim attacker

victim: victim.c attacker.h
	$(CC) victim.c -o victim $(CFLAGS)

spy.o: spy.S attacker.h
	$(CC) -c spy.S -o spy.o

attacker: attacker.c attacker.h spy.o
	$(CC) attacker.c spy.o -o attacker $(CFLAGS)

clean:
	rm -f victim attacker spy.o pipe.fifo timings.bin
	rm -rf raw_timings
