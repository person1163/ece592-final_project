CC = gcc
OPT ?= 3
CFLAGS = -O$(OPT) -Wall -march=native

all: victim attacker

victim: victim.c
	$(CC) victim.c -o victim $(CFLAGS)

attacker: attacker.c
	$(CC) attacker.c -o attacker $(CFLAGS)
	objdump -d attacker > assembly_dump.txt

clean:
	rm -f victim attacker timings*.bin assembly_dump.txt perf_0.txt perf_1.txt
