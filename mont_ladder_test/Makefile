CC = gcc
CFLAGS = -g -Wall -O2 -fno-inline

OPENSSL_ROOT_DIR = /usr/local/ssl
INC = -I$(OPENSSL_ROOT_DIR)/include
LIB = -L$(OPENSSL_ROOT_DIR)/lib -lcrypto -Wl,-rpath=$(OPENSSL_ROOT_DIR)/lib

all: spy mont_ladder

# -----------------------------------------------------
# Spy build (assembly + C wrapper)
# -----------------------------------------------------
spy: spy.c spy.S spy.h
	$(CC) $(CFLAGS) -o $@ spy.c spy.S

# -----------------------------------------------------
# Victim build
# -----------------------------------------------------
mont_ladder: mont_ladder.c
	$(CC) $(CFLAGS) $(INC) $(LIB) -o $@ mont_ladder.c

# -----------------------------------------------------
clean:
	rm -f spy mont_ladder timings.bin *.o *~ 

.PHONY: all clean
