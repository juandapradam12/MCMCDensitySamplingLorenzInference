CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11
LDFLAGS ?= -lm

# Enable OpenMP when the toolchain provides omp.h; otherwise build serial.
OMPFLAG := $(shell echo 'int main(void){return 0;}' | $(CC) -fopenmp -x c - -o /tmp/mcmc_omp_probe >/dev/null 2>&1 && echo -fopenmp || true)

BIN_DIR := bin
TARGET  := $(BIN_DIR)/metropolis_normal

.PHONY: all clean run

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): metropolis.c metropolis.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OMPFLAG) metropolis.c -o $@ $(LDFLAGS)

run: $(TARGET)
	cd $(BIN_DIR) && ./metropolis_normal 2000 4

clean:
	rm -rf $(BIN_DIR) sample_*.dat