CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra -std=c11 -fPIC
LDFLAGS ?= -lm

# Enable OpenMP when the toolchain provides it; otherwise build serial.
OMPFLAG := $(shell echo 'int main(void){return 0;}' | $(CC) -fopenmp -x c - -o /tmp/mcmc_omp_probe >/dev/null 2>&1 && echo -fopenmp || true)

BIN_DIR := bin
TARGET  := $(BIN_DIR)/metropolis_normal
LIB     := $(BIN_DIR)/libmetropolis.so

.PHONY: all clean run lib

all: $(TARGET) $(LIB)

lib: $(LIB)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(LIB): metropolis.c metropolis.h | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OMPFLAG) -shared metropolis.c -o $@ $(LDFLAGS)

$(TARGET): main.c $(LIB) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(OMPFLAG) main.c -L$(BIN_DIR) -lmetropolis -Wl,-rpath,'$$ORIGIN' -o $@ $(LDFLAGS)

run: $(TARGET)
	cd $(BIN_DIR) && ./metropolis_normal 2000 4

clean:
	rm -rf $(BIN_DIR) sample_*.dat
