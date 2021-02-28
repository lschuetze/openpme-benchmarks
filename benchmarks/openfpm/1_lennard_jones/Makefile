include ../../example.mk

CC=mpic++

LDIR =

OBJ = main.o
OBJ_VL = main_vl.o

all: md_dyn md_dyn_vl

%.o: %.cpp
	$(CC) -O3 -g -c --std=c++14 -o $@ $< $(INCLUDE_PATH)

md_dyn: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS_PATH) $(LIBS)

md_dyn_vl: $(OBJ_VL)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS_PATH) $(LIBS)

run: all
	mpirun -np 3 ./md_dyn && mpirun -np 3 ./md_dyn_vl;

.PHONY: clean all run

clean:
	rm -f *.o *~ core md_dyn md_dyn_vl particles_*
	rm -f ../*~ core md_dyn md_dyn_vl ../particles_*

