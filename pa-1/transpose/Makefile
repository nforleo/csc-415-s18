CXX=g++
CXXFLAGS=-g -Wall -pedantic -std=c++11

all: transpose

transpose: transpose.o
	$(CXX) -o $@ $(CXXFLAGS) $^

clean:
	rm -rf transpose core *.o *~
