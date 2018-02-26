CC=g++
CFLAGS=-g -Wall -pedantic -std=c++11

TARGET=transpose

all: $(TARGET)

p1: $(TARGET)
	$(CC) -o $(TARGET) $(CFLAGS) $(TARGET).cc

clean:
	-rm -rf core *.o *~ $(TARGET)
