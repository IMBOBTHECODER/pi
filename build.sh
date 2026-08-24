g++ -O3 -march=native -DNDEBUG chudnovsky.cpp \
    -I/usr/local/include -L/usr/local/lib -Wl,-rpath=/usr/local/lib \
    -lgmp -fopenmp -o main
ldd main | grep gmp   # verify /usr/local/lib