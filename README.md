# Chudnovsky π

Compute pi to specified precision with Chudnovsky Algorithm. Using optimisation 
such as **binary splitting**, **parallelisation (OpenMP)**, **GMP**, ...

## Installation

You need a C++ compiler with OpenMP (**g++** or **clang++**) and the **GMP**
development headers. On Windows, build under WSL.

**Debian / Ubuntu / WSL**
```sh
sudo apt update
sudo apt install g++ libgmp-dev
# optional: faster multithreaded allocator (see Build)
sudo apt install libgoogle-perftools-dev
```

**Fedora / RHEL**
```sh
sudo dnf install gcc-c++ gmp-devel
```

**Arch**
```sh
sudo pacman -S gcc gmp
```

**macOS** (Homebrew — Apple Clang ships without OpenMP, so use GCC)
```sh
brew install gcc gmp        # then build with g++-14 (or your version), not clang++
```

Check it worked: `g++ --version` and `echo '#include <gmp.h>' | g++ -x c++ -E - >/dev/null`
(no error means the headers are found).

The `chudnovsky.py` prototype is optional. It needs Python 3 and `gmpy2` — see `requirements.txt`:
```sh
pip install -r requirements.txt        # gmpy2 needs libgmp-dev installed first
```

## Build

```sh
g++ -O3 -march=native -flto -DNDEBUG -fopenmp -fno-math-errno \
    chudnovsky.cpp -lgmp -o main
```

- **`-fopenmp` is required** for any parallelism. Without it every `#pragma omp`
  is ignored and the whole program runs single-threaded (correct, just slow).
- `-fno-math-errno` lets the compiler treat `log2` as pure and fold the repeated
  calls in `p_bits`/`q_bits`/`t_bits`.
- Optional: link a faster multithreaded allocator, e.g. `-ltcmalloc_minimal`
  (`apt install libgoogle-perftools-dev`). GMP allocates heavily; on many cores
  the default allocator's lock contention costs more than the math.

## Run

```
./main            # prints the compute time
./main -p         # also writes the digits to a file (prompts for a name)
```

## Design

How the algorithm, parallelism, and preallocation bounds work — and why — is in
**[DESIGN.md](DESIGN.md)**.

## Files

- `chudnovsky.cpp` — the C++/GMP implementation (incremental + binary splitting).
- `chudnovsky.py` — the original Python/gmpy2 prototypes.
- `requirements.txt` — Python dependency (`gmpy2`) for the prototype.
- `DESIGN.md` — algorithm, optimisations, and derivations.
