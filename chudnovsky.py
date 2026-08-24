# Copyright (C) 2026  Pham Tien Dat

import gmpy2
import time
import math

mpfr_ = gmpy2.mpfr
mpz_ = gmpy2.mpz
fact = gmpy2.fac
div = gmpy2.div
sqrt = gmpy2.sqrt

# 1st design (Python, Only GMP)
def chudnovsky(n: int):
    """
    A function that uses the Chudnovsky algorithm to compute Pi

    n: number of digits
    """

    # 3.3219 bits per decimal place (64 guard bits)
    gmpy2.get_context().precision = int(n * 3.3219) + 64

    term = math.ceil(n / 14)

    pi = mpfr_(13591409) / sqrt(262537412640768000)

    sq640 = sqrt(640320)
    c640 = mpz_(640320)

    for k in range(1, term + 1):
        numerator = mpz_(1 - 2*(k & 1))
        numerator *= fact(6 * k)
        numerator *= (545140134*k + 13591409)

        denominator = fact(3*k)
        denominator *= fact(k) ** 3
        denominator *= c640 ** (3*k + 1) * sq640

        pi += div(numerator, denominator)

    return div(1, 12 * pi)

# 2nd design (PYthon, Incremental + GMP)
def incremental_chudnovsky(n: int):
    """
    A function that uses the Chudnovsky algorithm to compute Pi

    This function uses the difference between terms to compute each consecutive terms

    n: number of digits
    """

    # 3.3219 bits per decimal place (64 guard bits)
    gmpy2.get_context().precision = int(n * 3.3219) + 64
    term = math.ceil(n / 14)

    AkB = 13591409

    # Track the term value itself: t_k = t_{k-1} * num_k / den_k.
    # num_k and den_k are small exact ints, so each iteration only costs
    # big*small and big/small — no full-precision division.
    t = div(13591409, sqrt(262537412640768000))
    pi = t

    for k in range(1, term + 1):
        K6 = 6*k
        K3 = K6 >> 1

        # denominator factor uses the previous AkB
        den_k = K3*(K3 - 1)*(K3 - 2) * k*k*k * 262537412640768000 * AkB

        AkB += 545140134
        num_k = K6*(K6 - 1)*(K6 - 2)*(K6 - 3)*(K6 - 4)*(K6 - 5) * AkB

        t = div(-t * num_k, den_k)

        pi += t

    return div(1, 12 * pi)

# 3rd design (Python, Binary Splitting + GMP)
def bs_chudnovsky(n: int):
    """
    A function that uses the Chudnovsky algorithm with binary splitting to compute Pi

    A range of terms is kept as one exact fraction via a triple (P, Q, T); ranges
    merge with integer arithmetic only, so there is a single division at the end
    instead of one per term. This is the simple, serial reference — the optimised,
    parallel version lives in chudnovsky.cpp (see DESIGN.md).

    n: number of digits
    """
    gmpy2.get_context().precision = int(n * 3.3219) + 64
    terms = math.ceil(n / 14) + 1

    C3 = 262537412640768000  # 640320^3

    def binary_split(a, b):
        # (P, Q, T) for the term range [a, b)
        if b - a == 1:
            k = a
            if k == 0:
                P = mpz_(1)
                Q = mpz_(1)
            else:
                P = mpz_(-(6*k - 5)*(6*k - 4)*(6*k - 3)*(6*k - 2)*(6*k - 1)*(6*k))
                Q = mpz_((3*k - 2)*(3*k - 1)*(3*k) * k*k*k * C3)
            T = P * (545140134*k + 13591409)
            return P, Q, T

        m = (a + b) // 2
        lP, lQ, lT = binary_split(a, m)
        rP, rQ, rT = binary_split(m, b)
        return lP*rP, lQ*rQ, lT*rQ + lP*rT

    P, Q, T = binary_split(0, terms)

    # pi = 426880 * sqrt(10005) * Q / T
    return div(426880 * sqrt(mpfr_(10005)) * Q, T)

if __name__ == "__main__":
    x = int(input())
    
    # start = time.perf_counter()
    # chudnovsky(x)
    # print(f"Time taken: {time.perf_counter() - start:.8f}s")

    start = time.perf_counter()
    incremental_chudnovsky(x)
    print(f"Time taken: {time.perf_counter() - start:.8f}s")

# 2nd design (Incremental + GMP)