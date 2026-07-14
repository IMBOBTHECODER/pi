#include <gmp.h>
#include <chrono>
#include <cstdio>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// 640320^3, shared by the binary-splitting leaf (bs) and setup (bs_chudnovsky).
static mpz_t C3;

struct Triple {
    mpz_t P, Q, T;
    Triple()  { mpz_inits(P, Q, T, NULL); }
    ~Triple() { mpz_clears(P, Q, T, NULL); }
    // prevent accidental copies (they'd double-clear)
    Triple(const Triple&) = delete;
    Triple& operator=(const Triple&) = delete;
};


// 3nd design (C++, Incremental + GMP)
//
// Uses the ratio between consecutive terms to compute each term from the previous one
//
// `result` must already be initialized with enough precision (see main).   
// n: number of digits
void incremental_chudnovsky(mpf_t result, unsigned long n)
{
    mp_bitcnt_t prec = mpf_get_prec(result);
    unsigned long term = (n + 13) / 14; // ceil(n / 14)


    mpz_t AkB;
    mpz_init_set_ui(AkB, 13591409);

    // Per-iteration integer factors stay small (~60 digits), so build them
    // exactly in mpz and apply them to the big floats with one mpf_mul each.
    mpz_t num_k, den_k;
    mpz_init(num_k);
    mpz_init(den_k);

    // Track the term value itself instead of a separate numerator and
    // denominator: t_k = t_{k-1} * num_k / den_k. Both num_k and den_k are
    // small, so each iteration is O(precision) — no big/big division.
    mpf_t t, pi, tmp;
    mpf_init2(t, prec);
    mpf_init2(pi, prec);
    mpf_init2(tmp, prec);

    // t = 13591409 / sqrt(262537412640768000)
    mpf_set_z(tmp, C3);
    mpf_sqrt(tmp, tmp);
    mpf_set_ui(t, 13591409);
    mpf_div(t, t, tmp);

    mpf_set(pi, t);

    for (unsigned long k = 1; k <= term; ++k) {
        unsigned long K6 = 6 * k;
        unsigned long K3 = 3 * k;

        // denominator factor uses the previous AkB (AkB - 545140134 in Python)
        mpz_set_ui(den_k, K3);
        mpz_mul_ui(den_k, den_k, K3 - 1);
        mpz_mul_ui(den_k, den_k, K3 - 2);
        mpz_mul_ui(den_k, den_k, k);
        mpz_mul_ui(den_k, den_k, k);
        mpz_mul_ui(den_k, den_k, k);
        mpz_mul(den_k, den_k, C3);
        mpz_mul(den_k, den_k, AkB);

        mpz_add_ui(AkB, AkB, 545140134);

        mpz_set_ui(num_k, K6);
        mpz_mul_ui(num_k, num_k, K6 - 1);
        mpz_mul_ui(num_k, num_k, K6 - 2);
        mpz_mul_ui(num_k, num_k, K6 - 3);
        mpz_mul_ui(num_k, num_k, K6 - 4);
        mpz_mul_ui(num_k, num_k, K6 - 5);
        mpz_mul(num_k, num_k, AkB);

        mpf_neg(t, t);
        mpf_set_z(tmp, num_k);
        mpf_mul(t, t, tmp);
        mpf_set_z(tmp, den_k);
        mpf_div(t, t, tmp);

        mpf_add(pi, pi, t);
    }

    // result = 1 / (12 * pi)
    mpf_mul_ui(pi, pi, 12);
    mpf_ui_div(result, 1, pi);

    mpz_clear(AkB);
    mpz_clear(num_k);
    mpz_clear(den_k);
    mpf_clear(t);
    mpf_clear(pi);
    mpf_clear(tmp);
}

// Subtrees with fewer than this many terms run serially: task-spawn overhead
// isn't worth it for small leaves. Fixed floor — the depth cap is the primary
// granularity control.
static const unsigned long parallel_cutoff = 1024;

// depth: how many more levels to keep spawning parallel tasks.
// Below depth 0 the recursion runs serially (task overhead not worth it).
void bs(unsigned long a, unsigned long b, Triple& out, int depth) {
    if (b - a == 1) {
        // single-term interval [a, a+1): a is the term index k

        if (a == 0) {
            mpz_set_ui(out.P, 1);
            mpz_set_ui(out.Q, 1);
        } else {
            // P = -(6a-5)(6a-4)(6a-3)(6a-2)(6a-1)(6a)
            unsigned long K = 6*a;
            unsigned long K3 = K/2;

            mpz_set_ui(out.P, K - 5);
            mpz_mul_ui(out.P, out.P, K - 4);
            mpz_mul_ui(out.P, out.P, K - 3);
            mpz_mul_ui(out.P, out.P, K - 2);
            mpz_mul_ui(out.P, out.P, K - 1);
            mpz_mul_ui(out.P, out.P, K);
            mpz_neg(out.P, out.P);

            // Q = (3a-2)(3a-1)(3a) * a^3 * C3
            mpz_set_ui(out.Q, K3 - 2);
            mpz_mul_ui(out.Q, out.Q, K3 - 1);
            mpz_mul_ui(out.Q, out.Q, K3);
            mpz_mul_ui(out.Q, out.Q, a);
            mpz_mul_ui(out.Q, out.Q, a);
            mpz_mul_ui(out.Q, out.Q, a);
            mpz_mul(out.Q, out.Q, C3);
        }

        // a(k) = 545140134*a + 13591409  (fits in <96 bits; preallocate so the
        // mul/add below never reallocate)
        mpz_t ak;
        mpz_init2(ak, 96);
        mpz_set_ui(ak, 545140134UL);
        mpz_mul_ui(ak, ak, a);
        mpz_add_ui(ak, ak, 13591409UL);

        // T = a(k) * P
        mpz_mul(out.T, ak, out.P);

        mpz_clear(ak);
        return;
    }

    unsigned long m = (a + b) / 2;
    Triple L, R;                       // ctor mpz_init's P,Q,T; dtor mpz_clear's

    if (depth > 0 && (b - a) > parallel_cutoff) {
        // spawn the left child as a task, do the right child on this thread.
        // L is non-copyable, so it must be shared (not the default firstprivate).
        #pragma omp task shared(L)
        bs(a, m, L, depth - 1);

        bs(m, b, R, depth - 1);

        #pragma omp taskwait               // both children done before we combine
    } else {
        // small subtree (or out of depth budget): stay on this thread.
        bs(a, m, L, 0);
        bs(m, b, R, 0);
    }

    // out.P = L.P * R.P
    mpz_mul(out.P, L.P, R.P);
    // out.Q = L.Q * R.Q
    mpz_mul(out.Q, L.Q, R.Q);
    // out.T = L.T * R.Q + L.P * R.T
    mpz_mul(out.T, L.T, R.Q);
    mpz_addmul(out.T, L.P, R.T);
}

// Binary-splitting Chudnovsky, all-integer final step.
//
//   pi = 426880 * sqrt(10005) * Q / T
//
// where (P, Q, T) is the binary-split reduction of the first N terms. To stay
// on GMP's fast integer paths (no mpf), scale by 10^D and use integer sqrt/div:
//
//   S       = floor(sqrt(10005 * 10^(2D)))   ~ sqrt(10005) * 10^D
//   pi*10^D ~ 426880 * Q * S / T
//
// `result` (an initialized mpz_t) receives floor(pi * 10^digits): the integer
// "314159..." with `digits` places after the leading 3.
void bs_chudnovsky(mpz_t result, unsigned long digits)
{
    const unsigned long GUARD = 16;                 // guard digits absorb the floors
    unsigned long D = digits + GUARD;

    // ~14.18 digits per term; size N from D so the series carries the guards too
    unsigned long N = (unsigned long)(D / 14.1816474627254776555) + 1;
    if (N < 1) N = 1;

    // Binary splitting builds an almost perfectly balanced tree, so a few tasks
    // per core is plenty — over-decomposing only adds task + GMP allocation
    // overhead (which, given the low speedup ceiling, can go slower than serial).
    // depth is the real control: spawn the top log2(threads)+2 levels (~4 tasks
    // per core). The size cutoff is only a floor so tiny runs stay serial.
    int depth = 0;
#ifdef _OPENMP
    for (int nt = omp_get_max_threads(); (1 << depth) < nt; ++depth) {}
    depth += 2;
#endif

    Triple sum;
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp single nowait
#endif
    bs(0, N, sum, depth);

    // S = floor(sqrt(10005 * 10^(2D)))
    mpz_t S, scale;
    mpz_init(S);
    mpz_init(scale);
    mpz_ui_pow_ui(scale, 10, 2 * D);      // 10^(2D)
    mpz_mul_ui(S, scale, 10005);          // 10005 * 10^(2D)
    mpz_sqrt(S, S);                        // integer sqrt

    // result = 426880 * Q * S / T  ~ pi * 10^D
    mpz_mul(result, sum.Q, S);
    mpz_mul_ui(result, result, 426880);
    mpz_tdiv_q(result, result, sum.T);

    // drop the guard digits: / 10^GUARD  ->  floor(pi * 10^digits)
    mpz_ui_pow_ui(scale, 10, GUARD);
    mpz_tdiv_q(result, result, scale);

    mpz_clear(S);
    mpz_clear(scale);
}

// ---- Parallel base-10 conversion ------------------------------------------
// GMP's mpn_get_str is subquadratic but single-threaded. We split the number
// with the same divide-and-conquer the library uses, but run the two halves as
// OpenMP tasks so the conversion spreads across cores.

static const unsigned long OUT_BASE = 1024;   // convert chunks this small with GMP
static const unsigned long OUT_PAR  = 32768;   // spawn tasks above this width

// Write x (0 <= x < 10^width) as exactly `width` decimal chars into out[0,width),
// left-padded with '0'. pw[i] = 10^(2^i); only entries with 2^i < width are read.
static void to_decimal(const mpz_t x, unsigned long width, char *out, mpz_t *pw)
{
    if (width <= OUT_BASE) {
        char *s = mpz_get_str(NULL, 10, x);
        size_t len = strlen(s);
        memset(out, '0', width - len);         // left-pad
        memcpy(out + (width - len), s, len);
        void (*freefn)(void *, size_t);         // free via GMP's allocator
        mp_get_memory_functions(NULL, NULL, &freefn);
        freefn(s, len + 1);
        return;
    }

    // split off the largest power-of-ten-of-two below width: x = hi*10^(2^k) + lo
    int k = 0;
    while ((2UL << k) < width) ++k;             // 2^(k+1) < width
    unsigned long lo_width = 1UL << k;
    unsigned long hi_width = width - lo_width;

    mpz_t hi, lo;
    mpz_init(hi);
    mpz_init(lo);
    mpz_tdiv_qr(hi, lo, x, pw[k]);

    if (width > OUT_PAR) {
        #pragma omp task shared(hi)
        to_decimal(hi, hi_width, out, pw);
        to_decimal(lo, lo_width, out + hi_width, pw);
        #pragma omp taskwait
    } else {
        to_decimal(hi, hi_width, out, pw);
        to_decimal(lo, lo_width, out + hi_width, pw);
    }

    mpz_clear(hi);
    mpz_clear(lo);
}

// Write "3.<n digits>\n" of pi_int (= floor(pi*10^n)) to f.
static void write_pi(FILE *f, const mpz_t pi_int, unsigned long n)
{
    mpz_t tenN, frac;
    mpz_init(tenN);
    mpz_init(frac);
    mpz_ui_pow_ui(tenN, 10, n);
    mpz_tdiv_r(frac, pi_int, tenN);            // low n digits (the fractional part)

    char *buf = new char[n];

    if (n <= OUT_BASE) {
        to_decimal(frac, n, buf, NULL);        // base case only; pw unused
    } else {
        // pw[i] = 10^(2^i) for i = 0 .. Kmax-1, built by repeated squaring
        int Kmax = 0;
        while ((1UL << Kmax) < n) ++Kmax;
        mpz_t *pw = new mpz_t[Kmax];
        mpz_init_set_ui(pw[0], 10);
        for (int i = 1; i < Kmax; ++i) {
            mpz_init(pw[i]);
            mpz_mul(pw[i], pw[i - 1], pw[i - 1]);
        }

#ifdef _OPENMP
        #pragma omp parallel
        #pragma omp single nowait
#endif
        to_decimal(frac, n, buf, pw);

        for (int i = 0; i < Kmax; ++i) mpz_clear(pw[i]);
        delete[] pw;
    }

    fputc('3', f);
    fputc('.', f);
    fwrite(buf, 1, n, f);
    fputc('\n', f);

    delete[] buf;
    mpz_clear(tenN);
    mpz_clear(frac);
}
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
    unsigned long n;
    printf("Number of digits: ");
    fflush(stdout);
    if (scanf("%lu", &n) != 1)
        return 1;

    // pass -p to also write the digits to a file
    bool print = (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'p');

    // if -p, the next stdin line is the output filename (blank = default pi.txt)
    char filename[512] = "pi.txt";
    if (print) {
        printf("File name (blank for pi.txt): ");
        fflush(stdout);
        int c;
        while ((c = getchar()) != '\n' && c != EOF) {}   // finish n's line
        char line[512];
        if (fgets(line, sizeof line, stdin)) {
            size_t len = strlen(line);
            while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
                line[--len] = '\0';                      // trim newline / CR
            if (len > 0)
                memcpy(filename, line, len + 1);
        }
    }

    // 640320^3, shared constant used by bs. Init once here.
    mpz_init(C3);
    mpz_ui_pow_ui(C3, 640320, 3);

    mpz_t pi_bs;
    mpz_init(pi_bs);

    typedef std::chrono::steady_clock clock;
    typedef std::chrono::duration<double> secs;

    auto s = clock::now();
    bs_chudnovsky(pi_bs, n);
    double t_bs = secs(clock::now() - s).count();
    printf("binary splitting: %.8fs\n", t_bs);

    if (print) {
        // pi_bs is the integer "3" followed by n decimals; split off the
        // fractional part (zero-padded) and write "3.<digits>" to pi.txt.
        FILE *f = fopen(filename, "w");
        if (!f) {
            perror(filename);
        } else {
            auto so = clock::now();
            write_pi(f, pi_bs, n);
            double t_out = secs(clock::now() - so).count();
            fclose(f);
            printf("output:           %.8fs  (wrote %lu digits to %s)\n",
                   t_out, n, filename);
        }
    }

    mpz_clear(pi_bs);
    mpz_clear(C3);
    return 0;
}