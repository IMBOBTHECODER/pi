#include <gmp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// for raising the process scheduling priority (see main)
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#endif

// 640320^3 / 24 (exact) -- the reduced leaf denominator, see bs
static mpz_t C3_24;

struct Triple {
    mpz_t P, Q, T;
    // Preallocate P, Q, T each to their own size (see p_bits/q_bits/t_bits) so the
    // merge multiplies that fill them never realloc. Default 1 = GMP's minimum.
    explicit Triple(mp_bitcnt_t pb = 1, mp_bitcnt_t qb = 1, mp_bitcnt_t tb = 1) {
        mpz_init2(P, pb);
        mpz_init2(Q, qb);
        mpz_init2(T, tb);
    }
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

    // 640320^3. Local: bs uses the reduced C3/24, so this design is its only user.
    mpz_t C3;
    mpz_init(C3);
    mpz_ui_pow_ui(C3, 640320, 3);

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

    mpz_clear(C3);
    mpz_clear(AkB);
    mpz_clear(num_k);
    mpz_clear(den_k);
    mpf_clear(t);
    mpf_clear(pi);
    mpf_clear(tmp);
}

// Upper bounds (bits) on P, Q, T for a term range, so a Triple can be
// preallocated and its merge multiplies never realloc. Derivation in README.

// integral of log2 over [lo, hi)  (1/ln2 = 1.4426950408889634)
static double log2_integral(unsigned long long lo, unsigned long long hi)
{
    double a = (lo < 1) ? 1.0 : (double)lo;
    double b = (double)hi;
    return b * log2(b) - a * log2(a) - (b - a) * 1.4426950408889634;
}
// base = 3 * log2_integral(lo, hi), count = hi - lo. Passing the precomputed base
// means the integral (two log2 calls) is done once per range, not once per field.
static mp_bitcnt_t p_bits(double base, unsigned long long count)
{
    return (mp_bitcnt_t)(base + 7.0 * (double)count) + 64;
}
static mp_bitcnt_t q_bits(double base, unsigned long long count)
{
    return (mp_bitcnt_t)(base + 54.0 * (double)count) + 64;
}
static mp_bitcnt_t t_bits(double base, unsigned long long count, unsigned long long hi)
{
    return q_bits(base, count) + 2 * (mp_bitcnt_t)log2((double)hi) + 32;
}

// Below this many terms a subtree stops spawning tasks. On a balanced tree this
// leaf-size floor equals a depth limit, so it's folded into par_depth below.
static const unsigned long MIN_LEAF = 2048;

// Merge multiplies run as parallel tasks only when operands reach this many bits
// (near the root, where cores are free); smaller nodes merge serially. ~39k digits.
static const size_t COMBINE_PAR_BITS = 1u << 17;

// Binary-splitting core: (P, Q, T) for the term range [lo, hi). The leaf uses the
// reduced p/q -- the raw (6k)!/(3k)!(k!)^3 form shares a factor of 8(3k)(3k-1)(3k-2)
// between P and Q that cancels in T/Q; dropping it halves every operand. Merge rule:
//     P = left.P * right.P
//     Q = left.Q * right.Q
//     T = left.T * right.Q + left.P * right.T
// Subtrees are independent, so they run as OpenMP tasks. need_p: a node builds
// its P only if the parent uses it (the right spine, incl. the root, doesn't).
// See README for the algorithm and optimisations.
void bs(unsigned long long lo, unsigned long long hi, Triple& out, int par_depth, bool need_p = false)
{
    unsigned long long count = hi - lo;

    if (count == 1) {
        // one term, with index `lo`
        if (lo == 0) {
            mpz_set_ui(out.P, 1);
            mpz_set_ui(out.Q, 1);
        } else {
            unsigned long long k6 = 6 * lo;

            // P = -(6k-5)(2k-1)(6k-1),  k = lo
            mpz_set_ui(out.P, k6 - 5);
            mpz_mul_ui(out.P, out.P, 2 * lo - 1);
            mpz_mul_ui(out.P, out.P, k6 - 1);
            mpz_neg(out.P, out.P);

            // Q = k^3 * C3/24
            mpz_set_ui(out.Q, lo);
            mpz_mul_ui(out.Q, out.Q, lo);
            mpz_mul_ui(out.Q, out.Q, lo);
            mpz_mul(out.Q, out.Q, C3_24);
        }

        // T = (545140134*lo + 13591409) * P   (linear factor fits in <96 bits)
        mpz_t ak;
        mpz_init2(ak, 96);
        mpz_set_ui(ak, 545140134UL);
        mpz_mul_ui(ak, ak, lo);
        mpz_add_ui(ak, ak, 13591409UL);
        mpz_mul(out.T, ak, out.P);
        mpz_clear(ak);
        return;
    }

    unsigned long long mid = (lo + hi) / 2;
    unsigned long long cL = mid - lo, cR = hi - mid;
    double baseL = 3.0 * log2_integral(lo, mid), baseR = 3.0 * log2_integral(mid, hi);
    Triple left (p_bits(baseL, cL), q_bits(baseL, cL), t_bits(baseL, cL, mid)),
           right(p_bits(baseR, cR), q_bits(baseR, cR), t_bits(baseR, cR, hi));

    // Left child's P always feeds our T; the right child's P is needed only if
    // ours is. Spawn the left subtree as a task, keep the right on this thread.
    if (par_depth > 0) {
        #pragma omp task shared(left)      // non-copyable -> share, not firstprivate
        bs(lo, mid, left, par_depth - 1, true);

        bs(mid, hi, right, par_depth - 1, need_p);

        #pragma omp taskwait
    } else {
        bs(lo, mid, left, 0, true);
        bs(mid, hi, right, 0, need_p);
    }

    // Merge. The four products are independent (distinct outputs, read-only
    // inputs), so near the root run them as parallel tasks.
#ifdef _OPENMP
    if (mpz_sizeinbase(left.Q, 2) >= COMBINE_PAR_BITS) {
        mpz_t t_left, t_right;             // the two halves of T
        mpz_init2(t_left,  t_bits(baseL, cL, mid) + q_bits(baseR, cR));   // = left.T * right.Q
        mpz_init2(t_right, p_bits(baseL, cL) + t_bits(baseR, cR, hi));    // = left.P * right.T

        if (need_p) {
            #pragma omp task shared(out, left, right)
            mpz_mul(out.P, left.P, right.P);
        }
        #pragma omp task shared(out, left, right)
        mpz_mul(out.Q, left.Q, right.Q);
        #pragma omp task shared(t_left, left, right)
        mpz_mul(t_left, left.T, right.Q);
        #pragma omp task shared(t_right, left, right)
        mpz_mul(t_right, left.P, right.T);
        #pragma omp taskwait

        mpz_add(out.T, t_left, t_right);   // T = left.T*right.Q + left.P*right.T
        mpz_clear(t_left);
        mpz_clear(t_right);
        return;
    }
#endif

    // Serial merge (small node, or no OpenMP).
    if (need_p)
        mpz_mul(out.P, left.P, right.P);
    mpz_mul(out.Q, left.Q, right.Q);
    mpz_mul(out.T, left.T, right.Q);
    mpz_addmul(out.T, left.P, right.T);    // T += left.P * right.T
}

// Chudnovsky via binary splitting. All-integer final step: work at D digits and
// scale by 10^D, so S = floor(sqrt(10005*10^(2D))) and pi*10^D ~ 426880*Q*S/T.
// `result` receives floor(pi * 10^digits). See README.
void bs_chudnovsky(mpz_t result, unsigned long digits)
{
    const unsigned long GUARD = 16;               // guard digits absorb the floors
    unsigned long D = digits + GUARD;             // working precision, in digits

    unsigned long terms = (unsigned long)(D / 14.1816474627254776555) + 1;  // ~14.18 digits/term
    if (terms < 1) terms = 1;

    // Spawn tasks for the top par_depth levels only (~4 tasks/core is plenty on a
    // balanced tree), but not so deep that a leaf falls below MIN_LEAF terms.
    int par_depth = 0;
#ifdef _OPENMP
    for (int nt = omp_get_max_threads(); (1 << par_depth) < nt; ++par_depth) {}
    par_depth += 2;
#endif
    int max_depth = 0;
    while ((terms >> (max_depth + 1)) >= MIN_LEAF) ++max_depth;
    if (par_depth > max_depth) par_depth = max_depth;

    mpz_t S, scale;
    mpz_init(S);
    mpz_init(scale);

    double base = 3.0 * log2_integral(0, terms);
    Triple sum(p_bits(base, terms), q_bits(base, terms), t_bits(base, terms, terms));
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp single nowait
#endif
    {
        // S depends only on the precision, not on the series, so run it as a
        // sibling task that overlaps the tree instead of adding to the tail.
#ifdef _OPENMP
        #pragma omp task shared(S, scale)
#endif
        {
            mpz_ui_pow_ui(scale, 10, 2 * D);  // 10^(2D)
            mpz_mul_ui(S, scale, 10005);
            mpz_sqrt(S, S);                   // S = floor(sqrt(10005 * 10^(2D)))
        }

        bs(0, terms, sum, par_depth);   // root P is unused (need_p defaults false)

#ifdef _OPENMP
        #pragma omp taskwait                          // also waits for the S task
#endif
    }

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
// mpn_get_str is single-threaded, so split the number by powers of ten and
// convert the halves as OpenMP tasks. See README.

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

    // hi/lo have <= hi_width/lo_width digits; preallocate (log2(10) bits/digit,
    // over-estimated) + guard so mpz_tdiv_qr doesn't realloc. See README.
    mpz_t hi, lo;
    mpz_init2(hi, (mp_bitcnt_t)(hi_width * 3.3219280948873627) + 64);
    mpz_init2(lo, (mp_bitcnt_t)(lo_width * 3.3219280948873627) + 64);
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
    // 10^n and pi_int mod 10^n have <= n digits; preallocate (log2(10) bits/digit,
    // over-estimated) + guard so the pow/mod don't realloc. See README.
    mp_bitcnt_t bits = (mp_bitcnt_t)(n * 3.3219280948873627) + 64;
    mpz_t tenN, frac;
    mpz_init2(tenN, bits);
    mpz_init2(frac, bits);
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
    // Raise scheduling priority so background processes preempt us less. Doesn't
    // speed up the math itself; just trims scheduling jitter on a busy machine.
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);   // needs privilege to go negative
#endif

    unsigned long n;
    printf("Number of digits: ");
    fflush(stdout);
    if (scanf("%lu", &n) != 1) { return 1; }

    // pass -p to also write the digits to a file
    bool print = (argc > 1 && argv[1][0] == '-' && argv[1][1] == 'p');

    // if -p, the next stdin line is the output filename (blank = default pi.txt)
    char filename[512] = "pi.txt";
    if (print) {
        printf("File name (empty for pi.txt): ");
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

    // 640320^3 / 24, shared constant used by bs. Exact: 640320^3 is divisible by 24.
    mpz_init(C3_24);
    mpz_ui_pow_ui(C3_24, 640320, 3);
    mpz_divexact_ui(C3_24, C3_24, 24);

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
    mpz_clear(C3_24);
    return 0;
}