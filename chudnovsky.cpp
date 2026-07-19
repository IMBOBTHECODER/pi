#include <gmp.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

// for the process scheduling priority and the system info banner (see main)
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>          // peak working set -- link with -lpsapi
#else
#include <sys/resource.h>
#include <unistd.h>
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
    // Hand back a field's storage once it's been consumed. Can't mpz_clear it --
    // the destructor still owns it -- so shrink to GMP's minimum instead, which
    // frees the limbs and leaves a valid (zeroed) mpz behind.
    static void release(mpz_t x) { mpz_realloc2(x, 1); }
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
// preallocated and its merge multiplies never realloc. Derivation in DESIGN.md
// ("Preallocation bounds").

// Antiderivative of log2: F(x) = x*log2(x) - x/ln2  (1/ln2 = 1.4426950408889634),
// clamped at x = 1 since term 0 is trivial and log2(0) is undefined. One log2 call,
// so a node sharing an endpoint between two ranges evaluates it once (see bs).
static double log2_antideriv(unsigned long long v)
{
    double x = (v < 1) ? 1.0 : (double)v;
    return x * log2(x) - x * 1.4426950408889634;
}
// integral of log2 over [lo, hi)
static double log2_integral(unsigned long long lo, unsigned long long hi)
{
    return log2_antideriv(hi) - log2_antideriv(lo);
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
    // floor(log2(hi)) via clz, not libm: hi >= 1 here, so clzll is well-defined.
    return q_bits(base, count) + 2 * (mp_bitcnt_t)(63 - __builtin_clzll(hi)) + 32;
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
// See DESIGN.md for the algorithm and optimisations.
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
        mpz_init2(ak, 80);
        mpz_set_ui(ak, 545140134UL);
        mpz_mul_ui(ak, ak, lo);
        mpz_add_ui(ak, ak, 13591409UL);
        mpz_mul(out.T, ak, out.P);
        mpz_clear(ak);
        return;
    }

    unsigned long long mid = (lo + hi) / 2;
    unsigned long long cL = mid - lo, cR = hi - mid;
    // The two child ranges meet at mid, so F(mid) serves both: 3 log2 calls, not 4.
    double fLo = log2_antideriv(lo), fMid = log2_antideriv(mid), fHi = log2_antideriv(hi);
    double baseL = 3.0 * (fMid - fLo), baseR = 3.0 * (fHi - fMid);
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
// `result` receives floor(pi * 10^digits). See DESIGN.md ("All-integer final step").
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

    mpz_t S;
    mpz_init(S);

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
        #pragma omp task shared(S)
#endif
        {
            // Built in place: a separate 10^(2D) would sit allocated at ~2D digits
            // (~290 MB at 350M) for the whole tree, unread. sqrt then shrinks the
            // value to D digits but not the allocation, so hand that half back too
            // -- this is the peak, and it is concurrent with the tree.
            mpz_ui_pow_ui(S, 10, 2 * D);      // 10^(2D)
            mpz_mul_ui(S, S, 10005);
            mpz_sqrt(S, S);                   // S = floor(sqrt(10005 * 10^(2D)))
            mpz_realloc2(S, (mp_bitcnt_t)((double)D * 3.3219280948873627) + 64);
        }

        bs(0, terms, sum, par_depth);   // root P is unused (need_p defaults false)

#ifdef _OPENMP
        #pragma omp taskwait                          // also waits for the S task
#endif
    }

    // result = 426880 * Q * S / T  ~ pi * 10^D. Q, S and T are each ~n digits or
    // more and are dead the moment they're consumed, so release the storage before
    // the next step allocates. S is a local, so it clears outright.
    mpz_mul(result, sum.Q, S);
    mpz_clear(S);
    Triple::release(sum.Q);
    mpz_mul_ui(result, result, 426880);
    mpz_tdiv_q(result, result, sum.T);
    Triple::release(sum.T);

    // drop the guard digits: / 10^GUARD  ->  floor(pi * 10^digits). Declared here,
    // not up top: it holds 17 digits, so there is nothing to gain by sharing it
    // with the 10^(2D) above and ~2D digits to lose.
    mpz_t scale;
    mpz_init(scale);
    mpz_ui_pow_ui(scale, 10, GUARD);
    mpz_tdiv_q(result, result, scale);
    mpz_clear(scale);
}

// ---- Parallel base-10 conversion ------------------------------------------
// mpn_get_str is single-threaded, so split the number by powers of ten and
// convert the halves as OpenMP tasks. See DESIGN.md ("Parallel base-10 conversion").

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
    // over-estimated) + guard so mpz_tdiv_qr doesn't realloc. See DESIGN.md
    // ("Preallocation bounds", last paragraph).
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
    // pi_int is already the digit string we want: "3" followed by exactly n
    // decimals (3 <= pi < 4, so 10^n <= pi_int < 10^(n+1)). Convert it whole and
    // insert the point -- splitting the fractional part off first would cost a
    // 10^n and an n-digit remainder, both as large as the output itself.
    unsigned long width = n + 1;
    char *buf = new char[width];

    if (width <= OUT_BASE) {
        to_decimal(pi_int, width, buf, NULL);  // base case only; pw unused
    } else {
        // pw[i] = 10^(2^i) for i = 0 .. Kmax-1, built by repeated squaring.
        // Sized from width, not n: to_decimal reads up to pw[ceil(log2(width))-1].
        int Kmax = 0;
        while ((1UL << Kmax) < width) ++Kmax;
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
        to_decimal(pi_int, width, buf, pw);

        for (int i = 0; i < Kmax; ++i) mpz_clear(pw[i]);
        delete[] pw;
    }

    fputc(buf[0], f);                          // the leading '3'
    fputc('.', f);
    fwrite(buf + 1, 1, n, f);
    fputc('\n', f);

    delete[] buf;
}
// ---------------------------------------------------------------------------

// Render a duration for printing. Under a minute, keep full precision -- that's the
// benchmarking range. Past that, trim to 2dp (8 decimals of an hour-long run is
// noise) and add the larger units, since "3601s" doesn't read as an hour:
//   5.71246496s   /   90.12s | 1.50m   /   3601.00s | 60.02m | 1.00h
static void format_secs(char *buf, size_t n, double s)
{
    if (s < 60.0)
        snprintf(buf, n, "%.8fs", s);
    else if (s < 3600.0)
        snprintf(buf, n, "%.2fs | %.2fm", s, s / 60.0);
    else
        snprintf(buf, n, "%.2fs | %.2fm | %.2fh", s, s / 60.0, s / 3600.0);
}

// ---- System info ----------------------------------------------------------
// Printed before the run so benchmark output says what produced it.

// Pull one "key: value" field out of /proc/<file>. Returns false if not found.
#ifndef _WIN32
static bool proc_field(const char *path, const char *key, char *out, size_t n)
{
    FILE *f = fopen(path, "r");
    if (!f) return false;
    char line[512];
    size_t klen = strlen(key);
    bool found = false;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, klen) != 0) continue;
        char *v = strchr(line, ':');
        if (!v) continue;
        for (++v; *v == ' ' || *v == '\t'; ++v) {}       // skip separator space
        size_t len = strlen(v);
        while (len && (v[len - 1] == '\n' || v[len - 1] == '\r')) v[--len] = '\0';
        snprintf(out, n, "%s", v);
        found = true;
        break;
    }
    fclose(f);
    return found;
}
#endif

static void print_sysinfo(void)
{
    char cpu[160] = "unknown";
    double ram = 0, avail = 0, swap = 0;                 // GiB
    int logical = 1, cores = 0;                          // cores 0 = unknown

#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    logical = (int)si.dwNumberOfProcessors;
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        ram   = (double)ms.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        avail = (double)ms.ullAvailPhys / (1024.0 * 1024.0 * 1024.0);
    }
    DWORD sz = sizeof cpu;
    RegGetValueA(HKEY_LOCAL_MACHINE,
                 "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                 "ProcessorNameString", RRF_RT_REG_SZ, NULL, cpu, &sz);
#else
    logical = (int)sysconf(_SC_NPROCESSORS_ONLN);
    proc_field("/proc/cpuinfo", "model name", cpu, sizeof cpu);
    char buf[64];
    if (proc_field("/proc/cpuinfo", "cpu cores", buf, sizeof buf))
        cores = atoi(buf);
    // /proc/meminfo values are in kB
    if (proc_field("/proc/meminfo", "MemTotal", buf, sizeof buf))
        ram = atof(buf) / (1024.0 * 1024.0);
    if (proc_field("/proc/meminfo", "MemAvailable", buf, sizeof buf))
        avail = atof(buf) / (1024.0 * 1024.0);
    if (proc_field("/proc/meminfo", "SwapTotal", buf, sizeof buf))
        swap = atof(buf) / (1024.0 * 1024.0);
#endif

    int threads = logical;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    printf("CPU:     %s\n", cpu);
    // "cores" is whatever the OS claims. Under a VM (WSL2) that topology is
    // synthesised and can disagree with the physical chip -- a hybrid P/E-core
    // CPU gets flattened -- so it's labelled as reported, not asserted.
    if (cores > 0)
        printf("Threads: %d OpenMP | %d logical | %d cores (as reported)\n",
               threads, logical, cores);
    else
        printf("Threads: %d OpenMP | %d logical\n", threads, logical);
    printf("Memory:  %.1f GiB total | %.1f GiB available", ram, avail);
    if (swap > 0)
        printf(" | %.1f GiB swap", swap);
    printf("\n");
#ifndef _OPENMP
    printf("         (built without OpenMP -- single-threaded)\n");
#endif
}

// Peak resident set since process start, in MiB (0 if unavailable). This is a
// high-water mark the kernel maintains for us, so reading it at the end reports
// the true peak -- no sampling, and no need to catch the tree mid-descent.
static double peak_rss_mb(void)
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc))
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    return 0.0;
#else
    // VmHWM is the peak; VmRSS would only be the value at this instant.
    char buf[64];
    if (proc_field("/proc/self/status", "VmHWM", buf, sizeof buf))
        return atof(buf) / 1024.0;               // kB -> MiB
    // No /proc: ru_maxrss is the same figure, in kB on Linux.
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0)
        return (double)ru.ru_maxrss / 1024.0;
    return 0.0;
#endif
}

// Render a byte count the way the run is usually discussed: MiB up to ~1 GiB,
// GiB past that, since a peak in the GB range is what limits digit counts.
static void format_mem(char *buf, size_t n, double mb)
{
    if (mb < 1024.0)
        snprintf(buf, n, "%.1f MiB", mb);
    else
        snprintf(buf, n, "%.2f GiB (%.0f MiB)", mb / 1024.0, mb);
}

// Parse a digit count, accepting a k/M/G suffix and scientific notation so that
// "200M" or "2e8" can stand in for counting zeros. The mantissa may be
// fractional ("2.5M"); the result is rounded. Returns false on anything else,
// which is also how "-p" decides whether the word after it is a filename.
static bool parse_digits(const char *s, unsigned long *out)
{
    char *end;
    double v = strtod(s, &end);
    if (end == s) return false;
    switch (*end) {
        case 'k': case 'K': v *= 1e3; ++end; break;
        case 'm': case 'M': v *= 1e6; ++end; break;
        case 'g': case 'G': v *= 1e9; ++end; break;
        default: break;
    }
    // the range test also rejects inf/nan, which strtod would otherwise accept
    if (*end != '\0' || !(v >= 1.0) || v > 1e12) return false;
    *out = (unsigned long)(v + 0.5);
    return true;
}

// Does argv[i] name this flag? Matches both the separated form ("-t 8", *val
// left NULL for the caller to fill from argv[i+1]) and the joined one ("-t=8").
static bool is_flag(const char *arg, const char *flag, const char **val)
{
    size_t n = strlen(flag);
    if (strncmp(arg, flag, n) != 0) return false;
    if (arg[n] == '\0') { *val = NULL;        return true; }
    if (arg[n] == '=')  { *val = arg + n + 1; return true; }
    return false;                    // a longer flag that merely starts the same
}

static void usage(const char *prog)
{
    printf("Usage: %s [digits] [options]\n"
           "\n"
           "  digits            decimals of pi to compute; prompts if omitted.\n"
           "                    Accepts a k/M/G suffix or exponent: 50M, 2.5M, 1e8\n"
           "\n"
           "  -o, --out [FILE]  also write the digits out (default: pi.txt)\n"
           "                    -p and --print are accepted as aliases\n"
           "  -t, --threads N   OpenMP threads to use (default: all)\n"
           "  -q, --quiet       skip the system info banner\n"
           "  -h, --help        this message\n"
           "\n"
           "Options take their value joined or separated: -t8 is not accepted,\n"
           "but -t 8 and -t=8 both are.\n"
           "\n"
           "  %s 50M -t 8            50M digits on 8 threads, no output file\n"
           "  %s 1e6 -o pi1m.txt     one million digits, written to pi1m.txt\n"
           "\n"
           "With no digits argument the original interactive prompts are used.\n",
           prog, prog, prog);
}

int main(int argc, char **argv)
{
    // Raise scheduling priority so background processes preempt us less. Doesn't
    // speed up the math itself; just trims scheduling jitter on a busy machine.
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);   // needs privilege to go negative
#endif

// Note: capping glibc's per-thread malloc arenas (mallopt(M_ARENA_MAX, 2))
// cuts peak RSS by ~30% at 50M digits, but costs enough time to not be worth
// it -- the workers end up contending for the two arenas. Left out on purpose.
// It can still be tried from the outside with MALLOC_ARENA_MAX=2 in the env.

    unsigned long n = 0;
    bool have_n = false, print = false, have_file = false, quiet = false;
    int threads = 0;                     // 0 = leave OpenMP's default alone
    char filename[512] = "pi.txt";

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i], *val = NULL;
        char *end = NULL;

        if (is_flag(a, "-h", &val) || is_flag(a, "--help", &val)) {
            usage(argv[0]);
            return 0;
        } else if (is_flag(a, "-q", &val) || is_flag(a, "--quiet", &val)) {
            quiet = true;
        } else if (is_flag(a, "-o",     &val) || is_flag(a, "--out",   &val) ||
                   is_flag(a, "-p",     &val) || is_flag(a, "--print", &val)) {
            print = true;
            // The filename is optional. A following word is taken as one only if
            // it isn't the digit count: "./main -o 1M" means 1M digits to the
            // default pi.txt, not a file called "1M". Use -o=1M to force that.
            if (!val && i + 1 < argc && argv[i + 1][0] != '-') {
                unsigned long ignored;
                if (!parse_digits(argv[i + 1], &ignored))
                    val = argv[++i];
            }
            if (val && *val != '\0') {
                snprintf(filename, sizeof filename, "%s", val);
                have_file = true;
            }
        } else if (is_flag(a, "-t", &val) || is_flag(a, "--threads", &val)) {
            if (!val && i + 1 < argc) val = argv[++i];
            long v = val ? strtol(val, &end, 10) : 0;
            if (!val || *end != '\0' || v < 1 || v > 4096) {
                fprintf(stderr, "%s needs a thread count >= 1\n", a);
                return 1;
            }
            threads = (int)v;
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "unknown option: %s\n", a);
            usage(argv[0]);
            return 1;
        } else if (!have_n) {
            if (!parse_digits(a, &n)) {
                fprintf(stderr, "not a digit count: %s\n", a);
                return 1;
            }
            have_n = true;
        } else {
            fprintf(stderr, "unexpected argument: %s\n", a);
            usage(argv[0]);
            return 1;
        }
    }

#ifdef _OPENMP
    // Before anything reads omp_get_max_threads() -- print_sysinfo reports it and
    // bs_chudnovsky sizes par_depth from it. Same effect as OMP_NUM_THREADS.
    if (threads > 0) omp_set_num_threads(threads);
#else
    if (threads > 0)
        fprintf(stderr, "note: built without OpenMP, ignoring -t\n");
#endif

    // No digit count on the command line -> the original interactive path.
    if (!have_n) {
        printf("Number of digits: ");
        fflush(stdout);
        if (scanf("%lu", &n) != 1) { return 1; }

        // if -p, the next stdin line is the output filename (blank = default pi.txt)
        if (print && !have_file) {
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
    }

    if (!quiet) {
        print_sysinfo();
        printf("Run:     %lu digits%s%s\n\n", n,
               print ? " -> " : "", print ? filename : "");
    }

    // 640320^3 / 24, shared constant used by bs (640320^3 = 262537412640768000 is
    // divisible by 24). set_str, not set_ui: it exceeds a 32-bit unsigned long.
    mpz_init_set_str(C3_24, "10939058860032000", 10);

    mpz_t pi_bs;
    mpz_init(pi_bs);

    typedef std::chrono::steady_clock clock;
    typedef std::chrono::duration<double> secs;

    char tbuf[64], mbuf[64];

    auto s = clock::now();
    bs_chudnovsky(pi_bs, n);
    double t_bs = secs(clock::now() - s).count();
    // Sampled here, before write_pi allocates, so the two phases can be told
    // apart: this is the tree's high-water mark, which is what sets the ceiling.
    double peak_bs = peak_rss_mb();
    format_secs(tbuf, sizeof tbuf, t_bs);
    printf("binary splitting: %s\n", tbuf);

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
            format_secs(tbuf, sizeof tbuf, t_out);
            printf("output:           %s  (wrote %lu digits to %s)\n",
                   tbuf, n, filename);
        }
    }

    // Peak RSS, and whether output pushed past what the tree already held.
    double peak = peak_rss_mb();
    if (peak > 0.0) {
        format_mem(mbuf, sizeof mbuf, peak);
        printf("peak memory:      %s", mbuf);
        if (print && peak > peak_bs * 1.01) {
            format_mem(tbuf, sizeof tbuf, peak_bs);
            printf("  (%s in binary splitting, rest is output)", tbuf);
        }
        printf("\n");
    }

    mpz_clear(pi_bs);
    mpz_clear(C3_24);
    return 0;
}