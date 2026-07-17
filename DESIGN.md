# Design

Algorithm, parallelism, and preallocation details for the Chudnovsky π
implementation. For installation, build, and usage see [README.md](README.md).

## Algorithm

π from the Chudnovsky series is `426880·√10005·Q / T`, where `(P, Q, T)` is the
**binary-split reduction** of the first `N ≈ digits/14.18` terms.

### Binary splitting

The sum of terms `[lo, hi)` is kept as one exact fraction `T/Q` (`P` is a helper
product), so a whole range collapses to integer multiplies with a **single
division at the very end** instead of one division per term. Each range is a
triple `(P, Q, T)`; two adjacent ranges merge as:

```
P = left.P · right.P
Q = left.Q · right.Q
T = left.T · right.Q + left.P · right.T
```

The `T` rule: putting both sub-sums over the common denominator `Q = left.Q·right.Q`
scales the left numerator by `right.Q`; the right terms sit further along the
series so they already carry a factor of `left.P`. A single term `k` seeds the
recursion:

```
p(k) = -(6k)(6k-1)(6k-2)(6k-3)(6k-4)(6k-5)
q(k) = (3k-2)(3k-1)(3k)·k³·C3          (C3 = 640320³)
T(k) = (545140134k + 13591409)·p(k)
```

The recursion is a balanced binary tree, so subtrees are independent — that's
what makes it parallelisable.

### All-integer final step

To stay on GMP's fast integer paths (no `mpf`), work at `D = digits + GUARD`
digits and scale by `10^D`:

```
S       = floor(sqrt(10005 · 10^(2D)))   ~ sqrt(10005) · 10^D
pi·10^D ~ 426880 · Q · S / T
```

`GUARD = 16` extra digits absorb the floor errors from the integer sqrt and
division; they're divided back off at the end.

## Optimisations

**Parallel tree.** Subtrees run as OpenMP tasks for the top `par_depth` levels
(`log2(threads) + 2`, ~4 tasks/core — a balanced tree needs no more), then
serially. `par_depth` is also capped so no leaf task falls below `MIN_LEAF`
terms; this single depth is the only granularity knob.

**Skip P on the right spine (`need_p`).** `T` needs only the *left* child's `P`,
so a right child needs its `P` only if its parent does. The root's `P` is never
used, so `need_p` stays false down the entire rightmost spine, skipping the
largest `P` multiplies in the tree.

**Overlap the sqrt.** `S` depends only on the precision, not on the series, so
it runs as a sibling task alongside the tree — its `O(M(n))` cost is hidden
behind the tree's serial top-level merges instead of adding to the critical path.

**Parallel merge near the root.** A node's four merge multiplies are independent
(distinct outputs, read-only inputs). For nodes whose operands exceed
`COMBINE_PAR_BITS`, they run as parallel tasks — attacking the serial root merge
that otherwise caps scaling.

**Preallocated triples.** Each `mpz_t` in a `Triple` is `mpz_init2`'d to an upper
bound on its final size, so the merge multiplies never reallocate. See below.

**Parallel base-10 conversion.** `mpn_get_str` is subquadratic but
single-threaded, so output splits the number by powers of `10^(2^k)` (precomputed
by squaring) and converts halves as OpenMP tasks.

**Process priority.** `main` raises the OS scheduling priority (Windows
`HIGH_PRIORITY_CLASS` / POSIX `nice`) to reduce preemption on a busy machine.
This does not speed up the math itself.

## Preallocation bounds (`p_bits` / `q_bits` / `t_bits`)

Each field's bit length for a range `[lo, hi)` is bounded from the per-term
factors. Per term:

- `p(k) < (6k)⁶`      → `< 6·log₂k + 16` bits  (`log₂ 6⁶ = 15.51`)
- `q(k) < 27·C3·k⁶`   → `< 6·log₂k + 63` bits  (`log₂(27·C3) = 62.62`)

Summing an increasing function is bounded by its integral, so

```
P, Q bits < 6·∫_lo^hi log₂x dx + const·count + guard
```

which is exactly `p_bits`/`q_bits` (`log2_integral` computes `∫log₂ = x·log₂x − x/ln2`,
starting at 1 since term 0 is trivial).

For `T`, expanding the merge gives `T = Σ a(j)·P_{[lo,j]}·Q_{(j,hi)}`. Since
`|p(i)| < q(i)`, each product is `< Q`, so `|T| ≤ count·a(hi)·Q` with
`a(k) < 2³¹·k`. Hence `bits(T) < bits(Q) + 2·log₂(hi) + 30`, i.e.
`t_bits = q_bits + 2·log₂(hi) + 32`.

The guards (`+64`, `+32`) absorb the `+1` from bit-length and float rounding. The
bounds are only *upper* bounds — if one were ever short, GMP simply reallocates,
so results are never at risk. They were verified against exact binary-splitting
arithmetic over 200+ ranges (including narrow high-index ranges where `T > Q`);
minimum slack was ~63 bits, so allocation is tight but never undershoots.

The same idea sizes the radix conversion's operands: a `d`-digit value has
`≤ d·log₂10` bits, and `3.3219280948873627 ≥ log₂10` keeps the bound safe.

## Precision / limits

- `a`, `b`, and the leaf's `6k` are `unsigned long long` (64-bit). Realistic runs
  stay far below any overflow (1 billion digits ≈ 70M terms).
- Memory is the real ceiling: at `n` digits, `Q` and `T` are each ~`n·log₂10`
  bits, and the top-level merge needs several of them live at once.
