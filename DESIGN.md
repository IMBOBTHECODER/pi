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
p(k) = -(6k-5)(2k-1)(6k-1)
q(k) = k³·C3/24                (C3 = 640320³, so C3/24 = 10939058860032000)
T(k) = (545140134k + 13591409)·p(k)
```

The recursion is a balanced binary tree, so subtrees are independent — that's
what makes it parallelisable.

### Why the leaf is reduced

Read straight off the series, the seeds are

```
p(k) = -(6k)(6k-1)(6k-2)(6k-3)(6k-4)(6k-5)
q(k) = (3k-2)(3k-1)(3k)·k³·C3
```

but these share a large factor. Since `(6k)(6k-2)(6k-4) = 8(3k)(3k-1)(3k-2)` and
`(6k-3) = 3(2k-1)`, the whole `(3k)(3k-1)(3k-2)` cancels and `8·3 = 24` divides out
of `C3`:

```
p(k)/q(k) = -24(6k-5)(2k-1)(6k-1) / (k³·C3) = -(6k-5)(2k-1)(6k-1) / (k³·C3/24)
```

Only the **ratio** `p(k)/q(k)` affects `T/Q`, so `T(0,N)/Q(0,N)` is the identical
rational number and the final `426880·√10005·Q/T` is untouched. But the operands
shrink: `q(k)` drops from `6·log₂k + 63` bits to `3·log₂k + 54`, roughly halving
every multiply in the tree. Measured at 10M digits: peak RSS 386 → 277 MB, compute
8.27 → 5.71 s. (`C3` is divisible by 24, so `C3/24` is exact.)

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

**Reduced leaf.** The single biggest win — see *Why the leaf is reduced* above.

**Parallel tree.** Subtrees run as OpenMP tasks for the top `par_depth` levels
(`log2(threads) + 2`, ~4 tasks/core), then serially. `par_depth` is also capped so
no leaf task falls below `MIN_LEAF` terms; this single depth is the only
granularity knob. It is also the main **memory/speed dial**: each level keeps
roughly one root's worth of triples live at once (~24 MB at 10M digits), because in
a parallel descent both children are being filled and so both must be allocated.
Sibling subtrees are close to work-balanced despite unequal operand sizes (~6%
apart at the root, ~20% at depth 6), so coarser tasks cost less speed than the
oversubscription suggests.

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

**Release consumed operands.** `Q`, `S` and `T` are each ~2× the output length and
die the moment the final `426880·Q·S/T` consumes them, so their storage goes back
before the next step allocates. `S` is a local and clears outright; `Q` and `T`
belong to the `Triple` destructor, so `Triple::release` shrinks them to GMP's
minimum (`mpz_realloc2(x, 1)`) instead — an `mpz_clear` here would double-free,
since GMP leaves the struct dangling rather than resetting it. This trims the tail,
not the peak: by this point the tree has unwound.

**Convert the output whole.** `floor(pi·10^n)` is *already* the digit string —
"3" followed by exactly n decimals — so `write_pi` converts it at width `n+1` and
inserts the point, rather than building `10^n` and taking a remainder to split the
fractional part off first. That drops two n-digit numbers and an `O(M(n))` division.

**Parallel base-10 conversion.** `mpn_get_str` is subquadratic but
single-threaded, so output splits the number by powers of `10^(2^k)` (precomputed
by squaring) and converts halves as OpenMP tasks.

**Process priority.** `main` raises the OS scheduling priority (Windows
`HIGH_PRIORITY_CLASS` / POSIX `nice`) to reduce preemption on a busy machine.
This does not speed up the math itself.

## Preallocation bounds (`p_bits` / `q_bits` / `t_bits`)

Each field's bit length for a range `[lo, hi)` is bounded from the per-term
factors. Per term:

- `p(k) < 72k³`      → `< 3·log₂k + 7` bits   (`log₂ 72 = 6.17`)
- `q(k) = k³·C3/24`  → `< 3·log₂k + 54` bits  (`log₂(C3/24) = 53.28`)

Summing an increasing function is bounded by its integral, so

```
P, Q bits < 3·∫_lo^hi log₂x dx + const·count + guard
```

which is exactly `p_bits`/`q_bits`. The integral is evaluated as `F(hi) − F(lo)`
over the antiderivative `F(x) = x·log₂x − x/ln2`, clamped at `x = 1` since term 0
is trivial and `log₂0` is undefined. A node's two child ranges meet at `mid`, so
`F(mid)` serves both — three `log₂` calls per node rather than four.

For `T`, expanding the merge gives `T = Σ a(j)·P_{[lo,j]}·Q_{(j,hi)}`. Since
`|p(i)| < q(i)` (`72k³` against `1.09e16·k³`, comfortably), each product is `< Q`,
so `|T| ≤ count·a(hi)·Q` with `a(k) < 2³¹·k`. Hence `bits(T) < bits(Q) + 2·log₂(hi) + 30`,
i.e. `t_bits = q_bits + 2·log₂(hi) + 32`. The `log₂(hi)` there is only ever used
floored, so it comes from `__builtin_clzll` rather than libm.

The guards (`+64`, `+32`) absorb the `+1` from bit-length and float rounding. The
bounds are only *upper* bounds — if one were ever short, GMP simply reallocates,
so results are never at risk.

> **Caveat.** An earlier verification against exact binary-splitting arithmetic
> (200+ ranges, including narrow high-index ranges where `T > Q`; minimum slack
> ~63 bits) was performed against the **unreduced** leaf and its `6·log₂` constants.
> It has not been repeated since the reduction. The constants above are re-derived
> but not re-measured — correctness is unaffected either way, only the "never
> reallocates" claim.

The same idea sizes the radix conversion's operands: a `d`-digit value has
`≤ d·log₂10` bits, and `3.3219280948873627 ≥ log₂10` keeps the bound safe.

## Precision / limits

- `lo`, `hi`, and the leaf's `6k` are `unsigned long long` (64-bit). Realistic runs
  stay far below any overflow (1 billion digits ≈ 70M terms).
- Memory is the real ceiling, and it is set by the **tree descent**, not the final
  step. `Q` and `T` each run ~2.3× the output length in digits (at 10M digits, `Q`
  is ~22.8M digits ≈ 9.5 MB), and the top `par_depth` levels hold roughly one
  root's worth each — so the peak scales with `par_depth`, and trimming the tail
  cannot lower a high-water mark the tree already set.
