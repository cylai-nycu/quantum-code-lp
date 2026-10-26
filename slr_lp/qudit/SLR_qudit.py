#!/usr/bin/env python3
"""
SLR_qudit.py

Fast exact-integer A-only Shor--Laflamme--Rains feasibility solver for
q-ary stabilizer codes. It generalizes SLR_qudit.py to local dimension q.

For every prime-power q it imposes

    * q^2-ary Shor--Laflamme / MacWilliams identities,
    * distance constraints A_j=B_j below d and B_j>=A_j,
    * the generalized Rains shadow inequalities,
    * exact B-integrality and optional shell-capacity bounds.

For q=2 it also keeps the complete Type-I/II/III three-case counting
constraints of SLR_qudit.py. For q>2 these binary Arf/parity cases are not
forced at Hamming level. If ASSUME_FQ_LINEAR_STABILIZER is enabled (or the
--fq-linear flag is used), the q-ary linear-orbit divisibility constraints
A_j,B_j == 0 mod (q-1) for j>0 are added.

B and shadow variables are eliminated exactly. All exact equalities in A
are eliminated by rational RREF before Z3 is called.

Dependencies
------------
    pip install sympy z3-solver

Usage
-----
    python SLR_qudit.py q n k d
    python SLR_qudit.py q n k
    python SLR_qudit.py q n k d --fq-linear

Examples
--------
    python SLR_qudit.py 2 27 15 5
    python SLR_qudit.py 3 10 2 4 --fq-linear
    python SLR_qudit.py 4 8 2
"""

import sys
import argparse
import time
import math
import importlib.util
from fractions import Fraction
from math import comb
from pathlib import Path

try:
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
except ImportError as exc:
    raise SystemExit(
        "This program requires SymPy.\n"
        "Install it with:\n"
        "  pip install sympy"
    ) from exc

try:
    from z3 import Int, Or, Sum, SolverFor, sat, unsat, unknown
except ImportError as exc:
    raise SystemExit(
        "This program requires z3-solver.\n"
        "Install it with:\n"
        "  pip install z3-solver"
    ) from exc


# ============================================================
# USER OPTIONS
# ============================================================

# Print each surviving A-distribution in compact nonzero form.
PRINT_SOLUTIONS = True

# Keep solution collection separate from presentation.  Library callers
# (notably systematic_exclusion.py) need the full A distributions even when
# they suppress this module's detailed solution output.
COLLECT_SOLUTIONS = False

# Print rank/free-variable diagnostics for each case.
PRINT_PREPROCESSING = True

# Recompute B,S and verify every enumerated model in Python.
# Useful for debugging, but unnecessary for routine counting and slower.
VERIFY_EACH_SOLUTION = False

# None means enumerate all solutions exactly.
MAX_SOLUTIONS = None

# None means no Z3 timeout.  Unit: milliseconds.
TIMEOUT_MS = None

# Live progress while enumerating solutions.
SHOW_LIVE_PROGRESS = True

# Print a progress line after this many NEW solutions.
# Set to None to disable the count-based trigger.
PROGRESS_EVERY_SOLUTIONS = 10

# Also print a progress line when this many seconds have elapsed
# since the previous progress report.
PROGRESS_EVERY_SECONDS = 3600.0

# For explicit `python SLR_qudit.py n k d` survivor generation, interpret
# [[n,k,d]] as having exact minimum distance d (for k>0), so B_d>A_d.
# This matches the downstream logical-reference solver, which fixes a minimum
# logical operator of weight d.  The two-argument bound scan deliberately
# overrides this and tests distance >= d, preserving monotonicity in d.
REQUIRE_EXACT_DISTANCE = True

# Necessary shell-capacity bounds for stabilizer, normalizer, and shadow:
#     A_j, B_j, S_j <= 3^j * binom(n,j).
# These add only linear inequalities; no new variables.
USE_SHELL_CAPACITY_BOUNDS = True

# For q>2, a general stabilizer is only additive over the prime field.
# Set this to True only when restricting to F_q-linear stabilizer codes.
# Then every nonzero scalar orbit has size q-1, implying
#     A_j == B_j == 0 (mod q-1),  j>0.
ASSUME_FQ_LINEAR_STABILIZER = False

# In the binary case the normalized shadow enumerator is an actual coset
# weight distribution, so it is integral and obeys Pauli-shell capacities.
# For q>2 we impose only the universal generalized-shadow nonnegativity.
USE_BINARY_SHADOW_INTEGRALITY = True
USE_BINARY_SHADOW_CAPACITY = True

# Automatically save all SLR/qudit survivors into ONE persistent executable
# candidate database. The three refined solvers update this same file.
SAVE_CANDIDATE_DATABASE = True
CANDIDATE_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "candidates" / "qudit"
CANDIDATE_FILENAME = None


# ============================================================
# q-ary Krawtchouk and generalized-shadow transform matrices
# ============================================================

def is_prime_power(q):
    if q < 2:
        return False
    fac = sp.factorint(q)
    return len(fac) == 1


def krawtchouk(n, j, i, q):
    """q^2-ary Hamming Krawtchouk coefficient."""
    value = 0
    qm1 = q * q - 1
    for ell in range(j + 1):
        if ell <= i and j - ell <= n - i:
            value += (
                (-1) ** ell
                * qm1 ** (j - ell)
                * comb(i, ell)
                * comb(n - i, j - ell)
            )
    return value


def krawtchouk_matrix(n, q):
    return [
        [krawtchouk(n, j, i, q) for i in range(n + 1)]
        for j in range(n + 1)
    ]


def shadow_transform_coefficient(n, j, i, q):
    r"""
    Coefficient of x^(n-j)y^j in

        ((q-1)x + (q+1)y)^(n-i) (y-x)^i.

    With the normalized A-enumerator used here,

        S(x,y) = q^{-(n-k)} A((q-1)x+(q+1)y, y-x).

    For q=2 this equals (-1)^i K_j(i;n), recovering SLR/qudit exactly.
    """
    value = 0
    for b in range(i + 1):
        a = j - b
        if 0 <= a <= n - i:
            value += (
                comb(n - i, a)
                * (q - 1) ** (n - i - a)
                * (q + 1) ** a
                * comb(i, b)
                * (-1) ** (i - b)
            )
    return value


def shadow_transform_matrix(n, q):
    return [
        [shadow_transform_coefficient(n, j, i, q) for i in range(n + 1)]
        for j in range(n + 1)
    ]


def active_cases(q):
    # Binary Hamming weight parity is a quadratic refinement of the
    # symplectic form, producing the exact Type-I/II/III three cases.
    # For q>2 there is no direct ordinary-Hamming analogue.
    return (1, 2, 3) if q == 2 else (0,)


def case_name(q, case):
    if q == 2:
        return f"Parity Case {case}"
    return "General qudit case"

# ============================================================
# Small exact-linear-algebra helpers
# ============================================================

def gcd_many(values):
    g = 0

    for value in values:
        g = math.gcd(g, abs(int(value)))

    return g


def exact_rref_parameterization(nvars, eq_rows, eq_rhs):
    """
    Solve all exact equalities

        M A = b

    by rational RREF.

    Each original A_j is represented as

        A_j = const_j + sum_r coeff[j,r] F_r,

    where the F_r are free ORIGINAL A coordinates.
    """
    if not eq_rows:
        exprs = []

        for j in range(nvars):
            coeff = [Fraction(0)] * nvars
            coeff[j] = Fraction(1)
            exprs.append((Fraction(0), tuple(coeff)))

        return {
            "inconsistent": False,
            "rank": 0,
            "pivots": tuple(),
            "free_cols": list(range(nvars)),
            "exprs": exprs,
            "denominator_set": [1],
            "max_den": 1,
        }

    M = sp.Matrix(eq_rows)
    b = sp.Matrix(eq_rhs)
    Aug = M.row_join(b)

    D = DomainMatrix.from_Matrix(Aug)
    R, pivots_all = D.rref(method="FF")
    RM = R.to_Matrix()

    # Pivot in augmented RHS column => inconsistent.
    if nvars in pivots_all:
        return {
            "inconsistent": True,
            "rank": len([p for p in pivots_all if p < nvars]),
        }

    pivots = tuple(p for p in pivots_all if p < nvars)
    pivot_set = set(pivots)
    free_cols = [j for j in range(nvars) if j not in pivot_set]
    free_pos = {col: r for r, col in enumerate(free_cols)}
    nfree = len(free_cols)

    exprs = [None] * nvars

    # Free columns are literal original A variables.
    for col in free_cols:
        coeff = [Fraction(0)] * nfree
        coeff[free_pos[col]] = Fraction(1)
        exprs[col] = (Fraction(0), tuple(coeff))

    # Pivot expressions from RREF.
    for row_index, pivot_col in enumerate(pivots):
        const = Fraction(RM[row_index, nvars])
        coeff = []

        for free_col in free_cols:
            coeff.append(-Fraction(RM[row_index, free_col]))

        exprs[pivot_col] = (const, tuple(coeff))

    denominator_set = set()
    max_den = 1

    for const, coeff in exprs:
        den = const.denominator

        for c in coeff:
            den = math.lcm(den, c.denominator)

        denominator_set.add(den)
        max_den = max(max_den, den)

    return {
        "inconsistent": False,
        "rank": len(pivots),
        "pivots": pivots,
        "free_cols": free_cols,
        "exprs": exprs,
        "denominator_set": sorted(denominator_set),
        "max_den": max_den,
    }


def substitute_original_row(row, rhs, exprs, nfree):
    """
    Substitute A_j affine expressions into

        row . A - rhs.

    Returns
        const + coeff . F.
    """
    const = Fraction(-rhs)
    coeff = [Fraction(0)] * nfree

    for col, scale in enumerate(row):
        if scale == 0:
            continue

        e_const, e_coeff = exprs[col]
        const += Fraction(scale) * e_const

        for r, c in enumerate(e_coeff):
            coeff[r] += Fraction(scale) * c

    return const, tuple(coeff)


def integerize_affine(const, coeff):
    """
    Write a rational affine expression as

        (c0 + sum ci F_i) / den,

    with den>0 and all integers.
    """
    den = const.denominator

    for c in coeff:
        den = math.lcm(den, c.denominator)

    c0 = int(const * den)
    ci = tuple(int(c * den) for c in coeff)

    g = gcd_many([den, c0] + list(ci))

    if g > 1:
        den //= g
        c0 //= g
        ci = tuple(c // g for c in ci)

    return c0, ci, den


def canonical_linear(c0, ci):
    """
    Divide a linear inequality by the common positive gcd.
    """
    g = gcd_many([c0] + list(ci))

    if g > 1:
        c0 //= g
        ci = tuple(c // g for c in ci)

    return c0, ci


def z3_linear(c0, ci, free_vars):
    terms = []

    if c0:
        terms.append(c0)

    for c, var in zip(ci, free_vars):
        if c:
            terms.append(c * var)

    if not terms:
        return 0

    if len(terms) == 1:
        return terms[0]

    return Sum(terms)


# ============================================================
# Build the exact A-only mathematical problem
# ============================================================

def build_A_only_problem(q, n, k, d, case, K, SH, *, exact_distance=None,
                         assume_fq_linear=False):
    """
    Return the A-only exact feasibility system.

    B_j is eliminated using the q^2-ary MacWilliams transform. The
    generalized shadow S_j is eliminated using Rains' local-dimension-q
    shadow transform. B is always integral for stabilizer codes. For q>2
    the generalized shadow coefficients are used only through positivity;
    binary shadow integrality/coset-capacity is retained when q=2.
    """
    if case not in active_cases(q):
        raise ValueError(f"invalid case={case} for q={q}")

    if exact_distance is None:
        exact_distance = REQUIRE_EXACT_DISTANCE

    if q == 2 and n - k < 1:
        raise ValueError("The binary three-case formulas require n-k >= 1.")

    N = n + 1
    normalization = q ** (n - k)

    # B_j = NB_j / q^(n-k), S_j = NS_j / q^(n-k).
    NB_rows = [list(K[j]) for j in range(N)]
    NS_rows = [list(SH[j]) for j in range(N)]

    eq_rows = []
    eq_rhs = []
    inequalities = []
    congruences = []

    # A_0 = 1.
    row = [0] * N
    row[0] = 1
    eq_rows.append(row)
    eq_rhs.append(1)

    # sum A = |S| = q^(n-k).
    eq_rows.append([1] * N)
    eq_rhs.append(normalization)

    # Distance A_j=B_j for j<d.
    for j in range(d):
        row = NB_rows[j][:]
        row[j] -= normalization
        eq_rows.append(row)
        eq_rhs.append(0)

    # Exact minimum distance d when k>0.
    if exact_distance and k > 0:
        row = NB_rows[d][:]
        row[d] -= normalization
        inequalities.append((row, normalization))

    # Binary Type-I/II/III Hamming parity cases.
    if q == 2:
        even_indices = list(range(0, N, 2))
        odd_indices = list(range(1, N, 2))

        if case == 1:
            row = [0] * N
            for j in even_indices:
                row[j] = 1
            eq_rows.append(row)
            eq_rhs.append(2 ** (n - k - 1))

            row = [0] * N
            for j in even_indices:
                for i in range(N):
                    row[i] += NB_rows[j][i]
            eq_rows.append(row)
            eq_rhs.append(normalization * 2 ** (n + k - 1))

        elif case in (2, 3):
            for j in odd_indices:
                row = [0] * N
                row[j] = 1
                eq_rows.append(row)
                eq_rhs.append(0)

            sign = +1 if case == 2 else -1
            target_even_B = 2 ** (n - k - 1) * (4 ** k + sign * 2 ** k)

            row = [0] * N
            for j in even_indices:
                for i in range(N):
                    row[i] += NB_rows[j][i]
            eq_rows.append(row)
            eq_rhs.append(normalization * target_even_B)

    # B>=A.
    for j in range(N):
        row = NB_rows[j][:]
        row[j] -= normalization
        inequalities.append((row, 0))

    # Generalized-shadow positivity.
    for j in range(N):
        inequalities.append((NS_rows[j][:], 0))

    # A/B shell capacities. Binary shadow is also an affine Pauli coset.
    if USE_SHELL_CAPACITY_BOUNDS:
        for j in range(N):
            capacity = ((q * q - 1) ** j) * comb(n, j)

            rowA = [0] * N
            rowA[j] = -1
            inequalities.append((rowA, -capacity))

            inequalities.append(
                ([-c for c in NB_rows[j]], -normalization * capacity)
            )

            if q == 2 and USE_BINARY_SHADOW_CAPACITY:
                inequalities.append(
                    ([-c for c in NS_rows[j]], -normalization * capacity)
                )

    # B-integrality.
    for j in range(N):
        congruences.append((NB_rows[j][:], 0, normalization))

    # Binary shadow integrality.
    if q == 2 and USE_BINARY_SHADOW_INTEGRALITY:
        for j in range(N):
            congruences.append((NS_rows[j][:], 0, normalization))

    # Optional F_q-linear scalar-orbit divisibility.
    if assume_fq_linear and q > 2:
        orbit = q - 1
        for j in range(1, N):
            rowA = [0] * N
            rowA[j] = 1
            congruences.append((rowA, 0, orbit))
            congruences.append(
                (NB_rows[j][:], 0, normalization * orbit)
            )

    return {
        "q": q,
        "nvars": N,
        "normalization": normalization,
        "NB_rows": NB_rows,
        "NS_rows": NS_rows,
        "shadow_integral": (q == 2 and USE_BINARY_SHADOW_INTEGRALITY),
        "assume_fq_linear": bool(assume_fq_linear),
        "eq_rows": eq_rows,
        "eq_rhs": eq_rhs,
        "inequalities": inequalities,
        "congruences": congruences,
    }


# ============================================================
# Build reduced QF_LIA solver
# ============================================================

def build_reduced_solver(problem, param):
    nvars = problem["nvars"]
    normalization = problem["normalization"]

    free_cols = param["free_cols"]
    free_col_set = set(free_cols)
    exprs = param["exprs"]
    nfree = len(free_cols)

    free_vars = [
        Int(f"F_{r}_A{col}")
        for r, col in enumerate(free_cols)
    ]

    solver = SolverFor("QF_LIA")

    if TIMEOUT_MS is not None:
        solver.set(timeout=TIMEOUT_MS)

    # --------------------------------------------------------
    # Every free coordinate is an original A_j.
    #
    # From A>=0 and sum A=normalization:
    #
    #     0 <= A_j <= normalization.
    # --------------------------------------------------------

    for var in free_vars:
        solver.add(var >= 0)
        solver.add(var <= normalization)

    # --------------------------------------------------------
    # Pivot A_j: integrality + 0<=A_j<=normalization.
    # --------------------------------------------------------

    pivot_integrality = set()
    pivot_bounds = 0

    for col in range(nvars):
        if col in free_col_set:
            continue

        const, coeff = exprs[col]
        c0, ci, den = integerize_affine(const, coeff)
        lin = z3_linear(c0, ci, free_vars)

        if den > 1:
            signature = (
                c0 % den,
                tuple(c % den for c in ci),
                den,
            )

            if signature not in pivot_integrality:
                solver.add(lin % den == 0)
                pivot_integrality.add(signature)

        solver.add(lin >= 0)
        solver.add(lin <= den * normalization)
        pivot_bounds += 2

    # --------------------------------------------------------
    # B/S integrality congruences after substitution.
    # --------------------------------------------------------

    unique_modular = set()
    trivial_modular = 0
    impossible_modular = 0

    for row, rhs, modulus in problem["congruences"]:
        const, coeff = substitute_original_row(
            row,
            rhs,
            exprs,
            nfree,
        )

        c0, ci, den = integerize_affine(const, coeff)

        # (c0 + ci.F)/den == 0 mod modulus
        # iff numerator == 0 mod den*modulus.
        mod = den * modulus

        if all(c == 0 for c in ci):
            if c0 % mod == 0:
                trivial_modular += 1
            else:
                impossible_modular += 1
                solver.add(False)
            continue

        signature = (
            c0 % mod,
            tuple(c % mod for c in ci),
            mod,
        )

        if signature in unique_modular:
            continue

        unique_modular.add(signature)

        solver.add(
            z3_linear(c0, ci, free_vars) % mod == 0
        )

    # --------------------------------------------------------
    # B>=A and shadow positivity after substitution.
    # --------------------------------------------------------

    unique_ineq = set()
    trivial_true = 0
    trivial_false = 0

    for row, rhs in problem["inequalities"]:
        const, coeff = substitute_original_row(
            row,
            rhs,
            exprs,
            nfree,
        )

        c0, ci, den = integerize_affine(const, coeff)

        # den>0, so denominator may be dropped.
        c0, ci = canonical_linear(c0, ci)

        if all(c == 0 for c in ci):
            if c0 >= 0:
                trivial_true += 1
            else:
                trivial_false += 1
                solver.add(False)
            continue

        signature = (c0, ci)

        if signature in unique_ineq:
            continue

        unique_ineq.add(signature)

        solver.add(
            z3_linear(c0, ci, free_vars) >= 0
        )

    meta = {
        "free_vars": len(free_vars),
        "pivot_integrality": len(pivot_integrality),
        "unique_modular": len(unique_modular),
        "trivial_modular": trivial_modular,
        "impossible_modular": impossible_modular,
        "unique_ineq": len(unique_ineq),
        "trivial_true": trivial_true,
        "trivial_false": trivial_false,
        "pivot_bounds": pivot_bounds,
    }

    return solver, free_vars, meta


# ============================================================
# Reconstruct and verify one A solution
# ============================================================

def eval_affine_model(model, free_vars, expr):
    const, coeff = expr
    value = const

    for c, var in zip(coeff, free_vars):
        if c:
            value += (
                c
                * model.eval(
                    var,
                    model_completion=True,
                ).as_long()
            )

    if value.denominator != 1:
        raise AssertionError(
            f"Reconstructed A coefficient is nonintegral: {value}"
        )

    return value.numerator


def reconstruct_A(model, free_vars, param):
    return [
        eval_affine_model(model, free_vars, expr)
        for expr in param["exprs"]
    ]


def compute_B_S(A, problem):
    normalization = problem["normalization"]
    B = []
    S = []

    for row in problem["NB_rows"]:
        numerator = sum(c * a for c, a in zip(row, A))
        if numerator % normalization != 0:
            raise AssertionError("Internal error: reconstructed B is nonintegral.")
        B.append(numerator // normalization)

    for row in problem["NS_rows"]:
        numerator = sum(c * a for c, a in zip(row, A))
        if problem["shadow_integral"]:
            if numerator % normalization != 0:
                raise AssertionError(
                    "Internal error: reconstructed binary shadow is nonintegral."
                )
            S.append(numerator // normalization)
        else:
            S.append(Fraction(numerator, normalization))

    return B, S


def verify_solution(q, n, k, d, case, A, B, S, *, assume_fq_linear=False):
    normalization = q ** (n - k)

    assert A[0] == 1
    assert sum(A) == normalization
    assert all(a >= 0 for a in A)
    assert all(b >= 0 for b in B)
    assert all(s >= 0 for s in S)
    assert all(b >= a for a, b in zip(A, B))

    for j in range(d):
        assert A[j] == B[j]

    if REQUIRE_EXACT_DISTANCE and k > 0:
        assert B[d] >= A[d] + 1

    if USE_SHELL_CAPACITY_BOUNDS:
        for j in range(n + 1):
            capacity = ((q * q - 1) ** j) * comb(n, j)
            assert A[j] <= capacity
            assert B[j] <= capacity
            if q == 2 and USE_BINARY_SHADOW_CAPACITY:
                assert S[j] <= capacity

    if q == 2:
        even_A = sum(A[j] for j in range(0, n + 1, 2))
        even_B = sum(B[j] for j in range(0, n + 1, 2))

        if case == 1:
            assert even_A == 2 ** (n - k - 1)
            assert even_B == 2 ** (n + k - 1)
        elif case == 2:
            assert all(A[j] == 0 for j in range(1, n + 1, 2))
            assert even_B == 2 ** (n - k - 1) * (4 ** k + 2 ** k)
        elif case == 3:
            assert all(A[j] == 0 for j in range(1, n + 1, 2))
            assert even_B == 2 ** (n - k - 1) * (4 ** k - 2 ** k)

    if assume_fq_linear and q > 2:
        for j in range(1, n + 1):
            assert A[j] % (q - 1) == 0
            assert B[j] % (q - 1) == 0


def compact_A(A):
    return {
        j: value
        for j, value in enumerate(A)
        if value != 0
    }


# ============================================================
# Enumerate all distinct A distributions for one case
# ============================================================

def enumerate_A_solutions(q, n, k, d, case, K, SH, *, assume_fq_linear=False):
    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Mathematical A-only system.
    # --------------------------------------------------------

    build_start = time.perf_counter()

    if SHOW_LIVE_PROGRESS:
        print("  [1/4] Building A-only constraints...", flush=True)

    problem = build_A_only_problem(
        q, n, k, d, case, K, SH,
        exact_distance=REQUIRE_EXACT_DISTANCE,
        assume_fq_linear=assume_fq_linear,
    )

    build_time = time.perf_counter() - build_start

    # --------------------------------------------------------
    # Exact equality elimination.
    # --------------------------------------------------------

    rref_start = time.perf_counter()

    if SHOW_LIVE_PROGRESS:
        print(
            f"  [2/4] Exact RREF on {len(problem['eq_rows'])} equality rows...",
            flush=True,
        )

    param = exact_rref_parameterization(
        problem["nvars"],
        problem["eq_rows"],
        problem["eq_rhs"],
    )

    rref_time = time.perf_counter() - rref_start

    if param["inconsistent"]:
        elapsed = time.perf_counter() - total_start

        meta = {
            "build_time": build_time,
            "rref_time": rref_time,
            "reduce_time": 0.0,
            "solve_time": 0.0,
            "rank": param["rank"],
            "free": None,
            "denominators": [],
            "reduced_meta": None,
        }

        return unsat, [], elapsed, meta

    # --------------------------------------------------------
    # Reduced Z3 system.
    # --------------------------------------------------------

    reduce_start = time.perf_counter()

    if SHOW_LIVE_PROGRESS:
        print(
            f"  [3/4] Building reduced integer system "
            f"(rank={param['rank']}, free A={len(param['free_cols'])})...",
            flush=True,
        )

    solver, free_vars, reduced_meta = build_reduced_solver(
        problem,
        param,
    )

    reduce_time = time.perf_counter() - reduce_start

    # --------------------------------------------------------
    # Enumerate reduced free-coordinate models.
    # --------------------------------------------------------

    solve_start = time.perf_counter()

    solutions = []
    last_progress_time = solve_start
    last_progress_count = 0

    if SHOW_LIVE_PROGRESS:
        print(
            f"  [4/4] Enumerating distinct A distributions "
            f"with {len(free_vars)} free integer coordinate(s)...",
            flush=True,
        )

    while True:
        result = solver.check()

        if result == sat:
            model = solver.model()

            # Reconstruct the full A-distribution only if the caller needs
            # it, it will be printed, or it is explicitly verified.  For
            # pure counting, the free coordinates already identify the A
            # distribution uniquely, so this Python-side work is unnecessary.
            if (
                COLLECT_SOLUTIONS
                or PRINT_SOLUTIONS
                or VERIFY_EACH_SOLUTION
                or SAVE_CANDIDATE_DATABASE
            ):
                A_sol = reconstruct_A(
                    model,
                    free_vars,
                    param,
                )

                if VERIFY_EACH_SOLUTION:
                    B_sol, S_sol = compute_B_S(
                        A_sol,
                        problem,
                    )

                    verify_solution(
                        q, n, k, d, case,
                        A_sol, B_sol, S_sol,
                        assume_fq_linear=assume_fq_linear,
                    )

                solutions.append(A_sol)
            else:
                # Placeholder only; len(solutions) remains the exact
                # number of distinct A-distributions found.
                solutions.append(None)

            if SHOW_LIVE_PROGRESS:
                now = time.perf_counter()
                count_trigger = (
                    PROGRESS_EVERY_SOLUTIONS is not None
                    and len(solutions) - last_progress_count
                    >= PROGRESS_EVERY_SOLUTIONS
                )
                time_trigger = (
                    PROGRESS_EVERY_SECONDS is not None
                    and now - last_progress_time
                    >= PROGRESS_EVERY_SECONDS
                )

                if count_trigger or time_trigger:
                    elapsed_enum = now - solve_start
                    rate = (
                        len(solutions) / elapsed_enum
                        if elapsed_enum > 0
                        else 0.0
                    )
                    print(
                        f"        found {len(solutions)} solution(s) "
                        f"[{elapsed_enum:.1f} s, {rate:.2f} sol/s]",
                        flush=True,
                    )
                    last_progress_time = now
                    last_progress_count = len(solutions)

            if (
                MAX_SOLUTIONS is not None
                and len(solutions) >= MAX_SOLUTIONS
            ):
                result = sat
                break

            # The free coordinates are ORIGINAL A coordinates and
            # determine the full A distribution uniquely.
            if free_vars:
                current_free = [
                    model.eval(
                        var,
                        model_completion=True,
                    ).as_long()
                    for var in free_vars
                ]

                solver.add(
                    Or(
                        *[
                            var != value
                            for var, value in zip(
                                free_vars,
                                current_free,
                            )
                        ]
                    )
                )
            else:
                # Zero-dimensional affine space: at most one A.
                solver.add(False)

        elif result == unsat:
            break

        else:
            solve_time = time.perf_counter() - solve_start
            elapsed = time.perf_counter() - total_start

            meta = {
                "build_time": build_time,
                "rref_time": rref_time,
                "reduce_time": reduce_time,
                "solve_time": solve_time,
                "rank": param["rank"],
                "free": len(param["free_cols"]),
                "denominators": param["denominator_set"],
                "reduced_meta": reduced_meta,
                "reason_unknown": solver.reason_unknown(),
            }

            return unknown, solutions, elapsed, meta

    solve_time = time.perf_counter() - solve_start

    if SHOW_LIVE_PROGRESS:
        rate = (
            len(solutions) / solve_time
            if solve_time > 0
            else 0.0
        )
        print(
            f"        enumeration complete: {len(solutions)} solution(s) "
            f"[{solve_time:.3f} s, {rate:.2f} sol/s]",
            flush=True,
        )

    elapsed = time.perf_counter() - total_start

    meta = {
        "build_time": build_time,
        "rref_time": rref_time,
        "reduce_time": reduce_time,
        "solve_time": solve_time,
        "rank": param["rank"],
        "free": len(param["free_cols"]),
        "free_cols": param["free_cols"],
        "denominators": param["denominator_set"],
        "reduced_meta": reduced_meta,
    }

    if solutions:
        return sat, solutions, elapsed, meta

    return unsat, [], elapsed, meta


# ============================================================
# Reporting
# ============================================================

def print_case_detail(case, result, solutions, elapsed, meta):
    if PRINT_PREPROCESSING:
        print(
            f"  preprocessing: rank={meta['rank']}, "
            + (
                f"free A={meta['free']}, "
                if meta["free"] is not None
                else ""
            )
            + (
                "denominators="
                + ",".join(map(str, meta["denominators"]))
                + ", "
                if meta["denominators"]
                else ""
            )
            + f"build={meta['build_time']:.4f}s, "
            + f"RREF={meta['rref_time']:.4f}s, "
            + f"reduce={meta['reduce_time']:.4f}s, "
            + f"Z3={meta['solve_time']:.4f}s"
        )

        reduced = meta.get("reduced_meta")

        if reduced is not None:
            print(
                "  reduced system: "
                f"pivot congruences={reduced['pivot_integrality']}, "
                f"B/S congruences={reduced['unique_modular']}, "
                f"inequalities={reduced['unique_ineq']}"
            )

    if result == sat and PRINT_SOLUTIONS:
        for idx, A in enumerate(solutions, start=1):
            print(
                f"  Solution {idx}: "
                f"FIXED_A = {compact_A(A)}"
            )


def case_summary(q, case, result, count, elapsed):
    if result == sat:
        if MAX_SOLUTIONS is not None and count >= MAX_SOLUTIONS:
            return (
                f"{case_name(q, case)}: SAT "
                f"(at least {count} solution(s); enumeration capped) "
                f"[{elapsed:.6f} s]"
            )

        return (
            f"{case_name(q, case)}: SAT "
            f"({count} solution(s)) "
            f"[{elapsed:.6f} s]"
        )

    if result == unsat:
        return (
            f"{case_name(q, case)}: UNSAT "
            f"(0 solutions) "
            f"[{elapsed:.6f} s]"
        )

    return (
        f"{case_name(q, case)}: UNKNOWN "
        f"({count} solution(s) found before UNKNOWN) "
        f"[{elapsed:.6f} s]"
    )


# ============================================================
# Persistent executable candidate database
# ============================================================

REFINED_SOLVER_KEYS = (
    "hxrx_logical_reference",
    "hxrx_stabilizer_reference",
    "hxry_stabilizer_reference",
    "rxry_stabilizer_reference",
    "hxr_stabilizer_reference",
    "hxr_twisted_stabilizer_reference",
)


def result_label(result):
    if result == sat:
        return "SAT"
    if result == unsat:
        return "UNSAT"
    if result == unknown:
        return "UNKNOWN"
    return str(result).upper()


def candidate_output_path(q, n, k, d, assume_fq_linear=False):
    if CANDIDATE_FILENAME is None:
        filename = f"candidates_q{q}_n{n}_k{k}_d{d}" + ("_Fqlinear" if assume_fq_linear else "") + ".py"
    else:
        filename = str(CANDIDATE_FILENAME).format(q=q, n=n, k=k, d=d, fqlinear=int(bool(assume_fq_linear)))

    if CANDIDATE_OUTPUT_DIR is None:
        directory = Path.cwd()
    else:
        directory = Path(CANDIDATE_OUTPUT_DIR)

    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def _database_runtime_code():
    return r'''\

# =====================================================================
# Runtime helpers. Refined solvers may import this module and call
# pending_candidates(...) / record_test(...). This file is the single
# persistent source of candidate state for this [[n,k,d]]_q problem.
# =====================================================================

import argparse
import os
from pathlib import Path
from pprint import pformat

VALID_TEST_STATUS = {
    "not_tested", "running", "sat", "unsat", "unknown", "error"
}

_DB_BEGIN = "# === CANDIDATE DATABASE BEGIN ==="
_DB_END = "# === CANDIDATE DATABASE END ==="


def _refresh_exclusion(candidate):
    excluded_by = [
        name for name, info in candidate["tests"].items()
        if info.get("status") == "unsat"
    ]
    candidate["excluded_by"] = excluded_by
    candidate["excluded"] = bool(excluded_by)


def current_survivors(case=None):
    out = []
    for candidate in CANDIDATES:
        _refresh_exclusion(candidate)
        if candidate["excluded"]:
            continue
        if case is not None and candidate["case"] != int(case):
            continue
        out.append(candidate)
    return out


def pending_candidates(solver_name, solver_version=None, retry_unknown=True,
                       retry_error=True, include_excluded=False):
    if solver_name not in REFINED_SOLVER_KEYS:
        raise ValueError(f"Unknown solver_name: {solver_name}")

    pending = []
    for candidate in CANDIDATES:
        _refresh_exclusion(candidate)
        excluded_by_other = any(
            name != solver_name and test.get("status") == "unsat"
            for name, test in candidate["tests"].items()
        )
        if excluded_by_other and not include_excluded:
            continue

        info = candidate["tests"][solver_name]
        status = info.get("status", "not_tested")
        old_version = info.get("version")

        if status == "not_tested":
            pending.append(candidate)
            continue
        if status == "unknown" and retry_unknown:
            pending.append(candidate)
            continue
        if status == "error" and retry_error:
            pending.append(candidate)
            continue
        if status == "running":
            pending.append(candidate)
            continue
        if status in ("sat", "unsat"):
            if solver_version is not None and old_version != solver_version:
                pending.append(candidate)
            continue

    return pending


def _atomic_save():
    path = Path(__file__).resolve()
    text = path.read_text(encoding="utf-8")
    left = text.index(_DB_BEGIN) + len(_DB_BEGIN)
    right = text.index(_DB_END, left)

    database_text = "\nDATABASE = " + pformat(
        DATABASE, sort_dicts=False, width=120
    ) + "\n"
    new_text = text[:left] + database_text + text[right:]

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)


def record_test(candidate_id, solver_name, status, *, version=None,
                runtime=None, details=None, save=True):
    if solver_name not in REFINED_SOLVER_KEYS:
        raise ValueError(f"Unknown solver_name: {solver_name}")
    if status not in VALID_TEST_STATUS:
        raise ValueError(f"Invalid status: {status}")

    matches = [c for c in CANDIDATES if c["id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"Candidate id not unique/found: {candidate_id}")

    candidate = matches[0]
    info = candidate["tests"][solver_name]
    info["status"] = status
    info["version"] = version
    info["runtime"] = runtime
    info["details"] = details
    _refresh_exclusion(candidate)

    if save:
        _atomic_save()
    return candidate


def _fixed_A_text(fixed_A):
    return "{" + ", ".join(
        f"{int(w)}:{int(v)}" for w, v in fixed_A.items()
    ) + "}"


def _case_label(case):
    return f"Case {case}" if Q == 2 else "General qudit case"


def print_summary(show_status=False):
    survivors = current_survivors()
    excluded = len(CANDIDATES) - len(survivors)

    print("=" * 72)
    print(f"Candidate database for [[{N},{K},{D}]]_{Q}")
    print("=" * 72)
    print(f"Initial SLR candidates  : {len(CANDIDATES)}")
    print(f"Currently excluded      : {excluded}")
    print(f"Current survivors       : {len(survivors)}")
    print(f"F_q-linear assumption   : {ASSUME_FQ_LINEAR}")
    print()
    print("By case:")
    for case in CASE_LABELS:
        initial = sum(c["case"] == case for c in CANDIDATES)
        alive = len(current_survivors(case))
        print(f"  {_case_label(case)}: {alive} survivor(s) / {initial} initial")

    if show_status:
        print()
        print("Refined-solver status over all initial candidates:")
        order = ["not_tested", "running", "sat", "unsat", "unknown", "error"]
        for solver_name in REFINED_SOLVER_KEYS:
            counts = {status: 0 for status in order}
            for candidate in CANDIDATES:
                status = candidate["tests"][solver_name].get(
                    "status", "not_tested"
                )
                counts.setdefault(status, 0)
                counts[status] += 1
            shown = ", ".join(
                f"{status}={counts[status]}" for status in order
                if counts.get(status, 0)
            )
            print(f"  {solver_name:30s}: {shown}")
    print("=" * 72)


def print_candidates(*, show_all=False, case=None):
    rows = CANDIDATES if show_all else current_survivors(case)
    if show_all and case is not None:
        rows = [c for c in rows if c["case"] == int(case)]

    if not rows:
        print("No matching candidates.")
        return

    for candidate in rows:
        _refresh_exclusion(candidate)
        suffix = ""
        if show_all:
            if candidate["excluded"]:
                suffix = "  [EXCLUDED by " + ",".join(candidate["excluded_by"]) + "]"
            else:
                suffix = "  [SURVIVOR]"
        print(
            f"{candidate['id']} ({_case_label(candidate['case'])}): "
            f"FIXED_A = {_fixed_A_text(candidate['fixed_A'])}{suffix}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=f"Inspect persistent SLR/qudit candidates for [[{N},{K},{D}]]_{Q}."
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--show-case", type=int, choices=tuple(CASE_LABELS), default=None)
    args = parser.parse_args()

    print_summary(show_status=args.status)
    if args.show or args.all:
        print_candidates(show_all=args.all, case=args.show_case)


if __name__ == "__main__":
    main()
'''


def save_candidate_database(q, n, k, d, all_results, *, assume_fq_linear=False):
    cases = active_cases(q)
    path = candidate_output_path(q, n, k, d, assume_fq_linear)

    if path.exists():
        raise FileExistsError(
            f"Candidate database already exists: {path}\n"
            "Refusing to overwrite refined-solver history. Delete/rename it "
            "explicitly if you intentionally want to rebuild the baseline."
        )

    case_complete = {}
    for case in cases:
        info = all_results[case]
        capped = (
            MAX_SOLUTIONS is not None
            and info["result"] == sat
            and info["count"] >= MAX_SOLUTIONS
        )
        case_complete[case] = info["result"] != unknown and not capped

    complete = all(case_complete.values())
    candidates = []

    for case in cases:
        for idx, A in enumerate(all_results[case]["solutions"], start=1):
            if A is None:
                raise RuntimeError(
                    "Cannot save candidate database because a solution was not reconstructed."
                )
            candidates.append({
                "id": f"c{case}_{idx:04d}" if q == 2 else f"q{q}_{idx:04d}",
                "case": case,
                "fixed_A": compact_A(A),
                "excluded": False,
                "excluded_by": [],
                "tests": {
                    name: {
                        "status": "not_tested",
                        "version": None,
                        "runtime": None,
                        "details": None,
                    }
                    for name in REFINED_SOLVER_KEYS
                },
            })

    database = {
        "schema_version": 2,
        "code": {"q": q, "n": n, "k": k, "d": d},
        "ordinary_solver": {
            "name": "SLR_qudit",
            "exact_distance": bool(REQUIRE_EXACT_DISTANCE and k > 0),
            "shell_capacity_bounds": bool(USE_SHELL_CAPACITY_BOUNDS),
            "fq_linear_assumption": bool(assume_fq_linear),
            "complete": complete,
            "case_status": {
                case: result_label(all_results[case]["result"])
                for case in cases
            },
            "case_complete": case_complete,
            "case_counts": {
                case: all_results[case]["count"]
                for case in cases
            },
        },
        "refined_solver_keys": list(REFINED_SOLVER_KEYS),
        "candidates": candidates,
    }

    import pprint
    db_literal = pprint.pformat(database, sort_dicts=False, width=120)

    header = f'''#!/usr/bin/env python3
# Auto-generated by SLR_qudit.py.
# Persistent candidate database for [[{n},{k},{d}]]_{q}.

Q = {q}
N = {n}
K = {k}
D = {d}
ASSUME_FQ_LINEAR = {bool(assume_fq_linear)!r}
CASE_LABELS = {tuple(cases)!r}
REFINED_SOLVER_KEYS = {REFINED_SOLVER_KEYS!r}

# === CANDIDATE DATABASE BEGIN ===
DATABASE = {db_literal}
# === CANDIDATE DATABASE END ===

CANDIDATES = DATABASE["candidates"]
'''

    path.write_text(header + _database_runtime_code(), encoding="utf-8")
    return path


# ============================================================
# Check one parameter set and enumerate ALL A distributions
# ============================================================

def load_existing_candidate_database(path):
    spec = importlib.util.spec_from_file_location(
        f"slr3_candidates_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate database: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_one(q, n, k, d, *, assume_fq_linear=False):
    cases = active_cases(q)

    if SAVE_CANDIDATE_DATABASE:
        existing_path = candidate_output_path(q, n, k, d, assume_fq_linear)
        if existing_path.exists():
            print("=" * 76)
            print(f"Existing candidate database found for [[{n},{k},{d}]]_{q}")
            print(f"{existing_path}")
            print("SLR/qudit enumeration will NOT be rerun.")
            print("=" * 76)
            db = load_existing_candidate_database(existing_path)
            if hasattr(db, "print_summary"):
                db.print_summary(show_status=True)
            print(f"To list current survivors: python {existing_path.name} --show")
            return {"existing_database": str(existing_path)}

    print("=" * 76)
    if q == 2:
        print("SLR/qudit: Shor-Laflamme + Rains shadow + binary three cases")
    else:
        print("SLR/qudit: q-ary Shor-Laflamme + generalized Rains shadow")
    print(f"Parameters: [[{n},{k},{d}]]_{q}")
    print(
        "Distance mode: "
        + ("exact d (B_d>A_d)" if REQUIRE_EXACT_DISTANCE and k > 0 else "at least d")
    )
    print("Shell capacities: " + ("enabled" if USE_SHELL_CAPACITY_BOUNDS else "disabled"))
    print(f"F_q-linear scalar-orbit divisibility: {bool(assume_fq_linear)}")
    if q > 2:
        print("Binary Arf/parity cases: not imposed at ordinary Hamming level")
    print("Counting distinct ordinary A-weight distributions")
    print("=" * 76)

    Kmat = krawtchouk_matrix(n, q)
    SHmat = shadow_transform_matrix(n, q)

    all_results = {}
    for case in cases:
        print()
        print(case_name(q, case))
        print("-" * 76)

        result, solutions, elapsed, meta = enumerate_A_solutions(
            q, n, k, d, case, Kmat, SHmat,
            assume_fq_linear=assume_fq_linear,
        )

        all_results[case] = {
            "result": result,
            "count": len(solutions),
            "solutions": solutions,
            "elapsed": elapsed,
            "meta": meta,
        }

        print_case_detail(case, result, solutions, elapsed, meta)
        print(case_summary(q, case, result, len(solutions), elapsed))

    print()
    print("=" * 76)
    print("Summary")
    print("=" * 76)
    for case in cases:
        info = all_results[case]
        print(case_summary(q, case, info["result"], info["count"], info["elapsed"]))
    print("=" * 76)

    if SAVE_CANDIDATE_DATABASE:
        output_path = save_candidate_database(
            q, n, k, d, all_results, assume_fq_linear=assume_fq_linear
        )
        print(f"Candidate database saved to: {output_path}")
        print(f"Inspect later with: python {output_path.name}")

    return all_results


# ============================================================
# Fast feasibility-only scan over d
# ============================================================

def check_case_feasibility(q, n, k, d, case, Kmat, SHmat, *, assume_fq_linear=False):
    problem = build_A_only_problem(
        q, n, k, d, case, Kmat, SHmat,
        exact_distance=False,
        assume_fq_linear=assume_fq_linear,
    )

    param = exact_rref_parameterization(
        problem["nvars"], problem["eq_rows"], problem["eq_rhs"]
    )
    if param["inconsistent"]:
        return unsat

    solver, free_vars, reduced_meta = build_reduced_solver(problem, param)
    return solver.check()


def scan_d(q, n, k, *, assume_fq_linear=False):
    cases = active_cases(q)
    print("=" * 76)
    print("Fast q-ary Shor-Laflamme + generalized-shadow bound scan")
    print(f"Parameters: q={q}, n={n}, k={k}")
    print(f"F_q-linear scalar-orbit divisibility: {bool(assume_fq_linear)}")
    print("=" * 76)

    Kmat = krawtchouk_matrix(n, q)
    SHmat = shadow_transform_matrix(n, q)
    last_feasible_d = None

    for d in range(1, n + 1):
        start = time.perf_counter()
        case_results = [
            check_case_feasibility(
                q, n, k, d, case, Kmat, SHmat,
                assume_fq_linear=assume_fq_linear,
            )
            for case in cases
        ]
        elapsed = time.perf_counter() - start

        if q == 2:
            status = ", ".join(
                f"Case {case}={str(res).upper()}"
                for case, res in zip(cases, case_results)
            )
        else:
            status = f"General={str(case_results[0]).upper()}"

        print(f"[[{n},{k},{d}]]_{q} : {status} [{elapsed:.6f} s]")

        if sat in case_results:
            last_feasible_d = d
        elif all(r == unsat for r in case_results):
            break
        else:
            print("At least one case returned UNKNOWN.")
            break

    print()
    print("=" * 76)
    if last_feasible_d is not None:
        print(f"Last feasible d = {last_feasible_d}")
        if last_feasible_d < n:
            print(f"First infeasible d = {last_feasible_d + 1}")
    else:
        print("No feasible d found.")
    print("=" * 76)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exact q-ary Shor-Laflamme--Rains A-only feasibility solver."
    )
    parser.add_argument("q", type=int, help="local qudit dimension (prime power)")
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("d", type=int, nargs="?", default=None)
    parser.add_argument(
        "--fq-linear", action="store_true",
        help="assume the stabilizer is F_q-linear and add q-1 divisibility constraints"
    )
    args = parser.parse_args()

    q, n, k, d = args.q, args.n, args.k, args.d
    assume_fq_linear = bool(ASSUME_FQ_LINEAR_STABILIZER or args.fq_linear)

    if not is_prime_power(q):
        raise ValueError("q must be a prime power.")
    if not (0 <= k <= n):
        raise ValueError("k must satisfy 0 <= k <= n.")

    if d is None:
        scan_d(q, n, k, assume_fq_linear=assume_fq_linear)
    else:
        if not (1 <= d <= n):
            raise ValueError("d must satisfy 1 <= d <= n.")
        check_one(q, n, k, d, assume_fq_linear=assume_fq_linear)


if __name__ == "__main__":
    main()
