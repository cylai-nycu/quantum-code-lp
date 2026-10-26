#!/usr/bin/env python3
"""
SLR3_LP.py

Fast exact-integer A-only Shor-Laflamme--Rains three-case LP feasibility
solver for stabilizer codes, with enumeration of all distinct ordinary
A-weight distributions in the three parity cases.  The default exact-distance
mode also enforces B_d>A_d.  Survivors are saved into one persistent executable
candidate database for downstream refined solvers.

Main optimization
-----------------
The original formulation uses integer variables A_j, B_j, S_j.
Here B_j and S_j are eliminated exactly:

    N^B_j(A) = sum_i K_j(i;n) A_i
    N^S_j(A) = sum_i (-1)^i K_j(i;n) A_i

    B_j = N^B_j(A) / 2^(n-k)
    S_j = N^S_j(A) / 2^(n-k)

Their integrality is preserved by congruence constraints.  All exact
linear equalities in A are then eliminated by exact rational RREF before
Z3 is called.  Z3 sees only the free integer coordinates, congruences,
and reduced inequalities.

Dependencies
------------
    pip install sympy z3-solver

Usage
-----
    python SLR3_LP.py n k d
    python SLR3_LP.py n k

Examples
--------
    python SLR3_LP.py 23 1 9
    python SLR3_LP.py 27 15 5
    python SLR3_LP.py 20 1
"""

import sys
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
PROGRESS_EVERY_SECONDS = 5.0

# For explicit `python SLR3_LP.py n k d` survivor generation, interpret
# [[n,k,d]] as having exact minimum distance d (for k>0), so B_d>A_d.
# This matches the downstream logical-reference solver, which fixes a minimum
# logical operator of weight d.  The two-argument bound scan deliberately
# overrides this and tests distance >= d, preserving monotonicity in d.
REQUIRE_EXACT_DISTANCE = True

# Necessary shell-capacity bounds for stabilizer, normalizer, and shadow:
#     A_j, B_j, S_j <= 3^j * binom(n,j).
# These add only linear inequalities; no new variables.
USE_SHELL_CAPACITY_BOUNDS = True

# Automatically save all SLR3 survivors into ONE persistent executable
# candidate database. The three refined solvers update this same file.
SAVE_CANDIDATE_DATABASE = True
CANDIDATE_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "candidates" / "binary"
CANDIDATE_FILENAME = None


# ============================================================
# Krawtchouk polynomial / matrix
# ============================================================

def krawtchouk(n, j, i):
    value = 0

    for ell in range(j + 1):
        if ell <= i and j - ell <= n - i:
            value += (
                (-1) ** ell
                * 3 ** (j - ell)
                * comb(i, ell)
                * comb(n - i, j - ell)
            )

    return value


def krawtchouk_matrix(n):
    return [
        [krawtchouk(n, j, i) for i in range(n + 1)]
        for j in range(n + 1)
    ]


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

def build_A_only_problem(n, k, d, case, K, exact_distance=None):
    """
    Return:
      exact equalities in A,
      inequalities in A,
      modular constraints in A.

    This eliminates B,S exactly and then adds the enabled necessary
    stabilizer constraints (exact-distance and shell capacities).
    """
    if case not in (1, 2, 3):
        raise ValueError("case must be 1, 2, or 3.")

    if exact_distance is None:
        exact_distance = REQUIRE_EXACT_DISTANCE

    if n - k < 1:
        raise ValueError(
            "The parity-case formulas require n-k >= 1."
        )

    N = n + 1
    normalization = 2 ** (n - k)

    # MacWilliams and shadow numerators:
    #
    #   NB[j] = sum_i K[j][i] A_i
    #   NS[j] = sum_i (-1)^i K[j][i] A_i
    #
    NB_rows = [list(K[j]) for j in range(N)]
    NS_rows = [
        [((-1) ** i) * K[j][i] for i in range(N)]
        for j in range(N)
    ]

    eq_rows = []
    eq_rhs = []

    # Inequality convention:
    #     row . A >= rhs
    inequalities = []

    # Congruence convention:
    #     row . A == rhs (mod modulus)
    congruences = []

    # --------------------------------------------------------
    # A_0 = 1
    # --------------------------------------------------------

    row = [0] * N
    row[0] = 1
    eq_rows.append(row)
    eq_rhs.append(1)

    # --------------------------------------------------------
    # sum_j A_j = 2^(n-k)
    # --------------------------------------------------------

    eq_rows.append([1] * N)
    eq_rhs.append(normalization)

    # --------------------------------------------------------
    # Distance A_j = B_j for j<d
    #
    # NB_j(A) = normalization * A_j
    # --------------------------------------------------------

    for j in range(d):
        row = NB_rows[j][:]
        row[j] -= normalization

        eq_rows.append(row)
        eq_rhs.append(0)

    # If d denotes the exact minimum distance and k>0, there must be at
    # least one nontrivial logical operator of weight exactly d:
    #
    #     B_d - A_d >= 1.
    #
    # In numerator form:
    #
    #     NB_d(A) - normalization*A_d >= normalization.
    #
    # For k=0 there are no logical operators, so this condition is skipped.
    if exact_distance and k > 0:
        row = NB_rows[d][:]
        row[d] -= normalization
        inequalities.append((row, normalization))

    # --------------------------------------------------------
    # Parity cases
    # --------------------------------------------------------

    even_indices = list(range(0, N, 2))
    odd_indices = list(range(1, N, 2))

    if case == 1:
        row = [0] * N

        for j in even_indices:
            row[j] = 1

        eq_rows.append(row)
        eq_rhs.append(2 ** (n - k - 1))

        # sum_even B_j = 2^(n+k-1)
        #
        # sum_even NB_j(A)
        # = normalization * 2^(n+k-1)
        row = [0] * N

        for j in even_indices:
            for i in range(N):
                row[i] += NB_rows[j][i]

        eq_rows.append(row)
        eq_rhs.append(
            normalization * 2 ** (n + k - 1)
        )

    elif case in (2, 3):
        # A_j = 0 for odd j.
        for j in odd_indices:
            row = [0] * N
            row[j] = 1

            eq_rows.append(row)
            eq_rhs.append(0)

        sign = +1 if case == 2 else -1

        target_even_B = (
            2 ** (n - k - 1)
            * (4 ** k + sign * 2 ** k)
        )

        row = [0] * N

        for j in even_indices:
            for i in range(N):
                row[i] += NB_rows[j][i]

        eq_rows.append(row)
        eq_rhs.append(
            normalization * target_even_B
        )

    # --------------------------------------------------------
    # A >= 0
    #
    # Added after RREF as bounds on all original A variables.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # B >= A, including B >= 0 because A >= 0.
    #
    # NB_j(A) - normalization*A_j >= 0
    # --------------------------------------------------------

    for j in range(N):
        row = NB_rows[j][:]
        row[j] -= normalization

        inequalities.append((row, 0))

    # --------------------------------------------------------
    # Shadow positivity S_j >= 0:
    #
    # NS_j(A) >= 0.
    # --------------------------------------------------------

    for j in range(N):
        inequalities.append((NS_rows[j][:], 0))

    # --------------------------------------------------------
    # Pauli-shell capacity bounds.
    #
    # A stabilizer, its normalizer, and a stabilizer shadow are all subsets
    # (the shadow is an affine coset) of the phase-free n-qubit Pauli labels.
    # Hence each weight-j coefficient is at most
    #
    #     C_j = 3^j * binom(n,j).
    #
    # With B,S eliminated:
    #
    #     A_j <= C_j,
    #     NB_j(A) <= normalization*C_j,
    #     NS_j(A) <= normalization*C_j.
    # --------------------------------------------------------

    if USE_SHELL_CAPACITY_BOUNDS:
        for j in range(N):
            capacity = (3 ** j) * comb(n, j)

            rowA = [0] * N
            rowA[j] = -1
            inequalities.append((rowA, -capacity))

            inequalities.append(
                ([-c for c in NB_rows[j]], -normalization * capacity)
            )
            inequalities.append(
                ([-c for c in NS_rows[j]], -normalization * capacity)
            )

    # --------------------------------------------------------
    # B_j, S_j integral.
    #
    # NB_j(A) == 0 mod normalization
    # NS_j(A) == 0 mod normalization
    #
    # These preserve exactly the original Int(B_j), Int(S_j).
    # --------------------------------------------------------

    for j in range(N):
        congruences.append(
            (NB_rows[j][:], 0, normalization)
        )

        congruences.append(
            (NS_rows[j][:], 0, normalization)
        )

    return {
        "nvars": N,
        "normalization": normalization,
        "NB_rows": NB_rows,
        "NS_rows": NS_rows,
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
            raise AssertionError(
                "Internal error: reconstructed B is nonintegral."
            )

        B.append(numerator // normalization)

    for row in problem["NS_rows"]:
        numerator = sum(c * a for c, a in zip(row, A))

        if numerator % normalization != 0:
            raise AssertionError(
                "Internal error: reconstructed S is nonintegral."
            )

        S.append(numerator // normalization)

    return B, S


def verify_solution(n, k, d, case, A, B, S):
    normalization = 2 ** (n - k)

    assert A[0] == 1
    # assert A[1] == 0
    # assert A[2] == 0
    # assert A[3] == 0

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
            capacity = (3 ** j) * comb(n, j)
            assert A[j] <= capacity
            assert B[j] <= capacity
            assert S[j] <= capacity

    even_A = sum(A[j] for j in range(0, n + 1, 2))
    even_B = sum(B[j] for j in range(0, n + 1, 2))

    if case == 1:
        assert even_A == 2 ** (n - k - 1)
        assert even_B == 2 ** (n + k - 1)

    elif case == 2:
        assert all(A[j] == 0 for j in range(1, n + 1, 2))
        assert even_B == (
            2 ** (n - k - 1)
            * (4 ** k + 2 ** k)
        )

    elif case == 3:
        assert all(A[j] == 0 for j in range(1, n + 1, 2))
        assert even_B == (
            2 ** (n - k - 1)
            * (4 ** k - 2 ** k)
        )


def compact_A(A):
    return {
        j: value
        for j, value in enumerate(A)
        if value != 0
    }


# ============================================================
# Enumerate all distinct A distributions for one case
# ============================================================

def enumerate_A_solutions(n, k, d, case, K):
    total_start = time.perf_counter()

    # --------------------------------------------------------
    # Mathematical A-only system.
    # --------------------------------------------------------

    build_start = time.perf_counter()

    if SHOW_LIVE_PROGRESS:
        print("  [1/4] Building A-only constraints...", flush=True)

    problem = build_A_only_problem(
        n,
        k,
        d,
        case,
        K,
        exact_distance=REQUIRE_EXACT_DISTANCE,
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
                        n,
                        k,
                        d,
                        case,
                        A_sol,
                        B_sol,
                        S_sol,
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


def case_summary(case, result, count, elapsed):
    if result == sat:
        if MAX_SOLUTIONS is not None and count >= MAX_SOLUTIONS:
            return (
                f"Case {case}: SAT "
                f"(at least {count} solution(s); enumeration capped) "
                f"[{elapsed:.6f} s]"
            )

        return (
            f"Case {case}: SAT "
            f"({count} solution(s)) "
            f"[{elapsed:.6f} s]"
        )

    if result == unsat:
        return (
            f"Case {case}: UNSAT "
            f"(0 solutions) "
            f"[{elapsed:.6f} s]"
        )

    return (
        f"Case {case}: UNKNOWN "
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
)


def result_label(result):
    if result == sat:
        return "SAT"
    if result == unsat:
        return "UNSAT"
    if result == unknown:
        return "UNKNOWN"
    return str(result).upper()


def candidate_output_path(n, k, d):
    if CANDIDATE_FILENAME is None:
        filename = f"candidates_n{n}_k{k}_d{d}.py"
    else:
        filename = str(CANDIDATE_FILENAME).format(n=n, k=k, d=d)

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
# persistent source of candidate state for this [[n,k,d]].
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

        # Skip only when ANOTHER refined solver already excluded this
        # candidate.  If this solver itself produced an old UNSAT and its
        # version changed, allow the stale result to be rerun.
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


def print_summary(show_status=False):
    survivors = current_survivors()
    excluded = len(CANDIDATES) - len(survivors)

    print("=" * 72)
    print(f"Candidate database for [[{N},{K},{D}]]")
    print("=" * 72)
    print(f"Initial SLR3 candidates : {len(CANDIDATES)}")
    print(f"Currently excluded      : {excluded}")
    print(f"Current survivors       : {len(survivors)}")
    print()
    print("By case:")
    for case in (1, 2, 3):
        initial = sum(c["case"] == case for c in CANDIDATES)
        alive = len(current_survivors(case))
        print(f"  Case {case}: {alive} survivor(s) / {initial} initial")

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
            print(f"  {solver_name:18s}: {shown}")
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
            f"{candidate['id']} (Case {candidate['case']}): "
            f"FIXED_A = {_fixed_A_text(candidate['fixed_A'])}{suffix}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=f"Inspect persistent SLR3/refined candidates for [[{N},{K},{D}]]."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="print current surviving FIXED_A distributions"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="print every initial candidate, including excluded ones"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="show per-refined-solver status counts"
    )
    parser.add_argument(
        "--show-case", type=int, choices=(1, 2, 3), default=None,
        help="restrict --show/--all output to one parity case"
    )
    args = parser.parse_args()

    print_summary(show_status=args.status)
    if args.show or args.all:
        print_candidates(show_all=args.all, case=args.show_case)


if __name__ == "__main__":
    main()
'''


def save_candidate_database(n, k, d, all_results):
    path = candidate_output_path(n, k, d)

    if path.exists():
        raise FileExistsError(
            f"Candidate database already exists: {path}\n"
            "Refusing to overwrite refined-solver history. Delete/rename it "
            "explicitly if you intentionally want to rebuild the SLR3 baseline."
        )

    case_complete = {}
    for case in (1, 2, 3):
        info = all_results[case]
        capped = (
            MAX_SOLUTIONS is not None
            and info["result"] == sat
            and info["count"] >= MAX_SOLUTIONS
        )
        case_complete[case] = info["result"] != unknown and not capped

    complete = all(case_complete.values())
    candidates = []

    for case in (1, 2, 3):
        for idx, A in enumerate(all_results[case]["solutions"], start=1):
            if A is None:
                raise RuntimeError(
                    "Cannot save candidate database because a solution was not reconstructed."
                )
            candidates.append({
                "id": f"c{case}_{idx:04d}",
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
        "schema_version": 1,
        "code": {"n": n, "k": k, "d": d},
        "ordinary_solver": {
            "name": "SLR3_LP",
            "exact_distance": bool(REQUIRE_EXACT_DISTANCE and k > 0),
            "shell_capacity_bounds": bool(USE_SHELL_CAPACITY_BOUNDS),
            "complete": complete,
            "case_status": {
                case: result_label(all_results[case]["result"])
                for case in (1, 2, 3)
            },
            "case_complete": case_complete,
            "case_counts": {
                case: all_results[case]["count"]
                for case in (1, 2, 3)
            },
        },
        "refined_solver_keys": list(REFINED_SOLVER_KEYS),
        "candidates": candidates,
    }

    import pprint
    db_literal = pprint.pformat(database, sort_dicts=False, width=120)

    header = f'''#!/usr/bin/env python3
# Auto-generated by SLR3_LP.py.
# Persistent candidate database for [[{n},{k},{d}]].
#
# This is intentionally BOTH data and executable status viewer.
# Refined solvers update the DATABASE block in place; do not regenerate this
# file unless you intentionally want to discard all refined-solver history.

N = {n}
K = {k}
D = {d}
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


def check_one(n, k, d):
    if SAVE_CANDIDATE_DATABASE:
        existing_path = candidate_output_path(n, k, d)
        if existing_path.exists():
            print("=" * 76)
            print(f"Existing candidate database found for [[{n},{k},{d}]]")
            print(f"{existing_path}")
            print("SLR3 enumeration will NOT be rerun.")
            print("=" * 76)
            db = load_existing_candidate_database(existing_path)
            if hasattr(db, "print_summary"):
                db.print_summary(show_status=True)
            print(f"To list current survivors: python {existing_path.name} --show")
            print("Delete or rename the database explicitly only if you want to rebuild it.")
            return {"existing_database": str(existing_path)}

    print("=" * 76)
    print("SLR3 LP: Shor-Laflamme + Rains shadow + three parity cases")
    print(f"Parameters: [[{n},{k},{d}]]")
    print(
        "Distance mode: "
        + ("exact d (B_d>A_d)" if REQUIRE_EXACT_DISTANCE and k > 0
           else "at least d")
    )
    print(
        "Shell capacities: "
        + ("enabled" if USE_SHELL_CAPACITY_BOUNDS else "disabled")
    )
    print("Counting distinct ordinary A-weight distributions")
    print("=" * 76)

    K = krawtchouk_matrix(n)

    all_results = {}

    for case in (1, 2, 3):
        print()
        print(f"Parity Case {case}")
        print("-" * 76)

        result, solutions, elapsed, meta = enumerate_A_solutions(
            n,
            k,
            d,
            case,
            K,
        )

        all_results[case] = {
            "result": result,
            "count": len(solutions),
            "solutions": solutions,
            "elapsed": elapsed,
            "meta": meta,
        }

        print_case_detail(
            case,
            result,
            solutions,
            elapsed,
            meta,
        )

        print(
            case_summary(
                case,
                result,
                len(solutions),
                elapsed,
            )
        )

    print()
    print("=" * 76)
    print("Summary")
    print("=" * 76)

    for case in (1, 2, 3):
        info = all_results[case]

        print(
            case_summary(
                case,
                info["result"],
                info["count"],
                info["elapsed"],
            )
        )

    print("=" * 76)

    if SAVE_CANDIDATE_DATABASE:
        output_path = save_candidate_database(n, k, d, all_results)
        print(f"Candidate database saved to: {output_path}")
        print(f"Inspect later with: python {output_path.name}")

    return all_results


# ============================================================
# Fast feasibility-only scan over d
# ============================================================

def check_case_feasibility(n, k, d, case, K):
    """
    Feasibility only; no solution enumeration.
    """
    # Bound scan must test distance >= d (monotone in d), not exact distance.
    problem = build_A_only_problem(
        n, k, d, case, K, exact_distance=False
    )

    param = exact_rref_parameterization(
        problem["nvars"],
        problem["eq_rows"],
        problem["eq_rhs"],
    )

    if param["inconsistent"]:
        return unsat

    solver, free_vars, reduced_meta = build_reduced_solver(
        problem,
        param,
    )

    return solver.check()


def scan_d(n, k):
    print("=" * 76)
    print("Fast Shor-Laflamme + MacWilliams + Shadow bound scan")
    print(f"Parameters: n={n}, k={k}")
    print("=" * 76)

    K = krawtchouk_matrix(n)
    last_feasible_d = None

    for d in range(1, n + 1):
        start = time.perf_counter()

        case_results = [
            check_case_feasibility(n, k, d, case, K)
            for case in (1, 2, 3)
        ]

        elapsed = time.perf_counter() - start

        print(
            f"[[{n},{k},{d}]] : "
            f"Case 1={str(case_results[0]).upper()}, "
            f"Case 2={str(case_results[1]).upper()}, "
            f"Case 3={str(case_results[2]).upper()} "
            f"[{elapsed:.6f} s]"
        )

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
            print(
                f"First infeasible d = {last_feasible_d + 1}"
            )
    else:
        print("No feasible d found.")

    print("=" * 76)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) == 4:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
        d = int(sys.argv[3])

        if not (0 <= k <= n):
            raise ValueError(
                "k must satisfy 0 <= k <= n."
            )

        if not (1 <= d <= n):
            raise ValueError(
                "d must satisfy 1 <= d <= n."
            )

        check_one(n, k, d)

    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        k = int(sys.argv[2])

        if not (0 <= k <= n):
            raise ValueError(
                "k must satisfy 0 <= k <= n."
            )

        scan_d(n, k)

    else:
        print("Usage:")
        print("    python SLR3_LP.py n k")
        print("    python SLR3_LP.py n k d")
        print()
        print("Examples:")
        print("    python SLR3_LP.py 23 1 9")
        print("    python SLR3_LP.py 27 15 5")
        print("    python SLR3_LP.py 20 1")


if __name__ == "__main__":
    main()
