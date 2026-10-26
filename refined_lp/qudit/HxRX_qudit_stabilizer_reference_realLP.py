#!/usr/bin/env python3
r"""
HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py

GENERAL (non-MDS allowed) LAZY-CUT REAL-LP H x R_eta stabilizer-reference
necessary-condition solver for q-ary stabilizer codes, q in {2,3,4,5,7}.

Two modes
---------
1. DIRECT / standalone mode (default)

       python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py q n k d

   The ordinary stabilizer weight distribution A_w is left free.  The solver
   scans all possible nonzero stabilizer-reference weights u (and, for q=2,
   all three binary parity/quadratic cases).  A code is excluded only if every
   admissible direct branch is UNSAT.  A SAT branch merely means that these
   necessary conditions do not exclude the code.

2. SLR candidate-list / database mode

       python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py q n k d \\
           --database candidates_....py

   Reads the persistent candidate databases produced by SLR3_LP_lattice.py
   (binary) or SLR_qudit.py (q=2,3,4,5,7), and also accepts simple Python/JSON
   candidate lists.  A candidate with fixed A_w has a mandatory reference at
   every shell u with A_u>0.  Therefore one UNSAT mandatory shell excludes that
   candidate/case.  If a binary list does not record a parity case, all three
   cases are tested.

Projection / reference
----------------------
Choose a known weight-u stabilizer and normalize it locally to

    g = I^T eta^U,      |U|=u, |T|=n-u,

with eta=(1,0) in F_q^2.  T uses Hamming {0}/nonzero.  U uses

    R_eta = {0}, {c eta}_{c!=0}, C_r={a eta+r xi : a in F_q}_{r!=0},

with 2q-1 local cells (I/X/(YZ) for q=2).

Real-LP constraints
-------------------
* ordinary A_w normalization, or exact fixed A_w from an SLR candidate;
* projected H x R_eta MacWilliams constraints, B>=A, and A=B below d;
* all projection-compatible Weyl-character twisted Fourier positivity;
* generalized Rains split-Hamming shadow positivity;
* known-reference existence and stabilizer commutation support;
* additive/scalar orbit equalities when valid;
* cleaning/dimension constraints and projected-cell capacities;
* for q=2, the existing binary refined-shadow / matched-quadratic three-case
  sector is retained via the exact full sparse binary path.

No integrality, modular congruence, or orbit-divisibility condition is used.
Hence any UNSAT returned by this file is a genuine continuous real-LP
exclusion.

Backend
-------
For q=3,4,5,7 the default path is

    exact A-side orbit quotient
      -> small base LP
      -> HiGHS
      -> streaming MacWilliams/character separation
      -> add only actually violated cuts
      -> repeat.

The full refined transform matrix is never materialized at once.  For q=2,
the program currently routes to the earlier full sparse binary-special path so
that the matched quadratic/shadow sector is not weakened accidentally.

Dependencies
------------
    pip install numpy scipy sympy

Examples
--------
Direct, A_w free:
    python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py 3 10 2 4
    python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py 2 14 3 5

SLR database/list:
    python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py 2 14 3 5 \\
        --database candidates_n14_k3_d5.py
    python HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py 4 10 2 4 \\
        --database candidates_q4_n10_k2_d4.py

For q=4, --fq-linear explicitly restricts to the F_4-linear subclass.  If an
SLR_qudit database records that assumption, it is inherited automatically.
"""

# =====================================================================
# Imports
# =====================================================================

import argparse
import importlib.util
import itertools
import math
import os
import sys
import subprocess
import threading
import time
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

try:
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
except ImportError as exc:
    raise SystemExit(
        "This program requires SymPy.\nInstall it with:\n  pip install sympy"
    ) from exc




# =====================================================================
# USER CONFIGURATION
# =====================================================================

# These are overwritten by command-line arguments.
Q = 3
n = 14
k = 3
d = 5

# The solver is intended to consume a complete ordinary A distribution from
# an SLR_qudit candidate.  A direct FIXED_A can still be supplied for testing.
FIXED_A = None
FIXED_SPLIT_A = None  # optional exact MDS split A_{i,r}
FIXED_SPLIT_B = None  # optional exact MDS split B_{i,r}

# Core constraints.
USE_MACWILLIAMS = True
USE_CHARACTER_TWISTS = True
USE_GENERALIZED_SHADOW = True
USE_REFERENCE_EXISTENCE = True
USE_COMMUTATION = True
USE_ADDITIVE_ORBIT = True
USE_ADDITIVE_ORBIT_DIVISIBILITY = False
USE_SCALAR_ORBIT = True
USE_SCALAR_ORBIT_DIVISIBILITY = False
USE_CLEANING = True
USE_CELL_CAPACITY = True
B_INTEGRAL = False
TWIST_INTEGRAL = False

# q=2 regression/special structure.
USE_BINARY_REFINED_SHADOW_SECTOR = True
USE_BINARY_MATCHED_QUADRATIC_CASE = True

# q=4: full F_4-linearity is stronger than general additive stabilizer
# structure.  The default follows the candidate database.  Prime q is
# automatically F_q-linear because the prime field is F_q itself.
ASSUME_FQ_LINEAR = False

# LP/runtime controls.
TIMEOUT_MS = None  # legacy exact-helper default; real-LP path never sets a solver limit
SHOW_MODEL = False
SHOW_ZERO_MODEL_ENTRIES = False
SHOW_LIVE_PROGRESS = True
PROGRESS_EVERY_SECONDS = 3600.0

# Persistent DB identity.  This key already exists in SLR_qudit.py.
SOLVER_NAME = "hxr_twisted_stabilizer_reference"
SOLVER_VERSION = "2026-09-13-qdit-r1-character-twists-orbits"
DATABASE_FILENAME = None
REFERENCE_U_ORDER = "descending"


# =====================================================================
# Small utilities
# =====================================================================

def gcd_many(values):
    g = 0
    for v in values:
        g = math.gcd(g, abs(int(v)))
    return g


def zsum(items):
    items = list(items)
    return Sum(items) if items else 0


def is_prime(x):
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    r = int(math.isqrt(x))
    for p in range(3, r + 1, 2):
        if x % p == 0:
            return False
    return True


# =====================================================================
# Finite fields F_q for q=2,3,4,5,7
# =====================================================================

class SmallField:
    """Minimal exact finite-field backend for q=2,3,4,5,7."""

    def __init__(self, q):
        if q not in (2, 3, 4, 5, 7):
            raise ValueError(
                "Current HxR qudit backend supports q in {2,3,4,5,7}."
            )
        self.q = int(q)
        self.p = 2 if q in (2, 4) else q
        self.elements = tuple(range(q))
        self.zero = 0
        self.one = 1
        self.nonzero = tuple(range(1, q))

    def add(self, a, b):
        if self.q == 4:
            return int(a) ^ int(b)
        return (int(a) + int(b)) % self.q

    def neg(self, a):
        if self.q == 4:
            return int(a)
        return (-int(a)) % self.q

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        a = int(a)
        b = int(b)
        if self.q != 4:
            return (a * b) % self.q

        # GF(4)=GF(2)[x]/(x^2+x+1), encoded a0+a1*x by bits a0+2*a1.
        a0, a1 = a & 1, (a >> 1) & 1
        b0, b1 = b & 1, (b >> 1) & 1
        c0 = (a0 * b0) ^ (a1 * b1)      # x^2 = x+1
        c1 = (a0 * b1) ^ (a1 * b0) ^ (a1 * b1)
        return c0 | (c1 << 1)

    def pow(self, a, e):
        out = self.one
        base = int(a)
        e = int(e)
        while e:
            if e & 1:
                out = self.mul(out, base)
            base = self.mul(base, base)
            e >>= 1
        return out

    def trace(self, a):
        """Absolute trace F_q -> F_p, returned as an integer 0,...,p-1."""
        a = int(a)
        if self.q != 4:
            return a % self.p
        # Tr_{4/2}(a)=a+a^2, which lies in {0,1}.
        t = self.add(a, self.mul(a, a))
        if t not in (0, 1):
            raise AssertionError("GF(4) trace did not land in GF(2).")
        return t

    def prime_subfield(self):
        if self.q == 4:
            return (0, 1)
        return self.elements

    def fmt(self, a):
        if self.q != 4:
            return str(a)
        return {0: "0", 1: "1", 2: "w", 3: "w+1"}[int(a)]


FF = SmallField(Q)


def reset_field(q):
    global FF
    FF = SmallField(q)


def symplectic(u, v):
    """[u,v]=b*c-a*d for u=(a,b), v=(c,d)."""
    a, b = u
    c, dd = v
    return FF.sub(FF.mul(b, c), FF.mul(a, dd))


def vec_add(u, v):
    return (FF.add(u[0], v[0]), FF.add(u[1], v[1]))


def vec_scale(beta, u):
    return (FF.mul(beta, u[0]), FF.mul(beta, u[1]))


# =====================================================================
# Exact cyclotomic arithmetic Z[zeta_p]
# =====================================================================

# Coefficients are represented in the basis 1,zeta,...,zeta^(p-2).
# Since p is prime here, Phi_p=1+x+...+x^(p-1).

def cyclo_dim():
    return FF.p - 1


def czero():
    return (0,) * cyclo_dim()


def cone():
    out = [0] * cyclo_dim()
    out[0] = 1
    return tuple(out)


def cint(m):
    out = [0] * cyclo_dim()
    out[0] = int(m)
    return tuple(out)


def cadd(a, b):
    return tuple(x + y for x, y in zip(a, b))


def cscale(m, a):
    m = int(m)
    return tuple(m * x for x in a)


def root_power(e):
    p = FF.p
    e %= p
    dim = p - 1
    if e < dim:
        out = [0] * dim
        out[e] = 1
        return tuple(out)
    # zeta^(p-1)=-(1+...+zeta^(p-2)).
    return tuple(-1 for _ in range(dim))


@lru_cache(maxsize=None)
def cmul(a, b):
    p = FF.p
    out = [0] * (p - 1)
    for i, ai in enumerate(a):
        if not ai:
            continue
        for j, bj in enumerate(b):
            if not bj:
                continue
            term = root_power(i + j)
            scale = ai * bj
            for r, tr in enumerate(term):
                out[r] += scale * tr
    return tuple(out)


@lru_cache(maxsize=None)
def cpow(a, e):
    e = int(e)
    out = cone()
    base = tuple(a)
    while e:
        if e & 1:
            out = cmul(out, base)
        base = cmul(base, base)
        e >>= 1
    return out


def chi_value(x):
    return root_power(FF.trace(x))


# =====================================================================
# H x R_eta local cells
# =====================================================================

ETA = (1, 0)
XI = (0, 1)


def phase_space_labels():
    return tuple((a, b) for a in FF.elements for b in FF.elements)


def hamming_cells():
    zero = ((0, 0),)
    nonzero = tuple(v for v in phase_space_labels() if v != (0, 0))
    return (zero, nonzero)


def reta_cells():
    cells = [((0, 0),)]
    # Singleton scalar multiples c eta, c != 0.
    for c in FF.nonzero:
        cells.append(((c, 0),))
    # Affine fibers C_r={a eta+r xi : a in F_q}, r != 0.
    for r in FF.nonzero:
        cells.append(tuple((a, r) for a in FF.elements))
    return tuple(cells)


def reta_cell_names():
    names = ["I"]
    names.extend(f"X[{FF.fmt(c)}]" for c in FF.nonzero)
    names.extend(f"C[{FF.fmt(r)}]" for r in FF.nonzero)
    return tuple(names)


def identity_cell_index(cells):
    return cells.index(((0, 0),))


def cell_lookup(cells):
    out = {}
    for i, cell in enumerate(cells):
        for v in cell:
            if v in out:
                raise ValueError("Projection cells overlap.")
            out[v] = i
    if set(out) != set(phase_space_labels()):
        raise ValueError("Projection does not cover F_q^2 exactly.")
    return out


def projected_fourier_matrix(cells, twist=None):
    """
    Project the symplectic Fourier kernel to ``cells``.

    twist=None gives the ordinary Fourier transform.  For twist=t,
    input u is multiplied by chi([t,u]), so the full transform is

        F( chi_t * Lambda ),

    which is a phase-space translate of F(Lambda) and hence nonnegative.
    """
    rows = []
    for input_cell in cells:
        representative = None
        for u in input_cell:
            tw = cone() if twist is None else chi_value(symplectic(twist, u))
            row = []
            for output_cell in cells:
                s = czero()
                for v in output_cell:
                    s = cadd(s, chi_value(symplectic(u, v)))
                row.append(cmul(tw, s))
            row = tuple(row)
            if representative is None:
                representative = row
            elif row != representative:
                return None
        rows.append(representative)
    return tuple(rows)


def matrix_is_rational_integer(Qmat):
    return all(all(all(c == 0 for c in x[1:]) for x in row) for row in Qmat)


def matrix_constant_part(Qmat):
    if not matrix_is_rational_integer(Qmat):
        raise ValueError("Matrix has non-rational cyclotomic entries.")
    return tuple(tuple(x[0] for x in row) for row in Qmat)


# =====================================================================
# Weak compositions and polynomial expansion
# =====================================================================

@lru_cache(maxsize=None)
def weak_compositions(total, parts):
    total = int(total)
    parts = int(parts)
    if parts == 1:
        return ((total,),)
    out = []

    def rec(prefix, remaining, left):
        if left == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for x in range(remaining + 1):
            rec(prefix + [x], remaining - x, left - 1)

    rec([], total, parts)
    return tuple(out)


@lru_cache(maxsize=None)
def multinomial_terms_cyclo(power, coeffs):
    r = len(coeffs)
    out = {}
    for comp in weak_compositions(power, r):
        multinom = math.factorial(power)
        for x in comp:
            multinom //= math.factorial(x)
        coeff = cint(multinom)
        for c, e in zip(coeffs, comp):
            if e:
                coeff = cmul(coeff, cpow(c, e))
        if any(coeff):
            out[comp] = coeff
    return out


@lru_cache(maxsize=None)
def expand_composition_cyclo(comp, Qmat):
    r = len(comp)
    poly = {tuple([0] * r): cone()}
    for a, power in enumerate(comp):
        if power == 0:
            continue
        factor = multinomial_terms_cyclo(power, tuple(Qmat[a]))
        new_poly = {}
        for c1, v1 in poly.items():
            for c2, v2 in factor.items():
                oc = tuple(x + y for x, y in zip(c1, c2))
                vv = cmul(v1, v2)
                if oc in new_poly:
                    vv = cadd(new_poly[oc], vv)
                if any(vv):
                    new_poly[oc] = vv
                elif oc in new_poly:
                    del new_poly[oc]
        poly = new_poly
    return poly


@lru_cache(maxsize=None)
def multinomial_terms_int(power, coeffs):
    r = len(coeffs)
    out = {}
    for comp in weak_compositions(power, r):
        coeff = math.factorial(power)
        for x in comp:
            coeff //= math.factorial(x)
        for c, x in zip(coeffs, comp):
            coeff *= int(c) ** x
        if coeff:
            out[comp] = coeff
    return out


@lru_cache(maxsize=None)
def expand_composition_int(comp, Qmat):
    r = len(comp)
    poly = {tuple([0] * r): 1}
    for a, power in enumerate(comp):
        if power == 0:
            continue
        factor = multinomial_terms_int(power, tuple(Qmat[a]))
        new_poly = defaultdict(int)
        for c1, v1 in poly.items():
            for c2, v2 in factor.items():
                oc = tuple(x + y for x, y in zip(c1, c2))
                new_poly[oc] += v1 * v2
        poly = {c: v for c, v in new_poly.items() if v}
    return poly


# =====================================================================
# Split H x R_eta geometry
# =====================================================================

T_SIZE = 0
U_SIZE = 0


def set_reference_size(u):
    global T_SIZE, U_SIZE
    U_SIZE = int(u)
    T_SIZE = n - U_SIZE
    if T_SIZE < 0 or U_SIZE <= 0:
        raise ValueError("Reference weight u must satisfy 1 <= u <= n.")


def t_compositions():
    return weak_compositions(T_SIZE, 2)


def u_compositions():
    return weak_compositions(U_SIZE, 2 * Q - 1)


def all_buckets():
    return tuple(itertools.product(t_compositions(), u_compositions()))


def identity_bucket():
    return ((T_SIZE, 0), tuple([U_SIZE] + [0] * (2 * Q - 2)))


def reference_bucket():
    # eta=(1,0), and X[1] is the first nonzero singleton after I.
    uc = [0] * (2 * Q - 1)
    uc[1] = U_SIZE
    return ((T_SIZE, 0), tuple(uc))


def bucket_weight(bucket):
    tc, uc = bucket
    return tc[1] + U_SIZE - uc[0]


def multinomial_capacity(comp, cell_sizes):
    rem = sum(comp)
    m = 1
    for count in comp[:-1]:
        m *= math.comb(rem, count)
        rem -= count
    for count, s in zip(comp, cell_sizes):
        m *= int(s) ** int(count)
    return m


def bucket_capacity(bucket):
    tc, uc = bucket
    cap_t = multinomial_capacity(tc, (1, Q * Q - 1))
    u_sizes = tuple(len(c) for c in reta_cells())
    cap_u = multinomial_capacity(uc, u_sizes)
    return cap_t * cap_u


def bucket_label(bucket):
    tc, uc = bucket
    names = reta_cell_names()
    parts = [f"T(I={tc[0]},N={tc[1]})"]
    parts.append("U(" + ",".join(f"{nm}={c}" for nm, c in zip(names, uc) if c) + ")")
    return " | ".join(parts)


def normalized_fixed_A(fixed_A):
    if fixed_A is None:
        return None
    dist = [0] * (n + 1)
    items = fixed_A.items() if hasattr(fixed_A, "items") else enumerate(fixed_A)
    for w, value in items:
        w = int(w)
        value = int(value)
        if not 0 <= w <= n:
            raise ValueError(f"Invalid fixed weight {w}.")
        if value < 0:
            raise ValueError("A_w must be nonnegative.")
        dist[w] = value
    if dist[0] != 1:
        raise ValueError("A_0 must equal 1.")
    expected = Q ** (n - k)
    if sum(dist) != expected:
        raise ValueError(
            f"FIXED_A sums to {sum(dist)}, expected {expected}=q^(n-k)."
        )
    return dist


# =====================================================================
# Projection actions: commutation, additive reference, scalar action
# =====================================================================

def u_cell_symplectic_value(cell):
    """[eta,v] is constant on every R_eta cell."""
    vals = {symplectic(ETA, v) for v in cell}
    if len(vals) != 1:
        raise AssertionError("R_eta cell does not have constant eta pairing.")
    return next(iter(vals))


def bucket_reference_pairing(bucket):
    """[g,h] in F_q for g=I^T eta^U, visible on R_eta buckets."""
    _, uc = bucket
    total = 0
    for count, cell in zip(uc, reta_cells()):
        if count:
            total = FF.add(total, FF.mul(count % FF.p if Q == 4 else count % Q,
                                         u_cell_symplectic_value(cell)))
    return total


def bucket_commutes_with_reference(bucket, fq_linear):
    s = bucket_reference_pairing(bucket)
    if fq_linear:
        return s == 0
    return FF.trace(s) == 0


def induced_cell_permutation(cells, map_fn):
    sets = [frozenset(cell) for cell in cells]
    lookup = {s: i for i, s in enumerate(sets)}
    perm = []
    for cell in cells:
        image = frozenset(map_fn(v) for v in cell)
        if image not in lookup:
            return None
        perm.append(lookup[image])
    return tuple(perm)


def permute_composition(comp, perm):
    out = [0] * len(comp)
    for old, count in enumerate(comp):
        out[perm[old]] += count
    return tuple(out)


def additive_u_permutation(alpha):
    shift = vec_scale(alpha, ETA)
    return induced_cell_permutation(reta_cells(), lambda v: vec_add(v, shift))


def scalar_u_permutation(beta):
    return induced_cell_permutation(reta_cells(), lambda v: vec_scale(beta, v))


def additive_bucket_map(bucket, alpha):
    tc, uc = bucket
    perm = additive_u_permutation(alpha)
    if perm is None:
        raise AssertionError("R_eta not invariant under eta translation.")
    return (tc, permute_composition(uc, perm))


def scalar_bucket_map(bucket, beta):
    tc, uc = bucket
    perm = scalar_u_permutation(beta)
    if perm is None:
        raise AssertionError("R_eta not invariant under scalar multiplication.")
    return (tc, permute_composition(uc, perm))


def additive_action_elements(fq_linear):
    return FF.elements if fq_linear else FF.prime_subfield()


def additive_bucket_stabilizer_size(bucket, fq_linear):
    return sum(additive_bucket_map(bucket, a) == bucket for a in additive_action_elements(fq_linear))


def scalar_bucket_stabilizer_size(bucket):
    return sum(scalar_bucket_map(bucket, b) == bucket for b in FF.nonzero)


# =====================================================================
# Projected transform construction
# =====================================================================

def ordinary_local_matrices():
    QT = projected_fourier_matrix(hamming_cells(), twist=None)
    QU = projected_fourier_matrix(reta_cells(), twist=None)
    if QT is None or QU is None:
        raise AssertionError("Ordinary Fourier transform does not descend to H x R_eta.")
    return QT, QU


def character_local_matrices(alpha):
    QT, _ = ordinary_local_matrices()
    t = vec_scale(alpha, ETA)
    QU = projected_fourier_matrix(reta_cells(), twist=t)
    if QU is None:
        raise AssertionError(f"Character twist alpha={alpha} does not descend to R_eta.")
    return QT, QU


def build_cyclo_transform_rows(active_buckets, col_of, QT, QU):
    """
    rows[out_bucket][basis_index] is a sparse integer row in A variables.
    """
    dim = cyclo_dim()
    rows = {}

    # Cache ONLY compositions that actually occur in the active A support.
    # At q=5,u=7 there are 6435 possible U compositions but only ~1300
    # survive MDS/commutation/orbit pruning.  Expanding all 6435 was one of
    # the dominant time/memory pathologies of the earlier implementation.
    active_t = {b[0] for b in active_buckets}
    active_u = {b[1] for b in active_buckets}
    t_exp_cache = {tc: expand_composition_cyclo(tc, QT) for tc in active_t}
    u_exp_cache = {uc: expand_composition_cyclo(uc, QU) for uc in active_u}

    temp = defaultdict(lambda: [defaultdict(int) for _ in range(dim)])
    for in_bucket in active_buckets:
        col = col_of[in_bucket]
        tc, uc = in_bucket
        for otc, ct in t_exp_cache[tc].items():
            for ouc, cu in u_exp_cache[uc].items():
                coeff = cmul(ct, cu)
                if not any(coeff):
                    continue
                ob = (otc, ouc)
                comps = temp[ob]
                for r, val in enumerate(coeff):
                    if val:
                        comps[r][col] += val

    for ob, comps in temp.items():
        rows[ob] = tuple(dict(x) for x in comps)
    return dict(rows)


def row_add(a, b, scale_b=1):
    """Sparse integer row a + scale_b*b; zeros are deleted."""
    out = dict(a)
    for col, val in b.items():
        nv = out.get(col, 0) + scale_b * int(val)
        if nv:
            out[col] = nv
        elif col in out:
            del out[col]
    return out


def row_scale(a, scale):
    if scale == 0:
        return {}
    return {c: int(scale) * int(v) for c, v in a.items() if v}


def row_single(col, value=1):
    return {} if value == 0 else {int(col): int(value)}


def sum_sparse_rows(rows):
    out = {}
    for sparse in rows:
        out = row_add(out, sparse)
    return out


def cyclo_output_sparse(rows, bucket):
    # Read-only view: do NOT copy sparse dictionaries.  At q=5 the copies
    # themselves become a major transient-memory cost.
    comps = rows.get(bucket)
    if comps is None:
        return EMPTY_CYCLO_ROWS
    return comps

EMPTY_CYCLO_ROWS = tuple({} for _ in range(cyclo_dim()))

def cyclo_output_dense(rows, bucket, nvars):
    """Compatibility helper used only by the small q=2 special sector."""
    return tuple([sparse_to_dense_compat(nvars, s) for s in cyclo_output_sparse(rows, bucket)])

def sparse_to_dense_compat(nvars, sparse):
    row = [0] * nvars
    for col, coeff in sparse.items(): row[col] = int(coeff)
    return row

def add_rows(a, b, scale_b=1):
    return [x + scale_b*y for x,y in zip(a,b)]


# =====================================================================
# Generalized split-Hamming shadow transform
# =====================================================================

def build_split_shadow_rows(active_buckets, col_of):
    """
    Integer numerators for the generalized Rains shadow polynomial.

    The local homogeneous substitution is
        I-input      -> (q-1) x + (q+1) y,
        nonzero-input->      -x +       y.

    The common positive factor q^{-n} M^2 is irrelevant to positivity.
    We retain separate output Hamming weights on T and U.
    """
    Qsh = ((Q - 1, Q + 1), (-1, 1))
    rows = defaultdict(lambda: defaultdict(int))

    for bucket in active_buckets:
        col = col_of[bucket]
        tc, uc = bucket
        u_h = (uc[0], U_SIZE - uc[0])
        et = expand_composition_int(tc, Qsh)
        eu = expand_composition_int(u_h, Qsh)
        for otc, ct in et.items():
            for ouc, cu in eu.items():
                val = ct * cu
                if val:
                    rows[(otc, ouc)][col] += val

    return {b: dict(s) for b, s in rows.items()}


# =====================================================================
# Binary-only refined shadow-character sector
# =====================================================================

def binary_shadow_local_matrices(alpha):
    if Q != 2:
        raise ValueError("Binary shadow sector requested for q!=2.")
    QT, QU = character_local_matrices(alpha)

    # Multiply every nonidentity INPUT cell by -1: global binary shadow
    # quadratic twist.  On U, alpha=1 additionally supplies chi_eta.
    QT2 = []
    for a, row in enumerate(QT):
        sign = 1 if a == 0 else -1
        QT2.append(tuple(cscale(sign, x) for x in row))
    QU2 = []
    for a, row in enumerate(QU):
        sign = 1 if a == 0 else -1
        QU2.append(tuple(cscale(sign, x) for x in row))
    return tuple(QT2), tuple(QU2)


def binary_quadratic_bucket_parity(bucket, alpha):
    """Input sign exponent for shadow*character(alpha eta), q=2."""
    if Q != 2:
        raise ValueError
    tc, uc = bucket
    parity = tc[1] & 1  # shadow on T
    # shadow on U: every non-I; character alpha eta contributes [eta,v].
    for count, cell in zip(uc, reta_cells()):
        if count == 0:
            continue
        v = cell[0]
        bit = 0 if v == (0, 0) else 1
        if alpha:
            bit ^= FF.trace(symplectic(ETA, v))
        parity ^= (count * bit) & 1
    return parity


# =====================================================================
# Build mathematical problem
# =====================================================================

def build_problem(fixed_A, case, fq_linear):
    build_start = time.perf_counter()
    fixed_A = normalized_fixed_A(fixed_A)
    norm = Q ** (n - k)
    universe = all_buckets()
    idb = identity_bucket()

    # ---------------------------------------------------------------
    # Active A buckets: fixed zero shells and reference commutation.
    # ---------------------------------------------------------------
    active = []
    for bucket in universe:
        w = bucket_weight(bucket)
        if fixed_A is not None and fixed_A[w] == 0:
            continue
        if FIXED_SPLIT_A is not None:
            i_split = bucket[0][1]
            r_split = U_SIZE - bucket[1][0]
            if FIXED_SPLIT_A.get((i_split, r_split), 0) == 0:
                continue
        if USE_COMMUTATION and not bucket_commutes_with_reference(bucket, fq_linear):
            continue
        active.append(bucket)
    active = tuple(active)
    col_of = {b: i for i, b in enumerate(active)}
    nvars = len(active)
    if idb not in col_of:
        raise ValueError("Identity bucket was pruned.")

    eq_rows, eq_rhs = [], []
    inequalities = []          # row.x >= rhs
    congruences = []           # row.x == rhs mod modulus
    log = {"basic": [], "additional": [], "skipped": []}

    def phase(label):
        print(
            f"        [build] {label:<34} vars={nvars} eq={len(eq_rows)} "
            f"ineq={len(inequalities)} congr={len(congruences)} "
            f"elapsed={time.perf_counter()-build_start:.3f}s",
            flush=True,
        )

    phase("active buckets / pruning")

    def add_eq(row, rhs=0):
        eq_rows.append(row)
        eq_rhs.append(int(rhs))

    # A_identity=1.
    add_eq(row_single(col_of[idb]), 1)

    if FIXED_SPLIT_A is not None:
        for (ii, rr), target in sorted(FIXED_SPLIT_A.items()):
            if target == 0:
                continue
            row = {c: 1 for b, c in col_of.items() if b[0][1] == ii and U_SIZE - b[1][0] == rr}
            add_eq(row, target)
        log["basic"].append("exact MDS split A_{i,r} distribution")
    elif fixed_A is not None:
        for w, target in enumerate(fixed_A):
            if target == 0:
                continue
            row = {c: 1 for b, c in col_of.items() if bucket_weight(b) == w}
            add_eq(row, target)
        log["basic"].append("fixed ordinary A_w distribution")
    else:
        add_eq({c: 1 for c in range(nvars)}, norm)
        log["basic"].append(f"sum A=q^(n-k)={norm}")

    # ---------------------------------------------------------------
    phase("A normalization / split A")

    # ---------------------------------------------------------------
    QT, QU = ordinary_local_matrices()
    B_rows = build_cyclo_transform_rows(active, col_of, QT, QU)

    if USE_MACWILLIAMS:
        for ob in universe:
            comp_rows = cyclo_output_sparse(B_rows, ob)
            # B must be rational: all nonconstant cyclotomic coordinates vanish.
            for rr in comp_rows[1:]:
                add_eq(rr, 0)
            row0 = dict(comp_rows[0])
            acol = col_of.get(ob)
            cmp_row = dict(row0)
            if acol is not None:
                nv = cmp_row.get(acol, 0) - norm
                if nv: cmp_row[acol] = nv
                elif acol in cmp_row: del cmp_row[acol]
            if bucket_weight(ob) < d:
                add_eq(cmp_row, 0)
            else:
                inequalities.append((cmp_row, 0))
            if B_INTEGRAL:
                congruences.append((row0, 0, norm))
            if USE_CELL_CAPACITY:
                cap = bucket_capacity(ob)
                inequalities.append((row_scale(row0, -1), -(norm * cap)))
        log["basic"].append(
            "exact projected HxR_eta MacWilliams; cyclotomic non-rational parts=0; B>=A; A=B below d"
        )

    phase("MacWilliams / refined B")

    # Exact MDS split B_{i,r} collapse.  B_rows are numerators, so the
    # target is norm * B_{i,r}.  This is stronger than ordinary shell fixing
    # and is the main MDS-specific reduction.
    if USE_MACWILLIAMS and FIXED_SPLIT_B is not None:
        split_rows = {}
        for ob in universe:
            ii = ob[0][1]
            rr = U_SIZE - ob[1][0]
            key = (ii, rr)
            comp_rows = cyclo_output_sparse(B_rows, ob)
            # nonconstant cyclotomic components already vanish cellwise above.
            row0 = comp_rows[0]
            if key not in split_rows:
                split_rows[key] = {}
            split_rows[key] = row_add(split_rows[key], row0)
        for key, target in sorted(FIXED_SPLIT_B.items()):
            add_eq(split_rows.get(key, {}), norm * int(target))
        log["basic"].append("exact MDS split B_{i,r} distribution")

    # ---------------------------------------------------------------
    phase("exact MDS split B")

    # All R_eta-compatible Weyl-character twisted Fourier transforms.
    # alpha=0 is ordinary B and is skipped here as duplicate.
    # ---------------------------------------------------------------
    twist_rows = {}
    if USE_CHARACTER_TWISTS:
        for alpha in FF.nonzero:
            QTa, QUa = character_local_matrices(alpha)
            rows = build_cyclo_transform_rows(active, col_of, QTa, QUa)
            # Process this character sector immediately and discard it.
            # Keeping all q-1 transforms simultaneously is catastrophic at q=5+.
            for ob in universe:
                comp_rows = cyclo_output_sparse(rows, ob)
                for rr in comp_rows[1:]:
                    add_eq(rr, 0)
                row0 = dict(comp_rows[0])
                inequalities.append((row0, 0))
                if TWIST_INTEGRAL:
                    congruences.append((row0, 0, norm))
                # IMPORTANT: a nontrivial character-twisted Fourier
                # coefficient is not a projected cell cardinality.  It is
                # universally nonnegative, but there is no combinatorial
                # bucket-capacity upper bound analogous to ordinary B.
            del rows
        log["additional"].append(
            f"all {Q} compatible character twists t=alpha*eta on U (including ordinary alpha=0; {Q-1} added)"
        )

    # ---------------------------------------------------------------
    phase("character twists")

    # Generalized Rains shadow at split-Hamming resolution.
    # ---------------------------------------------------------------
    shadow_rows = {}
    if USE_GENERALIZED_SHADOW:
        shadow_rows = build_split_shadow_rows(active, col_of)
        out_hamming = tuple(itertools.product(weak_compositions(T_SIZE, 2), weak_compositions(U_SIZE, 2)))
        for ob in out_hamming:
            inequalities.append((dict(shadow_rows.get(ob, {})), 0))
        log["basic"].append(
            "generalized Rains shadow inequalities at split-Hamming resolution"
        )

    # ---------------------------------------------------------------
    phase("generalized shadow")

    # q=2: refined shadow + shadow-character alpha=1.
    # ---------------------------------------------------------------
    binary_shadow_rows = {}
    if Q == 2 and USE_BINARY_REFINED_SHADOW_SECTOR:
        for alpha in FF.elements:
            QTs, QUs = binary_shadow_local_matrices(alpha)
            rows = build_cyclo_transform_rows(active, col_of, QTs, QUs)
            binary_shadow_rows[alpha] = rows
            for ob in universe:
                comp_rows = cyclo_output_sparse(rows, ob)
                row0 = dict(comp_rows[0])
                inequalities.append((row0, 0))
                congruences.append((row0, 0, norm))
                # alpha=0 is the genuine refined Rains shadow and retains
                # the projected-cell capacity bound.  alpha=1 is the
                # shadow-character / (111,100) blockwise twisted spectrum;
                # it is not a cell count and must NOT receive that bound.
                if USE_CELL_CAPACITY and alpha == 0:
                    cap = bucket_capacity(ob)
                    inequalities.append((row_scale(row0, -1), -(norm * cap)))
        log["additional"].append(
            "q=2 refined shadow sector: alpha=0 shadow and alpha=1 shadow-character twist"
        )

        # Matched binary quadratic three-case constraint for alpha=1.
        if USE_BINARY_MATCHED_QUADRATIC_CASE and case in (1, 2, 3):
            alpha = 1
            # The three-case counting theorem partitions C and C^perp by
            # the quadratic parity q_{111,100}.  Therefore B_even is a sum
            # of ORDINARY dual-enumerator coefficients B_bucket over the
            # q-even output buckets.  Do not sum the (111,100) transformed
            # spectrum itself here.
            rows = B_rows
            A_even = [0] * nvars
            B_even_num = [0] * nvars
            A_odd = [0] * nvars
            for b, c in col_of.items():
                if binary_quadratic_bucket_parity(b, alpha) == 0:
                    A_even[c] += 1
                else:
                    A_odd[c] += 1
            for ob in universe:
                if binary_quadratic_bucket_parity(ob, alpha) == 0:
                    rr = cyclo_output_dense(rows, ob, nvars)[0]
                    B_even_num = add_rows(B_even_num, rr)
            if case == 1:
                add_eq(A_even, 2 ** (n - k - 1))
                add_eq(B_even_num, norm * (2 ** (n + k - 1)))
            elif case == 2:
                add_eq(A_odd, 0)
                target = 2 ** (n - k - 1) * (4 ** k + 2 ** k)
                add_eq(B_even_num, norm * target)
            elif case == 3:
                add_eq(A_odd, 0)
                target = 2 ** (n - k - 1) * (4 ** k - 2 ** k)
                add_eq(B_even_num, norm * target)
            log["additional"].append(
                f"q=2 matched quadratic/Gauss-sum case {case} for shadow-character alpha=1"
            )

    # ---------------------------------------------------------------
    phase("binary special sectors")

    # Known reference exists.
    # ---------------------------------------------------------------
    rb = reference_bucket()
    if USE_REFERENCE_EXISTENCE:
        if rb not in col_of:
            add_eq({}, 1)
        else:
            inequalities.append((row_single(col_of[rb]), 1))
        log["additional"].append("known stabilizer reference bucket A_g>=1")

    # ---------------------------------------------------------------
    phase("reference existence")

    # Commutation support on B.
    # ---------------------------------------------------------------
    if USE_COMMUTATION:
        bz = 0
        for ob in universe:
            if not bucket_commutes_with_reference(ob, fq_linear):
                comp_rows = cyclo_output_sparse(B_rows, ob)
                for rr in comp_rows:
                    add_eq(dict(rr), 0)
                bz += 1
        mode = "F_q-valued [g,h]=0" if fq_linear else "trace-symplectic Tr[g,h]=0"
        log["additional"].append(f"reference commutation support ({mode}; {bz} B-zero buckets)")

    # ---------------------------------------------------------------
    phase("commutation support")

    # Additive reference action h -> h+alpha g.
    # ---------------------------------------------------------------
    add_group = additive_action_elements(fq_linear)
    if USE_ADDITIVE_ORBIT:
        Aeq = Beq = 0
        seenA, seenB = set(), set()
        for alpha in add_group:
            if alpha == 0:
                continue
            for b in universe:
                bp = additive_bucket_map(b, alpha)
                key = tuple(sorted((repr(b), repr(bp))))
                # A equality.
                if key not in seenA:
                    seenA.add(key)
                    c1, c2 = col_of.get(b), col_of.get(bp)
                    if c1 is not None or c2 is not None:
                        row = {}
                        if c1 is not None: row[c1] = row.get(c1, 0) + 1
                        if c2 is not None: row[c2] = row.get(c2, 0) - 1
                        row = {c:v for c,v in row.items() if v}
                        add_eq(row, 0)
                        Aeq += 1
                # B equality.
                if key not in seenB:
                    seenB.add(key)
                    r1 = cyclo_output_sparse(B_rows, b)[0]
                    r2 = cyclo_output_sparse(B_rows, bp)[0]
                    if b != bp:
                        add_eq(row_add(r1, r2, -1), 0)
                        Beq += 1
        log["additional"].append(
            f"additive reference orbit over {len(add_group)} scalars (A eq={Aeq}, B eq={Beq})"
        )

    if USE_ADDITIVE_ORBIT_DIVISIBILITY:
        Ac = Bc = 0
        for b in universe:
            s = additive_bucket_stabilizer_size(b, fq_linear)
            if s <= 1:
                continue
            col = col_of.get(b)
            if col is not None:
                congruences.append((row_single(col), 0, s))
                Ac += 1
            rowB = dict(cyclo_output_sparse(B_rows, b)[0])
            congruences.append((rowB, 0, norm * s))
            Bc += 1
        log["additional"].append(
            f"additive orbit-stabilizer divisibility (A buckets={Ac}, B buckets={Bc})"
        )

    # ---------------------------------------------------------------
    phase("additive reference action")

    # F_q^x scalar action, valid only for F_q-linear stabilizers.
    # ---------------------------------------------------------------
    if fq_linear and USE_SCALAR_ORBIT:
        Aeq = Beq = 0
        for beta in FF.nonzero:
            if beta == 1:
                continue
            seen = set()
            for b in universe:
                bp = scalar_bucket_map(b, beta)
                key = tuple(sorted((repr(b), repr(bp))))
                if key in seen:
                    continue
                seen.add(key)
                c1, c2 = col_of.get(b), col_of.get(bp)
                if c1 is not None or c2 is not None:
                    row = {}
                    if c1 is not None: row[c1] = row.get(c1, 0) + 1
                    if c2 is not None: row[c2] = row.get(c2, 0) - 1
                    row = {c:v for c,v in row.items() if v}
                    add_eq(row, 0)
                    Aeq += 1
                if b != bp:
                    r1 = cyclo_output_sparse(B_rows, b)[0]
                    r2 = cyclo_output_sparse(B_rows, bp)[0]
                    add_eq(row_add(r1, r2, -1), 0)
                    Beq += 1
        log["additional"].append(
            f"F_q^x scalar orbit symmetry (A eq={Aeq}, B eq={Beq})"
        )
    elif USE_SCALAR_ORBIT:
        log["skipped"].append("F_q^x scalar orbit requires F_q-linear stabilizer")

    if fq_linear and USE_SCALAR_ORBIT_DIVISIBILITY:
        Ac = Bc = 0
        for b in universe:
            if bucket_weight(b) == 0:
                continue
            s = scalar_bucket_stabilizer_size(b)
            if s <= 1:
                continue
            col = col_of.get(b)
            if col is not None:
                congruences.append((row_single(col), 0, s))
                Ac += 1
            rowB = dict(cyclo_output_sparse(B_rows, b)[0])
            congruences.append((rowB, 0, norm * s))
            Bc += 1
        log["additional"].append(
            f"F_q^x orbit-stabilizer divisibility (A buckets={Ac}, B buckets={Bc})"
        )

    # ---------------------------------------------------------------
    phase("scalar action")

    # Cleaning / local dimension relations.
    # ---------------------------------------------------------------
    if USE_CLEANING:
        # Blocks indexed 0=T, 1=U.  Correctable if size<d.
        sizes = (T_SIZE, U_SIZE)
        correctable = [i for i, s in enumerate(sizes) if s < d]
        id_t = (T_SIZE, 0)
        id_u = tuple([U_SIZE] + [0] * (2 * Q - 2))
        for ci in correctable:
            # S_C: other block identity; S_O: correctable block identity.
            row_SC = {}
            row_SO = {}
            for b, col in col_of.items():
                tc, uc = b
                if ci == 0:
                    if uc == id_u: row_SC[col] = 1
                    if tc == id_t: row_SO[col] = 1
                else:
                    if tc == id_t: row_SC[col] = 1
                    if uc == id_u: row_SO[col] = 1
            csize = sizes[ci]
            exponent = n - k - 2 * csize
            if exponent >= 0:
                add_eq(row_add(row_SO, row_SC, -(Q ** exponent)), 0)
            else:
                add_eq(row_add(row_scale(row_SO, Q ** (-exponent)), row_SC, -1), 0)

            # N_O: B with correctable block identity.
            selected = []
            for ob in universe:
                tc, uc = ob
                is_id = (tc == id_t) if ci == 0 else (uc == id_u)
                if is_id:
                    selected.append(cyclo_output_sparse(B_rows, ob)[0])
            B_O_num = sum_sparse_rows(selected)
            # |N_O| = q^(2k) |S_O|.
            add_eq(row_add(B_O_num, row_SO, -(norm * (Q ** (2 * k)))), 0)
        if correctable:
            log["additional"].append(
                "cleaning/dimension constraints for correctable blocks " + ",".join("T" if i == 0 else "U" for i in correctable)
            )

    # ---------------------------------------------------------------
    phase("cleaning / local dimensions")

    # A upper bounds.
    # ---------------------------------------------------------------
    upper_bounds = {}
    for b, col in col_of.items():
        cap = bucket_capacity(b)
        w = bucket_weight(b)
        shell = fixed_A[w] if fixed_A is not None else norm
        upper_bounds[col] = min(shell, cap) if USE_CELL_CAPACITY else shell

    phase("upper bounds / finalize")

    return {
        "q": Q,
        "fq_linear": fq_linear,
        "case": case,
        "T_SIZE": T_SIZE,
        "U_SIZE": U_SIZE,
        "universe": universe,
        "active_buckets": active,
        "col_of": col_of,
        "nvars": nvars,
        "norm": norm,
        "fixed_A": fixed_A,
        "eq_rows": eq_rows,
        "eq_rhs": eq_rhs,
        "inequalities": inequalities,
        "congruences": congruences,
        "upper_bounds": upper_bounds,
        "constraint_log": log,
        "B_rows": B_rows,
        "twist_rows": {},
        "shadow_rows": {},
        "binary_shadow_rows": binary_shadow_rows,
    }


# =====================================================================
# Exact RREF parameterization
# =====================================================================

def exact_rref_parameterization(nvars, eq_rows, eq_rhs):
    if not eq_rows:
        exprs = []
        for j in range(nvars):
            coeff = [Fraction(0)] * nvars
            coeff[j] = Fraction(1)
            exprs.append((Fraction(0), tuple(coeff)))
        return {
            "inconsistent": False, "rank": 0, "pivots": tuple(),
            "free_cols": list(range(nvars)), "exprs": exprs,
            "denominator_set": [1], "max_den": 1,
        }

    M = sp.Matrix(eq_rows)
    b = sp.Matrix(eq_rhs)
    Aug = M.row_join(b)
    Dm = DomainMatrix.from_Matrix(Aug)
    R, pivots_all = Dm.rref(method="GJ")
    RM = R.to_Matrix()

    if nvars in pivots_all:
        return {
            "inconsistent": True,
            "rank": len([p for p in pivots_all if p < nvars]),
        }

    pivots = tuple(p for p in pivots_all if p < nvars)
    pset = set(pivots)
    free_cols = [j for j in range(nvars) if j not in pset]
    free_pos = {c: i for i, c in enumerate(free_cols)}
    nfree = len(free_cols)
    exprs = [None] * nvars

    for col in free_cols:
        coeff = [Fraction(0)] * nfree
        coeff[free_pos[col]] = Fraction(1)
        exprs[col] = (Fraction(0), tuple(coeff))

    for ri, pc in enumerate(pivots):
        const = Fraction(RM[ri, nvars])
        coeff = tuple(-Fraction(RM[ri, fc]) for fc in free_cols)
        exprs[pc] = (const, coeff)

    dens = set()
    max_den = 1
    for const, coeff in exprs:
        den = const.denominator
        for c in coeff:
            den = math.lcm(den, c.denominator)
        dens.add(den)
        max_den = max(max_den, den)

    return {
        "inconsistent": False,
        "rank": len(pivots),
        "pivots": pivots,
        "free_cols": free_cols,
        "exprs": exprs,
        "denominator_set": sorted(dens),
        "max_den": max_den,
    }


def substitute_original_row(row, rhs, exprs, nfree):
    const = Fraction(-rhs)
    coeff = [Fraction(0)] * nfree
    for col, scale in enumerate(row):
        if not scale:
            continue
        ec, ev = exprs[col]
        const += Fraction(scale) * ec
        for r, c in enumerate(ev):
            coeff[r] += Fraction(scale) * c
    return const, tuple(coeff)


def integerize_affine(const, coeff):
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


def canonical_geq(c0, ci):
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
    return Sum(terms) if len(terms) > 1 else terms[0]


# =====================================================================
# Reduced Z3 model
# =====================================================================

def build_reduced_solver(problem, param):
    nvars = problem["nvars"]
    active = problem["active_buckets"]
    upper = problem["upper_bounds"]
    free_cols = param["free_cols"]
    exprs = param["exprs"]
    nfree = len(free_cols)
    fset = set(free_cols)
    free_vars = [Int(f"F_{r}") for r, _ in enumerate(free_cols)]
    solver = SolverFor("QF_LIA")
    if TIMEOUT_MS is not None:
        solver.set(timeout=int(TIMEOUT_MS))

    for var, col in zip(free_vars, free_cols):
        solver.add(var >= 0)
        solver.add(var <= upper[col])

    pivot_congruences = set()
    for col in range(nvars):
        if col in fset:
            continue
        const, coeff = exprs[col]
        c0, ci, den = integerize_affine(const, coeff)
        lin = z3_linear(c0, ci, free_vars)
        if den > 1:
            sig = (c0 % den, tuple(c % den for c in ci), den)
            if sig not in pivot_congruences:
                solver.add(lin % den == 0)
                pivot_congruences.add(sig)
        solver.add(lin >= 0)
        solver.add(lin <= den * upper[col])

    unique_mod = set()
    for row, rhs, modulus in problem["congruences"]:
        const, coeff = substitute_original_row(row, rhs, exprs, nfree)
        c0, ci, den = integerize_affine(const, coeff)
        mod = den * int(modulus)
        sig = (c0 % mod, tuple(c % mod for c in ci), mod)
        if sig in unique_mod:
            continue
        unique_mod.add(sig)
        solver.add(z3_linear(c0, ci, free_vars) % mod == 0)

    unique_ineq = set()
    trivial_false = 0
    for row, rhs in problem["inequalities"]:
        const, coeff = substitute_original_row(row, rhs, exprs, nfree)
        c0, ci, den = integerize_affine(const, coeff)
        c0, ci = canonical_geq(c0, ci)
        if all(c == 0 for c in ci):
            if c0 < 0:
                solver.add(False)
                trivial_false += 1
            continue
        sig = (c0, ci)
        if sig in unique_ineq:
            continue
        unique_ineq.add(sig)
        solver.add(z3_linear(c0, ci, free_vars) >= 0)

    meta = {
        "free_vars": len(free_vars),
        "pivot_congruences": len(pivot_congruences),
        "original_modular": len(unique_mod),
        "unique_ineq": len(unique_ineq),
        "trivial_false": trivial_false,
    }
    return solver, free_vars, meta


# =====================================================================
# Optional model reconstruction
# =====================================================================

def eval_expr(model, free_vars, expr):
    const, coeff = expr
    value = const
    for c, var in zip(coeff, free_vars):
        if c:
            value += c * model.eval(var, model_completion=True).as_long()
    if value.denominator != 1:
        raise AssertionError(f"Nonintegral reconstructed A={value}.")
    return value.numerator


def print_model(problem, param, solver, free_vars):
    model = solver.model()
    print("Nonzero projected A coefficients:")
    for b, expr in zip(problem["active_buckets"], param["exprs"]):
        v = eval_expr(model, free_vars, expr)
        if v == 0 and not SHOW_ZERO_MODEL_ENTRIES:
            continue
        print(f"  A[{bucket_label(b)}] = {v}")


# =====================================================================
# Internal mathematical regression tests
# =====================================================================

def internal_self_test(q):
    reset_field(q)
    global Q
    Q = q

    H = hamming_cells()
    R = reta_cells()
    QH = projected_fourier_matrix(H)
    QR = projected_fourier_matrix(R)
    assert QH is not None and QR is not None

    # Hamming MacWilliams matrix must be [[1,q^2-1],[1,-1]].
    assert matrix_constant_part(QH) == ((1, q*q - 1), (1, -1))

    # R_eta has exactly 2q-1 cells and every eta character descends.
    assert len(R) == 2*q - 1
    for alpha in FF.elements:
        assert projected_fourier_matrix(R, vec_scale(alpha, ETA)) is not None

    # Additive eta translations and nonzero scalar multiplication preserve R_eta.
    for alpha in FF.elements:
        assert additive_u_permutation(alpha) is not None
    for beta in FF.nonzero:
        assert scalar_u_permutation(beta) is not None

    # q=2 must reduce to I/X/(two-element C_1) and four binary sectors.
    if q == 2:
        assert tuple(len(c) for c in R) == (1, 1, 2)


# =====================================================================
# Progress helper
# =====================================================================

def run_stage_with_heartbeat(label, fn):
    if not SHOW_LIVE_PROGRESS:
        return fn()
    done = threading.Event()
    box = {}

    def worker():
        try:
            box["value"] = fn()
        except BaseException as exc:
            box["exc"] = exc
        finally:
            done.set()

    th = threading.Thread(target=worker, daemon=True)
    start = time.perf_counter()
    th.start()
    while not done.wait(PROGRESS_EVERY_SECONDS):
        print(f"        {label}: {time.perf_counter()-start:.1f}s elapsed ...", flush=True)
    th.join()
    if "exc" in box:
        raise box["exc"]
    return box.get("value")



def run_native_stage_with_heartbeat(label, fn):
    """Heartbeat for native calls that may hold the Python GIL (notably HiGHS).

    A separate Python subprocess prints directly to the inherited console, so
    the heartbeat remains visible even if scipy/HiGHS blocks the parent Python
    interpreter inside C/C++ for a long time.
    """
    if not SHOW_LIVE_PROGRESS:
        return fn()
    interval = float(PROGRESS_EVERY_SECONDS)
    hb_code = (
        "import sys,time\n"
        "label=sys.argv[1]; interval=float(sys.argv[2]); start=time.monotonic()\n"
        "while True:\n"
        "    time.sleep(interval)\n"
        "    print(f'        {label}: {time.monotonic()-start:.1f}s elapsed ... still running', flush=True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", hb_code, str(label), str(interval)],
        stdout=None, stderr=None,
    )
    try:
        return fn()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


# =====================================================================
# Candidate database helpers
# =====================================================================

def load_candidate_database(path):
    spec = importlib.util.spec_from_file_location(f"hxr_qdit_db_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate database: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_database_solver_key(db, db_path):
    if SOLVER_NAME in db.REFINED_SOLVER_KEYS:
        return False
    # Newer SLR_qudit already includes this key.  Refuse silent schema surgery
    # if an old DB is supplied: regenerating/migrating explicitly is safer.
    raise ValueError(
        f"Candidate DB {db_path} does not contain solver key {SOLVER_NAME!r}. "
        "Regenerate it with the current SLR_qudit.py or add the key explicitly."
    )


def positive_reference_weights(fixed_A, order="descending"):
    vals = sorted((int(w) for w, v in fixed_A.items() if int(w) > 0 and int(v) > 0))
    if order == "descending":
        vals.reverse()
    return vals


def effective_fq_linearity(q, db_assumption):
    # Prime-q stabilizers are additive over F_q automatically.
    return is_prime(q) or bool(db_assumption)


# =====================================================================
# One reference branch / one candidate
# =====================================================================

def _rows_to_csr(rows, nvars, scale=1.0):
    """Convert sparse dict rows directly to CSR without dense Python rows."""
    if not rows:
        return None
    data, indices, indptr = [], [], [0]
    for row in rows:
        for j, value in sorted(row.items()):
            if value:
                data.append(float(scale * value))
                indices.append(int(j))
        indptr.append(len(data))
    return csr_matrix((np.asarray(data, dtype=float),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(rows), nvars))


def _classify_highs_result(res):
    """Map SciPy/HiGHS result to sat/unsat/unknown for a pure feasibility LP."""
    if res.status == 0:
        return "sat"
    if res.status == 2:
        return "unsat"
    return "unknown"


def _highs_attempt(method, nvars, A_ub, b_ub, A_eq, b_eq, bounds):
    """Run one HiGHS algorithm with NO time limit."""
    print(f"        [solve] starting {method}; time limit = NONE", flush=True)
    t0 = time.perf_counter()
    res = run_native_stage_with_heartbeat(
        f"HiGHS/{method}",
        lambda: linprog(
            c=np.zeros(nvars, dtype=float),
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            method=method,
            options={"presolve": True},
        ),
    )
    dt = time.perf_counter() - t0
    status = _classify_highs_result(res)
    print(
        f"        [solve] {method} finished: {status.upper()} "
        f"(scipy status={res.status}) [{dt:.3f}s]",
        flush=True,
    )
    print(f"                {res.message}", flush=True)
    return status, res, dt


def _solve_highs_robust(nvars, A_ub, b_ub, A_eq, b_eq, bounds):
    """Use HiGHS without a time limit and retry numerical/Unknown exits.

    The generic 'highs' driver may occasionally return SciPy status=4 with a
    HiGHS model_status Unknown even though its primal status says Infeasible.
    Such a numerical/driver Unknown is NOT accepted as an exclusion.  Retry
    explicitly with dual simplex and IPM; only a genuine status=2 is UNSAT.
    """
    attempts = []
    for method in ("highs", "highs-ds", "highs-ipm"):
        status, res, dt = _highs_attempt(method, nvars, A_ub, b_ub, A_eq, b_eq, bounds)
        attempts.append((method, status, res, dt))
        if status in ("sat", "unsat"):
            return status, res, attempts
        print(f"        [solve] {method} returned UNKNOWN; retrying with another HiGHS algorithm ...", flush=True)
    return "unknown", attempts[-1][2], attempts


def solve_reference_branch(fixed_A, case, u, fq_linear):
    """Solve one mandatory stabilizer-reference branch as a REAL LP.

    No solver time limit is ever set.  Build and solve stages emit a heartbeat
    every PROGRESS_EVERY_SECONDS (default 3600 s).
    """
    set_reference_size(u)
    start = time.perf_counter()
    print(f"    reference u={u}: T={T_SIZE}, U={U_SIZE}, fq_linear={fq_linear}", flush=True)
    print(f"      heartbeat interval={PROGRESS_EVERY_SECONDS:g}s; solver time limit=NONE", flush=True)

    t0 = time.perf_counter()
    problem = run_stage_with_heartbeat("build", lambda: build_problem(fixed_A, case, fq_linear))
    build_time = time.perf_counter() - t0
    nvars = problem["nvars"]
    print(
        f"      build complete: raw={len(problem['universe'])}, active A={nvars}, "
        f"eq={len(problem['eq_rows'])}, ineq={len(problem['inequalities'])}, "
        f"ignored congruences={len(problem['congruences'])} [{build_time:.3f}s]",
        flush=True,
    )
    if problem["constraint_log"]["basic"]:
        print("      basic constraint families:", flush=True)
        for item in problem["constraint_log"]["basic"]:
            print(f"        - {item}", flush=True)
    if problem["constraint_log"]["additional"]:
        print("      additional constraint families:", flush=True)
        for item in problem["constraint_log"]["additional"]:
            print(f"        - {item}", flush=True)
    if problem["constraint_log"]["skipped"]:
        print("      skipped:", flush=True)
        for item in problem["constraint_log"]["skipped"]:
            print(f"        - {item}", flush=True)

    # HiGHS form: A_ub x <= b_ub. Builder stores row.x >= rhs.
    eq_rows = problem["eq_rows"]
    eq_rhs = np.asarray(problem["eq_rhs"], dtype=float)
    ub_rows = [row for row, _ in problem["inequalities"]]
    ub_rhs = [-float(rhs) for _, rhs in problem["inequalities"]]

    print("      [matrix] converting exact rows to sparse CSR ...", flush=True)
    t0 = time.perf_counter()
    A_eq, A_ub = run_stage_with_heartbeat(
        "CSR assembly",
        lambda: (_rows_to_csr(eq_rows, nvars), _rows_to_csr(ub_rows, nvars, scale=-1.0)),
    )
    matrix_time = time.perf_counter() - t0
    b_ub = np.asarray(ub_rhs, dtype=float) if ub_rhs else None
    bounds = [(0.0, float(problem["upper_bounds"][j])) for j in range(nvars)]
    eq_nnz = 0 if A_eq is None else int(A_eq.nnz)
    ineq_nnz = 0 if A_ub is None else int(A_ub.nnz)
    sparse_bytes = 0
    for mat in (A_eq, A_ub):
        if mat is not None:
            sparse_bytes += mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
    print(
        f"      [matrix] vars={nvars}, eq={0 if A_eq is None else A_eq.shape[0]}, "
        f"ineq={0 if A_ub is None else A_ub.shape[0]}, nnz(eq)={eq_nnz}, "
        f"nnz(ineq)={ineq_nnz}, CSR~{sparse_bytes/2**20:.1f} MiB [{matrix_time:.3f}s]",
        flush=True,
    )

    status, res, attempts = _solve_highs_robust(
        nvars, A_ub, b_ub, A_eq, eq_rhs if eq_rows else None, bounds
    )
    solve_time = sum(dt for _, _, _, dt in attempts)
    elapsed = time.perf_counter() - start
    print(
        f"      => FINAL {status.upper()} after {len(attempts)} HiGHS attempt(s) "
        f"[solve={solve_time:.3f}s, total={elapsed:.3f}s]",
        flush=True,
    )
    if status == "sat" and SHOW_MODEL:
        nz = [(problem['active_buckets'][j], x) for j, x in enumerate(res.x) if abs(x) > 1e-9]
        print(f"      nonzero A buckets: {len(nz)}")
        for b, x in nz:
            print(f"        {bucket_label(b)} = {x:.12g}")

    return status, {
        "u": u, "t": n-u,
        "raw_buckets": len(problem["universe"]),
        "active_A_vars": nvars,
        "eq_rows": len(eq_rows),
        "inequalities": len(ub_rows),
        "eq_nnz": eq_nnz,
        "ineq_nnz": ineq_nnz,
        "runtime_build": build_time,
        "runtime_matrix": matrix_time,
        "runtime_solve": solve_time,
        "runtime_total": elapsed,
        "highs_attempts": [
            {"method": m, "status": s, "scipy_status": int(r.status), "message": str(r.message), "seconds": dt}
            for m, s, r, dt in attempts
        ],
        "relaxation": "continuous real LP; all integrality/modular/divisibility constraints omitted",
        "time_limit": None,
    }

def solve_candidate_all_references(fixed_A, case, fq_linear, *, order="descending",
                                   min_u=None, max_u=None, max_u_tests=None):
    uvals = positive_reference_weights(fixed_A, order=order)
    if min_u is not None:
        uvals = [u for u in uvals if u >= min_u]
    if max_u is not None:
        uvals = [u for u in uvals if u <= max_u]
    if max_u_tests is not None:
        uvals = uvals[:max_u_tests]
    if not uvals:
        return "error", {"reason": "no_positive_reference_shell", "tested_u": []}

    tests = []
    saw_unknown = False
    for u in uvals:
        try:
            status, info = solve_reference_branch(fixed_A, case, u, fq_linear)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            status, info = "error", {"u": u, "error": repr(exc)}
            print(f"      => ERROR {exc!r}")
        tests.append({"status": status, **info})
        if status == "unsat":
            return "unsat", {"excluded_u": u, "tested_u": uvals, "branches": tests}
        if status in ("unknown", "error"):
            saw_unknown = True
    return ("unknown" if saw_unknown else "sat"), {"tested_u": uvals, "branches": tests}


# =====================================================================
# Direct mode (without candidate DB), useful for development
# =====================================================================

def direct_solve(fixed_A, case, fq_linear, u):
    status, info = solve_reference_branch(fixed_A, case, u, fq_linear)
    print(status.upper(), info)
    return status



def standalone_reference_weights(order="descending", min_u=None, max_u=None, max_u_tests=None):
    """All possible nonidentity stabilizer weights for a standalone sweep."""
    vals = list(range(1, n + 1))
    if order == "descending":
        vals.reverse()
    if min_u is not None:
        vals = [u for u in vals if u >= min_u]
    if max_u is not None:
        vals = [u for u in vals if u <= max_u]
    if max_u_tests is not None:
        vals = vals[:max_u_tests]
    return vals


def standalone_cases():
    """Global quadratic cases that must be covered when no SLR DB fixes one."""
    if Q == 2 and USE_BINARY_MATCHED_QUADRATIC_CASE:
        return (1, 2, 3)
    return (0,)


def solve_standalone_all_references(fq_linear, *, order="descending",
                                    min_u=None, max_u=None, max_u_tests=None,
                                    dry_run=False):
    """
    Feasibility test without an SLR candidate database.

    The ordinary A distribution is left free.  A nontrivial stabilizer code
    (n>k) has at least one nonidentity stabilizer element, hence at least one
    reference weight u in {1,...,n}.  Therefore:

      * one SAT (case,u) branch => NOT EXCLUDED by this solver;
      * EXCLUDED is valid only if every admissible branch is UNSAT;
      * any UNKNOWN/ERROR, or a restricted u sweep with no SAT, is INCONCLUSIVE.
    """
    if n <= k:
        return "not_applicable", {
            "reason": "n_equals_k_trivial_stabilizer_has_no_nonidentity_reference",
            "branches": [],
        }

    uvals = standalone_reference_weights(
        order=order, min_u=min_u, max_u=max_u, max_u_tests=max_u_tests
    )
    cases = standalone_cases()
    restricted = (min_u is not None or max_u is not None or max_u_tests is not None)

    if not uvals:
        return "inconclusive", {
            "reason": "empty_reference_weight_sweep",
            "branches": [],
            "restricted": restricted,
        }

    if dry_run:
        print("Standalone branches (A_w free):")
        print(f"  cases = {list(cases)}")
        print(f"  u     = {uvals}")
        print(f"  total = {len(cases) * len(uvals)} branch(es)")
        return "dry_run", {
            "cases": list(cases), "tested_u": uvals, "restricted": restricted,
            "branches": [],
        }

    tests = []
    saw_unknown = False
    saw_error = False
    total = len(cases) * len(uvals)
    pos = 0

    for case in cases:
        for u in uvals:
            pos += 1
            case_label = f"case={case}, " if Q == 2 else ""
            print("\n" + "-" * 104)
            print(f"Standalone branch [{pos}/{total}]: {case_label}u={u}; ordinary A_w free")
            print("-" * 104)
            try:
                status, info = solve_reference_branch(
                    None, case, u, fq_linear
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                status, info = "error", {"u": u, "case": case, "error": repr(exc)}
                print(f"      => ERROR {exc!r}")

            entry = {"status": status, "case": case, **info}
            tests.append(entry)

            if status == "sat":
                return "not_excluded", {
                    "sat_case": case,
                    "sat_u": u,
                    "tested_u": uvals,
                    "restricted": restricted,
                    "branches": tests,
                }
            if status == "unknown":
                saw_unknown = True
            elif status == "error":
                saw_error = True

    if restricted:
        return "inconclusive", {
            "reason": "restricted_reference_sweep_no_sat",
            "tested_u": uvals,
            "restricted": True,
            "branches": tests,
        }
    if saw_unknown or saw_error:
        return "inconclusive", {
            "reason": "unknown_or_error_branch_prevents_exclusion",
            "tested_u": uvals,
            "restricted": False,
            "branches": tests,
        }
    return "excluded", {
        "reason": "all_reference_branches_unsat",
        "tested_u": uvals,
        "restricted": False,
        "branches": tests,
    }


# =====================================================================
# Main
# =====================================================================

def exact_support_mds_count(length, distance, alphabet):
    if length == 0:
        return 1
    if length < distance:
        return 0
    return (alphabet - 1) * sum(
        (-1) ** ell * math.comb(length - 1, ell) * alphabet ** (length - distance - ell)
        for ell in range(length - distance + 1)
    )


def mds_distribution(length, distance, alphabet):
    return [
        1 if w == 0 else math.comb(length, w) * exact_support_mds_count(w, distance, alphabet)
        for w in range(length + 1)
    ]


def mds_split_distribution(t, u, distance, alphabet):
    out = {}
    for i in range(t + 1):
        for r in range(u + 1):
            w = i + r
            mult = 1 if w == 0 else exact_support_mds_count(w, distance, alphabet)
            out[(i, r)] = math.comb(t, i) * math.comb(u, r) * mult
    return out


def main():
    global Q, n, k, d, ASSUME_FQ_LINEAR, FIXED_SPLIT_A, FIXED_SPLIT_B, PROGRESS_EVERY_SECONDS
    parser = argparse.ArgumentParser(
        description="QMDS H x R_eta (HxRX) stabilizer-reference SPARSE REAL-LP solver for q-ary codes"
    )
    parser.add_argument("q", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument("--u", type=int, action="append", default=None)
    parser.add_argument("--u-order", choices=("ascending","descending"), default="ascending")
    parser.add_argument("--continue-after-unsat", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=3600.0,
                        help="progress heartbeat interval; default 3600 s; does NOT impose a solver timeout")
    parser.add_argument("--fq-linear", action="store_true",
                        help="for nonprime q, restrict to F_q-linear stabilizers")
    args = parser.parse_args()

    Q, n, k, d = args.q, args.n, args.k, args.d
    reset_field(Q)
    expected_k = n - 2 * (d - 1)
    if k != expected_k:
        raise SystemExit(f"Not QMDS parameters: require k=n-2(d-1)={expected_k}, got {k}.")
    if k < 0 or d < 1 or d > n:
        raise SystemExit("Invalid QMDS parameters.")
    if args.heartbeat_seconds <= 0:
        raise SystemExit("--heartbeat-seconds must be positive.")
    PROGRESS_EVERY_SECONDS = float(args.heartbeat_seconds)

    ASSUME_FQ_LINEAR = True if is_prime(Q) else bool(args.fq_linear)
    fq_linear = effective_fq_linearity(Q, ASSUME_FQ_LINEAR)
    internal_self_test(Q)

    alphabet = Q * Q
    dA = n - d + 2
    avec = mds_distribution(n, dA, alphabet)
    bvec = mds_distribution(n, d, alphabet)
    cardA = Q ** (n-k)
    cardB = Q ** (n+k)
    if sum(avec) != cardA:
        raise AssertionError(f"MDS stabilizer distribution cardinality {sum(avec)} != {cardA}")
    if sum(bvec) != cardB:
        raise AssertionError(f"MDS dual distribution cardinality {sum(bvec)} != {cardB}")

    mandatory = [u for u in range(1,n+1) if avec[u] > 0]
    if args.u:
        wanted=set(args.u); mandatory=[u for u in mandatory if u in wanted]
    mandatory.sort(reverse=(args.u_order=="descending"))

    print("="*108)
    print("QUDIT QMDS H x R_eta (HxRX) STABILIZER-REFERENCE -- LAZY-CUT REAL LP (HiGHS)")
    print(f"Code: [[{n},{k},{d}]]_{Q}; local Pauli alphabet={alphabet}")
    print(f"Stabilizer MDS distance D_A={dA}; F_q-linear={fq_linear}")
    print("A_w = {" + ", ".join(f"{w}:{v}" for w,v in enumerate(avec) if v) + "}")
    print("B_w = {" + ", ".join(f"{w}:{v}" for w,v in enumerate(bvec) if v) + "}")
    print(f"Mandatory stabilizer-reference weights ({args.u_order}): {mandatory}")
    print("MDS specialization: exact ordinary/split A and B distributions; zero split-A cells pruned before transforms")
    print("Relaxation: continuous real A variables; integrality/Mod/divisibility constraints omitted")
    print(f"Runtime: NO solver time limit; heartbeat every {PROGRESS_EVERY_SECONDS:g} s")
    print(f"R_eta local cells={2*Q-1}; compatible character sectors={Q}")
    print("="*108)

    any_unknown=False; excluded_by=None
    for idx_u,u in enumerate(mandatory,1):
        set_reference_size(u)
        FIXED_SPLIT_A = mds_split_distribution(n-u, u, dA, alphabet)
        FIXED_SPLIT_B = mds_split_distribution(n-u, u, d, alphabet)
        raw = (n-u+1) * math.comb(u + (2*Q-1)-1, (2*Q-1)-1)
        nonzero_split = sum(v>0 for v in FIXED_SPLIT_A.values())
        print("\n"+"-"*108)
        print(f"[{idx_u}/{len(mandatory)}] reference u={u}: T={n-u}, U={u}, A_u={avec[u]}")
        print(f"    raw composition buckets before MDS/commutation pruning={raw}; positive A split shells={nonzero_split}")
        status, details = solve_reference_branch(avec, 0, u, fq_linear)
        if status == "unsat":
            excluded_by=u
            print(f"    >>> mandatory reference u={u} UNSAT; QMDS code EXCLUDED.")
            if not args.continue_after_unsat:
                break
        elif status == "unknown":
            any_unknown=True

    print("\n"+"="*108)
    if excluded_by is not None:
        print(f"FINAL: EXCLUDED by mandatory stabilizer-reference weight u={excluded_by}.")
    elif any_unknown:
        print("FINAL: INCONCLUSIVE (no UNSAT found, at least one UNKNOWN).")
    else:
        print("FINAL: SURVIVES all tested QMDS HxR_eta REAL-LP stabilizer-reference constraints.")
    print("="*108)



# =====================================================================
# REDUCED-COORDINATE REAL-LP BACKEND (2026-09-20)
#
# Architecture inherited from the exact/integer solvers:
#   1) prune fixed-zero/commutation cells;
#   2) quotient A variables by exact visible stabilizer/scalar orbits;
#   3) build ONLY core equalities and exact-RREF them;
#   4) build only the non-rational character-twist equalities in the reduced
#      coordinates and exact-RREF a second time;
#   5) only then generate positivity/capacity/shadow inequalities directly
#      in the final free coordinates;
#   6) give HiGHS the reduced continuous LP.  No integer/Mod/divisibility
#      information is used anywhere in this backend.
#
# The old sparse backend remains above for regression and for q=2 fallback,
# but solve_reference_branch() is overridden below for q>=3.
# =====================================================================

_LEGACY_SPARSE_SOLVE_REFERENCE_BRANCH = solve_reference_branch


def _F(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    return Fraction(x)


def _clean_sparse_fraction_row(row):
    out = {}
    for j, v in row.items():
        fv = _F(v)
        if fv:
            out[int(j)] = out.get(int(j), Fraction(0)) + fv
    return {j: v for j, v in out.items() if v}


def _normalized_equation_signature(row, rhs):
    """Scale-invariant exact signature for row*x=rhs."""
    row = _clean_sparse_fraction_row(row)
    rhs = _F(rhs)
    if not row:
        return (), rhs
    j0 = min(row)
    a0 = row[j0]
    # Normalize the first nonzero coefficient to +1 exactly.
    nrow = tuple((j, row[j] / a0) for j in sorted(row))
    nrhs = rhs / a0
    return nrow, nrhs


def exact_sparse_rref_parameterization(nvars, eq_rows, eq_rhs, label="RREF"):
    """Exact rational RREF from sparse rows, returning sparse affine expressions.

    Each original variable x_j is represented as
        x_j = const_j + sum_r coeff_j[r] * y_r,
    where y are the RREF free coordinates.
    """
    t0 = time.perf_counter()
    seen = set()
    rows = []
    rhs2 = []
    trivial = 0
    duplicates = 0

    for row, rhs in zip(eq_rows, eq_rhs):
        row = _clean_sparse_fraction_row(row)
        rhs = _F(rhs)
        if not row:
            if rhs != 0:
                return {
                    "inconsistent": True, "rank": 0, "free_cols": [], "exprs": [],
                    "input_rows": len(eq_rows), "unique_rows": len(rows),
                    "duplicates": duplicates, "trivial": trivial,
                    "elapsed": time.perf_counter()-t0,
                    "reason": f"{label}: 0 = {rhs}",
                }
            trivial += 1
            continue
        sig = _normalized_equation_signature(row, rhs)
        if sig in seen:
            duplicates += 1
            continue
        seen.add(sig)
        rows.append(row)
        rhs2.append(rhs)

    if not rows:
        exprs = [(Fraction(0), {j: Fraction(1)}) for j in range(nvars)]
        return {
            "inconsistent": False, "rank": 0, "pivots": tuple(),
            "free_cols": list(range(nvars)), "exprs": exprs,
            "input_rows": len(eq_rows), "unique_rows": 0,
            "duplicates": duplicates, "trivial": trivial,
            "max_expr_nnz": 1 if nvars else 0,
            "total_expr_nnz": nvars,
            "elapsed": time.perf_counter()-t0,
        }

    # Sparse augmented matrix over QQ.  Using DomainMatrix keeps the exact
    # arithmetic path from the integer solvers without materializing Python
    # dense coefficient rows first.
    entries = {}
    for i, (row, rhs) in enumerate(zip(rows, rhs2)):
        for j, v in row.items():
            entries[(i, j)] = sp.Rational(v.numerator, v.denominator)
        if rhs:
            entries[(i, nvars)] = sp.Rational(rhs.numerator, rhs.denominator)
    Aug = sp.SparseMatrix(len(rows), nvars + 1, entries)
    Dm = DomainMatrix.from_Matrix(Aug)
    R, pivots_all = Dm.rref(method="GJ")

    pivots_all = tuple(int(p) for p in pivots_all)
    if nvars in pivots_all:
        return {
            "inconsistent": True,
            "rank": len([p for p in pivots_all if p < nvars]),
            "free_cols": [], "exprs": [],
            "input_rows": len(eq_rows), "unique_rows": len(rows),
            "duplicates": duplicates, "trivial": trivial,
            "elapsed": time.perf_counter()-t0,
            "reason": f"{label}: augmented-column pivot",
        }

    pivots = tuple(p for p in pivots_all if p < nvars)
    pset = set(pivots)
    free_cols = [j for j in range(nvars) if j not in pset]
    free_pos = {c: i for i, c in enumerate(free_cols)}
    nfree = len(free_cols)

    # Avoid a dense RM[rank,free] scan.  Pull only nonzeros from the RREF.
    RM = R.to_Matrix()
    dok = RM.todok() if hasattr(RM, "todok") else {(i,j): RM[i,j] for i in range(RM.rows) for j in range(RM.cols) if RM[i,j]}
    by_row = defaultdict(dict)
    for (ri, cj), val in dok.items():
        by_row[int(ri)][int(cj)] = _F(val)

    exprs = [None] * nvars
    for col in free_cols:
        exprs[col] = (Fraction(0), {free_pos[col]: Fraction(1)})

    for ri, pc in enumerate(pivots):
        rd = by_row.get(ri, {})
        const = rd.get(nvars, Fraction(0))
        coeff = {}
        for fc in free_cols:
            v = rd.get(fc)
            if v:
                coeff[free_pos[fc]] = -v
        exprs[pc] = (const, coeff)

    total_expr_nnz = sum(len(c) for _, c in exprs)
    max_expr_nnz = max((len(c) for _, c in exprs), default=0)
    return {
        "inconsistent": False,
        "rank": len(pivots),
        "pivots": pivots,
        "free_cols": free_cols,
        "exprs": exprs,
        "input_rows": len(eq_rows),
        "unique_rows": len(rows),
        "duplicates": duplicates,
        "trivial": trivial,
        "max_expr_nnz": max_expr_nnz,
        "total_expr_nnz": total_expr_nnz,
        "elapsed": time.perf_counter()-t0,
    }


def _substitute_sparse_row(row, rhs, param):
    """Return affine expression row*x-rhs in param free coordinates."""
    const = -_F(rhs)
    coeff = {}
    exprs = param["exprs"]
    for col, scale0 in row.items():
        scale = _F(scale0)
        if not scale:
            continue
        ec, ev = exprs[int(col)]
        const += scale * ec
        for r, c in ev.items():
            nv = coeff.get(r, Fraction(0)) + scale * c
            if nv:
                coeff[r] = nv
            elif r in coeff:
                del coeff[r]
    return const, coeff


def _compose_parameterizations(param1, param2):
    """Compose x(y) with y(z), returning x(z)."""
    exprs2 = param2["exprs"]
    out = []
    for c0, cv in param1["exprs"]:
        const = c0
        coeff = {}
        for y, scale in cv.items():
            ec, ev = exprs2[y]
            const += scale * ec
            for z, a in ev.items():
                nv = coeff.get(z, Fraction(0)) + scale * a
                if nv:
                    coeff[z] = nv
                elif z in coeff:
                    del coeff[z]
        out.append((const, coeff))
    return {
        "inconsistent": False,
        "rank": param1.get("rank", 0) + param2.get("rank", 0),
        "free_cols": list(range(len(param2["free_cols"]))),
        "exprs": out,
        "stage1": param1,
        "stage2": param2,
        "nfree": len(param2["free_cols"]),
    }


def _accum_col(row, col, value=1):
    if col is None or not value:
        return
    nv = row.get(col, 0) + value
    if nv:
        row[col] = nv
    elif col in row:
        del row[col]


def _build_active_orbit_quotient(fixed_A, fq_linear):
    """Prune impossible A cells, then quotient exact visible A-orbit equalities.

    If an orbit maps an otherwise-active bucket into a pruned zero bucket, the
    whole active orbit component is forced to zero and is removed before any
    transform is built.  This is exactly what the old explicit A_b=A_b' rows
    implied, but without carrying those rows into RREF/HiGHS.
    """
    universe = all_buckets()
    initial = []
    for bucket in universe:
        w = bucket_weight(bucket)
        if fixed_A is not None and fixed_A[w] == 0:
            continue
        if FIXED_SPLIT_A is not None:
            i_split = bucket[0][1]
            r_split = U_SIZE - bucket[1][0]
            if FIXED_SPLIT_A.get((i_split, r_split), 0) == 0:
                continue
        if USE_COMMUTATION and not bucket_commutes_with_reference(bucket, fq_linear):
            continue
        initial.append(bucket)

    idx = {b: i for i, b in enumerate(initial)}
    parent = list(range(len(initial)))
    rank = [0] * len(initial)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    forced_idx = set()
    actions = []
    if USE_ADDITIVE_ORBIT:
        for alpha in additive_action_elements(fq_linear):
            if alpha != 0:
                actions.append(("add", alpha))
    if fq_linear and USE_SCALAR_ORBIT:
        for beta in FF.nonzero:
            if beta != 1:
                actions.append(("scale", beta))

    for b in initial:
        i = idx[b]
        for typ, a in actions:
            bp = additive_bucket_map(b, a) if typ == "add" else scalar_bucket_map(b, a)
            j = idx.get(bp)
            if j is None:
                forced_idx.add(i)
            else:
                union(i, j)

    forced_roots = {find(i) for i in forced_idx}
    members = defaultdict(list)
    for i, b in enumerate(initial):
        r = find(i)
        if r not in forced_roots:
            members[r].append(b)

    roots = sorted(members, key=lambda r: repr(min(members[r], key=repr)))
    col_of = {}
    orbit_members = []
    for col, r in enumerate(roots):
        mm = tuple(sorted(members[r], key=repr))
        orbit_members.append(mm)
        for b in mm:
            col_of[b] = col

    active = tuple(b for mm in orbit_members for b in mm)
    forced_zero = len(initial) - len(active)
    return {
        "universe": universe,
        "initial_active": tuple(initial),
        "active_buckets": active,
        "col_of": col_of,
        "orbit_members": tuple(orbit_members),
        "nvars": len(orbit_members),
        "forced_zero_buckets": forced_zero,
        "actions": tuple(actions),
    }


def _build_core_reduced_data(fixed_A, case, fq_linear):
    """Build only exact linear equalities needed before expensive positivity."""
    start = time.perf_counter()
    fixed_A = normalized_fixed_A(fixed_A)
    norm = Q ** (n-k)
    qdata = _build_active_orbit_quotient(fixed_A, fq_linear)
    universe = qdata["universe"]
    active = qdata["active_buckets"]
    col_of = qdata["col_of"]
    orbit_members = qdata["orbit_members"]
    nvars = qdata["nvars"]
    idb = identity_bucket()
    if idb not in col_of:
        raise ValueError("UNSAT: identity bucket was removed by pruning/orbit quotient")

    eq_rows, eq_rhs = [], []
    log = []
    core_contradiction = [None]

    def add_eq(row, rhs=0):
        row = {int(j): int(v) if isinstance(v, (int, np.integer)) else v for j,v in row.items() if v}
        if not row:
            if rhs != 0:
                core_contradiction[0] = f"trivial core contradiction 0 = {rhs}"
            return
        eq_rows.append(row)
        eq_rhs.append(rhs)

    # Identity and exact split-A totals.  Because several original cells may
    # share one quotient variable, multiplicities MUST be retained.
    add_eq({col_of[idb]: 1}, 1)
    if FIXED_SPLIT_A is not None:
        for (ii, rr), target in sorted(FIXED_SPLIT_A.items()):
            if target == 0:
                continue
            row = {}
            for b in active:
                if b[0][1] == ii and U_SIZE - b[1][0] == rr:
                    _accum_col(row, col_of[b], 1)
            add_eq(row, int(target))
        log.append("exact MDS split A")
    elif fixed_A is not None:
        for w, target in enumerate(fixed_A):
            if target == 0:
                continue
            row = {}
            for b in active:
                if bucket_weight(b) == w:
                    _accum_col(row, col_of[b], 1)
            add_eq(row, int(target))
        log.append("fixed ordinary A")
    else:
        row = {}
        for b in active:
            _accum_col(row, col_of[b], 1)
        add_eq(row, norm)

    QT, QU = ordinary_local_matrices()
    B_rows = build_cyclo_transform_rows(active, col_of, QT, QU)

    # Ordinary MacWilliams rationality + exact distance equalities.
    if USE_MACWILLIAMS:
        for ob in universe:
            comps = cyclo_output_sparse(B_rows, ob)
            for rr in comps[1:]:
                add_eq(dict(rr), 0)
            if bucket_weight(ob) < d:
                row = dict(comps[0])
                acol = col_of.get(ob)
                if acol is not None:
                    _accum_col(row, acol, -norm)
                add_eq(row, 0)
        log.append("ordinary projected MacWilliams rationality + A=B below d")

    # Exact fixed split-B totals.
    if USE_MACWILLIAMS and FIXED_SPLIT_B is not None:
        split_rows = {}
        for ob in universe:
            key = (ob[0][1], U_SIZE - ob[1][0])
            rr = cyclo_output_sparse(B_rows, ob)[0]
            split_rows[key] = row_add(split_rows.get(key, {}), rr)
        for key, target in sorted(FIXED_SPLIT_B.items()):
            add_eq(split_rows.get(key, {}), norm * int(target))
        log.append("exact MDS split B")

    # B commutation zeros.
    if USE_COMMUTATION:
        bz = 0
        for ob in universe:
            if not bucket_commutes_with_reference(ob, fq_linear):
                for rr in cyclo_output_sparse(B_rows, ob):
                    add_eq(dict(rr), 0)
                bz += 1
        log.append(f"B commutation support ({bz} zero output buckets)")

    # B translation/scalar orbit equalities remain useful independent core
    # equalities.  A-side orbit equalities were already quotient-eliminated.
    if USE_ADDITIVE_ORBIT:
        seen = set(); cnt = 0
        for alpha in additive_action_elements(fq_linear):
            if alpha == 0:
                continue
            for b in universe:
                bp = additive_bucket_map(b, alpha)
                key = tuple(sorted((repr(b), repr(bp))))
                if key in seen or b == bp:
                    continue
                seen.add(key)
                r1 = cyclo_output_sparse(B_rows, b)[0]
                r2 = cyclo_output_sparse(B_rows, bp)[0]
                add_eq(row_add(r1, r2, -1), 0); cnt += 1
        log.append(f"B additive-reference orbit equalities ({cnt})")

    if fq_linear and USE_SCALAR_ORBIT:
        seen = set(); cnt = 0
        for beta in FF.nonzero:
            if beta == 1:
                continue
            for b in universe:
                bp = scalar_bucket_map(b, beta)
                key = tuple(sorted((repr(b), repr(bp))))
                if key in seen or b == bp:
                    continue
                seen.add(key)
                r1 = cyclo_output_sparse(B_rows, b)[0]
                r2 = cyclo_output_sparse(B_rows, bp)[0]
                add_eq(row_add(r1, r2, -1), 0); cnt += 1
        log.append(f"B F_q^x scalar orbit equalities ({cnt})")

    # Cleaning/dimension equalities.
    if USE_CLEANING:
        sizes = (T_SIZE, U_SIZE)
        correctable = [i for i,s in enumerate(sizes) if s < d]
        id_t = (T_SIZE, 0)
        id_u = tuple([U_SIZE] + [0] * (2*Q-2))
        for ci in correctable:
            row_SC = {}; row_SO = {}
            for b in active:
                tc, uc = b; col = col_of[b]
                if ci == 0:
                    if uc == id_u: _accum_col(row_SC, col, 1)
                    if tc == id_t: _accum_col(row_SO, col, 1)
                else:
                    if tc == id_t: _accum_col(row_SC, col, 1)
                    if uc == id_u: _accum_col(row_SO, col, 1)
            csize = sizes[ci]
            exponent = n-k-2*csize
            if exponent >= 0:
                add_eq(row_add(row_SO, row_SC, -(Q**exponent)), 0)
            else:
                add_eq(row_add(row_scale(row_SO, Q**(-exponent)), row_SC, -1), 0)

            selected = []
            for ob in universe:
                tc, uc = ob
                is_id = (tc == id_t) if ci == 0 else (uc == id_u)
                if is_id:
                    selected.append(cyclo_output_sparse(B_rows, ob)[0])
            B_O_num = sum_sparse_rows(selected)
            add_eq(row_add(B_O_num, row_SO, -(norm*(Q**(2*k)))), 0)
        if correctable:
            log.append("cleaning/dimension equalities")

    upper_bounds = {}
    for col, mm in enumerate(orbit_members):
        ub = None
        for b in mm:
            cap = bucket_capacity(b)
            shell = fixed_A[bucket_weight(b)] if fixed_A is not None else norm
            here = min(shell, cap) if USE_CELL_CAPACITY else shell
            ub = here if ub is None else min(ub, here)
        upper_bounds[col] = int(ub if ub is not None else norm)

    return {
        **qdata,
        "fixed_A": fixed_A,
        "norm": norm,
        "eq_rows": eq_rows,
        "eq_rhs": eq_rhs,
        "core_contradiction": core_contradiction[0],
        "upper_bounds": upper_bounds,
        "core_log": log,
        "build_core_elapsed": time.perf_counter()-start,
    }


def _build_twist_rationality_equalities_original(core):
    """Build only nonconstant character-twist equalities in orbit-quotient x coordinates.

    Crucially, do NOT substitute an RREF parameterization here: that can turn a
    sparse transform row into a dense row in hundreds/thousands of free
    coordinates.  We instead append these exact equalities to the core equality
    subsystem and eliminate everything together once.
    """
    eq_rows = []; eq_rhs = []
    generated = 0; nonzero = 0
    for alpha in FF.nonzero:
        QTa, QUa = character_local_matrices(alpha)
        rows = build_cyclo_transform_rows(core["active_buckets"], core["col_of"], QTa, QUa)
        a_nonzero = 0
        for ob in core["universe"]:
            comps = cyclo_output_sparse(rows, ob)
            for rr in comps[1:]:
                generated += 1
                if rr:
                    eq_rows.append(dict(rr)); eq_rhs.append(0)
                    nonzero += 1; a_nonzero += 1
        print(f"        [twist-eq] alpha={alpha}: nonzero={a_nonzero}, accumulated={nonzero}", flush=True)
        del rows
    return eq_rows, eq_rhs, {"generated": generated, "nonzero": nonzero, "sectors": len(FF.nonzero)}


def _build_twist_rationality_equalities(core, param1):
    """Generate only nonconstant cyclotomic character-twist coordinates."""
    eq_rows = []; eq_rhs = []
    sectors = 0
    generated = 0
    for alpha in FF.nonzero:
        QTa, QUa = character_local_matrices(alpha)
        rows = build_cyclo_transform_rows(core["active_buckets"], core["col_of"], QTa, QUa)
        for ob in core["universe"]:
            comps = cyclo_output_sparse(rows, ob)
            for rr in comps[1:]:
                const, coeff = _substitute_sparse_row(rr, 0, param1)
                # const + coeff*y = 0  -> coeff*y = -const
                if not coeff:
                    if const != 0:
                        return None, None, {"inconsistent": True, "reason": f"twist alpha={alpha}: constant non-rational component {const}"}
                    continue
                eq_rows.append(coeff)
                eq_rhs.append(-const)
                generated += 1
        sectors += 1
        del rows
        print(f"        [twist-eq] alpha={alpha}: accumulated exact rationality eq={generated}", flush=True)
    return eq_rows, eq_rhs, {"inconsistent": False, "sectors": sectors, "generated": generated}


def _append_final_geq(rows, rhs_list, const, coeff, stats, family):
    """Append exact affine inequality const+coeff*z >=0 as scaled float CSR row.

    HiGHS receives (-coeff) z <= const.  Each row is independently normalized
    to O(1) scale to reduce the numerical pathology seen in the unreduced LP.
    """
    coeff = {int(j): _F(v) for j,v in coeff.items() if v}
    const = _F(const)
    if not coeff:
        if const < 0:
            stats["contradiction"] = f"{family}: constant inequality {const} >= 0"
            return False
        stats["trivial"] += 1
        return True

    # Convert only after exact zero/contradiction handling.
    vals = [abs(float(const))] + [abs(float(v)) for v in coeff.values()]
    scale = max(vals)
    if not math.isfinite(scale) or scale == 0:
        stats["contradiction"] = f"{family}: nonfinite/zero scaling"
        return False
    row = {j: -float(v)/scale for j,v in coeff.items()}
    brhs = float(const)/scale
    rows.append(row)
    rhs_list.append(brhs)
    stats[family] = stats.get(family, 0) + 1
    return True


def _original_geq_to_final(row, rhs, final_param):
    # row*x >= rhs -> (row*x-rhs) >=0
    return _substitute_sparse_row(row, rhs, final_param)


def _build_final_reduced_inequalities(core, final_param, case, fq_linear):
    start = time.perf_counter()
    rows = []; rhs = []
    stats = {"trivial": 0, "contradiction": None}
    norm = core["norm"]
    universe = core["universe"]
    col_of = core["col_of"]

    def add_orig_geq(row, rrhs, family):
        const, coeff = _original_geq_to_final(row, rrhs, final_param)
        return _append_final_geq(rows, rhs, const, coeff, stats, family)

    # Exact nonnegativity/capacity of every quotient A variable.
    for col in range(core["nvars"]):
        if not add_orig_geq({col:1}, 0, "A>=0"):
            return None, None, stats
        if not add_orig_geq({col:-1}, -core["upper_bounds"][col], "A<=cap"):
            return None, None, stats

    # Ordinary B>=A and B cell-capacity inequalities.  Rebuild the ordinary
    # transform here rather than keeping its huge Python-dict representation
    # alive across the exact-RREF / twist stages.
    QT0, QU0 = ordinary_local_matrices()
    B_rows = build_cyclo_transform_rows(core["active_buckets"], col_of, QT0, QU0)
    for ob in universe:
        comps = cyclo_output_sparse(B_rows, ob)
        row0 = dict(comps[0])
        if bucket_weight(ob) >= d:
            cmp_row = dict(row0)
            acol = col_of.get(ob)
            if acol is not None:
                _accum_col(cmp_row, acol, -norm)
            if not add_orig_geq(cmp_row, 0, "B>=A"):
                return None, None, stats
        if USE_CELL_CAPACITY:
            if not add_orig_geq(row_scale(row0,-1), -(norm*bucket_capacity(ob)), "B<=cap"):
                return None, None, stats
    del B_rows
    import gc as _gc
    _gc.collect()

    # Character-twist positivity/capacity.  The non-rational coordinates were
    # already killed exactly in stage 2, so only the constant cyclotomic row is
    # emitted here.  Each sector is built and discarded immediately.
    if USE_CHARACTER_TWISTS:
        for alpha in FF.nonzero:
            QTa, QUa = character_local_matrices(alpha)
            trows = build_cyclo_transform_rows(core["active_buckets"], col_of, QTa, QUa)
            before = len(rows)
            for ob in universe:
                row0 = dict(cyclo_output_sparse(trows, ob)[0])
                if not add_orig_geq(row0, 0, "twist>=0"):
                    return None, None, stats
                if USE_CELL_CAPACITY:
                    if not add_orig_geq(row_scale(row0,-1), -(norm*bucket_capacity(ob)), "twist<=cap"):
                        return None, None, stats
            print(f"        [twist-ineq] alpha={alpha}: added {len(rows)-before} reduced inequalities", flush=True)
            del trows

    # Generalized split-Hamming shadow.
    if USE_GENERALIZED_SHADOW:
        srows = build_split_shadow_rows(core["active_buckets"], col_of)
        out_hamming = tuple(itertools.product(weak_compositions(T_SIZE,2), weak_compositions(U_SIZE,2)))
        for ob in out_hamming:
            if not add_orig_geq(dict(srows.get(ob, {})), 0, "shadow>=0"):
                return None, None, stats
        del srows

    # Reference existence.
    if USE_REFERENCE_EXISTENCE:
        rb = reference_bucket()
        acol = col_of.get(rb)
        if acol is None:
            stats["contradiction"] = "reference bucket was pruned/forced zero"
            return None, None, stats
        if not add_orig_geq({acol:1}, 1, "reference"):
            return None, None, stats

    stats["elapsed"] = time.perf_counter()-start
    stats["rows"] = len(rows)
    stats["nnz"] = sum(len(r) for r in rows)
    return rows, rhs, stats


def _reconstruct_original_float(final_param, z):
    vals = []
    for c0, cv in final_param["exprs"]:
        v = float(c0)
        for j,a in cv.items():
            v += float(a) * float(z[j])
        vals.append(v)
    return vals


def solve_reference_branch(fixed_A, case, u, fq_linear):
    """Reduced-coordinate REAL LP with equality-first exact elimination.

    q=2 is routed to the previous sparse backend because its extra binary
    quadratic sectors need a separate staged derivation.  For q>=3:

        A-orbit quotient
          -> build ALL exact equalities only (ordinary + distance + split-B +
             commutation + B-orbits + cleaning + twist rationality)
          -> one sparse exact GJ RREF
          -> generate all inequalities directly in final free coordinates
          -> HiGHS continuous feasibility.
    """
    if Q == 2:
        print("      [reduced backend] q=2 uses legacy sparse path for binary-special sectors", flush=True)
        return _LEGACY_SPARSE_SOLVE_REFERENCE_BRANCH(fixed_A, case, u, fq_linear)

    set_reference_size(u)
    start = time.perf_counter()
    print(f"    reference u={u}: T={T_SIZE}, U={U_SIZE}, fq_linear={fq_linear}", flush=True)
    print("      backend=orbit quotient -> ALL exact equalities -> sparse GJ RREF -> reduced sparse HiGHS", flush=True)
    print(f"      heartbeat interval={PROGRESS_EVERY_SECONDS:g}s; solver time limit=NONE", flush=True)

    # Stage 1: core equalities and A-orbit quotient.
    t0=time.perf_counter()
    core=run_stage_with_heartbeat("core build",lambda:_build_core_reduced_data(fixed_A,case,fq_linear))
    core_time=time.perf_counter()-t0
    print(
        f"      [core] raw={len(core['universe'])}, initially-active={len(core['initial_active'])}, "
        f"forced-zero-by-orbit={core['forced_zero_buckets']}, A-orbit-vars={core['nvars']}, "
        f"core-eq={len(core['eq_rows'])} [{core_time:.3f}s]",flush=True)
    for item in core['core_log']:
        print(f"        - {item}",flush=True)
    if core.get('core_contradiction'):
        elapsed=time.perf_counter()-start
        print(f"      => UNSAT in core construction: {core['core_contradiction']} [total={elapsed:.3f}s]",flush=True)
        return 'unsat',{'u':u,'stage':'core-build','runtime_total':elapsed}

    # No later stage needs the ordinary transform itself.  _build_core... did
    # not return it, so its large dict storage is now reclaimable.
    import gc as _gc
    _gc.collect()

    # Stage 2: build ONLY character rationality equalities.  For characteristic
    # two (notably q=4), additive characters are +/-1 and cyclo_dim()==1, so
    # there are no nonconstant cyclotomic coordinates at all: skip the whole
    # transform build here.
    t0=time.perf_counter()
    if cyclo_dim() == 1:
        teq,trhs,tmeta=[],[],{'sectors':len(FF.nonzero),'generated':0,'nonzero':0}
        print("      [twist equalities] cyclo_dim=1 -> none required",flush=True)
    else:
        teq,trhs,tmeta=run_stage_with_heartbeat("twist equality build",lambda:_build_twist_rationality_equalities_original(core))
    twist_eq_time=time.perf_counter()-t0
    print(
        f"      [twist equalities] sectors={tmeta['sectors']}, generated={tmeta['generated']}, "
        f"nonzero={tmeta['nonzero']} [{twist_eq_time:.3f}s]",flush=True)

    all_eq=core['eq_rows']+teq
    all_rhs=core['eq_rhs']+trhs
    del teq,trhs

    # Stage 3: one exact sparse GJ elimination after all equality families are known.
    t0=time.perf_counter()
    param=run_native_stage_with_heartbeat(
        "exact equality RREF",
        lambda: exact_sparse_rref_parameterization(core['nvars'],all_eq,all_rhs,"all-equality RREF")
    )
    rref_time=time.perf_counter()-t0
    if param['inconsistent']:
        elapsed=time.perf_counter()-start
        print(f"      => UNSAT at exact equality RREF: {param.get('reason','inconsistent')} [total={elapsed:.3f}s]",flush=True)
        return 'unsat',{'u':u,'stage':'equality-rref','runtime_total':elapsed}
    print(
        f"      [RREF] input={param['input_rows']}, unique={param['unique_rows']}, rank={param['rank']}, "
        f"free={len(param['free_cols'])}, duplicates={param['duplicates']}, trivial={param['trivial']}, "
        f"expr-nnz={param['total_expr_nnz']} max-row={param['max_expr_nnz']} [{rref_time:.3f}s]",flush=True)

    nfree=len(param['free_cols'])

    # Stage 4: NOW materialize inequalities, already in final coordinates.
    t0=time.perf_counter()
    ub_rows,ub_rhs,istats=run_stage_with_heartbeat(
        "reduced inequality build",
        lambda:_build_final_reduced_inequalities(core,param,case,fq_linear)
    )
    ineq_time=time.perf_counter()-t0
    if istats.get('contradiction'):
        elapsed=time.perf_counter()-start
        print(f"      => UNSAT while reducing inequalities: {istats['contradiction']} [total={elapsed:.3f}s]",flush=True)
        return 'unsat',{'u':u,'stage':'ineq-reduction','runtime_total':elapsed}
    print(
        f"      [reduced inequalities] rows={istats['rows']}, nnz={istats['nnz']}, "
        f"trivial-dropped={istats['trivial']} [{ineq_time:.3f}s]",flush=True)
    fams=[kk for kk,vv in istats.items() if isinstance(vv,int) and kk not in ('trivial','rows','nnz')]
    if fams:
        print('        families: '+', '.join(f"{kk}={istats[kk]}" for kk in fams),flush=True)

    if nfree==0:
        elapsed=time.perf_counter()-start
        print(f"      => FINAL SAT (zero-dimensional feasible point) [total={elapsed:.3f}s]",flush=True)
        return 'sat',{'u':u,'free_vars':0,'runtime_total':elapsed}

    print("      [matrix] assembling ONLY reduced inequality CSR ...",flush=True)
    t0=time.perf_counter()
    A_ub=run_stage_with_heartbeat("reduced CSR assembly",lambda:_rows_to_csr(ub_rows,nfree))
    matrix_time=time.perf_counter()-t0
    b_ub=np.asarray(ub_rhs,dtype=float)
    sparse_bytes=A_ub.data.nbytes+A_ub.indices.nbytes+A_ub.indptr.nbytes
    print(
        f"      [matrix] free-vars={nfree}, ineq={A_ub.shape[0]}, nnz={A_ub.nnz}, "
        f"CSR~{sparse_bytes/2**20:.1f} MiB [{matrix_time:.3f}s]",flush=True)

    bounds=[(None,None)]*nfree
    status,res,attempts=_solve_highs_robust(nfree,A_ub,b_ub,None,None,bounds)
    solve_time=sum(dt for _,_,_,dt in attempts)
    elapsed=time.perf_counter()-start
    print(
        f"      => FINAL {status.upper()} after {len(attempts)} HiGHS attempt(s) "
        f"[solve={solve_time:.3f}s, total={elapsed:.3f}s]",flush=True)

    if status=='sat' and SHOW_MODEL:
        vals=_reconstruct_original_float(param,res.x)
        nz=[]
        for col,v in enumerate(vals):
            if abs(v)>1e-9:
                nz.append((col,v,core['orbit_members'][col]))
        print(f"      nonzero quotient A variables: {len(nz)}",flush=True)
        for col,v,mm in nz:
            print(f"        orbit[{col}] = {v:.12g}; representative={bucket_label(mm[0])}; size={len(mm)}",flush=True)

    return status,{
        'u':u,'t':n-u,
        'raw_buckets':len(core['universe']),
        'initial_active_buckets':len(core['initial_active']),
        'forced_zero_by_orbit':core['forced_zero_buckets'],
        'orbit_A_vars':core['nvars'],
        'core_eq_rows':len(core['eq_rows']),
        'twist_eq_rows':tmeta['nonzero'],
        'eq_unique':param['unique_rows'],
        'eq_rank':param['rank'],
        'free_vars':nfree,
        'inequalities':len(ub_rows),
        'ineq_nnz':int(A_ub.nnz),
        'csr_mib':sparse_bytes/2**20,
        'runtime_core_build':core_time,
        'runtime_twist_eq_build':twist_eq_time,
        'runtime_rref':rref_time,
        'runtime_ineq_build':ineq_time,
        'runtime_matrix':matrix_time,
        'runtime_solve':solve_time,
        'runtime_total':elapsed,
        'relaxation':'continuous real LP; exact equality elimination retained; all integrality/modular/divisibility omitted',
        'time_limit':None,
    }



# =====================================================================
# LAZY-CUT / SEPARATION BACKEND
# =====================================================================
# This is the default backend below.  It deliberately avoids materializing
# the full refined transform matrices.  The exact A-side orbit quotient is
# done up front; transform-derived equalities/inequalities are separated at
# candidate LP points and only violated rows are reconstructed and added.
# =====================================================================

_PRE_LAZY_SOLVE_REFERENCE_BRANCH = solve_reference_branch
LAZY_CUTS_PER_ROUND = 256
LAZY_SEP_TOL = 1e-8
LAZY_MAX_ROUNDS = 0   # 0 = unlimited; not a time limit


def _lazy_base_problem(fixed_A, fq_linear):
    fixed_A = normalized_fixed_A(fixed_A)
    norm = Q ** (n-k)
    qd = _build_active_orbit_quotient(fixed_A, fq_linear)
    active = qd['active_buckets']; col_of = qd['col_of']; nvars = qd['nvars']
    idb = identity_bucket()
    if idb not in col_of:
        return {**qd, 'inconsistent':'identity bucket pruned'}

    eq_rows=[]; eq_rhs=[]
    def add_eq(row,rhs=0):
        row={j:v for j,v in row.items() if v}
        if not row:
            if rhs != 0:
                raise ValueError(f'UNSAT base equality 0={rhs}')
            return
        eq_rows.append(row); eq_rhs.append(float(rhs))

    add_eq({col_of[idb]:1},1)
    if FIXED_SPLIT_A is not None:
        for (ii,rr),target in sorted(FIXED_SPLIT_A.items()):
            if target==0: continue
            row={}
            for b in active:
                if b[0][1]==ii and U_SIZE-b[1][0]==rr:
                    _accum_col(row,col_of[b],1)
            add_eq(row,target)
    elif fixed_A is not None:
        for w,target in enumerate(fixed_A):
            if target==0: continue
            row={}
            for b in active:
                if bucket_weight(b)==w: _accum_col(row,col_of[b],1)
            add_eq(row,target)
    else:
        row={}
        for b in active: _accum_col(row,col_of[b],1)
        add_eq(row,norm)

    # A-only part of cleaning can be imposed without any transform.
    if USE_CLEANING:
        sizes=(T_SIZE,U_SIZE)
        id_t=(T_SIZE,0); id_u=tuple([U_SIZE]+[0]*(2*Q-2))
        for ci,csize in enumerate(sizes):
            if csize >= d: continue
            row_SC={}; row_SO={}
            for b in active:
                tc,uc=b; c=col_of[b]
                if ci==0:
                    if uc==id_u: _accum_col(row_SC,c,1)
                    if tc==id_t: _accum_col(row_SO,c,1)
                else:
                    if tc==id_t: _accum_col(row_SC,c,1)
                    if uc==id_u: _accum_col(row_SO,c,1)
            exp=n-k-2*csize
            if exp>=0:
                add_eq(row_add(row_SO,row_SC,-(Q**exp)),0)
            else:
                add_eq(row_add(row_scale(row_SO,Q**(-exp)),row_SC,-1),0)

    upper={}
    for col,mm in enumerate(qd['orbit_members']):
        ub=None
        for b in mm:
            shell=fixed_A[bucket_weight(b)] if fixed_A is not None else norm
            here=min(shell,bucket_capacity(b)) if USE_CELL_CAPACITY else shell
            ub=here if ub is None else min(ub,here)
        upper[col]=float(ub if ub is not None else norm)

    rb=reference_bucket(); rcol=col_of.get(rb)
    if USE_REFERENCE_EXISTENCE and rcol is None:
        return {**qd,'inconsistent':'reference bucket pruned/forced zero'}

    bounds=[]
    for j in range(nvars):
        lb=1.0 if USE_REFERENCE_EXISTENCE and j==rcol else 0.0
        bounds.append((lb,upper[j]))

    # Small generalized shadow rows are cheap enough to include from the start.
    shadow_geq=[]
    if USE_GENERALIZED_SHADOW:
        srows=build_split_shadow_rows(active,col_of)
        for ob in itertools.product(weak_compositions(T_SIZE,2),weak_compositions(U_SIZE,2)):
            row=dict(srows.get(ob,{}))
            if row: shadow_geq.append((row,0.0,('shadow',ob)))

    return {
        **qd,'fixed_A':fixed_A,'norm':norm,'eq_rows':eq_rows,'eq_rhs':eq_rhs,
        'bounds':bounds,'reference_col':rcol,'shadow_geq':shadow_geq,
        'inconsistent':None,
    }


def _lazy_geometry(base):
    tcs=tuple(t_compositions()); ucs=tuple(u_compositions())
    ti={c:i for i,c in enumerate(tcs)}; ui={c:i for i,c in enumerate(ucs)}
    by_uc=defaultdict(list)
    for b in base['active_buckets']:
        tc,uc=b; by_uc[uc].append((ti[tc],base['col_of'][b]))
    return {'tcs':tcs,'ucs':ucs,'ti':ti,'ui':ui,'by_uc':dict(by_uc)}


def _lazy_t_comp_matrix(QT,geo):
    nt=len(geo['tcs'])
    M=np.zeros((nt,nt),dtype=float)
    for j,tc in enumerate(geo['tcs']):
        exp=expand_composition_cyclo(tc,QT)
        for otc,val in exp.items():
            if any(val[r] for r in range(1,len(val))):
                raise AssertionError('T Hamming transform unexpectedly cyclotomic')
            M[geo['ti'][otc],j]=float(val[0])
    return M


def _lazy_eval_transform(base,geo,x,alpha=None,label='transform'):
    QT,QU = ordinary_local_matrices() if alpha is None else character_local_matrices(alpha)
    Tm=_lazy_t_comp_matrix(QT,geo)
    nt,nu,dim=len(geo['tcs']),len(geo['ucs']),cyclo_dim()
    vals=np.zeros((nt,nu,dim),dtype=float)
    tstart=time.perf_counter(); last=tstart
    items=list(geo['by_uc'].items())
    for idx,(uc, entries) in enumerate(items,1):
        ain=np.zeros(nt,dtype=float)
        for tj,col in entries: ain[tj]+=float(x[col])
        aout=Tm@ain
        if np.any(aout):
            eu=expand_composition_cyclo(uc,QU)
            if eu:
                inds=np.fromiter((geo['ui'][ouc] for ouc in eu.keys()),dtype=np.int64,count=len(eu))
                coeff=np.asarray(list(eu.values()),dtype=float)  # m x dim
                vals[:,inds,:] += aout[:,None,None]*coeff[None,:,:]
        now=time.perf_counter()
        if now-last >= PROGRESS_EVERY_SECONDS:
            print(f"        [{label}] {idx}/{len(items)} active U compositions; {now-tstart:.1f}s elapsed ...",flush=True)
            last=now
    return vals


def _lazy_ob_pos(ob,geo):
    return geo['ti'][ob[0]],geo['ui'][ob[1]]


def _lazy_norm_residual(v,rhs=0.0):
    return abs(v-rhs)/max(1.0,abs(v),abs(rhs))


def _lazy_collect_ordinary_violations(base,geo,x,vals,existing,limit_hint):
    out=[]; norm=base['norm']; col_of=base['col_of']; fq_linear=base['fq_linear'] if 'fq_linear' in base else effective_fq_linearity(Q,ASSUME_FQ_LINEAR)
    universe=base['universe']

    def add_eq(key,val,rhs,positions,direct=None):
        if key in existing: return
        sev=_lazy_norm_residual(val,rhs)
        if sev>LAZY_SEP_TOL:
            out.append((sev,{'key':key,'relation':'eq','rhs':rhs,'positions':positions,'direct':direct or {},'family':None}))
    def add_geq(key,val,rhs,positions,direct=None):
        if key in existing: return
        scale=max(1.0,abs(val),abs(rhs)); sev=(rhs-val)/scale
        if sev>LAZY_SEP_TOL:
            out.append((sev,{'key':key,'relation':'geq','rhs':rhs,'positions':positions,'direct':direct or {},'family':None}))

    # Cellwise ordinary MacWilliams / rationality / distance / capacities.
    for ob in universe:
        ti,ui=_lazy_ob_pos(ob,geo); vv=vals[ti,ui]
        for r in range(1,len(vv)):
            add_eq(('B-cyclo',ob,r),vv[r],0.0,[(ti,ui,r,1)])
        v0=vv[0]
        if USE_COMMUTATION and not bucket_commutes_with_reference(ob,fq_linear):
            add_eq(('B-commute',ob),v0,0.0,[(ti,ui,0,1)])
            continue
        acol=col_of.get(ob); aval=float(x[acol]) if acol is not None else 0.0
        if bucket_weight(ob)<d:
            add_eq(('B=A',ob),v0-norm*aval,0.0,[(ti,ui,0,1)],({acol:-norm} if acol is not None else {}))
        else:
            add_geq(('B>=A',ob),v0-norm*aval,0.0,[(ti,ui,0,1)],({acol:-norm} if acol is not None else {}))
        if USE_CELL_CAPACITY:
            add_geq(('Bcap',ob),-v0,-norm*bucket_capacity(ob),[(ti,ui,0,-1)])

    # Exact split-B totals: usually redundant once split-A + MacWilliams hold,
    # but keep them as a rigorous regression/separation family.
    if FIXED_SPLIT_B is not None:
        buckets_by_split=defaultdict(list)
        for ob in universe:
            buckets_by_split[(ob[0][1],U_SIZE-ob[1][0])].append(ob)
        for key,target in FIXED_SPLIT_B.items():
            obs=buckets_by_split[key]
            val=sum(vals[_lazy_ob_pos(ob,geo)][0] for ob in obs)
            positions=[(*_lazy_ob_pos(ob,geo),0,1) for ob in obs]
            add_eq(('splitB',key),val,norm*float(target),positions)

    # B translation orbit equalities.
    if USE_ADDITIVE_ORBIT:
        seen=set()
        for alpha in additive_action_elements(fq_linear):
            if alpha==0: continue
            for b in universe:
                bp=additive_bucket_map(b,alpha)
                key=tuple(sorted((repr(b),repr(bp))))
                if b==bp or key in seen: continue
                seen.add(key)
                p1=_lazy_ob_pos(b,geo);p2=_lazy_ob_pos(bp,geo)
                val=vals[p1][0]-vals[p2][0]
                add_eq(('B-add-orbit',key),val,0.0,[(*p1,0,1),(*p2,0,-1)])

    if fq_linear and USE_SCALAR_ORBIT:
        seen=set()
        for beta in FF.nonzero:
            if beta==1: continue
            for b in universe:
                bp=scalar_bucket_map(b,beta)
                key=tuple(sorted((repr(b),repr(bp))))
                if b==bp or key in seen: continue
                seen.add(key)
                p1=_lazy_ob_pos(b,geo);p2=_lazy_ob_pos(bp,geo)
                val=vals[p1][0]-vals[p2][0]
                add_eq(('B-scale-orbit',key),val,0.0,[(*p1,0,1),(*p2,0,-1)])

    # B side of cleaning.
    if USE_CLEANING:
        sizes=(T_SIZE,U_SIZE); id_t=(T_SIZE,0); id_u=tuple([U_SIZE]+[0]*(2*Q-2))
        for ci,csize in enumerate(sizes):
            if csize>=d: continue
            positions=[]; val=0.0; row_SO={}; aval_SO=0.0
            for ob in universe:
                tc,uc=ob; is_id=(tc==id_t) if ci==0 else (uc==id_u)
                if is_id:
                    pp=_lazy_ob_pos(ob,geo); positions.append((*pp,0,1)); val+=vals[pp][0]
            for b in base['active_buckets']:
                tc,uc=b; is_id=(tc==id_t) if ci==0 else (uc==id_u)
                if is_id:
                    c=base['col_of'][b]; _accum_col(row_SO,c,1); aval_SO+=float(x[c])
            coeff=norm*(Q**(2*k)); val-=coeff*aval_SO
            direct={c:-coeff*v for c,v in row_SO.items()}
            add_eq(('cleanB',ci),val,0.0,positions,direct)

    out.sort(key=lambda z:z[0],reverse=True)
    return out


def _lazy_collect_twist_violations(base,geo,x,vals,alpha,existing):
    out=[];norm=base['norm']
    for ob in base['universe']:
        ti,ui=_lazy_ob_pos(ob,geo);vv=vals[ti,ui]
        for r in range(1,len(vv)):
            key=('T-cyclo',alpha,ob,r)
            if key not in existing:
                sev=_lazy_norm_residual(vv[r],0.0)
                if sev>LAZY_SEP_TOL:
                    out.append((sev,{'key':key,'relation':'eq','rhs':0.0,'positions':[(ti,ui,r,1)],'direct':{},'family':alpha}))
        v0=vv[0]
        key=('T-pos',alpha,ob)
        if key not in existing:
            sev=(-v0)/max(1.0,abs(v0))
            if sev>LAZY_SEP_TOL:
                out.append((sev,{'key':key,'relation':'geq','rhs':0.0,'positions':[(ti,ui,0,1)],'direct':{},'family':alpha}))
        # No bucket-capacity upper bound is valid for a nontrivial
        # character-twisted Fourier coefficient.  Positivity is the
        # universal constraint; capacity belongs to ordinary B (and, in the
        # binary alpha=0 case, the genuine refined shadow) only.
    out.sort(key=lambda z:z[0],reverse=True)
    return out


def _lazy_reconstruct_rows(base,geo,descs,alpha=None):
    """Reconstruct only selected violated transform rows in quotient A coords."""
    if not descs: return {}
    QT,QU=ordinary_local_matrices() if alpha is None else character_local_matrices(alpha)
    Tm=_lazy_t_comp_matrix(QT,geo)
    rows={d['key']:dict(d.get('direct',{})) for d in descs}
    # Reverse selected output-U positions to descriptors.
    rev=defaultdict(list)
    for d in descs:
        for ti,ui,comp,scale in d['positions']:
            rev[ui].append((d['key'],ti,comp,scale))
    selected_ui=set(rev)
    tstart=time.perf_counter();last=tstart
    items=list(geo['by_uc'].items())
    for idx,(uc,entries) in enumerate(items,1):
        eu=expand_composition_cyclo(uc,QU)
        for ouc,cv in eu.items():
            ui=geo['ui'][ouc]
            if ui not in selected_ui: continue
            for key,oti,comp,scale in rev[ui]:
                e=cv[comp]
                if not e: continue
                row=rows[key]
                for iti,col in entries:
                    a=Tm[oti,iti]
                    if a:
                        nv=row.get(col,0)+int(scale)*int(e)*int(round(a))
                        if nv: row[col]=nv
                        elif col in row: del row[col]
        now=time.perf_counter()
        if now-last>=PROGRESS_EVERY_SECONDS:
            print(f"        [row reconstruction] {idx}/{len(items)} U compositions; {now-tstart:.1f}s elapsed ...",flush=True);last=now
    return rows


def _normalize_float_eq(row,rhs):
    if not row: return {},float(rhs)
    scale=max([abs(float(rhs))]+[abs(float(v)) for v in row.values()])
    if scale==0: scale=1.0
    return {j:float(v)/scale for j,v in row.items()},float(rhs)/scale


def _lazy_add_descriptors(descs,base,geo,eqcuts,geqcuts):
    if not descs: return 0
    # Group by transform family: None=ordinary, scalar alpha=character twist.
    fams=defaultdict(list)
    for d in descs: fams[d['family']].append(d)
    added=0
    for fam,dd in fams.items():
        rows=_lazy_reconstruct_rows(base,geo,dd,alpha=fam)
        for d in dd:
            row=rows[d['key']]; rhs=d['rhs']
            nrow,nrhs=_normalize_float_eq(row,rhs)
            if d['relation']=='eq': eqcuts[d['key']]=(nrow,nrhs)
            else: geqcuts[d['key']]=(nrow,nrhs)
            added+=1
        del rows
    return added


def _lazy_rows_to_csr_pairs(pairs,nvars,geq=False):
    rows=[];rhs=[]
    for row,r in pairs:
        if geq:
            rows.append({j:-v for j,v in row.items()});rhs.append(-float(r))
        else:
            rows.append(row);rhs.append(float(r))
    A=_rows_to_csr(rows,nvars) if rows else None
    b=np.asarray(rhs,dtype=float) if rhs else None
    return A,b


def _lazy_solve_current(base,eqcuts,geqcuts):
    nvars=base['nvars']
    eqpairs=list(zip(base['eq_rows'],base['eq_rhs']))+list(eqcuts.values())
    geqpairs=list(base['shadow_geq'])
    # shadow_geq has triples row,rhs,key
    geqpairs=[(a,b) for a,b,_ in geqpairs]+list(geqcuts.values())
    Aeq,beq=_lazy_rows_to_csr_pairs(eqpairs,nvars,geq=False)
    Aub,bub=_lazy_rows_to_csr_pairs(geqpairs,nvars,geq=True)
    status,res,attempts=_solve_highs_robust(nvars,Aub,bub,Aeq,beq,base['bounds'])
    return status,res,attempts,Aeq,Aub


def solve_reference_branch(fixed_A, case, u, fq_linear):
    set_reference_size(u)
    start=time.perf_counter()
    print(f"    reference u={u}: T={T_SIZE}, U={U_SIZE}, fq_linear={fq_linear}",flush=True)
    print("      backend=exact A-orbit quotient + lazy transform separation; NO full refined matrix materialization",flush=True)
    print(f"      cuts/round={LAZY_CUTS_PER_ROUND}; sep_tol={LAZY_SEP_TOL:g}; solver time limit=NONE",flush=True)

    t0=time.perf_counter();base=_lazy_base_problem(fixed_A,fq_linear);bt=time.perf_counter()-t0
    base['fq_linear']=fq_linear
    if base.get('inconsistent'):
        print(f"      => UNSAT in A-side preprocessing: {base['inconsistent']}",flush=True)
        return 'unsat',{'u':u,'stage':'A-preprocess','runtime_total':time.perf_counter()-start}
    print(
        f"      [A preprocess] raw={len(base['universe'])}, initial-active={len(base['initial_active'])}, "
        f"forced-zero={base['forced_zero_buckets']}, orbit-vars={base['nvars']}, base-eq={len(base['eq_rows'])}, "
        f"shadow={len(base['shadow_geq'])} [{bt:.3f}s]",flush=True)
    geo=_lazy_geometry(base)
    print(f"      [geometry] active U compositions={len(geo['by_uc'])}/{len(geo['ucs'])}",flush=True)

    eqcuts={};geqcuts={};existing=set();round_no=0
    while True:
        round_no+=1
        if LAZY_MAX_ROUNDS and round_no>LAZY_MAX_ROUNDS:
            print(f"      => UNKNOWN: reached lazy-cut round cap {LAZY_MAX_ROUNDS}",flush=True)
            return 'unknown',{'u':u,'rounds':round_no-1,'runtime_total':time.perf_counter()-start}
        print(f"\n      [round {round_no}] solve with dynamic eq={len(eqcuts)}, geq={len(geqcuts)}",flush=True)
        status,res,attempts,Aeq,Aub=_lazy_solve_current(base,eqcuts,geqcuts)
        if status=='unsat':
            elapsed=time.perf_counter()-start
            print(f"      => FINAL UNSAT: current relaxation already infeasible [rounds={round_no}, total={elapsed:.3f}s]",flush=True)
            return 'unsat',{'u':u,'rounds':round_no,'eqcuts':len(eqcuts),'geqcuts':len(geqcuts),'runtime_total':elapsed}
        if status!='sat':
            elapsed=time.perf_counter()-start
            print(f"      => FINAL UNKNOWN from HiGHS [rounds={round_no}, total={elapsed:.3f}s]",flush=True)
            return 'unknown',{'u':u,'rounds':round_no,'runtime_total':elapsed}
        x=np.asarray(res.x,dtype=float)

        candidates=[]
        # Ordinary transform first.  If it already gives enough cuts, do not
        # waste this round evaluating character sectors.
        print("        [separate] ordinary MacWilliams/commutation ...",flush=True)
        vals=_lazy_eval_transform(base,geo,x,alpha=None,label='ordinary separation')
        vio=_lazy_collect_ordinary_violations(base,geo,x,vals,existing,LAZY_CUTS_PER_ROUND)
        del vals
        candidates.extend(vio)
        print(f"        [separate] ordinary violations={len(vio)}",flush=True)

        if len(candidates)<LAZY_CUTS_PER_ROUND and USE_CHARACTER_TWISTS:
            for alpha in FF.nonzero:
                print(f"        [separate] character alpha={alpha} ...",flush=True)
                vals=_lazy_eval_transform(base,geo,x,alpha=alpha,label=f'twist alpha={alpha}')
                vv=_lazy_collect_twist_violations(base,geo,x,vals,alpha,existing)
                del vals
                candidates.extend(vv)
                print(f"        [separate] alpha={alpha} violations={len(vv)}; accumulated={len(candidates)}",flush=True)
                if len(candidates)>=LAZY_CUTS_PER_ROUND: break

        if not candidates:
            elapsed=time.perf_counter()-start
            print(f"      => FINAL SAT: candidate satisfies every streamed transform constraint [rounds={round_no}, total={elapsed:.3f}s]",flush=True)
            return 'sat',{'u':u,'rounds':round_no,'eqcuts':len(eqcuts),'geqcuts':len(geqcuts),'runtime_total':elapsed}

        candidates.sort(key=lambda z:z[0],reverse=True)
        chosen=[]
        for sev,dsc in candidates:
            if dsc['key'] in existing: continue
            chosen.append(dsc)
            if len(chosen)>=LAZY_CUTS_PER_ROUND: break
        print(f"        [cuts] reconstructing {len(chosen)} most violated rows (max normalized violation={candidates[0][0]:.3e})",flush=True)
        added=_lazy_add_descriptors(chosen,base,geo,eqcuts,geqcuts)
        for dsc in chosen: existing.add(dsc['key'])
        print(f"        [cuts] added={added}; totals eq={len(eqcuts)}, geq={len(geqcuts)}",flush=True)
        if added==0:
            elapsed=time.perf_counter()-start
            print("      => UNKNOWN: separator found violations but no new cuts were addable",flush=True)
            return 'unknown',{'u':u,'rounds':round_no,'runtime_total':elapsed}



# =====================================================================
# GENERAL NON-MDS DRIVER / DATABASE ADAPTER
# =====================================================================

# The older q=2 binary-special builder still emits a few dense Python rows.
# Keep one CSR adapter that accepts both sparse dict rows and dense sequences.
def _rows_to_csr(rows, nvars, scale=1.0):
    if not rows:
        return None
    data, indices, indptr = [], [], [0]
    for row in rows:
        items = row.items() if hasattr(row, "items") else enumerate(row)
        for j, value in items:
            if value:
                data.append(float(scale * value))
                indices.append(int(j))
        indptr.append(len(data))
    return csr_matrix((np.asarray(data, dtype=float),
                       np.asarray(indices, dtype=np.int32),
                       np.asarray(indptr, dtype=np.int64)),
                      shape=(len(rows), nvars))

# Preserve the q>=3 lazy separator defined immediately above.  Binary q=2 is
# deliberately routed through the older sparse builder because that builder
# carries the extra refined-shadow and matched quadratic case constraints.
_GENERAL_LAZY_QGE3_SOLVE_REFERENCE_BRANCH = solve_reference_branch


def solve_reference_branch(fixed_A, case, u, fq_linear):
    global FIXED_SPLIT_A, FIXED_SPLIT_B
    # General codes have no analytic MDS split distribution.  Keep these off
    # even if a caller imported this module after using the MDS sibling.
    FIXED_SPLIT_A = None
    FIXED_SPLIT_B = None
    if Q == 2 and USE_BINARY_REFINED_SHADOW_SECTOR:
        print(
            "      backend=q=2 full sparse binary-special path "
            "(refined shadow + matched quadratic cases retained)",
            flush=True,
        )
        return _PRE_LAZY_SOLVE_REFERENCE_BRANCH(fixed_A, case, u, fq_linear)
    return _GENERAL_LAZY_QGE3_SOLVE_REFERENCE_BRANCH(fixed_A, case, u, fq_linear)


GENERAL_SOLVER_NAME = "hxrx_stabilizer_reference_realLP_lazy"
GENERAL_SOLVER_VERSION = "2026-09-21-general-lazy-r4-persistent-safeio"
PERSISTENT_CANONICAL_SOLVER_KEY = "hxrx_stabilizer_reference"


def _load_python_module(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"hxrx_real_lp_input_{path.stem}_{abs(hash(str(path))) & 0xfffffff:x}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_like_list(obj):
    if not isinstance(obj, (list, tuple)):
        return False
    if not obj:
        return True
    x = obj[0]
    if isinstance(x, dict):
        return (
            "fixed_A" in x or "A" in x or "a" in x or
            all(isinstance(k, (int, str)) for k in x.keys())
        )
    return isinstance(x, (list, tuple))


def _extract_candidate_source(path):
    """Load SLR3/SLR_qudit database or a simple Python/JSON list.

    Returns (candidates, metadata, module_or_None).
    Each returned candidate is normalized only structurally here; fixed-A
    cardinality is checked later by normalized_fixed_A().
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    module = None
    metadata = {"path": str(path), "complete": None, "fq_linear": None}

    if path.suffix.lower() == ".json":
        import json
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            db = obj.get("DATABASE", obj)
            cand = db.get("candidates", db.get("CANDIDATES", None))
            if cand is None:
                for key in ("survivors", "solutions", "list"):
                    if key in db and _candidate_like_list(db[key]):
                        cand = db[key]
                        break
            code = db.get("code", {}) if isinstance(db, dict) else {}
            ordinary = db.get("ordinary_solver", {}) if isinstance(db, dict) else {}
            metadata.update({
                "q": code.get("q"), "n": code.get("n"),
                "k": code.get("k"), "d": code.get("d"),
                "complete": ordinary.get("complete"),
                "fq_linear": ordinary.get("fq_linear_assumption"),
            })
        elif _candidate_like_list(obj):
            cand = obj
        else:
            raise ValueError("JSON input is neither a candidate database nor a candidate list.")
    else:
        module = _load_python_module(path)
        db = getattr(module, "DATABASE", None)
        cand = getattr(module, "CANDIDATES", None)
        if cand is None and isinstance(db, dict):
            cand = db.get("candidates")
        if cand is None:
            for name in (
                "SURVIVORS", "survivors", "SOLUTIONS", "solutions",
                "CANDIDATE_LIST", "candidate_list", "LIST", "candidates",
            ):
                obj = getattr(module, name, None)
                if _candidate_like_list(obj):
                    cand = obj
                    break
        if cand is None:
            # Last-resort introspection for a unique candidate-like module list.
            possible = []
            for name, obj in vars(module).items():
                if name.startswith("_"):
                    continue
                if _candidate_like_list(obj) and obj:
                    possible.append((name, obj))
            if len(possible) == 1:
                cand = possible[0][1]
            elif len(possible) > 1:
                names = ", ".join(name for name, _ in possible[:10])
                raise ValueError(
                    "Several candidate-like lists were found; expose one as "
                    f"CANDIDATES. Found: {names}"
                )
        if cand is None:
            raise ValueError(
                "No candidate list found. Expected CANDIDATES, DATABASE['candidates'], "
                "SURVIVORS, or SOLUTIONS."
            )

        ordinary = db.get("ordinary_solver", {}) if isinstance(db, dict) else {}
        code = db.get("code", {}) if isinstance(db, dict) else {}
        metadata.update({
            "q": getattr(module, "Q", code.get("q")),
            "n": getattr(module, "N", code.get("n")),
            "k": getattr(module, "K", code.get("k")),
            "d": getattr(module, "D", code.get("d")),
            "complete": ordinary.get("complete") if ordinary else None,
            "fq_linear": (
                getattr(module, "ASSUME_FQ_LINEAR", None)
                if hasattr(module, "ASSUME_FQ_LINEAR")
                else ordinary.get("fq_linear_assumption")
            ),
        })

    if cand is None:
        raise ValueError("Candidate source did not contain a candidate list.")

    normalized = []
    for idx, item in enumerate(cand, start=1):
        if isinstance(item, dict) and (
            "fixed_A" in item or "A" in item or "a" in item or "case" in item or "id" in item
        ):
            fa = item.get("fixed_A", item.get("A", item.get("a")))
            if fa is None:
                # A bare {weight: multiplicity} dictionary should also work.
                if all(str(key).lstrip("-").isdigit() for key in item.keys()):
                    fa = item
                else:
                    raise ValueError(f"Candidate #{idx} has no fixed_A/A field.")
            normalized.append({
                "id": str(item.get("id", f"c{idx:04d}")),
                "case": item.get("case", None),
                "fixed_A": fa,
                "excluded": bool(item.get("excluded", False)),
                "excluded_by": list(item.get("excluded_by", [])),
                "raw": item,
            })
        elif isinstance(item, dict):
            # Treat as bare compact A dictionary.
            normalized.append({
                "id": f"c{idx:04d}", "case": None,
                "fixed_A": item, "excluded": False, "excluded_by": [], "raw": item,
            })
        elif isinstance(item, (list, tuple)):
            normalized.append({
                "id": f"c{idx:04d}", "case": None,
                "fixed_A": list(item), "excluded": False, "excluded_by": [], "raw": item,
            })
        else:
            raise ValueError(f"Unsupported candidate entry #{idx}: {type(item).__name__}")

    return normalized, metadata, module



def _persistent_db_solver_key(db_module):
    """Return the solver key to use for persistent SLR3/SLR_qudit write-back.

    Historical candidate databases already reserve ``hxrx_stabilizer_reference``
    for the HxRX stage.  Prefer a dedicated real-LP key only if the database
    schema explicitly contains it; otherwise reuse the canonical HxRX key so
    that current_survivors() immediately sees UNSAT candidates as excluded.
    """
    if db_module is None:
        return None
    if not hasattr(db_module, "record_test") or not hasattr(db_module, "CANDIDATES"):
        return None
    keys = tuple(getattr(db_module, "REFINED_SOLVER_KEYS", ()))
    for key in (GENERAL_SOLVER_NAME, PERSISTENT_CANONICAL_SOLVER_KEY):
        if key in keys:
            return key
    return None


def _persistent_db_candidate_ids(db_module, solver_key, args):
    """Select persistent-DB candidates, respecting pending/rerun semantics."""
    if db_module is None or solver_key is None:
        return None

    # Explicit --rerun means recompute the HxRX stage even if the same version
    # already has SAT/UNSAT.  Otherwise use the database's pending_candidates()
    # when available, exactly like the older integer HxRX driver.
    if args.rerun or not hasattr(db_module, "pending_candidates"):
        pool = []
        for cand in db_module.CANDIDATES:
            if not args.include_excluded:
                excluded_by_other = any(
                    name != solver_key and info.get("status") == "unsat"
                    for name, info in cand.get("tests", {}).items()
                )
                if excluded_by_other:
                    continue
            pool.append(cand)
    else:
        try:
            pool = db_module.pending_candidates(
                solver_key,
                solver_version=GENERAL_SOLVER_VERSION,
                include_excluded=args.include_excluded,
            )
        except TypeError:
            # Compatibility with older generated DB runtimes.
            pool = db_module.pending_candidates(
                solver_key,
                GENERAL_SOLVER_VERSION,
                include_excluded=args.include_excluded,
            )
    return {str(c["id"]) for c in pool}


def _db_status_from_general(status):
    return {
        "survives": "sat",
        "excluded": "unsat",
        "inconclusive": "unknown",
        "error": "error",
    }.get(status, "unknown")


def _flush_persistent_db(db_module, *, attempts=6, final=False):
    """Best-effort persistent flush tolerant of transient Windows/cloud locks.

    Generated SLR/SLR3 databases save atomically by writing ``.tmp`` and then
    calling ``os.replace(tmp, db.py)``.  Google Drive / OneDrive / antivirus
    software can transiently lock the destination on Windows and raise
    ``PermissionError`` at the replace step.  The DATABASE object has already
    been updated in memory, so a later successful flush writes *all* pending
    candidate updates at once.
    """
    if db_module is None or not hasattr(db_module, "_atomic_save"):
        return True

    attempts = max(1, int(attempts))
    last_exc = None
    for j in range(attempts):
        try:
            db_module._atomic_save()
            if j:
                print(f"      [DB] save succeeded after {j+1} attempts", flush=True)
            return True
        except PermissionError as exc:
            last_exc = exc
            # Short bounded backoff.  Do not abort a long candidate batch for
            # a transient sync-client lock.
            delay = min(2.0, 0.15 * (2 ** j))
            print(
                f"      [DB] destination temporarily locked; retry {j+1}/{attempts} "
                f"in {delay:.2f}s ({exc})",
                flush=True,
            )
            time.sleep(delay)

    tag = "FINAL" if final else "DEFERRED"
    print(
        f"      [DB] {tag} SAVE FAILED after {attempts} attempts: {last_exc}\n"
        "      [DB] in-memory DATABASE is still current; a later successful "
        "flush will include all pending updates.",
        flush=True,
    )
    return False


def _record_persistent_status(db_module, solver_key, cid, status, *, runtime=None, details=None,
                              flush=True):
    """Update one HxRX result and robustly flush the generated candidate DB."""
    if db_module is None or solver_key is None:
        return True

    # Current generated SLR3/SLR_qudit runtimes support save=False.  Updating
    # memory first decouples solver progress from the fragile os.replace step.
    try:
        db_module.record_test(
            cid,
            solver_key,
            status,
            version=GENERAL_SOLVER_VERSION,
            runtime=runtime,
            details=details,
            save=False,
        )
    except TypeError:
        # Very old DB runtime without the ``save`` keyword.  Fall back to its
        # canonical API, but retry transient PermissionError around the whole
        # call rather than crashing immediately.
        last_exc = None
        for j in range(6):
            try:
                db_module.record_test(
                    cid,
                    solver_key,
                    status,
                    version=GENERAL_SOLVER_VERSION,
                    runtime=runtime,
                    details=details,
                )
                return True
            except PermissionError as exc:
                last_exc = exc
                delay = min(2.0, 0.15 * (2 ** j))
                print(f"      [DB] old-runtime save locked; retry {j+1}/6 in {delay:.2f}s", flush=True)
                time.sleep(delay)
        print(f"      [DB] DEFERRED SAVE FAILED: {last_exc}", flush=True)
        return False

    return _flush_persistent_db(db_module) if flush else True


def _compact_A_dict(fixed_A):
    dist = normalized_fixed_A(fixed_A)
    return {w: int(v) for w, v in enumerate(dist) if int(v) != 0}


def _positive_reference_weights_any(fixed_A, order="ascending"):
    dist = normalized_fixed_A(fixed_A)
    vals = [u for u in range(1, n + 1) if dist[u] > 0]
    vals.sort(reverse=(order == "descending"))
    return vals


def _selected_u_values(vals, args):
    original = list(vals)
    if args.u:
        wanted = set(int(x) for x in args.u)
        vals = [u for u in vals if u in wanted]
    if args.min_u is not None:
        vals = [u for u in vals if u >= args.min_u]
    if args.max_u is not None:
        vals = [u for u in vals if u <= args.max_u]
    if args.max_u_tests is not None:
        vals = vals[:args.max_u_tests]
    return vals, vals != original


def _candidate_cases(candidate):
    c = candidate.get("case")
    if Q != 2:
        return (0,)
    if c is None:
        return (1, 2, 3)
    try:
        c = int(c)
    except Exception:
        return (1, 2, 3)
    if c in (1, 2, 3):
        return (c,)
    return (1, 2, 3)


def _solve_fixed_candidate(candidate, fq_linear, args):
    cid = candidate["id"]
    fixed_A = _compact_A_dict(candidate["fixed_A"])
    all_u = _positive_reference_weights_any(fixed_A, args.u_order)
    selected_u, restricted_u = _selected_u_values(all_u, args)
    cases = _candidate_cases(candidate)

    if not all_u:
        return "error", {
            "candidate": cid,
            "reason": "fixed_A has no nonidentity shell although n>k",
            "fixed_A": fixed_A,
        }
    if not selected_u:
        return "skipped", {
            "candidate": cid,
            "reason": "no selected mandatory reference shell",
            "fixed_A": fixed_A,
            "all_u": all_u,
        }

    case_records = []
    any_case_survives = False
    any_case_inconclusive = False

    for case in cases:
        print("\n" + "#" * 110)
        print(f"{cid}: case={case if Q == 2 else 'general'}; mandatory u={selected_u}")
        print("#" * 110)
        branch_records = []
        case_excluded = False
        case_unknown = False

        for pos, u in enumerate(selected_u, start=1):
            print("\n" + "-" * 110)
            print(
                f"{cid} case={case}: mandatory reference [{pos}/{len(selected_u)}] "
                f"u={u}, A_u={fixed_A.get(u, 0)}"
            )
            print("-" * 110)
            try:
                status, info = solve_reference_branch(fixed_A, case, u, fq_linear)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                status, info = "error", {"u": u, "error": repr(exc)}
                print(f"      => ERROR {exc!r}", flush=True)
            branch_records.append({"u": u, "status": status, "details": info})

            # One impossible positive shell is enough to exclude this fixed-A case.
            if status == "unsat":
                case_excluded = True
                print(
                    f"      >>> CASE EXCLUDED: mandatory positive shell u={u} is UNSAT.",
                    flush=True,
                )
                if not args.all_u:
                    break
            elif status in ("unknown", "error"):
                case_unknown = True

        tested = {r["u"] for r in branch_records}
        complete_u = (tested >= set(all_u)) or case_excluded

        if case_excluded:
            cstatus = "excluded"
        elif case_unknown:
            cstatus = "inconclusive"
            any_case_inconclusive = True
        elif not complete_u or restricted_u:
            cstatus = "inconclusive"
            any_case_inconclusive = True
        else:
            # Every mandatory positive shell was feasible for this case.
            cstatus = "survives"
            any_case_survives = True

        case_records.append({
            "case": case, "status": cstatus,
            "tested_u": sorted(tested), "all_u": all_u,
            "branches": branch_records,
        })

        # If the parity case itself is not fixed by the input list, one
        # surviving compatible case is enough for the fixed A distribution to
        # survive these necessary conditions.
        if cstatus == "survives" and candidate.get("case") is None and not args.all_cases:
            break

    if any_case_survives:
        overall = "survives"
    elif any_case_inconclusive:
        overall = "inconclusive"
    else:
        overall = "excluded"

    return overall, {
        "candidate": cid,
        "fixed_A": fixed_A,
        "cases": case_records,
        "fq_linear": fq_linear,
    }


def _write_results(path, results, metadata):
    path = Path(path)
    if path.suffix.lower() == ".json":
        import json
        path.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2, default=str),
            encoding="utf-8",
        )
    else:
        import pprint
        survivors = [r for r in results if r["status"] == "survives"]
        excluded = [r for r in results if r["status"] == "excluded"]
        inconclusive = [r for r in results if r["status"] == "inconclusive"]
        body = (
            "# Auto-generated by HxRX_qudit_stabilizer_reference_realLP_lazy_v4.py\n"
            f"METADATA = {pprint.pformat(metadata, sort_dicts=False, width=120)}\n\n"
            f"RESULTS = {pprint.pformat(results, sort_dicts=False, width=120)}\n\n"
            f"SURVIVORS = {pprint.pformat(survivors, sort_dicts=False, width=120)}\n\n"
            f"EXCLUDED = {pprint.pformat(excluded, sort_dicts=False, width=120)}\n\n"
            f"INCONCLUSIVE = {pprint.pformat(inconclusive, sort_dicts=False, width=120)}\n"
        )
        path.write_text(body, encoding="utf-8")
    print(f"Results written to: {path.resolve()}")


def main():
    global Q, n, k, d, ASSUME_FQ_LINEAR, FIXED_SPLIT_A, FIXED_SPLIT_B
    global PROGRESS_EVERY_SECONDS

    parser = argparse.ArgumentParser(
        description=(
            "General q-ary HxRX stabilizer-reference lazy real-LP solver; "
            "supports direct A-free mode and SLR3/SLR_qudit candidate lists."
        )
    )
    parser.add_argument("q", type=int, choices=(2, 3, 4, 5, 7))
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument(
        "--database", "--list", dest="database", type=str, default=None,
        help="SLR3/SLR_qudit persistent .py database, simple Python candidate list, or JSON list",
    )
    parser.add_argument(
        "--candidate", action="append", default=None,
        help="candidate id to test; may be repeated (database/list mode)",
    )
    parser.add_argument(
        "--include-excluded", action="store_true",
        help="include candidates already marked excluded in a persistent SLR database",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help=("persistent DB mode: rerun this HxRX stage even if the current "
              "solver version already has SAT/UNSAT recorded"),
    )
    parser.add_argument("--u", type=int, action="append", default=None,
                        help="test only this reference weight; may be repeated")
    parser.add_argument("--u-order", choices=("ascending", "descending"), default="ascending")
    parser.add_argument("--min-u", type=int, default=None)
    parser.add_argument("--max-u", type=int, default=None)
    parser.add_argument("--max-u-tests", type=int, default=None)
    parser.add_argument("--all-u", action="store_true",
                        help="continue through all mandatory u even after a decisive UNSAT")
    parser.add_argument("--all-cases", action="store_true",
                        help="for binary candidates without a recorded case, continue after a surviving case")
    parser.add_argument("--fq-linear", action="store_true",
                        help="for q=4, impose the F_4-linear subclass (prime q is automatically F_q-linear)")
    parser.add_argument("--heartbeat-seconds", type=float, default=3600.0,
                        help="progress heartbeat interval; no solver time limit is imposed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="optional .py or .json result file")
    args = parser.parse_args()

    Q, n, k, d = int(args.q), int(args.n), int(args.k), int(args.d)
    if not (0 <= k <= n):
        raise SystemExit("Require 0 <= k <= n.")
    if not (1 <= d <= n):
        raise SystemExit("Require 1 <= d <= n.")
    if args.heartbeat_seconds <= 0:
        raise SystemExit("--heartbeat-seconds must be positive.")
    if args.max_u_tests is not None and args.max_u_tests <= 0:
        raise SystemExit("--max-u-tests must be positive.")

    reset_field(Q)
    FIXED_SPLIT_A = None
    FIXED_SPLIT_B = None
    PROGRESS_EVERY_SECONDS = float(args.heartbeat_seconds)
    internal_self_test(Q)

    # ---------------------------------------------------------------
    # Direct mode: ordinary A_w is free.
    # ---------------------------------------------------------------
    if args.database is None:
        ASSUME_FQ_LINEAR = bool(args.fq_linear)
        fq_linear = effective_fq_linearity(Q, ASSUME_FQ_LINEAR)
        print("=" * 110)
        print("GENERAL QUDIT HxRX STABILIZER-REFERENCE -- REAL LP")
        print(f"Code: [[{n},{k},{d}]]_{Q}")
        print("Mode: DIRECT / A_w free")
        print(f"F_q-linear={fq_linear}")
        print("Arithmetic relaxation: continuous real LP; NO integrality/Mod/divisibility")
        if Q == 2:
            print("q=2 backend: full sparse binary-special sector (three parity/quadratic cases)")
        else:
            print("Backend: A-orbit quotient + lazy streamed MacWilliams/character cuts")
        print(f"Runtime: NO solver time limit; heartbeat every {PROGRESS_EVERY_SECONDS:g} s")
        print("=" * 110)

        # standalone helper handles case/u semantics.  Repeated --u needs an
        # arbitrary-set filter, so handle that explicitly here when supplied.
        if args.u:
            uvals = list(range(1, n + 1))
            uvals.sort(reverse=(args.u_order == "descending"))
            wanted = set(args.u)
            uvals = [u for u in uvals if u in wanted]
            if args.min_u is not None:
                uvals = [u for u in uvals if u >= args.min_u]
            if args.max_u is not None:
                uvals = [u for u in uvals if u <= args.max_u]
            if args.max_u_tests is not None:
                uvals = uvals[:args.max_u_tests]
            cases = standalone_cases()
            if args.dry_run:
                print(f"Direct selected cases={list(cases)}, u={uvals}")
                return
            records=[]; saw_unknown=False
            for case in cases:
                for u in uvals:
                    print("\n" + "-"*110)
                    print(f"Direct branch: case={case}, u={u}; A_w free")
                    print("-"*110)
                    st,info=solve_reference_branch(None,case,u,fq_linear)
                    records.append({"case":case,"u":u,"status":st,"details":info})
                    if st=="sat":
                        print("\nFINAL: NOT EXCLUDED by selected direct branch(es).")
                        if args.output:
                            _write_results(args.output,[{"id":"direct","status":"survives","details":records}],
                                           {"mode":"direct","q":Q,"n":n,"k":k,"d":d,"restricted":True})
                        return
                    if st in ("unknown","error"): saw_unknown=True
            print("\nFINAL: INCONCLUSIVE -- selected direct branches contained no SAT, but the u sweep was restricted.")
            if args.output:
                _write_results(args.output,[{"id":"direct","status":"inconclusive","details":records}],
                               {"mode":"direct","q":Q,"n":n,"k":k,"d":d,"restricted":True})
            return

        status, details = solve_standalone_all_references(
            fq_linear,
            order=args.u_order,
            min_u=args.min_u,
            max_u=args.max_u,
            max_u_tests=args.max_u_tests,
            dry_run=args.dry_run,
        )
        print("\n" + "=" * 110)
        print(f"DIRECT FINAL: {status.upper()}")
        print("=" * 110)
        if args.output:
            mapped = "excluded" if status == "excluded" else (
                "survives" if status == "not_excluded" else "inconclusive"
            )
            _write_results(
                args.output,
                [{"id":"direct","status":mapped,"details":details}],
                {"mode":"direct","q":Q,"n":n,"k":k,"d":d},
            )
        return

    # ---------------------------------------------------------------
    # Candidate-list / database mode.
    # ---------------------------------------------------------------
    candidates, meta, db_module = _extract_candidate_source(args.database)
    persistent_solver_key = _persistent_db_solver_key(db_module)

    # If this is a genuine generated SLR3/SLR_qudit database, use its pending
    # bookkeeping rather than blindly rerunning every candidate.  Simple Python
    # lists/JSON files remain read-only inputs.
    pending_ids = _persistent_db_candidate_ids(
        db_module, persistent_solver_key, args
    ) if persistent_solver_key is not None else None
    if pending_ids is not None:
        candidates = [c for c in candidates if c["id"] in pending_ids]

    # Validate metadata when the source supplies it.  Binary SLR3 databases
    # historically omit Q, in which case q=2 from the CLI is authoritative.
    for label, requested, supplied in (
        ("q", Q, meta.get("q")), ("n", n, meta.get("n")),
        ("k", k, meta.get("k")), ("d", d, meta.get("d")),
    ):
        if supplied is not None and int(supplied) != int(requested):
            raise ValueError(
                f"Candidate source has {label}={supplied}, but command line requests {requested}."
            )

    db_fq = bool(meta.get("fq_linear")) if meta.get("fq_linear") is not None else False
    ASSUME_FQ_LINEAR = bool(args.fq_linear or db_fq)
    fq_linear = effective_fq_linearity(Q, ASSUME_FQ_LINEAR)

    if not args.include_excluded and persistent_solver_key is None:
        # Simple lists have no pending_candidates() semantics; honor their
        # stored excluded flag.  For a genuine persistent DB we already used
        # pending_candidates(), which intentionally allows a candidate killed
        # by THIS HxRX key to be retried after a solver-version bump while
        # still skipping candidates excluded by other solvers.
        if db_module is not None and hasattr(db_module, "current_survivors"):
            try:
                survivor_ids = {str(c["id"]) for c in db_module.current_survivors()}
                candidates = [c for c in candidates if c["id"] in survivor_ids]
            except Exception:
                candidates = [c for c in candidates if not c.get("excluded", False)]
        else:
            candidates = [c for c in candidates if not c.get("excluded", False)]
    if args.candidate:
        wanted = set(args.candidate)
        known = {c["id"] for c in candidates}
        missing = wanted - known
        if missing:
            raise ValueError("Unknown candidate id(s): " + ", ".join(sorted(missing)))
        candidates = [c for c in candidates if c["id"] in wanted]

    print("=" * 110)
    print("GENERAL QUDIT HxRX STABILIZER-REFERENCE -- REAL LP -- SLR/LIST MODE")
    print(f"Code: [[{n},{k},{d}]]_{Q}")
    print(f"Source: {Path(args.database).resolve()}")
    print(f"Candidates selected: {len(candidates)}")
    if persistent_solver_key is not None:
        print(f"Persistent write-back: ENABLED -> tests[{persistent_solver_key!r}]")
        print(f"Solver version: {GENERAL_SOLVER_VERSION}")
    else:
        print("Persistent write-back: disabled (simple list/JSON or unsupported DB schema)")
    print(f"Source enumeration complete: {meta.get('complete')}")
    print(f"F_q-linear={fq_linear} (source={db_fq}, CLI --fq-linear={args.fq_linear})")
    print("Arithmetic relaxation: continuous real LP; NO integrality/Mod/divisibility")
    if Q == 2:
        print("q=2 backend: full sparse binary-special sector")
    else:
        print("Backend: A-orbit quotient + lazy streamed MacWilliams/character cuts")
    print(f"Reference order={args.u_order}; solver time limit=NONE; heartbeat={PROGRESS_EVERY_SECONDS:g}s")
    print("=" * 110)

    if args.dry_run:
        for c in candidates:
            fa = _compact_A_dict(c["fixed_A"])
            uv = _positive_reference_weights_any(fa, args.u_order)
            sel, restricted = _selected_u_values(uv, args)
            print(f"{c['id']}: cases={_candidate_cases(c)}, mandatory_u={uv}, selected_u={sel}, FIXED_A={fa}")
        return

    results=[]
    counts={"survives":0,"excluded":0,"inconclusive":0,"error":0,"skipped":0}
    for pos,candidate in enumerate(candidates,start=1):
        print("\n" + "="*110)
        print(f"CANDIDATE [{pos}/{len(candidates)}] {candidate['id']}")
        print(f"FIXED_A = {_compact_A_dict(candidate['fixed_A'])}")
        if Q==2:
            print(f"Recorded parity case = {candidate.get('case')}")
        print("="*110)
        cid = candidate["id"]

        # A restricted --u/--min-u/--max-u selection may miss every positive
        # stabilizer shell of this candidate.  That is NOT an HxRX UNKNOWN;
        # nothing was tested.  Skip without touching the persistent DB entry.
        _fa_now = _compact_A_dict(candidate["fixed_A"])
        _all_u_now = _positive_reference_weights_any(_fa_now, args.u_order)
        _sel_u_now, _restricted_now = _selected_u_values(_all_u_now, args)
        if not _sel_u_now:
            details = {
                "candidate": cid,
                "reason": "no selected mandatory reference shell",
                "fixed_A": _fa_now,
                "all_u": _all_u_now,
            }
            status = "skipped"
            counts[status] = counts.get(status, 0) + 1
            results.append({
                "id":cid,"status":status,"db_status":None,"runtime":0.0,
                "fixed_A":_fa_now,"details":details,
            })
            print(
                f"\n>>> {cid}: SKIPPED (selected u filter misses all positive A_u); "
                f"available mandatory u={_all_u_now}",
                flush=True,
            )
            continue

        if persistent_solver_key is not None:
            _record_persistent_status(
                db_module, persistent_solver_key, cid, "running",
                runtime=None, details=None,
            )
        t0=time.perf_counter()
        try:
            status, details = _solve_fixed_candidate(candidate,fq_linear,args)
        except KeyboardInterrupt:
            # Deliberately leave the DB entry as ``running`` so the generated
            # database's pending_candidates() will retry it later.
            print("Interrupted; persistent candidate remains marked running for retry.")
            raise
        except Exception as exc:
            status="error";details={"candidate":cid,"error":repr(exc)}
            print(f"ERROR: {exc!r}")
        elapsed=time.perf_counter()-t0
        counts[status]=counts.get(status,0)+1

        db_status = None if status == "skipped" else _db_status_from_general(status)
        if persistent_solver_key is not None and db_status is not None:
            _record_persistent_status(
                db_module, persistent_solver_key, cid, db_status,
                runtime=elapsed, details=details,
            )

        results.append({
            "id":cid,"status":status,"db_status":db_status,"runtime":elapsed,
            "fixed_A":_compact_A_dict(candidate["fixed_A"]),"details":details,
        })
        print(f"\n>>> {cid}: {status.upper()} / DB={(db_status.upper() if db_status else 'UNCHANGED')}  [{elapsed:.3f}s]")

    print("\n" + "="*110)
    print("LIST/DATABASE SUMMARY")
    for key in ("survives","excluded","inconclusive","error","skipped"):
        print(f"  {key:12s}: {counts.get(key,0)}")
    if meta.get("complete") is True and counts.get("survives",0)==0 and counts.get("inconclusive",0)==0 and counts.get("error",0)==0:
        print("  All candidates in a COMPLETE SLR enumeration were excluded by this real-LP filter.")
    elif meta.get("complete") is not True:
        print("  Source is not marked complete; per-candidate exclusions are valid, but this alone is not a full parameter exclusion.")
    print("="*110)

    if persistent_solver_key is not None:
        # One final, longer retry window.  Because record_test(save=False)
        # mutates the in-memory DATABASE, one successful flush here persists
        # every update that may have been deferred by transient cloud locks.
        _flush_persistent_db(db_module, attempts=12, final=True)
        if hasattr(db_module, "print_summary"):
            print("\nPersistent candidate database after HxRX write-back:")
            try:
                db_module.print_summary(show_status=True)
            except TypeError:
                db_module.print_summary()
        elif hasattr(db_module, "current_survivors"):
            alive = db_module.current_survivors()
            print(f"Persistent current survivors after write-back: {len(alive)}")

    if args.output:
        _write_results(
            args.output, results,
            {
                "mode":"database","solver":GENERAL_SOLVER_NAME,
                "version":GENERAL_SOLVER_VERSION,
                "q":Q,"n":n,"k":k,"d":d,
                "source":str(Path(args.database).resolve()),
                "source_complete":meta.get("complete"),
                "fq_linear":fq_linear,
            },
        )


if __name__ == "__main__":
    main()
