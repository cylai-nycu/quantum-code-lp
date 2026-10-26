#!/usr/bin/env python3
r"""
binary_RXxRY_stabilizer_reference_realLP_optimized.py

Optimized binary R_X x R_Y stabilizer-reference continuous real-LP survivor filter
with exact Z3-QF_LRA verification of every numerical UNSAT before exclusion.

Intended role
-------------
This is deliberately NOT the baseline solver.  The intended pipeline is

    SLR3 lattice
        -> HxRX stabilizer lattice
        -> HxRY stabilizer lattice
        -> RXxRY stabilizer lattice.

By default this program reads a persistent SLR3 candidate database and runs
EVERY candidate that is still alive: a candidate is skipped only if some
refined solver has already returned UNSAT.  In particular, HxRX/HxRY may be
SAT, UNKNOWN, running, or not yet tested; none of those statuses blocks RXxRY.

This follows the survivor-pipeline rule used throughout this project:

    no known UNSAT  -> keep pushing the candidate forward.

The joint RXxRY filter therefore does not wait for one-sided prerequisites.

Geometry
--------
For each positive stabilizer shell A_u>0 choose an actual stabilizer reference

    g = I^(n-u) X^u.

Split the qubits as |T|=n-u and |U|=u, and use

    T : R_X = I / X / (Y Z),
    U : R_Y = I / Y / (X Z).

The actual X^U reference is aligned with R_X on T only trivially (I^T) and is
transverse to R_Y on U.  Therefore direct coefficientwise X^U
commutation/orbit constraints on the U projection are intentionally NOT used.

Strong joint/transverse constraints retained
--------------------------------------------
The source is the existing strong RXxRY stabilizer-reference solver.  It keeps:

  * ordinary/projected MacWilliams identities;
  * refined/projected Rains shadow identity;
  * global parity/shadow Cases 1/2/3 inherited from the SLR3 candidate;
  * six extra same-parity RXxRY blockwise twists:
        (000,101), (011,000), (011,101),
        (100,010), (100,111), (111,010);
  * transverse q_Y three-case branching for (111,010);
  * exact compatibility between global and q_Y quadratic types;
  * exhaustive Y^U membership branching:
        Y^U outside S^perp,
        Y^U in S^perp \ S,
        Y^U in S;
  * all projection-visible A/B/shadow commutation, orbit and self-parity
    consequences allowed by the selected Y^U membership branch;
  * coarse existence of the actual X^U stabilizer reference;
  * continuous real A/B variables (no integrality/Mod/divisibility);
  * exact RXxRY cell capacities;
  * cleaning whenever either split block is strictly correctable.

A mandatory reference weight u is excluded only if EVERY compatible
(q_Y case, Y^U membership) subbranch is UNSAT.  A candidate is excluded as
soon as ONE mandatory positive A_u shell is excluded.

Real-LP + exact-verification backend
-----------------------------------
The expensive joint projected constraints are built once per reference shell u.
HiGHS is used only as a fast floating-point screening solver.  A HiGHS
"infeasible" result is NEVER allowed to kill a candidate by itself.  The exact
rational subsystem used in that solve (equalities, bounds, and the active lazy
cuts or full inequality family) is rebuilt and decided with Z3 QF_LRA.

Only Z3 exact UNSAT is promoted to a mathematical UNSAT.  If Z3 finds SAT,
returns UNKNOWN, or is unavailable, the branch is treated as UNKNOWN and the
candidate is not excluded.

Usage
-----
Typical final-stage run:

    python binary_RXxRY_stabilizer_lattice.py 14 3 5 \
        --database candidates_n14_k3_d5.py

Inspect which candidates would run:

    python binary_RXxRY_stabilizer_lattice.py 14 3 5 \
        --database candidates_n14_k3_d5.py --dry-run


Dependencies
------------
    pip install sympy scipy numpy z3-solver
"""


# =====================================================================
# USER CONFIGURATION
# =====================================================================

# ---------------------------------------------------------------------
# Code parameters
# ---------------------------------------------------------------------

n = 14
k = 3
d = 5


# ---------------------------------------------------------------------
# Ordinary stabilizer weight distribution
#
# If a complete ordinary A_w distribution is already known, enter all
# nonzero weights here.  Omitted weights are interpreted as zero.
#
# Set FIXED_A = None to let the projected enumerator determine A_w.
# In that case the solver uses only A_0=1 and sum_w A_w=2^(n-k),
# together with the other enabled constraints.
# ---------------------------------------------------------------------

FIXED_A = None  # supplied candidate-by-candidate from the SLR3 database


# ---------------------------------------------------------------------
# Geometry: split or unsplit
# ---------------------------------------------------------------------

USE_SPLIT = True

# Used only when USE_SPLIT=True.
# The two blocks are called T and U.
U_SIZE = 1
T_SIZE = n - U_SIZE


# ---------------------------------------------------------------------
# Pauli projection
#
# Each list is a partition of {"I","X","Y","Z"}.
#
# Common choices:
#
# Full CWE:
#   [["I"], ["X"], ["Y"], ["Z"]]
#
# Hamming:
#   [["I"], ["X","Y","Z"]]
#
# LA18-style refinement:
#   [["I"], ["X"], ["Y","Z"]]
# ---------------------------------------------------------------------

# Used when USE_SPLIT=False.
PROJECTION = [
    ["I"],
    ["X"],
    ["Y"],
    ["Z"],
]

# Used when USE_SPLIT=True.
PROJECTION_T = [
    ["I"],
    ["X"],
    ["Y", "Z"],
]

PROJECTION_U = [
    ["I"],
    ["Y"],
    ["X", "Z"],
]


# ---------------------------------------------------------------------
# Ordinary/global constraints
# ---------------------------------------------------------------------

USE_MACWILLIAMS = True

# None, 1, 2, or 3.
#
# Case 1:
#   sum_even A = 2^(n-k-1),
#   sum_even B = 2^(n+k-1).
#
# Case 2:
#   odd A = 0,
#   sum_even B = 2^(n-k-1)(4^k+2^k).
#
# Case 3:
#   odd A = 0,
#   sum_even B = 2^(n-k-1)(4^k-2^k).
GLOBAL_PARITY_CASE = None


# ---------------------------------------------------------------------
# Shadow identity
#
# Shadow is a basic Shor--Laflamme constraint.  Whenever tau=111
# descends to the selected projection, impose the projected/refined
# shadow identity coefficientwise using the same A variables.  Shadow
# coefficients are actual cardinalities, hence nonnegative integers.
# If tau=111 is not projection-compatible, fall back to ordinary SI.
# ---------------------------------------------------------------------

USE_SHADOW_IDENTITY = True


# ---------------------------------------------------------------------
# Twisted complete-weight identities
#
# The master theory has 8 global twists.
#
# TWISTS may be:
#   "all"
# or an explicit list such as
#   [(0,0,0), (1,1,1)]
#
# The program automatically keeps only twists that descend to the
# selected projection(s).  000 is already the ordinary MacWilliams
# transform; nonzero compatible twists add coefficientwise positivity.
# ---------------------------------------------------------------------

USE_TWISTED_CWE = False
TWISTS = "all"


# ---------------------------------------------------------------------
# Blockwise twisted positivity
#
# In split mode the two blocks may use different twists.  A blockwise
# transform is universally positive when both local twists descend to
# their chosen projections and have the same parity |tau| mod 2.  The
# transformed coefficients are linear forms in the SAME A variables;
# no extra weight-enumerator variables are introduced.
#
# For transverse R_X x R_Y = I/X/(YZ) x I/Y/(XZ), enabling this adds
# exactly the six new positivity families
#
#     (000,101), (011,000), (011,101),
#     (100,010), (100,111), (111,010),
#
# in addition to ordinary MI (000,000) and shadow (111,111).
# The distinguished odd pair (111,010) is the transverse quadratic refinement
#
#     q_Y = wt(T) + wt_Y(U)  (mod 2).
#
# The pair (111,010) differs from global shadow by the linear character
# <Y^U,.>, so it is the same q_Y used in H x R_Y.  The other compatible
# same-parity pairs are also imposed coefficientwise for positivity.
# Positivity is exact and projection-compatible.
# ---------------------------------------------------------------------

USE_BLOCKWISE_TWISTS = True


# ---------------------------------------------------------------------
# Blockwise quadratic parity/type cases
#
# Every odd-odd blockwise twist pair defines a quadratic refinement
#
#     q(g) = q_{tau_T}(g_T) + q_{tau_U}(g_U)  (mod 2),
#
# whose polarization is the full symplectic form.  Therefore, for a
# stabilizer code, the same three parity/type cases used for the global
# shadow parity also apply to q.
#
# Specify any desired cases as a dictionary
#
#     { (tau_T, tau_U): case }
#
# with case in {1,2,3}.  Example for R_X x R_Y = I/X/(YZ) x I/Y/(XZ):
#
#     BLOCKWISE_PARITY_CASES = {
#         ((1,1,1), (0,1,0)): 1,
#     }
#
# Here q_Y(g)=wt(g_T)+wt_Y(g_U) mod 2 for the distinguished pair
# (111,010).  The persistent driver automatically branches all compatible
# q_Y cases and unions them.  The other two non-global odd-odd pairs
# (100,010) and (100,111) contribute positivity by default; their separate
# three-case type branching is deliberately not enabled in this first
# transverse solver because it would multiply the branch count.  The
# global pair (111,111) is controlled by
# GLOBAL_PARITY_CASE and should not be repeated here.
#
# These constraints use the existing A variables and ordinary B rows;
# they do NOT introduce a new enumerator.
# ---------------------------------------------------------------------

BLOCKWISE_PARITY_CASES = {}


# ---------------------------------------------------------------------
# Known stabilizer
#
# A known constant-Pauli pattern on each block.
#
# For example, if A_u>0 and a weight-u stabilizer is normalized to
#
#       g = I^T_SIZE X^U_SIZE,
#
# use:
#
#   USE_KNOWN_STABILIZER = True
#   STABILIZER_ON_T = "I"
#   STABILIZER_ON_U = "X"
#
# In unsplit mode, KNOWN_STABILIZER_SYMBOL means P^n.
# To represent a weight-u<n stabilizer, use split mode.
# ---------------------------------------------------------------------

USE_KNOWN_STABILIZER = True

STABILIZER_ON_T = "I"
STABILIZER_ON_U = "X"

KNOWN_STABILIZER_SYMBOL = "X"   # used only if USE_SPLIT=False

# Require the projected bucket containing the canonical X^U stabilizer
# to contain at least one A element.  In R_Y this bucket is coarse
# ({X,Z}^U), so this is valid existence information but not a singleton.
USE_REFERENCE_EXISTENCE = True

# Force all normalizer/stabilizer buckets that anticommute with the
# known stabilizer to vanish, when the projection retains that parity.
# IMPORTANT: for the transverse R_Y partition, the canonical X^U
# reference commutation bit does NOT descend because the {X,Z} cell mixes
# commuting X and anticommuting Z.  Keep this OFF; valid transverse
# commutation consequences are added dynamically only when Y^U membership
# in S^perp/S has been established by the quadratic branches.
USE_COMMUTATION = False

# Multiplication by the X^U reference likewise does NOT descend to R_Y:
# the {X,Z} cell splits under multiplication by X.
USE_ORBIT = False
USE_SELF_ORBIT_PARITY = False


# ---------------------------------------------------------------------
# Local / cleaning constraints
#
# Implemented only in split mode, with T interpreted as the correctable
# region and U=T^c.
#
# Requires T_SIZE < d and the projection to isolate I on both blocks.
# ---------------------------------------------------------------------

USE_CLEANING = False  # set dynamically when T or U has size < d


# ---------------------------------------------------------------------
# Integrality
# ---------------------------------------------------------------------

# A coefficients are always nonnegative integers in this solver.
A_INTEGRAL = False  # optimized real-LP: never build/use integrality

# Usually unnecessary and can make the problem stronger/harder.
# If enabled, every ordinary B coefficient is required to be integral.
B_INTEGRAL = False  # optimized real-LP: never build/use integrality
REAL_LP_SKIP_CONGRUENCES = True
HIGHS_METHOD = "highs"
INEQ_MODE = "auto"
LAZY_CUTS_PER_ROUND = 256
LAZY_FALLBACK_ROUNDS = 3
LAZY_FALLBACK_CUTS = 768
EXACT_VERIFY_TIMEOUT_MS = 0  # 0 = no timeout; numerical UNSAT never kills without exact verification

# Real-LP infeasibility certification.  Numerical HiGHS UNSAT is first
# converted into a Farkas certificate and verified with exact rational
# arithmetic.  Z3 QF_LRA is only a fallback if certificate reconstruction
# does not succeed.
USE_EXACT_FARKAS_CERTIFICATE = True
FARKAS_NUMERIC_METHOD = "highs-ds"
FARKAS_SUPPORT_REL_TOLS = (1e-8, 1e-10, 1e-12, 1e-14)
FARKAS_DIRECT_MAX_DENOMINATORS = (10**6, 10**9, 10**12, 2**45)
FARKAS_EXACT_SUPPORT_RECONSTRUCTION = True
FARKAS_EXACT_SUPPORT_MAX_VARS = 450
FARKAS_FALLBACK_TO_Z3 = True

# Exact projected-cell capacity bounds for A, B, and refined shadow.
USE_REFINED_CELL_CAPACITY = True

# Branch over all transverse q_Y three-case types compatible with the
# fixed global shadow case.  The odd pair is (111,010).
USE_TRANSVERSE_QY_CASES = True

# Use q_Y + q_shadow = <Y^U,.> to branch exact membership of Y^U:
# outside S^perp, in S^perp\S, or in S.  When membership permits, add
# all projection-visible A/B/shadow commutation, orbit, and self-parity
# consequences under multiplication by Y^U.
USE_TRANSVERSE_MEMBERSHIP_BRANCHING = True

# Persistent candidate-database identity.  Bump this whenever a
# mathematical constraint or its implementation changes.
SOLVER_NAME = "rxry_stabilizer_reference"
SOLVER_VERSION = "2026-09-24-realLP-r5-farkas-exact"
DATABASE_FILENAME = None  # default: candidates_n{n}_k{k}_d{d}.py

# Reference-shell order.  Descending usually makes |T| smaller first.
REFERENCE_U_ORDER = "descending"


# ---------------------------------------------------------------------
# Solver / output
# ---------------------------------------------------------------------

SHOW_MODEL = False
SHOW_ZERO_MODEL_ENTRIES = False
SHOW_FREE_COORDINATES = True

# Live progress / heartbeat.  Long RREF, reduced-system construction,
# and Z3 solving stages report elapsed time periodically so that a long
# computation does not look frozen.
SHOW_LIVE_PROGRESS = True
PROGRESS_EVERY_SECONDS = 5.0

TIMEOUT_MS = None


# =====================================================================
# END USER CONFIGURATION
# =====================================================================


import sys
import argparse
import importlib.util
from pathlib import Path
import itertools
import math
import time
import threading
import gc
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack, hstack

try:
    import z3
except ImportError:
    z3 = None
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache

try:
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix
except ImportError as exc:
    raise SystemExit(
        "This program requires SymPy.\n"
        "Install it with:\n"
        "  pip install sympy"
    ) from exc

# Z3 is deliberately not required by this continuous real-LP build.
sat = "sat"
unsat = "unsat"
unknown = "unknown"


# =====================================================================
# Basic Pauli data
# =====================================================================

PAULIS = ("I", "X", "Y", "Z")
P_INDEX = {p: i for i, p in enumerate(PAULIS)}

FOURIER = (
    (1,  1,  1,  1),
    (1,  1, -1, -1),
    (1, -1,  1, -1),
    (1, -1, -1,  1),
)

# Phase-free multiplication.
PAULI_MUL = {
    ("I", "I"): "I", ("I", "X"): "X", ("I", "Y"): "Y", ("I", "Z"): "Z",
    ("X", "I"): "X", ("X", "X"): "I", ("X", "Y"): "Z", ("X", "Z"): "Y",
    ("Y", "I"): "Y", ("Y", "X"): "Z", ("Y", "Y"): "I", ("Y", "Z"): "X",
    ("Z", "I"): "Z", ("Z", "X"): "Y", ("Z", "Y"): "X", ("Z", "Z"): "I",
}

# Symplectic commutation bit.
ANTICOMMUTE = {
    ("I", "I"): 0, ("I", "X"): 0, ("I", "Y"): 0, ("I", "Z"): 0,
    ("X", "I"): 0, ("X", "X"): 0, ("X", "Y"): 1, ("X", "Z"): 1,
    ("Y", "I"): 0, ("Y", "X"): 1, ("Y", "Y"): 0, ("Y", "Z"): 1,
    ("Z", "I"): 0, ("Z", "X"): 1, ("Z", "Y"): 1, ("Z", "Z"): 0,
}


# =====================================================================
# Small utilities
# =====================================================================

def yesno(flag):
    return "yes" if flag else "no"


def check_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False.")


def parse_twists():
    if TWISTS == "all":
        return list(itertools.product((0, 1), repeat=3))

    out = []

    for tau in TWISTS:
        tau = tuple(tau)

        if len(tau) != 3 or any(x not in (0, 1) for x in tau):
            raise ValueError(f"Invalid twist {tau}.")

        out.append(tau)

    return out


def tau_name(tau):
    return "".join(str(x) for x in tau)


def zsum(items):
    items = list(items)
    return Sum(items) if items else 0


def gcd_many(values):
    g = 0
    for v in values:
        g = math.gcd(g, abs(v))
    return g


@lru_cache(maxsize=None)
def q4_krawtchouk(n_total, j, i):
    """q=4 Hamming Krawtchouk coefficient K_j(i;n)."""
    total = 0

    ell_min = max(0, j - (n_total - i))
    ell_max = min(j, i)

    for ell in range(ell_min, ell_max + 1):
        total += (
            (-1) ** ell
            * (3 ** (j - ell))
            * math.comb(i, ell)
            * math.comb(n_total - i, j - ell)
        )

    return total


# =====================================================================
# Weak compositions
# =====================================================================

@lru_cache(maxsize=None)
def weak_compositions(total, parts):
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


# =====================================================================
# Projection / block specification
# =====================================================================

def normalize_projection(projection):
    cells = [tuple(cell) for cell in projection]

    flat = [p for cell in cells for p in cell]

    if sorted(flat) != sorted(PAULIS):
        raise ValueError(
            "Each projection must partition exactly I,X,Y,Z once. "
            f"Got {projection}."
        )

    if ("I",) not in cells:
        raise ValueError(
            "This solver currently requires I to be a singleton cell "
            "so that identity, Hamming weight, and distance constraints "
            "remain coefficientwise well defined."
        )

    return tuple(cells)


def projection_name(cells):
    pieces = []

    for cell in cells:
        if len(cell) == 1:
            pieces.append(cell[0])
        else:
            pieces.append("(" + "".join(cell) + ")")

    return "/".join(pieces)


def cell_of_symbol(cells):
    out = {}
    for i, cell in enumerate(cells):
        for p in cell:
            out[p] = i
    return out


def identity_cell(cells):
    return cells.index(("I",))


def twisted_fourier_matrix(tau):
    tx, ty, tz = tau
    signs = (1, (-1) ** tx, (-1) ** ty, (-1) ** tz)

    return tuple(
        tuple(signs[r] * FOURIER[r][c] for c in range(4))
        for r in range(4)
    )


def twist_parity(tau):
    """Return |tau| mod 2."""
    return sum(tau) & 1


def compatible_twists_for_block(block):
    """All twists that descend to this block projection."""
    return tuple(
        tau
        for tau in itertools.product((0, 1), repeat=3)
        if block.projected_transform(tau) is not None
    )


def compatible_blockwise_twist_pairs(blocks):
    """
    Same-parity projected blockwise twists for a two-block split.

    Same parity is exactly the universal-positivity condition: even-even
    gives local Pauli conjugation, while odd-odd gives full transpose
    followed by local Pauli conjugation.  Mixed parity would contain a
    partial transpose and is therefore excluded.
    """
    if len(blocks) != 2:
        return tuple()

    left = compatible_twists_for_block(blocks[0])
    right = compatible_twists_for_block(blocks[1])

    return tuple(
        (tau_left, tau_right)
        for tau_left in left
        for tau_right in right
        if twist_parity(tau_left) == twist_parity(tau_right)
    )


def blockwise_tau_name(pair):
    return f"({tau_name(pair[0])},{tau_name(pair[1])})"


def twist_cell_bits(cells, tau):
    """
    Return the twist exponent bit on each projection cell.

    For tau=(tau_X,tau_Y,tau_Z), q_tau(p) is 0 on I and tau_P on
    P in {X,Y,Z}.  The parity q_tau is visible on a projected cell
    exactly when this bit is constant throughout that cell.
    """
    tx, ty, tz = tau
    symbol_bit = {
        "I": 0,
        "X": tx,
        "Y": ty,
        "Z": tz,
    }

    bits = []
    for cell in cells:
        values = {symbol_bit[p] for p in cell}
        if len(values) != 1:
            return None
        bits.append(next(iter(values)))

    return tuple(bits)


def bucket_blockwise_twist_parity(blocks, bucket, pair):
    """Return q_pair(bucket) mod 2 for a projection-visible twist pair."""
    parity = 0

    for block, comp, tau in zip(blocks, bucket, pair):
        bits = twist_cell_bits(block.cells, tau)
        if bits is None:
            return None

        parity ^= sum(count * bit for count, bit in zip(comp, bits)) & 1

    return parity


def projected_transform(cells, tau):
    """
    Return the projected local transform Q if the twisted Fourier
    transform descends to the partition.

    For input cell C and output cell D,
        Q[C,D] = sum_{q in D} M[p,q]
    must be independent of p in C.
    """
    M = twisted_fourier_matrix(tau)
    r = len(cells)
    Q = [[0] * r for _ in range(r)]

    for a, input_cell in enumerate(cells):
        representative_values = None

        for p in input_cell:
            row = []

            for output_cell in cells:
                row.append(
                    sum(
                        M[P_INDEX[p]][P_INDEX[q]]
                        for q in output_cell
                    )
                )

            if representative_values is None:
                representative_values = row
            elif row != representative_values:
                return None

        Q[a] = representative_values

    return tuple(tuple(row) for row in Q)


def induced_generator_cell_permutation(cells, generator_symbol):
    """
    Return cell permutation induced by p -> p*generator_symbol if the
    Pauli partition is invariant under that permutation.
    """
    cell_sets = [frozenset(cell) for cell in cells]
    lookup = {cell: i for i, cell in enumerate(cell_sets)}
    perm = []

    for cell in cells:
        image = frozenset(
            PAULI_MUL[(p, generator_symbol)]
            for p in cell
        )

        if image not in lookup:
            return None

        perm.append(lookup[image])

    return tuple(perm)


def commutation_cell_bits(cells, generator_symbol):
    """
    Return the common anticommutation bit of each projection cell
    against generator_symbol, or None if a cell mixes commuting and
    anticommuting Paulis.
    """
    bits = []

    for cell in cells:
        values = {
            ANTICOMMUTE[(p, generator_symbol)]
            for p in cell
        }

        if len(values) != 1:
            return None

        bits.append(next(iter(values)))

    return tuple(bits)


class Block:
    def __init__(self, name, size, projection, generator_symbol=None):
        self.name = name
        self.size = size
        self.cells = normalize_projection(projection)
        self.r = len(self.cells)
        self.cell_of = cell_of_symbol(self.cells)
        self.i_cell = identity_cell(self.cells)
        self.generator_symbol = generator_symbol

    def all_compositions(self):
        return weak_compositions(self.size, self.r)

    def identity_composition(self):
        comp = [0] * self.r
        comp[self.i_cell] = self.size
        return tuple(comp)

    def symbol_composition(self, symbol):
        comp = [0] * self.r
        comp[self.cell_of[symbol]] = self.size
        return tuple(comp)

    def weight_of_composition(self, comp):
        return self.size - comp[self.i_cell]

    def projected_transform(self, tau):
        return projected_transform(self.cells, tau)

    def orbit_cell_permutation(self):
        if self.generator_symbol is None:
            return None
        return induced_generator_cell_permutation(
            self.cells,
            self.generator_symbol,
        )

    def commutation_bits(self):
        if self.generator_symbol is None:
            return None
        return commutation_cell_bits(
            self.cells,
            self.generator_symbol,
        )



def internal_projection_self_test():
    """Silent regression checks for R_X x R_Y and the transverse Y action."""
    refined_rx = (("I",), ("X",), ("Y", "Z"))
    refined_ry = (("I",), ("Y",), ("X", "Z"))

    expected_R = (
        (1, 1, 2),
        (1, 1, -2),
        (1, -1, 0),
    )
    assert projected_transform(refined_rx, (0, 0, 0)) == expected_R
    assert projected_transform(refined_ry, (0, 0, 0)) == expected_R

    rx_twists = {
        tau for tau in itertools.product((0, 1), repeat=3)
        if projected_transform(refined_rx, tau) is not None
    }
    ry_twists = {
        tau for tau in itertools.product((0, 1), repeat=3)
        if projected_transform(refined_ry, tau) is not None
    }

    assert rx_twists == {(0,0,0), (0,1,1), (1,0,0), (1,1,1)}
    assert ry_twists == {(0,0,0), (0,1,0), (1,0,1), (1,1,1)}

    # Eight same-parity R_X x R_Y pairs; six are extra beyond MI/shadow.
    test_blocks = [Block("T", 1, refined_rx), Block("U", 1, refined_ry)]
    names = {blockwise_tau_name(p) for p in compatible_blockwise_twist_pairs(test_blocks)}
    assert names == {
        "(000,000)", "(000,101)", "(011,000)", "(011,101)",
        "(100,010)", "(100,111)", "(111,010)", "(111,111)",
    }

    # Canonical actual stabilizer remains X^U.  The U-side R_Y projection
    # mixes X and Z, so direct X^U commutation/orbit still do NOT descend.
    tblock_x = Block("T", 2, refined_rx, "I")
    ublock_x = Block("U", 3, refined_ry, "X")
    assert tblock_x.orbit_cell_permutation() == (0,1,2)
    assert ublock_x.orbit_cell_permutation() is None
    assert tblock_x.commutation_bits() == (0,0,0)
    assert ublock_x.commutation_bits() is None
    x_bucket = (tblock_x.symbol_composition("I"), ublock_x.symbol_composition("X"))
    assert bucket_capacity([tblock_x, ublock_x], x_bucket) == 2**3

    # Y^U is a singleton visible direction of R_Y.  Once a q_Y branch
    # establishes its S^perp/S membership, its action and commutation parity
    # descend exactly.
    tblock_y = Block("T", 2, refined_rx, "I")
    ublock_y = Block("U", 3, refined_ry, "Y")
    assert tblock_y.orbit_cell_permutation() == (0,1,2)
    assert ublock_y.orbit_cell_permutation() == (1,0,2)
    assert tblock_y.commutation_bits() == (0,0,0)
    assert ublock_y.commutation_bits() == (0,0,1)
    y_bucket = (tblock_y.symbol_composition("I"), ublock_y.symbol_composition("Y"))
    assert bucket_capacity([tblock_y, ublock_y], y_bucket) == 1


# =====================================================================
# Enumerator geometry
# =====================================================================

def build_blocks():
    if not (0 <= k <= n):
        raise ValueError("Require 0 <= k <= n.")
    if not (1 <= d <= n + 1):
        raise ValueError("Require 1 <= d <= n+1.")

    if USE_SPLIT:
        if T_SIZE < 0 or U_SIZE < 0 or T_SIZE + U_SIZE != n:
            raise ValueError(
                "Require T_SIZE,U_SIZE >=0 and T_SIZE+U_SIZE=n."
            )

        gT = STABILIZER_ON_T if USE_KNOWN_STABILIZER else None
        gU = STABILIZER_ON_U if USE_KNOWN_STABILIZER else None

        return [
            Block("T", T_SIZE, PROJECTION_T, gT),
            Block("U", U_SIZE, PROJECTION_U, gU),
        ]

    g = KNOWN_STABILIZER_SYMBOL if USE_KNOWN_STABILIZER else None

    return [
        Block("All", n, PROJECTION, g),
    ]


def all_buckets(blocks):
    comp_lists = [block.all_compositions() for block in blocks]
    return tuple(itertools.product(*comp_lists))


def identity_bucket(blocks):
    return tuple(
        block.identity_composition()
        for block in blocks
    )


def bucket_weight(blocks, bucket):
    return sum(
        block.weight_of_composition(comp)
        for block, comp in zip(blocks, bucket)
    )


def bucket_capacity(blocks, bucket):
    """Exact number of phase-free Pauli strings in one projected bucket."""
    total = 1
    for block, comp in zip(blocks, bucket):
        rem = block.size
        multinom = 1
        for count in comp[:-1]:
            multinom *= math.comb(rem, count)
            rem -= count
        total *= multinom
        for count, cell in zip(comp, block.cells):
            total *= len(cell) ** count
    return total


def bucket_label(blocks, bucket):
    pieces = []

    for block, comp in zip(blocks, bucket):
        pieces.append(
            f"{block.name}{tuple(comp)}"
        )

    return " | ".join(pieces)


# =====================================================================
# Known-generator actions on buckets
# =====================================================================

def permute_composition(comp, cell_perm):
    """
    cell_perm[a]=b means old cell a maps to new cell b.
    """
    out = [0] * len(comp)

    for a, count in enumerate(comp):
        out[cell_perm[a]] += count

    return tuple(out)


def bucket_orbit_map(blocks, bucket, perms):
    return tuple(
        permute_composition(comp, perm)
        for comp, perm in zip(bucket, perms)
    )


def bucket_commutation_parity(blocks, bucket, bit_lists):
    parity = 0

    for comp, bits in zip(bucket, bit_lists):
        parity ^= (
            sum(c * b for c, b in zip(comp, bits)) & 1
        )

    return parity


def known_stabilizer_bucket(blocks):
    if not USE_KNOWN_STABILIZER:
        return None

    return tuple(
        block.symbol_composition(block.generator_symbol)
        for block in blocks
    )


def known_stabilizer_is_identity(blocks):
    if not USE_KNOWN_STABILIZER:
        return False

    return all(
        block.generator_symbol == "I"
        for block in blocks
    )


# =====================================================================
# Fixed ordinary A distribution
# =====================================================================

def normalized_fixed_A():
    if FIXED_A is None:
        return None

    dist = [0] * (n + 1)

    for w, value in FIXED_A.items():
        w = int(w)
        value = int(value)

        if not 0 <= w <= n:
            raise ValueError(f"Invalid fixed weight {w}.")
        if value < 0:
            raise ValueError("A_w must be nonnegative.")

        dist[w] = value

    if dist[0] != 1:
        raise ValueError("A_0 must equal 1.")

    expected = 2 ** (n - k)
    actual = sum(dist)

    if actual != expected:
        raise ValueError(
            f"FIXED_A sums to {actual}, expected {expected}=2^(n-k)."
        )

    return dist


# =====================================================================
# Projected block transform
# =====================================================================

@lru_cache(maxsize=None)
def multinomial_terms(power, coeffs):
    """
    Expand (sum_j coeffs[j] y_j)^power.

    Return dict[output_composition] = integer coefficient.
    """
    r = len(coeffs)
    out = {}

    for comp in weak_compositions(power, r):
        coeff = math.factorial(power)
        for x in comp:
            coeff //= math.factorial(x)

        for c, x in zip(coeffs, comp):
            coeff *= c ** x

        if coeff:
            out[comp] = coeff

    return out


@lru_cache(maxsize=None)
def expand_block_composition_cached(comp, Q):
    """
    Expand prod_a (sum_b Q[a,b] y_b)^comp[a].
    """
    r = len(comp)

    poly = {
        tuple([0] * r): 1
    }

    for a, power in enumerate(comp):
        if power == 0:
            continue

        factor = multinomial_terms(power, tuple(Q[a]))
        new_poly = defaultdict(int)

        for c1, v1 in poly.items():
            for c2, v2 in factor.items():
                out_comp = tuple(
                    x + y for x, y in zip(c1, c2)
                )
                new_poly[out_comp] += v1 * v2

        poly = {
            c: v for c, v in new_poly.items()
            if v
        }

    return poly


def expand_global_bucket(blocks, bucket, Qs):
    """
    Return dict[output_bucket] = transform coefficient.
    """
    block_expansions = [
        expand_block_composition_cached(comp, Q)
        for comp, Q in zip(bucket, Qs)
    ]

    out = {tuple(): 1}

    for expansion in block_expansions:
        new_out = defaultdict(int)

        for prefix, c1 in out.items():
            for out_comp, c2 in expansion.items():
                new_out[prefix + (out_comp,)] += c1 * c2

        out = {
            b: c for b, c in new_out.items()
            if c
        }

    return out


def build_blockwise_transform_rows(
    blocks,
    active_buckets,
    col_of,
    taus,
):
    """
    Sparse rows for a projected blockwise twisted transform.

    ``taus[b]`` is the twist applied to block ``b``.  The row space is
    built from the same active A buckets and therefore introduces no new
    primal variables.
    """
    if len(taus) != len(blocks):
        raise ValueError("Need one twist per block.")

    Qs = []

    for block, tau in zip(blocks, taus):
        Q = block.projected_transform(tau)

        if Q is None:
            return None

        Qs.append(Q)

    rows = defaultdict(dict)

    for in_bucket in active_buckets:
        col = col_of[in_bucket]

        expansion = expand_global_bucket(
            blocks,
            in_bucket,
            tuple(Qs),
        )

        for out_bucket, coeff in expansion.items():
            if coeff:
                rows[out_bucket][col] = coeff

    return dict(rows)


def build_transform_rows(
    blocks,
    active_buckets,
    col_of,
    tau,
):
    """Sparse rows for a global twist, identical on every block."""
    return build_blockwise_transform_rows(
        blocks,
        active_buckets,
        col_of,
        tuple(tau for _ in blocks),
    )


# =====================================================================
# Original linear system representation
# =====================================================================

def sparse_row(nvars, terms=None):
    row = [0] * nvars

    if terms:
        for col, coeff in terms.items():
            row[col] += coeff

    return row


def add_rows(row1, row2, scale2=1):
    return [
        a + scale2 * b
        for a, b in zip(row1, row2)
    ]


def row_from_sparse(nvars, sparse):
    row = [0] * nvars
    for col, coeff in sparse.items():
        row[col] = coeff
    return row


def sum_sparse_rows(nvars, sparse_rows):
    row = [0] * nvars

    for sparse in sparse_rows:
        for col, coeff in sparse.items():
            row[col] += coeff

    return row


# =====================================================================
# Build mathematical problem
# =====================================================================

def build_problem():
    blocks = build_blocks()
    fixed_A = normalized_fixed_A()
    norm = 2 ** (n - k)

    universe = all_buckets(blocks)
    id_bucket = identity_bucket(blocks)

    # ---------------------------------------------------------------
    # Determine generator compatibility.
    # ---------------------------------------------------------------

    generator_orbit_available = False
    generator_comm_available = False
    orbit_perms = None
    comm_bits = None

    if USE_KNOWN_STABILIZER:
        orbit_perms_tmp = [
            block.orbit_cell_permutation()
            for block in blocks
        ]

        if all(x is not None for x in orbit_perms_tmp):
            generator_orbit_available = True
            orbit_perms = tuple(orbit_perms_tmp)

        comm_bits_tmp = [
            block.commutation_bits()
            for block in blocks
        ]

        if all(x is not None for x in comm_bits_tmp):
            generator_comm_available = True
            comm_bits = tuple(comm_bits_tmp)

    # ---------------------------------------------------------------
    # Choose active A buckets.
    #
    # If ordinary A is fixed, zero shells are pruned immediately.
    # If commutation is available and enabled, anticommuting A buckets
    # are also identically zero.
    # ---------------------------------------------------------------

    active_buckets = []

    for bucket in universe:
        w = bucket_weight(blocks, bucket)

        if fixed_A is not None and fixed_A[w] == 0:
            continue

        if (
            USE_KNOWN_STABILIZER
            and USE_COMMUTATION
            and generator_comm_available
            and bucket_commutation_parity(
                blocks,
                bucket,
                comm_bits,
            )
        ):
            continue

        active_buckets.append(bucket)

    active_buckets = tuple(active_buckets)
    col_of = {
        bucket: col
        for col, bucket in enumerate(active_buckets)
    }
    nvars = len(active_buckets)

    if id_bucket not in col_of:
        raise ValueError(
            "Identity bucket was pruned. Check FIXED_A/projection."
        )

    # ---------------------------------------------------------------
    # Check twisted-transform compatibility.
    # ---------------------------------------------------------------

    requested_twists = parse_twists()
    compatible_twists = []
    skipped_twists = []

    for tau in requested_twists:
        if all(
            block.projected_transform(tau) is not None
            for block in blocks
        ):
            compatible_twists.append(tau)
        else:
            skipped_twists.append(tau)

    # Ordinary MI is tau=000; the shadow identity is tau=111.
    tau000 = (0, 0, 0)
    tau111 = (1, 1, 1)

    if not isinstance(BLOCKWISE_PARITY_CASES, dict):
        raise ValueError(
            "BLOCKWISE_PARITY_CASES must be a dictionary mapping "
            "((tau_T),(tau_U)) to case 1,2,or3."
        )

    need_blockwise_pairs = (
        USE_BLOCKWISE_TWISTS or bool(BLOCKWISE_PARITY_CASES)
    )

    if need_blockwise_pairs and (not USE_SPLIT or len(blocks) != 2):
        raise ValueError(
            "Blockwise twists/parity cases currently require exactly "
            "two split blocks."
        )

    blockwise_pairs = (
        compatible_blockwise_twist_pairs(blocks)
        if need_blockwise_pairs
        else tuple()
    )

    normalized_blockwise_parity_cases = {}
    for pair_raw, case in BLOCKWISE_PARITY_CASES.items():
        try:
            pair = tuple(tuple(tau) for tau in pair_raw)
        except TypeError as exc:
            raise ValueError(
                f"Invalid blockwise parity pair {pair_raw}."
            ) from exc

        if (
            len(pair) != 2
            or any(len(tau) != 3 for tau in pair)
            or any(x not in (0, 1) for tau in pair for x in tau)
        ):
            raise ValueError(
                f"Invalid blockwise parity pair {pair_raw}."
            )

        if case not in (1, 2, 3):
            raise ValueError(
                f"Blockwise parity case for {pair} must be 1,2,or3."
            )

        if pair == (tau111, tau111):
            raise ValueError(
                "Use GLOBAL_PARITY_CASE for the global (111,111) pair; "
                "do not duplicate it in BLOCKWISE_PARITY_CASES."
            )

        if not all(twist_parity(tau) == 1 for tau in pair):
            raise ValueError(
                f"Blockwise three-case classification requires an "
                f"odd-odd quadratic pair; got {blockwise_tau_name(pair)}."
            )

        if pair not in blockwise_pairs:
            raise ValueError(
                f"Blockwise parity pair {blockwise_tau_name(pair)} does "
                "not descend to the selected projections."
            )

        # The parity itself must be visible coefficientwise in the chosen
        # projected variables.  (For the compatible HxR odd pair 111,100
        # this is wt_T + wt_X,U mod 2.)
        if any(
            twist_cell_bits(block.cells, tau) is None
            for block, tau in zip(blocks, pair)
        ):
            raise ValueError(
                f"Parity for {blockwise_tau_name(pair)} is not visible "
                "on the selected projection cells."
            )

        normalized_blockwise_parity_cases[pair] = case

    # MI and projected shadow are already imposed separately.  They are
    # kept in the theoretical pair list but omitted from the additional
    # blockwise-positivity rows to avoid duplicates.
    enabled_blockwise_pairs = []
    for pair in blockwise_pairs:
        if not USE_BLOCKWISE_TWISTS:
            continue
        if pair == (tau000, tau000):
            continue
        if USE_SHADOW_IDENTITY and pair == (tau111, tau111):
            continue
        # If the same global twist is already requested through the old
        # global-twist machinery, do not duplicate its inequalities.
        if (
            USE_TWISTED_CWE
            and pair[0] == pair[1]
            and pair[0] in compatible_twists
        ):
            continue
        enabled_blockwise_pairs.append(pair)

    shadow_projected_compatible = all(
        block.projected_transform(tau111) is not None
        for block in blocks
    )

    if USE_SHADOW_IDENTITY and not USE_MACWILLIAMS:
        raise ValueError(
            "USE_SHADOW_IDENTITY currently requires USE_MACWILLIAMS=True."
        )

    if USE_MACWILLIAMS:
        if not all(
            block.projected_transform(tau000) is not None
            for block in blocks
        ):
            raise ValueError(
                "The selected projection is not closed under the "
                "ordinary complete MacWilliams transform. Use a finer "
                "projection, such as full CWE."
            )

    # ---------------------------------------------------------------
    # Build transform rows only for transforms actually needed.
    # ---------------------------------------------------------------

    needed_taus = set()

    if USE_MACWILLIAMS:
        needed_taus.add(tau000)

    if USE_SHADOW_IDENTITY and shadow_projected_compatible:
        needed_taus.add(tau111)

    if USE_TWISTED_CWE:
        needed_taus.update(
            tau for tau in compatible_twists
            if tau != tau000
        )

    transform_rows = {}

    for tau in sorted(needed_taus):
        transform_rows[tau] = build_transform_rows(
            blocks,
            active_buckets,
            col_of,
            tau,
        )

    B_rows = transform_rows.get(tau000, {})
    shadow_rows = transform_rows.get(tau111, {})

    blockwise_transform_rows = {}
    for pair in enabled_blockwise_pairs:
        rows = build_blockwise_transform_rows(
            blocks,
            active_buckets,
            col_of,
            pair,
        )
        if rows is None:
            raise AssertionError(
                f"Blockwise pair {pair} passed compatibility but failed row construction."
            )
        blockwise_transform_rows[pair] = rows

    # ---------------------------------------------------------------
    # Mathematical constraints.
    # ---------------------------------------------------------------

    eq_rows = []
    eq_rhs = []

    inequalities = []   # row.x >= rhs
    congruences = []    # row.x == rhs (mod modulus)

    constraint_log = {
        "basic": [],
        "additional": [],
        "skipped": [],
    }

    # A_0=1.
    row = [0] * nvars
    row[col_of[id_bucket]] = 1
    eq_rows.append(row)
    eq_rhs.append(1)

    constraint_log["basic"].append("A_identity = 1")
    constraint_log["basic"].append("A coefficients nonnegative integers")

    # Fixed ordinary A shells or total stabilizer size.
    if fixed_A is not None:
        for w, target in enumerate(fixed_A):
            if target == 0:
                continue

            row = [0] * nvars

            for bucket, col in col_of.items():
                if bucket_weight(blocks, bucket) == w:
                    row[col] += 1

            eq_rows.append(row)
            eq_rhs.append(target)

        constraint_log["basic"].append(
            "fixed ordinary stabilizer distribution A_w"
        )
    else:
        row = [1] * nvars
        eq_rows.append(row)
        eq_rhs.append(norm)

        constraint_log["basic"].append(
            "sum A = 2^(n-k)"
        )

    # ---------------------------------------------------------------
    # Ordinary MacWilliams / distance / B>=A.
    # ---------------------------------------------------------------

    if USE_MACWILLIAMS:
        constraint_log["basic"].append(
            "projected MacWilliams identity (B eliminated symbolically)"
        )
        constraint_log["basic"].append(
            "B >= A coefficientwise"
        )
        constraint_log["basic"].append(
            "A = B for projected buckets of weight < d"
        )

        for out_bucket in universe:
            sparse = B_rows.get(out_bucket, {})
            row_num = row_from_sparse(nvars, sparse)

            a_col = col_of.get(out_bucket)

            row = row_num[:]

            if a_col is not None:
                row[a_col] -= norm

            if bucket_weight(blocks, out_bucket) < d:
                eq_rows.append(row)
                eq_rhs.append(0)
            else:
                inequalities.append((row, 0))

    # ---------------------------------------------------------------
    # Shadow identity (SI).
    #
    # Projected/refined form:
    #     norm * S_beta = N^sh_beta(A),  norm=2^(n-k).
    # Thus N^sh_beta >= 0 and N^sh_beta == 0 (mod norm).
    # No new primal variables are needed.
    # ---------------------------------------------------------------

    shadow_mode = None

    if USE_SHADOW_IDENTITY:
        if shadow_projected_compatible:
            shadow_mode = "projected"

            for out_bucket in universe:
                rowS = row_from_sparse(
                    nvars, shadow_rows.get(out_bucket, {})
                )
                inequalities.append((rowS, 0))
                if not REAL_LP_SKIP_CONGRUENCES:
                    congruences.append((rowS, 0, norm))

            constraint_log["basic"].append(
                "projected/refined shadow identity "
                "(S eliminated; coefficientwise S>=0 and integral)"
            )
        else:
            shadow_mode = "ordinary-fallback"

            for out_w in range(n + 1):
                rowS = [0] * nvars
                for bucket, col in col_of.items():
                    in_w = bucket_weight(blocks, bucket)
                    rowS[col] += (
                        (-1) ** in_w
                        * q4_krawtchouk(n, out_w, in_w)
                    )
                inequalities.append((rowS, 0))
                if not REAL_LP_SKIP_CONGRUENCES:
                    congruences.append((rowS, 0, norm))

            constraint_log["basic"].append(
                "ordinary shadow identity fallback "
                "(projected tau=111 is not closed)"
            )
            constraint_log["skipped"].append(
                "coefficientwise projected shadow identity: "
                "tau=111 does not descend to selected projection"
            )

    # ---------------------------------------------------------------
    # Global parity cases.
    # ---------------------------------------------------------------

    if GLOBAL_PARITY_CASE is not None:
        if GLOBAL_PARITY_CASE not in (1, 2, 3):
            raise ValueError(
                "GLOBAL_PARITY_CASE must be None,1,2,or3."
            )

        if n - k < 1:
            raise ValueError(
                "Parity cases currently assume n-k >= 1."
            )

        A_even_row = [0] * nvars
        A_odd_row = [0] * nvars

        for bucket, col in col_of.items():
            if bucket_weight(blocks, bucket) % 2 == 0:
                A_even_row[col] += 1
            else:
                A_odd_row[col] += 1

        if not USE_MACWILLIAMS:
            raise ValueError(
                "GLOBAL_PARITY_CASE requires USE_MACWILLIAMS=True."
            )

        B_even_num = sum_sparse_rows(
            nvars,
            [
                B_rows.get(bucket, {})
                for bucket in universe
                if bucket_weight(blocks, bucket) % 2 == 0
            ],
        )

        if GLOBAL_PARITY_CASE == 1:
            eq_rows.append(A_even_row)
            eq_rhs.append(2 ** (n - k - 1))

            eq_rows.append(B_even_num)
            eq_rhs.append(
                norm * (2 ** (n + k - 1))
            )

        elif GLOBAL_PARITY_CASE == 2:
            eq_rows.append(A_odd_row)
            eq_rhs.append(0)

            target = (
                2 ** (n - k - 1)
                * (4 ** k + 2 ** k)
            )

            eq_rows.append(B_even_num)
            eq_rhs.append(norm * target)

        elif GLOBAL_PARITY_CASE == 3:
            eq_rows.append(A_odd_row)
            eq_rhs.append(0)

            target = (
                2 ** (n - k - 1)
                * (4 ** k - 2 ** k)
            )

            eq_rows.append(B_even_num)
            eq_rhs.append(norm * target)

        constraint_log["additional"].append(
            f"global parity/type case {GLOBAL_PARITY_CASE}"
        )

    # ---------------------------------------------------------------
    # Blockwise quadratic parity/type cases.
    #
    # For an odd-odd pair (tau_T,tau_U),
    #
    #   q(g)=q_{tau_T}(g_T)+q_{tau_U}(g_U) mod 2
    #
    # is a quadratic refinement of the full symplectic form.  Hence the
    # stabilizer/normalizer q-parity counts obey exactly the same three
    # cases as the global shadow parity:
    #
    #   Case 1: q|_S nonzero -> A and B split equally.
    #   Case 2: q|_S = 0, plus type.
    #   Case 3: q|_S = 0, minus type.
    #
    # B is eliminated through the ordinary MacWilliams rows, so no new
    # enumerator variables are introduced.
    # ---------------------------------------------------------------

    if normalized_blockwise_parity_cases:
        if n - k < 1:
            raise ValueError(
                "Blockwise parity cases currently assume n-k >= 1."
            )

        if not USE_MACWILLIAMS:
            raise ValueError(
                "BLOCKWISE_PARITY_CASES requires USE_MACWILLIAMS=True."
            )

        for pair, case in sorted(normalized_blockwise_parity_cases.items()):
            A_even_row = [0] * nvars
            A_odd_row = [0] * nvars

            for bucket, col in col_of.items():
                qbit = bucket_blockwise_twist_parity(
                    blocks, bucket, pair
                )
                if qbit is None:
                    raise AssertionError(
                        f"Lost parity visibility for {pair}."
                    )
                if qbit == 0:
                    A_even_row[col] += 1
                else:
                    A_odd_row[col] += 1

            B_even_num = sum_sparse_rows(
                nvars,
                [
                    B_rows.get(bucket, {})
                    for bucket in universe
                    if bucket_blockwise_twist_parity(
                        blocks, bucket, pair
                    ) == 0
                ],
            )

            if case == 1:
                eq_rows.append(A_even_row)
                eq_rhs.append(2 ** (n - k - 1))

                eq_rows.append(B_even_num)
                eq_rhs.append(
                    norm * (2 ** (n + k - 1))
                )

            elif case == 2:
                eq_rows.append(A_odd_row)
                eq_rhs.append(0)

                target = (
                    2 ** (n - k - 1)
                    * (4 ** k + 2 ** k)
                )
                eq_rows.append(B_even_num)
                eq_rhs.append(norm * target)

            elif case == 3:
                eq_rows.append(A_odd_row)
                eq_rhs.append(0)

                target = (
                    2 ** (n - k - 1)
                    * (4 ** k - 2 ** k)
                )
                eq_rows.append(B_even_num)
                eq_rhs.append(norm * target)

            constraint_log["additional"].append(
                "blockwise quadratic parity/type case "
                f"{case} for {blockwise_tau_name(pair)}"
            )

    # ---------------------------------------------------------------
    # Twisted CWE positivity.
    # ---------------------------------------------------------------

    if USE_TWISTED_CWE:
        enabled_nonzero = [
            tau for tau in compatible_twists
            if tau != tau000
            and not (USE_SHADOW_IDENTITY and tau == tau111)
        ]

        for tau in enabled_nonzero:
            rows = transform_rows[tau]

            for out_bucket in universe:
                row = row_from_sparse(
                    nvars,
                    rows.get(out_bucket, {}),
                )

                inequalities.append((row, 0))

        if enabled_nonzero:
            constraint_log["additional"].append(
                "twisted-CWE positivity for "
                + ",".join(tau_name(t) for t in enabled_nonzero)
            )

        if skipped_twists:
            constraint_log["skipped"].append(
                "twists not closed on selected projection: "
                + ",".join(tau_name(t) for t in skipped_twists)
            )

    # ---------------------------------------------------------------
    # Blockwise twisted positivity.
    #
    # These are same-parity local twists on the two split blocks.  Each
    # transformed coefficient is a fixed linear form in the existing A
    # variables, so only inequalities are added.
    # ---------------------------------------------------------------

    if USE_BLOCKWISE_TWISTS:
        for pair in enabled_blockwise_pairs:
            rows = blockwise_transform_rows[pair]

            for out_bucket in universe:
                row = row_from_sparse(
                    nvars,
                    rows.get(out_bucket, {}),
                )
                inequalities.append((row, 0))

        if enabled_blockwise_pairs:
            constraint_log["additional"].append(
                "blockwise twisted positivity for "
                + ",".join(
                    blockwise_tau_name(pair)
                    for pair in enabled_blockwise_pairs
                )
                + " (same A variables; no new enumerator variables)"
            )
        else:
            constraint_log["skipped"].append(
                "no additional same-parity blockwise twist remains after "
                "removing MI/shadow/global-twist duplicates"
            )

    # ---------------------------------------------------------------
    # Known stabilizer: existence.
    # ---------------------------------------------------------------

    if USE_KNOWN_STABILIZER and USE_REFERENCE_EXISTENCE:
        ref_bucket = known_stabilizer_bucket(blocks)

        if ref_bucket in col_of:
            row = [0] * nvars
            row[col_of[ref_bucket]] = 1
            inequalities.append((row, 1))

            constraint_log["additional"].append(
                "known stabilizer reference bucket A_g >= 1"
            )
        else:
            # Its ordinary shell or commutation pruning says zero.
            eq_rows.append([0] * nvars)
            eq_rhs.append(1)

            constraint_log["additional"].append(
                "known stabilizer reference bucket existence "
                "(immediately inconsistent with active support)"
            )

    # ---------------------------------------------------------------
    # Known stabilizer: commutation.
    # ---------------------------------------------------------------

    if USE_KNOWN_STABILIZER and USE_COMMUTATION:
        if not generator_comm_available:
            constraint_log["skipped"].append(
                "generator commutation: projection mixes commuting "
                "and anticommuting Pauli symbols"
            )
        else:
            if not USE_MACWILLIAMS:
                raise ValueError(
                    "Generator commutation on B requires "
                    "USE_MACWILLIAMS=True."
                )

            # A anticommuting buckets were already pruned.
            # Force B anticommuting output buckets to zero.
            count = 0

            for out_bucket in universe:
                if bucket_commutation_parity(
                    blocks,
                    out_bucket,
                    comm_bits,
                ):
                    row = row_from_sparse(
                        nvars,
                        B_rows.get(out_bucket, {}),
                    )

                    eq_rows.append(row)
                    eq_rhs.append(0)
                    count += 1

            constraint_log["additional"].append(
                f"known-generator commutation (A pruned; {count} B-zero equalities)"
            )

            # The shadow is the affine character coset satisfying
            # <h,g> = wt(g) mod 2.  Hence wrong-parity shadow buckets
            # vanish whenever commutation parity is visible.
            if USE_SHADOW_IDENTITY and shadow_projected_compatible:
                g_weight_parity = (
                    sum(block.size for block in blocks
                        if block.generator_symbol != "I") & 1
                )
                shadow_comm_zero = 0
                for out_bucket in universe:
                    if bucket_commutation_parity(
                        blocks, out_bucket, comm_bits
                    ) != g_weight_parity:
                        rowS = row_from_sparse(
                            nvars, shadow_rows.get(out_bucket, {})
                        )
                        eq_rows.append(rowS)
                        eq_rhs.append(0)
                        shadow_comm_zero += 1
                constraint_log["additional"].append(
                    "known-generator shadow character support "
                    f"({shadow_comm_zero} shadow-zero equalities)"
                )

    # ---------------------------------------------------------------
    # Known stabilizer: orbit symmetry.
    # ---------------------------------------------------------------

    if USE_KNOWN_STABILIZER and USE_ORBIT:
        if not generator_orbit_available:
            constraint_log["skipped"].append(
                "generator orbit: projection is not invariant under "
                "multiplication by the known stabilizer"
            )
        else:
            if not USE_MACWILLIAMS:
                raise ValueError(
                    "B-orbit symmetry requires USE_MACWILLIAMS=True."
                )

            seen_pairs = set()
            A_orbit_count = 0

            for bucket in universe:
                partner = bucket_orbit_map(
                    blocks,
                    bucket,
                    orbit_perms,
                )

                key = tuple(sorted(
                    (repr(bucket), repr(partner))
                ))

                if key in seen_pairs:
                    continue

                seen_pairs.add(key)

                c1 = col_of.get(bucket)
                c2 = col_of.get(partner)

                if c1 is None and c2 is None:
                    continue

                row = [0] * nvars

                if c1 is not None:
                    row[c1] += 1
                if c2 is not None:
                    row[c2] -= 1

                eq_rows.append(row)
                eq_rhs.append(0)
                A_orbit_count += 1

            seen_pairs.clear()
            B_orbit_count = 0

            for bucket in universe:
                partner = bucket_orbit_map(
                    blocks,
                    bucket,
                    orbit_perms,
                )

                key = tuple(sorted(
                    (repr(bucket), repr(partner))
                ))

                if key in seen_pairs:
                    continue

                seen_pairs.add(key)

                if bucket == partner:
                    continue

                row1 = row_from_sparse(
                    nvars,
                    B_rows.get(bucket, {}),
                )
                row2 = row_from_sparse(
                    nvars,
                    B_rows.get(partner, {}),
                )

                eq_rows.append(
                    add_rows(row1, row2, -1)
                )
                eq_rhs.append(0)
                B_orbit_count += 1

            constraint_log["additional"].append(
                f"generator A-orbit symmetry ({A_orbit_count} equalities)"
            )
            constraint_log["additional"].append(
                f"generator B-orbit symmetry ({B_orbit_count} equalities)"
            )

            # Sh(S) is invariant under multiplication by g in S,
            # so the same coefficient orbit holds for refined shadow.
            if USE_SHADOW_IDENTITY and shadow_projected_compatible:
                seen_pairs.clear()
                S_orbit_count = 0
                for bucket in universe:
                    partner = bucket_orbit_map(blocks, bucket, orbit_perms)
                    key = tuple(sorted((repr(bucket), repr(partner))))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    if bucket == partner:
                        continue
                    row1 = row_from_sparse(
                        nvars, shadow_rows.get(bucket, {})
                    )
                    row2 = row_from_sparse(
                        nvars, shadow_rows.get(partner, {})
                    )
                    eq_rows.append(add_rows(row1, row2, -1))
                    eq_rhs.append(0)
                    S_orbit_count += 1
                constraint_log["additional"].append(
                    f"generator shadow-orbit symmetry ({S_orbit_count} equalities)"
                )

    # ---------------------------------------------------------------
    # Self-orbit parity.
    # ---------------------------------------------------------------

    if USE_KNOWN_STABILIZER and USE_SELF_ORBIT_PARITY:
        if known_stabilizer_is_identity(blocks):
            raise ValueError(
                "Self-orbit parity requires a nonidentity known stabilizer."
            )

        if not generator_orbit_available:
            constraint_log["skipped"].append(
                "self-orbit parity: generator orbit does not descend "
                "to the projection"
            )
        else:
            if not USE_MACWILLIAMS:
                raise ValueError(
                    "B self-orbit parity requires USE_MACWILLIAMS=True."
                )

            A_self = 0
            B_self = 0

            for bucket in universe:
                partner = bucket_orbit_map(
                    blocks,
                    bucket,
                    orbit_perms,
                )

                if bucket != partner:
                    continue

                col = col_of.get(bucket)

                if col is not None:
                    row = [0] * nvars
                    row[col] = 1
                    congruences.append((row, 0, 2))
                    A_self += 1

                # B_bucket = numerator/norm must be even:
                # numerator == 0 mod (2*norm).
                rowB = row_from_sparse(
                    nvars,
                    B_rows.get(bucket, {}),
                )
                congruences.append(
                    (rowB, 0, 2 * norm)
                )
                B_self += 1

            constraint_log["additional"].append(
                f"self-orbit parity for A ({A_self} buckets)"
            )
            constraint_log["additional"].append(
                f"self-orbit parity for B ({B_self} buckets)"
            )

            if USE_SHADOW_IDENTITY and shadow_projected_compatible:
                S_self = 0
                for bucket in universe:
                    partner = bucket_orbit_map(blocks, bucket, orbit_perms)
                    if bucket != partner:
                        continue
                    rowS = row_from_sparse(
                        nvars, shadow_rows.get(bucket, {})
                    )
                    # S_bucket = rowS/norm must be even.
                    congruences.append((rowS, 0, 2 * norm))
                    S_self += 1
                constraint_log["additional"].append(
                    f"self-orbit parity for shadow ({S_self} buckets)"
                )

    # ---------------------------------------------------------------
    # Optional B integrality.
    # ---------------------------------------------------------------

    if B_INTEGRAL:
        if not USE_MACWILLIAMS:
            raise ValueError(
                "B_INTEGRAL requires USE_MACWILLIAMS=True."
            )

        for bucket in universe:
            rowB = row_from_sparse(
                nvars,
                B_rows.get(bucket, {}),
            )
            congruences.append((rowB, 0, norm))

        constraint_log["additional"].append(
            "ordinary B coefficient integrality"
        )

    # ---------------------------------------------------------------
    # Cleaning constraints for split geometry.
    #
    # Either block may be the correctable region.  If C is correctable,
    # |C|<d, and O=C^c is the other block, then
    #
    #     dim S_O - dim S_C = n-k-2|C|,
    #     |N_O| = 4^k |S_O|.
    #
    # Both sides are imposed if both split blocks have size < d.
    # ---------------------------------------------------------------

    if USE_CLEANING:
        if not USE_SPLIT or len(blocks) != 2:
            raise ValueError(
                "USE_CLEANING currently requires split mode."
            )

        if not USE_MACWILLIAMS:
            raise ValueError(
                "Cleaning constraints require USE_MACWILLIAMS=True."
            )

        correctable_indices = [
            idx for idx, block in enumerate(blocks)
            if block.size < d
        ]

        if not correctable_indices:
            raise ValueError(
                "Cleaning constraints require at least one split block "
                "with size < d."
            )

        identity_comps = [
            block.identity_composition()
            for block in blocks
        ]

        for c_idx in correctable_indices:
            o_idx = 1 - c_idx
            C = blocks[c_idx]
            O = blocks[o_idx]

            # S_C: supported only on correctable C -> O is identity.
            row_SC = [0] * nvars

            # S_O: supported only on O -> C is identity.
            row_SO = [0] * nvars

            for bucket, col in col_of.items():
                if bucket[o_idx] == identity_comps[o_idx]:
                    row_SC[col] += 1
                if bucket[c_idx] == identity_comps[c_idx]:
                    row_SO[col] += 1

            exponent = n - k - 2 * C.size

            if exponent >= 0:
                # |S_O| = 2^exponent |S_C|.
                row = add_rows(
                    row_SO,
                    row_SC,
                    -(2 ** exponent),
                )
            else:
                # 2^(-exponent)|S_O| = |S_C|.
                row = add_rows(
                    [x * (2 ** (-exponent)) for x in row_SO],
                    row_SC,
                    -1,
                )

            eq_rows.append(row)
            eq_rhs.append(0)

            # N_O consists of B buckets with the correctable block C
            # equal to identity.
            B_O_num = sum_sparse_rows(
                nvars,
                [
                    B_rows.get(bucket, {})
                    for bucket in universe
                    if bucket[c_idx] == identity_comps[c_idx]
                ],
            )

            # sum B_O = 4^k sum A_O:
            # B_O_num / norm = 4^k * row_SO.
            row = add_rows(
                B_O_num,
                row_SO,
                -(norm * (4 ** k)),
            )

            eq_rows.append(row)
            eq_rhs.append(0)

            constraint_log["additional"].append(
                f"local dimension relation on correctable {C.name}: "
                f"rho_{O.name}-rho_{C.name}=n-k-2|{C.name}|"
            )
            constraint_log["additional"].append(
                f"cleaning from {C.name}: |N_{O.name}|=4^k |S_{O.name}|"
            )

    # ---------------------------------------------------------------
    # Exact projected-cell capacities.
    # ---------------------------------------------------------------

    if USE_REFINED_CELL_CAPACITY:
        if not USE_MACWILLIAMS:
            raise ValueError("USE_REFINED_CELL_CAPACITY requires USE_MACWILLIAMS=True.")

        cap_count = 0
        for bucket in universe:
            cap = bucket_capacity(blocks, bucket)

            # B_bucket = rowB/norm <= cap.
            rowB = row_from_sparse(nvars, B_rows.get(bucket, {}))
            inequalities.append(([-x for x in rowB], -(norm * cap)))

            # Shadow_bucket = rowS/norm <= cap whenever projected shadow closes.
            if USE_SHADOW_IDENTITY and shadow_projected_compatible:
                rowS = row_from_sparse(nvars, shadow_rows.get(bucket, {}))
                inequalities.append(([-x for x in rowS], -(norm * cap)))

            cap_count += 1

        constraint_log["additional"].append(
            f"exact RXxRY cell capacities for A/B/shadow ({cap_count} buckets)"
        )

    # ---------------------------------------------------------------
    # Upper bounds for every active A variable.
    # ---------------------------------------------------------------

    upper_bounds = {}

    for bucket, col in col_of.items():
        w = bucket_weight(blocks, bucket)
        cap = bucket_capacity(blocks, bucket)

        if fixed_A is not None:
            upper_bounds[col] = min(fixed_A[w], cap) if USE_REFINED_CELL_CAPACITY else fixed_A[w]
        else:
            upper_bounds[col] = min(norm, cap) if USE_REFINED_CELL_CAPACITY else norm

    metadata = {
        "blocks": blocks,
        "universe": universe,
        "active_buckets": active_buckets,
        "col_of": col_of,
        "fixed_A": fixed_A,
        "norm": norm,
        "nvars": nvars,
        "eq_rows": eq_rows,
        "eq_rhs": eq_rhs,
        "inequalities": inequalities,
        "congruences": congruences,
        "upper_bounds": upper_bounds,
        "constraint_log": constraint_log,
        "requested_twists": requested_twists,
        "compatible_twists": compatible_twists,
        "skipped_twists": skipped_twists,
        "blockwise_pairs": blockwise_pairs,
        "enabled_blockwise_pairs": tuple(enabled_blockwise_pairs),
        "blockwise_parity_cases": dict(normalized_blockwise_parity_cases),
        "blockwise_transform_rows": blockwise_transform_rows,
        "generator_orbit_available": generator_orbit_available,
        "generator_comm_available": generator_comm_available,
        "orbit_perms": orbit_perms,
        "comm_bits": comm_bits,
        "B_rows": B_rows,
        "shadow_rows": shadow_rows,
        "shadow_mode": shadow_mode,
        "shadow_projected_compatible": shadow_projected_compatible,
    }

    return metadata


# =====================================================================
# Exact RREF parameterization
# =====================================================================

def exact_rref_parameterization(nvars, eq_rows, eq_rhs):
    M = sp.Matrix(eq_rows)
    b = sp.Matrix(eq_rhs)
    Aug = M.row_join(b)

    D = DomainMatrix.from_Matrix(Aug)
    R, pivots_all = D.rref(method="FF")
    RM = R.to_Matrix()

    if nvars in pivots_all:
        return {
            "inconsistent": True,
            "rank": len(
                [p for p in pivots_all if p < nvars]
            ),
        }

    pivots = tuple(
        p for p in pivots_all
        if p < nvars
    )
    pivot_set = set(pivots)

    free_cols = [
        j for j in range(nvars)
        if j not in pivot_set
    ]

    free_pos = {
        col: pos
        for pos, col in enumerate(free_cols)
    }

    nfree = len(free_cols)
    exprs = [None] * nvars

    for col in free_cols:
        coeff = [Fraction(0) for _ in range(nfree)]
        coeff[free_pos[col]] = Fraction(1)
        exprs[col] = (
            Fraction(0),
            tuple(coeff),
        )

    for row_index, pivot_col in enumerate(pivots):
        const = Fraction(RM[row_index, nvars])
        coeff = []

        for free_col in free_cols:
            coeff.append(
                -Fraction(RM[row_index, free_col])
            )

        exprs[pivot_col] = (
            const,
            tuple(coeff),
        )

    denominator_set = set()
    max_den = 1

    for const, coeff in exprs:
        den = const.denominator

        for c in coeff:
            den = math.lcm(
                den,
                c.denominator,
            )

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


# =====================================================================
# Affine substitution
# =====================================================================

def substitute_original_row(
    row,
    rhs,
    exprs,
    nfree,
):
    const = Fraction(-rhs)
    coeff = [Fraction(0) for _ in range(nfree)]

    for col, scale in enumerate(row):
        if not scale:
            continue

        e_const, e_coeff = exprs[col]
        const += Fraction(scale) * e_const

        for r, c in enumerate(e_coeff):
            coeff[r] += Fraction(scale) * c

    return const, tuple(coeff)


def integerize_affine(const, coeff):
    den = const.denominator

    for c in coeff:
        den = math.lcm(
            den,
            c.denominator,
        )

    c0 = int(const * den)
    ci = tuple(
        int(c * den)
        for c in coeff
    )

    g = gcd_many(
        [den, c0] + list(ci)
    )

    if g > 1:
        den //= g
        c0 //= g
        ci = tuple(c // g for c in ci)

    return c0, ci, den


def canonical_geq(c0, ci):
    g = gcd_many(
        [c0] + list(ci)
    )

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
# Exact modular / affine-lattice preprocessing
# =====================================================================

def _v2_part(modulus):
    """Return (largest power of two dividing modulus, odd cofactor)."""
    modulus = int(modulus)
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    p2 = modulus & -modulus
    return p2, modulus // p2


def _center_mod(value, modulus):
    value %= modulus
    return value - modulus if value > modulus // 2 else value


def _transform_integer_affine(c0, ci, offset, basis):
    """
    Compose F = offset + basis*Y into c0 + ci.F.
    Return (new_c0, new_ci).
    """
    nfree = len(offset)

    new_c0 = int(c0) + sum(
        int(ci[i]) * int(offset[i])
        for i in range(nfree)
    )

    new_ci = []
    for j in range(nfree):
        new_ci.append(
            sum(
                int(ci[i]) * int(basis[i][j])
                for i in range(nfree)
            )
        )

    return new_c0, tuple(new_ci)


def _normalize_modular_integer_affine(c0, ci, modulus):
    """
    Simplify c0 + ci.Y == 0 mod modulus by the gcd shared with modulus.

    Returns:
      ("true",  c0, ci, modulus)
      ("false", c0, ci, modulus)
      ("active",c0, ci, modulus)
    """
    modulus = int(modulus)
    if modulus <= 0:
        raise ValueError("modulus must be positive")

    c0 = int(c0)
    ci = tuple(int(x) for x in ci)

    g = modulus
    for value in ci:
        g = math.gcd(g, abs(value))

    rhs = (-c0) % modulus
    if rhs % g != 0:
        return "false", c0, ci, modulus

    modulus //= g
    c0 //= g
    ci = tuple(value // g for value in ci)

    if modulus == 1:
        return "true", c0, ci, modulus

    c0 %= modulus
    ci = tuple(value % modulus for value in ci)

    if all(value == 0 for value in ci):
        if c0 % modulus == 0:
            return "true", c0, ci, modulus
        return "false", c0, ci, modulus

    return "active", c0, ci, modulus


def absorb_power2_congruences(nvars, congruences):
    """
    Exact full-rank affine-lattice parameterization for power-of-two
    congruences.

    Each congruence is
        (c0, ci, modulus, label)
    meaning
        c0 + ci.F == 0 mod modulus.

    Returns
        offset, basis, stats
    such that every integer solution is represented exactly as
        F = offset + basis.Y,  Y in Z^n.
    """
    offset = [0] * nvars
    basis = [
        [1 if i == j else 0 for j in range(nvars)]
        for i in range(nvars)
    ]

    remaining = list(congruences)
    active_steps = 0
    tautologies = 0
    by_type_input = {}
    by_type_active = {}

    for _, _, _, label in remaining:
        by_type_input[label] = by_type_input.get(label, 0) + 1

    def transformed_data(item):
        c0, ci, modulus, label = item

        if modulus <= 0 or (modulus & (modulus - 1)) != 0:
            raise ValueError(
                f"2-adic lattice requires power-of-two modulus; "
                f"got {modulus} for {label}"
            )

        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )

        g = modulus
        for value in tci:
            g = math.gcd(g, abs(int(value)))

        rhs = (-tc0) % modulus
        if rhs % g != 0:
            raise ValueError(
                f"UNSAT in 2-adic lattice preprocessing ({label}): "
                f"gcd={g} does not divide rhs={rhs}"
            )

        q = modulus // g
        nz = sum(value != 0 for value in tci)
        return tc0, tci, g, q, label, nz

    while remaining:
        evaluated = []
        still_active = []

        # Batch-remove all congruences made tautological by prior substitutions.
        for item in remaining:
            data = transformed_data(item)
            if data[3] == 1:
                tautologies += 1
            else:
                evaluated.append((item, data))
                still_active.append(item)

        remaining = still_active
        if not evaluated:
            break

        # Strongest 2-adic restriction first; sparse tie-break.
        best_pos = max(
            range(len(evaluated)),
            key=lambda idx: (
                evaluated[idx][1][3],
                -evaluated[idx][1][5],
            ),
        )
        best_item, data = evaluated[best_pos]
        remaining.remove(best_item)

        tc0, tci, g, q, label, _ = data

        reduced = [int(value) // g for value in tci]
        rhs = ((-tc0) // g) % q

        odd_cols = [
            j for j, value in enumerate(reduced)
            if value & 1
        ]
        if not odd_cols:
            raise AssertionError(
                f"No odd pivot after gcd reduction for {label}; "
                f"q={q}, gcd={g}"
            )

        pivot = min(
            odd_cols,
            key=lambda j: abs(_center_mod(reduced[j], q)),
        )

        inv = pow(reduced[pivot] % q, -1, q)
        r0 = _center_mod(inv * rhs, q)

        r = [0] * nvars
        for j in range(nvars):
            if j == pivot:
                continue
            r[j] = _center_mod(
                -inv * (reduced[j] % q),
                q,
            )

        old_col = [basis[i][pivot] for i in range(nvars)]

        offset = [
            offset[i] + old_col[i] * r0
            for i in range(nvars)
        ]

        new_basis = [[0] * nvars for _ in range(nvars)]
        for i in range(nvars):
            for j in range(nvars):
                if j == pivot:
                    new_basis[i][j] = basis[i][pivot] * q
                else:
                    new_basis[i][j] = (
                        basis[i][j]
                        + basis[i][pivot] * r[j]
                    )

        basis = new_basis
        active_steps += 1
        by_type_active[label] = by_type_active.get(label, 0) + 1

    # Exact final regression.
    for c0, ci, modulus, label in congruences:
        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )
        if tc0 % modulus != 0:
            raise AssertionError(
                f"Lattice constant regression failed for {label}"
            )
        if any(value % modulus != 0 for value in tci):
            raise AssertionError(
                f"Lattice coefficient regression failed for {label}"
            )

    max_basis = max(
        (abs(v) for row in basis for v in row),
        default=0,
    )
    max_offset = max(
        (abs(v) for v in offset),
        default=0,
    )

    return offset, basis, {
        "input": len(congruences),
        "active": active_steps,
        "tautologies": tautologies,
        "max_basis_bits": max_basis.bit_length(),
        "max_offset_bits": max_offset.bit_length(),
        "by_type_input": by_type_input,
        "by_type_active": by_type_active,
    }


def _free_values_from_lattice_model(model, yvars, param):
    offset = param.get("lattice_offset")
    basis = param.get("lattice_basis")

    if offset is None or basis is None:
        return [
            model.eval(v, model_completion=True).as_long()
            for v in yvars
        ]

    y = [
        model.eval(v, model_completion=True).as_long()
        for v in yvars
    ]

    return [
        int(offset[i])
        + sum(int(basis[i][j]) * y[j] for j in range(len(y)))
        for i in range(len(offset))
    ]


def self_test_lattice_backend():
    """Exhaustive tiny residue regression for the exact 2-adic engine."""
    tests = [
        (-3, (1, 2), 8, "sat1"),
        (-6, (4, 2), 16, "sat2"),
        (-1, (0, 1), 4, "sat3"),
    ]

    off, H, _ = absorb_power2_congruences(2, tests)

    brute = set()
    image = set()
    M = 16

    for x0 in range(M):
        for x1 in range(M):
            if all(
                (c0 + ci[0] * x0 + ci[1] * x1) % mod == 0
                for c0, ci, mod, _ in tests
            ):
                brute.add((x0, x1))

    for y0 in range(M):
        for y1 in range(M):
            x0 = (
                off[0] + H[0][0] * y0 + H[0][1] * y1
            ) % M
            x1 = (
                off[1] + H[1][0] * y0 + H[1][1] * y1
            ) % M
            image.add((x0, x1))

    assert image == brute

    caught = False
    try:
        absorb_power2_congruences(
            1,
            [(-2, (4,), 8, "unsat")],
        )
    except ValueError as exc:
        caught = "UNSAT in 2-adic lattice preprocessing" in str(exc)
    assert caught


# =====================================================================
# Reduced Z3 model
# =====================================================================

def build_reduced_solver(problem, param):
    """
    Exact reduced backend:
      RREF free coordinates F
        -> collect every integrality congruence
        -> absorb all 2-adic factors into F=offset+basis*Y
        -> leave only residual odd modular factors in Z3.
    """
    nvars = problem["nvars"]
    upper_bounds = problem["upper_bounds"]

    free_cols = param["free_cols"]
    exprs = param["exprs"]
    nfree = len(free_cols)

    # ---------------------------------------------------------------
    # Collect pivot integrality + original modular constraints.
    # ---------------------------------------------------------------
    raw_congruences = []
    pivot_integrality_count = 0
    original_modular_count = 0

    free_col_set = set(free_cols)

    for col in range(nvars):
        if col in free_col_set:
            continue

        const, coeff = exprs[col]
        c0, ci, den = integerize_affine(const, coeff)

        if den > 1:
            raw_congruences.append(
                (c0, ci, den, "pivot-integrality")
            )
            pivot_integrality_count += 1

    for row, rhs, modulus in problem["congruences"]:
        const, coeff = substitute_original_row(
            row, rhs, exprs, nfree
        )
        c0, ci, den = integerize_affine(const, coeff)

        raw_congruences.append(
            (
                c0,
                ci,
                den * modulus,
                "refined-modular",
            )
        )
        original_modular_count += 1

    # ---------------------------------------------------------------
    # Normalize/deduplicate, split 2-adic vs odd coprime factors.
    # ---------------------------------------------------------------
    seen_full = set()
    power2_congruences = []
    odd_congruences = []
    trivial_modular = 0
    impossible_modular = 0

    for c0, ci, modulus, label in raw_congruences:
        state, nc0, nci, nmod = _normalize_modular_integer_affine(
            c0, ci, modulus
        )

        if state == "true":
            trivial_modular += 1
            continue

        if state == "false":
            impossible_modular += 1
            continue

        sig = (
            nc0 % nmod,
            tuple(v % nmod for v in nci),
            nmod,
        )
        if sig in seen_full:
            continue
        seen_full.add(sig)

        p2, odd = _v2_part(nmod)

        if p2 > 1:
            power2_congruences.append(
                (nc0, nci, p2, label)
            )
        if odd > 1:
            odd_congruences.append(
                (nc0, nci, odd, label)
            )

    solver = SolverFor("QF_LIA")
    if impossible_modular:
        solver.add(False)

    # ---------------------------------------------------------------
    # Exact affine 2-adic lattice.
    # ---------------------------------------------------------------
    offset, basis, lattice_stats = absorb_power2_congruences(
        nfree,
        power2_congruences,
    )

    param["lattice_offset"] = offset
    param["lattice_basis"] = basis

    yvars = [
        Int(f"Y_{r}_from_F{col}")
        for r, col in enumerate(free_cols)
    ]

    # ---------------------------------------------------------------
    # Bounds for EVERY original projected A coordinate.
    # ---------------------------------------------------------------
    unique_bounds = set()
    original_A_bounds = 0

    for col in range(nvars):
        const, coeff = exprs[col]
        c0, ci, den = integerize_affine(const, coeff)

        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )

        lin = z3_linear(tc0, tci, yvars)

        lo_sig = (tc0, tci, ">=", 0)
        hi_sig = (
            tc0, tci, "<=", den * int(upper_bounds[col])
        )

        if lo_sig not in unique_bounds:
            solver.add(lin >= 0)
            unique_bounds.add(lo_sig)
            original_A_bounds += 1

        if hi_sig not in unique_bounds:
            solver.add(lin <= den * int(upper_bounds[col]))
            unique_bounds.add(hi_sig)
            original_A_bounds += 1

    # ---------------------------------------------------------------
    # Residual ODD modular constraints only.
    # ---------------------------------------------------------------
    unique_odd = set()
    odd_trivial = 0
    odd_impossible = 0

    for c0, ci, modulus, label in odd_congruences:
        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )

        state, nc0, nci, nmod = _normalize_modular_integer_affine(
            tc0, tci, modulus
        )

        if state == "true":
            odd_trivial += 1
            continue

        if state == "false":
            odd_impossible += 1
            solver.add(False)
            continue

        sig = (
            nc0 % nmod,
            tuple(v % nmod for v in nci),
            nmod,
        )
        if sig in unique_odd:
            continue
        unique_odd.add(sig)

        solver.add(
            z3_linear(nc0, nci, yvars) % nmod == 0
        )

    # ---------------------------------------------------------------
    # All inequalities after lattice substitution.
    # ---------------------------------------------------------------
    unique_ineq = set()
    trivial_true = 0
    trivial_false = 0

    for row, rhs in problem["inequalities"]:
        const, coeff = substitute_original_row(
            row, rhs, exprs, nfree
        )
        c0, ci, den = integerize_affine(const, coeff)

        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )
        tc0, tci = canonical_geq(tc0, tci)

        if all(v == 0 for v in tci):
            if tc0 >= 0:
                trivial_true += 1
            else:
                trivial_false += 1
                solver.add(False)
            continue

        sig = (tc0, tci)
        if sig in unique_ineq:
            continue
        unique_ineq.add(sig)

        solver.add(z3_linear(tc0, tci, yvars) >= 0)

    # Exact regression for every absorbed 2-adic congruence.
    for c0, ci, modulus, label in power2_congruences:
        tc0, tci = _transform_integer_affine(
            c0, ci, offset, basis
        )
        if tc0 % modulus != 0:
            raise AssertionError(
                f"2-adic regression constant failed for {label}"
            )
        if any(v % modulus != 0 for v in tci):
            raise AssertionError(
                f"2-adic regression coefficient failed for {label}"
            )

    meta = {
        "free_vars": len(yvars),
        "pivot_congruences": pivot_integrality_count,
        "original_modular": original_modular_count,
        "unique_full_modular": len(seen_full),
        "power2_input": len(power2_congruences),
        "odd_input": len(odd_congruences),
        "odd_left_z3": len(unique_odd),
        "trivial_modular": trivial_modular,
        "impossible_modular": impossible_modular,
        "odd_trivial": odd_trivial,
        "odd_impossible": odd_impossible,
        "pivot_bounds": 0,
        "original_A_bounds": original_A_bounds,
        "unique_ineq": len(unique_ineq),
        "trivial_true": trivial_true,
        "trivial_false": trivial_false,
        "lattice": lattice_stats,
    }

    return solver, yvars, meta


# =====================================================================
# Model reconstruction
# =====================================================================

def eval_expr(model, free_vars, expr, param=None):
    const, coeff = expr

    if param is not None and param.get("lattice_offset") is not None:
        F_values = _free_values_from_lattice_model(
            model, free_vars, param
        )

        value = const
        for c, f_value in zip(coeff, F_values):
            if c:
                value += c * f_value
    else:
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
            f"Nonintegral reconstructed A={value}."
        )

    return value.numerator


def print_model(problem, param, solver, free_vars):
    model = solver.model()
    blocks = problem["blocks"]
    active_buckets = problem["active_buckets"]

    print()
    print("Nonzero projected A coefficients:")

    shown = 0

    for bucket, expr in zip(
        active_buckets,
        param["exprs"],
    ):
        value = eval_expr(
            model,
            free_vars,
            expr,
            param=param,
        )

        if value == 0 and not SHOW_ZERO_MODEL_ENTRIES:
            continue

        print(
            f"  A[{bucket_label(blocks,bucket)}] = {value}"
        )
        shown += 1

    if shown == 0:
        print("  (none)")


# =====================================================================
# Reporting
# =====================================================================

def print_configuration(problem):
    blocks = problem["blocks"]
    fixed_A = problem["fixed_A"]
    log = problem["constraint_log"]

    print("=" * 82)
    print("Quantum weight-enumerator integer feasibility")
    print("=" * 82)
    print(f"Code:                         [[{n},{k},{d}]]")

    if USE_SPLIT:
        print("Geometry:                     split")
        print(
            f"Blocks:                       T={T_SIZE}, U={U_SIZE}"
        )
        print(
            f"Projection T:                 {projection_name(blocks[0].cells)}"
        )
        print(
            f"Projection U:                 {projection_name(blocks[1].cells)}"
        )
    else:
        print("Geometry:                     unsplit")
        print(
            f"Projection:                   {projection_name(blocks[0].cells)}"
        )

    print(
        "Fixed ordinary A_w:           "
        + ("yes" if fixed_A is not None else "no")
    )

    if fixed_A is not None:
        print(
            "Nonzero A_w:                  "
            + ", ".join(
                f"A_{w}={a}"
                for w, a in enumerate(fixed_A)
                if a
            )
        )

    print(
        f"Original projected A variables:{problem['nvars']:>8}"
    )
    print(
        f"All projected output buckets: {len(problem['universe']):>8}"
    )

    if USE_KNOWN_STABILIZER:
        if USE_SPLIT:
            print(
                "Known stabilizer pattern:     "
                f"{STABILIZER_ON_T}^{T_SIZE} "
                f"{STABILIZER_ON_U}^{U_SIZE}"
            )
        else:
            print(
                "Known stabilizer pattern:     "
                f"{KNOWN_STABILIZER_SYMBOL}^{n}"
            )

        print(
            "Generator orbit compatible:     "
            + yesno(problem["generator_orbit_available"])
        )
        print(
            "Generator commutation visible:  "
            + yesno(problem["generator_comm_available"])
        )

    print("-" * 82)
    print("Basic mathematical constraints:")

    for item in log["basic"]:
        print(f"  [x] {item}")

    print("-" * 82)
    print("Additional constraints enabled:")

    if log["additional"]:
        for item in log["additional"]:
            print(f"  [x] {item}")
    else:
        print("  (none)")

    if USE_TWISTED_CWE:
        print(
            "  Twists requested:            "
            + ",".join(
                tau_name(t)
                for t in problem["requested_twists"]
            )
        )
        print(
            "  Twists compatible:           "
            + ",".join(
                tau_name(t)
                for t in problem["compatible_twists"]
            )
        )

    if USE_BLOCKWISE_TWISTS:
        print(
            "  Blockwise positive pairs:     "
            + ",".join(
                blockwise_tau_name(pair)
                for pair in problem["blockwise_pairs"]
            )
        )
        print(
            "  Blockwise extra pairs:        "
            + (
                ",".join(
                    blockwise_tau_name(pair)
                    for pair in problem["enabled_blockwise_pairs"]
                )
                or "(none)"
            )
        )

    if problem.get("blockwise_parity_cases"):
        print(
            "  Blockwise parity cases:       "
            + ",".join(
                f"{blockwise_tau_name(pair)}:case{case}"
                for pair, case in sorted(
                    problem["blockwise_parity_cases"].items()
                )
            )
        )

    if log["skipped"]:
        print("-" * 82)
        print("Requested constraints skipped:")

        for item in log["skipped"]:
            print(f"  [!] {item}")

    print("=" * 82)


def print_preprocessing(problem, param, reduced_meta, times):
    print()
    print("Solver preprocessing:")
    print(
        f"  Equality rows before RREF:      {len(problem['eq_rows'])}"
    )
    print(
        f"  Equality rank:                  {param['rank']}"
    )
    print(
        f"  Free integer coordinates:       {len(param['free_cols'])}"
    )
    print(
        f"  Eliminated A coordinates:       {param['rank']}"
    )
    print(
        "  Pivot denominators:             "
        + ",".join(
            map(str, param["denominator_set"])
        )
    )
    print(
        f"  Maximum pivot denominator:      {param['max_den']}"
    )
    print(
        f"  Pivot-integrality congruences:  {reduced_meta['pivot_congruences']}"
    )
    print(
        f"  Extra modular constraints:      {reduced_meta['original_modular']}"
    )
    print(
        f"  Reduced unique inequalities:    {reduced_meta['unique_ineq']}"
    )
    print(
        f"  Trivial inequalities removed:   {reduced_meta['trivial_true']}"
    )
    print(
        f"  Mathematical build time:        {times['math']:.3f} s"
    )
    print(
        f"  Exact RREF time:                {times['rref']:.3f} s"
    )
    print(
        f"  Reduced-system build time:      {times['reduce']:.3f} s"
    )

    if SHOW_FREE_COORDINATES and len(param["free_cols"]) <= 40:
        print()
        print("Free projected A coordinates:")

        for col in param["free_cols"]:
            bucket = problem["active_buckets"][col]
            print(
                "  "
                + bucket_label(
                    problem["blocks"],
                    bucket,
                )
            )


# =====================================================================
# Live progress / heartbeat
# =====================================================================

def run_stage_with_heartbeat(stage_label, fn, heartbeat_text="still working"):
    """
    Run a blocking stage while a lightweight daemon thread periodically
    reports elapsed wall-clock time.  The heartbeat thread never touches
    SymPy or Z3 objects; it only prints timing information.
    """
    start = time.perf_counter()

    if SHOW_LIVE_PROGRESS:
        print(stage_label, flush=True)

    stop_event = threading.Event()
    heartbeat_thread = None

    interval = PROGRESS_EVERY_SECONDS
    heartbeat_enabled = (
        SHOW_LIVE_PROGRESS
        and interval is not None
        and interval > 0
    )

    if heartbeat_enabled:
        def heartbeat():
            while not stop_event.wait(interval):
                elapsed = time.perf_counter() - start
                print(
                    f"        {heartbeat_text}... {elapsed:.1f} s elapsed",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            daemon=True,
        )
        heartbeat_thread.start()

    try:
        result = fn()
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.2)

    elapsed = time.perf_counter() - start

    if SHOW_LIVE_PROGRESS:
        print(f"        done in {elapsed:.3f} s", flush=True)

    return result, elapsed


# =====================================================================
# Persistent candidate-database transverse RXxRY stabilizer-reference driver
# =====================================================================


def load_candidate_database(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"candidate_db_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import candidate database: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_database_solver_key(db, db_path):
    """
    Add this solver key to an existing SLR3_v2 executable database in place.

    Older candidate databases know only the original three refined solvers.
    Re-enumerating SLR3 just to add one bookkeeping key would discard useful
    history, so this solver performs a narrow schema-compatible migration:

      * append SOLVER_NAME to REFINED_SOLVER_KEYS;
      * add a not_tested entry to every candidate;
      * update DATABASE["refined_solver_keys"];
      * atomically rewrite the DATABASE block and the generated header tuple.

    No ordinary candidate or previous refined result is changed.
    """
    db_path = Path(db_path).resolve()
    old_keys = list(getattr(db, "REFINED_SOLVER_KEYS", ()))

    already = (
        SOLVER_NAME in old_keys
        and all(SOLVER_NAME in c.get("tests", {}) for c in db.CANDIDATES)
    )
    if already:
        return False

    new_keys = list(old_keys)
    if SOLVER_NAME not in new_keys:
        new_keys.append(SOLVER_NAME)

    default_test = {
        "status": "not_tested",
        "version": None,
        "runtime": None,
        "details": None,
    }
    for candidate in db.CANDIDATES:
        candidate.setdefault("tests", {})
        candidate["tests"].setdefault(SOLVER_NAME, dict(default_test))

    db.REFINED_SOLVER_KEYS = tuple(new_keys)
    db.DATABASE["refined_solver_keys"] = list(new_keys)

    if not hasattr(db, "_atomic_save"):
        raise RuntimeError(
            "Candidate database does not provide _atomic_save(); "
            "cannot safely migrate the refined-solver key."
        )
    db._atomic_save()

    # _atomic_save() intentionally rewrites only the DATABASE block.  Patch
    # the generated top-level tuple as well so future imports know the key.
    text = db_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith("REFINED_SOLVER_KEYS = "):
            ending = "\n" if line.endswith("\n") else ""
            lines[i] = f"REFINED_SOLVER_KEYS = {tuple(new_keys)!r}" + ending
            replaced = True
            break
    if not replaced:
        raise RuntimeError(
            "Migrated DATABASE block but could not find generated "
            "REFINED_SOLVER_KEYS header line."
        )

    tmp = db_path.with_suffix(db_path.suffix + ".migrate.tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(db_path)
    return True


def status_name(result):
    if result == sat:
        return "sat"
    if result == unsat:
        return "unsat"
    if result == unknown:
        return "unknown"
    return "error"


def positive_reference_weights(fixed_A, order=None):
    order = REFERENCE_U_ORDER if order is None else order
    if order not in ("ascending", "descending"):
        raise ValueError("reference-u order must be ascending or descending")
    vals = [int(u) for u, a in fixed_A.items() if int(u) > 0 and int(a) > 0]
    vals.sort(reverse=(order == "descending"))
    return vals


def qy_case_compatible(global_case, qy_case, u):
    """
    Compatibility forced by q_Y + q_sh = <Y^U,.> and the known X^U in S.

    If both restrictions vanish (cases 2/3), Y^U lies in S^perp.  Since
    X^U is an actual stabilizer, this requires u even.  For even u the
    two quotient Gauss signs are equal, hence the 2/3 types must match.
    """
    global_case = int(global_case)
    qy_case = int(qy_case)
    u = int(u)
    if global_case in (2,3) and qy_case in (2,3):
        return (u % 2 == 0) and (global_case == qy_case)
    return True


def transverse_membership_modes(global_case, qy_case, u):
    """
    Return exhaustive possibilities for Y^U relative to S subset S^perp.

      outside_normalizer      : Y^U notin S^perp
      normalizer_logical      : Y^U in S^perp \\ S
      normalizer_stabilizer   : Y^U in S

    When exactly one q restriction is case 1, the difference is nonzero,
    so Y^U is outside S^perp.  When both are 2/3, membership in S^perp is
    forced.  When both are case 1, normalizer membership is undetermined.
    The known X^U stabilizer rules out Y^U in S^perp for odd u.
    """
    g = int(global_case)
    q = int(qy_case)
    u = int(u)

    if not qy_case_compatible(g, q, u):
        return []

    if (g == 1) ^ (q == 1):
        return ["outside_normalizer"]

    if g in (2,3) and q in (2,3):
        # Compatible implies u even and same Gauss type.
        return ["normalizer_logical", "normalizer_stabilizer"]

    # both case 1
    if u % 2:
        return ["outside_normalizer"]
    return ["outside_normalizer", "normalizer_logical", "normalizer_stabilizer"]


def configure_reference_branch(fixed_A, case, u, qy_case, timeout_ms=None):
    global FIXED_A, GLOBAL_PARITY_CASE, T_SIZE, U_SIZE, USE_CLEANING
    global BLOCKWISE_PARITY_CASES, TIMEOUT_MS

    FIXED_A = {int(w): int(v) for w, v in fixed_A.items()}
    GLOBAL_PARITY_CASE = int(case)
    U_SIZE = int(u)
    T_SIZE = n - U_SIZE

    USE_CLEANING = (T_SIZE < d or U_SIZE < d)

    if USE_TRANSVERSE_QY_CASES:
        BLOCKWISE_PARITY_CASES = {
            ((1,1,1), (0,1,0)): int(qy_case),
        }
    else:
        BLOCKWISE_PARITY_CASES = {}

    if timeout_ms is not None:
        TIMEOUT_MS = timeout_ms


def _transverse_y_action(blocks):
    """Return R_Y-visible multiplication and commutation data for Y^U."""
    perms = []
    bits = []
    for block in blocks:
        symbol = "I" if block.name == "T" else "Y"
        p = induced_generator_cell_permutation(block.cells, symbol)
        b = commutation_cell_bits(block.cells, symbol)
        if p is None or b is None:
            raise AssertionError("Y^U action/commutation must descend to canonical RXxRY.")
        perms.append(p)
        bits.append(b)
    return tuple(perms), tuple(bits)


def _transverse_y_bucket(blocks):
    comps = []
    for block in blocks:
        symbol = "I" if block.name == "T" else "Y"
        comps.append(block.symbol_composition(symbol))
    return tuple(comps)


def _add_zero_A_on_parity(problem, bit_lists, parity=1, label=None):
    count = 0
    for bucket, col in problem["col_of"].items():
        if bucket_commutation_parity(problem["blocks"], bucket, bit_lists) == parity:
            row = [0] * problem["nvars"]
            row[col] = 1
            problem["eq_rows"].append(row)
            problem["eq_rhs"].append(0)
            count += 1
    if label:
        problem["constraint_log"]["additional"].append(f"{label} ({count} A-zero equalities)")
    return count


def _add_zero_transform_on_parity(problem, rows_key, bit_lists, forbidden_parity, label):
    count = 0
    rows = problem[rows_key]
    for bucket in problem["universe"]:
        if bucket_commutation_parity(problem["blocks"], bucket, bit_lists) == forbidden_parity:
            row = row_from_sparse(problem["nvars"], rows.get(bucket, {}))
            problem["eq_rows"].append(row)
            problem["eq_rhs"].append(0)
            count += 1
    problem["constraint_log"]["additional"].append(f"{label} ({count} zero equalities)")
    return count


def _add_A_orbit(problem, perms, label):
    seen = set(); count = 0
    for bucket in problem["universe"]:
        partner = bucket_orbit_map(problem["blocks"], bucket, perms)
        key = tuple(sorted((repr(bucket), repr(partner))))
        if key in seen: continue
        seen.add(key)
        c1 = problem["col_of"].get(bucket); c2 = problem["col_of"].get(partner)
        if c1 is None and c2 is None: continue
        row = [0] * problem["nvars"]
        if c1 is not None: row[c1] += 1
        if c2 is not None: row[c2] -= 1
        problem["eq_rows"].append(row); problem["eq_rhs"].append(0); count += 1
    problem["constraint_log"]["additional"].append(f"{label} ({count} equalities)")
    return count


def _add_transform_orbit(problem, rows_key, perms, label):
    seen = set(); count = 0; rows = problem[rows_key]
    for bucket in problem["universe"]:
        partner = bucket_orbit_map(problem["blocks"], bucket, perms)
        key = tuple(sorted((repr(bucket), repr(partner))))
        if key in seen: continue
        seen.add(key)
        if bucket == partner: continue
        r1 = row_from_sparse(problem["nvars"], rows.get(bucket, {}))
        r2 = row_from_sparse(problem["nvars"], rows.get(partner, {}))
        problem["eq_rows"].append(add_rows(r1, r2, -1)); problem["eq_rhs"].append(0); count += 1
    problem["constraint_log"]["additional"].append(f"{label} ({count} equalities)")
    return count


def _add_A_self_parity(problem, perms, label):
    count = 0
    for bucket, col in problem["col_of"].items():
        if bucket_orbit_map(problem["blocks"], bucket, perms) == bucket:
            row = [0] * problem["nvars"]; row[col] = 1
            problem["congruences"].append((row, 0, 2)); count += 1
    problem["constraint_log"]["additional"].append(f"{label} ({count} buckets)")
    return count


def _add_transform_self_parity(problem, rows_key, perms, label):
    count = 0; rows = problem[rows_key]; norm = problem["norm"]
    for bucket in problem["universe"]:
        if bucket_orbit_map(problem["blocks"], bucket, perms) == bucket:
            row = row_from_sparse(problem["nvars"], rows.get(bucket, {}))
            problem["congruences"].append((row, 0, 2*norm)); count += 1
    problem["constraint_log"]["additional"].append(f"{label} ({count} buckets)")
    return count


def add_transverse_membership_constraints(problem, global_case, qy_case, u, mode):
    """Add exact consequences of one exhaustive Y^U membership branch."""
    blocks = problem["blocks"]
    norm = problem["norm"]
    yb = _transverse_y_bucket(blocks)
    if bucket_capacity(blocks, yb) != 1:
        raise AssertionError("Y^U must be a singleton RXxRY bucket.")

    acol = problem["col_of"].get(yb)
    brow = row_from_sparse(problem["nvars"], problem["B_rows"].get(yb, {}))

    if mode == "outside_normalizer":
        # Y^U not in S^perp, hence also not in S.
        if acol is not None:
            row=[0]*problem["nvars"]; row[acol]=1
            problem["eq_rows"].append(row); problem["eq_rhs"].append(0)
        problem["eq_rows"].append(brow); problem["eq_rhs"].append(0)
        problem["constraint_log"]["additional"].append(
            "transverse membership branch: Y^U outside S^perp (A_Y=B_Y=0)"
        )
        return

    if mode not in ("normalizer_logical", "normalizer_stabilizer"):
        raise ValueError(f"Unknown transverse membership mode {mode}")

    # Y^U in S^perp: singleton B_Y=1 and every stabilizer commutes with Y^U.
    problem["eq_rows"].append(brow); problem["eq_rhs"].append(norm)
    perms, bits = _transverse_y_action(blocks)
    _add_zero_A_on_parity(problem, bits, 1, "Y^U in S^perp: stabilizer commutation support")

    # S^perp and its shadow affine coset are invariant under translation by
    # any Y^U in S^perp.  Add B/shadow orbit and fixed-bucket evenness.
    _add_transform_orbit(problem, "B_rows", perms, "Y^U normalizer B-orbit symmetry")
    _add_transform_self_parity(problem, "B_rows", perms, "Y^U normalizer B self-orbit parity")
    if USE_SHADOW_IDENTITY and problem["shadow_projected_compatible"]:
        _add_transform_orbit(problem, "shadow_rows", perms, "Y^U normalizer shadow-orbit symmetry")
        _add_transform_self_parity(problem, "shadow_rows", perms, "Y^U normalizer shadow self-orbit parity")

    if mode == "normalizer_logical":
        if acol is not None:
            row=[0]*problem["nvars"]; row[acol]=1
            problem["eq_rows"].append(row); problem["eq_rhs"].append(0)
        problem["constraint_log"]["additional"].append(
            "transverse membership branch: Y^U in S^perp\\S (A_Y=0,B_Y=1)"
        )
        return

    # Y^U in S.  This is possible only for even u because X^U is already in S.
    if u % 2:
        problem["eq_rows"].append([0]*problem["nvars"]); problem["eq_rhs"].append(1)
        problem["constraint_log"]["additional"].append(
            "Y^U in S impossible for odd u because it anticommutes with known X^U"
        )
        return

    if acol is None:
        problem["eq_rows"].append([0]*problem["nvars"]); problem["eq_rhs"].append(1)
    else:
        row=[0]*problem["nvars"]; row[acol]=1
        problem["eq_rows"].append(row); problem["eq_rhs"].append(1)

    # Stabilizer membership upgrades normalizer consequences to full A/B
    # commutation support and A orbit/self parity.  Shadow has the fixed
    # affine commutation bit <h,Y^U>=wt(Y^U)=u mod 2.
    _add_zero_transform_on_parity(problem, "B_rows", bits, 1,
                                  "Y^U in S: normalizer commutation support")
    if USE_SHADOW_IDENTITY and problem["shadow_projected_compatible"]:
        required = u & 1
        _add_zero_transform_on_parity(problem, "shadow_rows", bits, 1-required,
                                      "Y^U in S: shadow character support")
    _add_A_orbit(problem, perms, "Y^U stabilizer A-orbit symmetry")
    _add_A_self_parity(problem, perms, "Y^U stabilizer A self-orbit parity")
    problem["constraint_log"]["additional"].append(
        "transverse membership branch: Y^U in S (A_Y=B_Y=1)"
    )


def validate_strong_configuration():
    required_true = {
        "USE_SPLIT": USE_SPLIT,
        "USE_MACWILLIAMS": USE_MACWILLIAMS,
        "USE_SHADOW_IDENTITY": USE_SHADOW_IDENTITY,
        "USE_BLOCKWISE_TWISTS": USE_BLOCKWISE_TWISTS,
        "USE_KNOWN_STABILIZER": USE_KNOWN_STABILIZER,
        "USE_REFERENCE_EXISTENCE": USE_REFERENCE_EXISTENCE,
        "USE_REFINED_CELL_CAPACITY": USE_REFINED_CELL_CAPACITY,
        "USE_TRANSVERSE_QY_CASES": USE_TRANSVERSE_QY_CASES,
        "USE_TRANSVERSE_MEMBERSHIP_BRANCHING": USE_TRANSVERSE_MEMBERSHIP_BRANCHING,
    }
    bad=[name for name,value in required_true.items() if value is not True]
    if bad:
        raise ValueError("Strong RXxRY transverse configuration disabled: "+", ".join(bad))
    # These X-reference actions do not descend to R_Y and must stay off.
    if USE_COMMUTATION or USE_ORBIT or USE_SELF_ORBIT_PARITY:
        raise ValueError("Direct X^U commutation/orbit/self-parity must remain OFF in transverse R_Y.")
    if USE_TWISTED_CWE:
        raise ValueError("Full/global CWE twists must remain OFF in this solver.")
    if PROJECTION_T != [["I"], ["X"], ["Y","Z"]]:
        raise ValueError("RXxRY stabilizer solver requires R_X=I/X/(YZ) on T.")
    if PROJECTION_U != [["I"], ["Y"], ["X","Z"]]:
        raise ValueError("RXxRY stabilizer solver requires R_Y=I/Y/(XZ) on U.")
    if STABILIZER_ON_T != "I" or STABILIZER_ON_U != "X":
        raise ValueError("Canonical actual stabilizer reference must remain g=I^T X^U.")



def _row_items_real(row):
    if isinstance(row, dict):
        return ((int(j), float(v)) for j, v in row.items() if v)
    return ((j, float(v)) for j, v in enumerate(row) if v)


def _rows_to_csr_real(rows, nvars, *, negate=False):
    data=[]; indices=[]; indptr=[0]
    sign=-1.0 if negate else 1.0
    for row in rows:
        for j,v in _row_items_real(row):
            data.append(sign*v); indices.append(j)
        indptr.append(len(data))
    return csr_matrix((np.asarray(data,dtype=float), np.asarray(indices,dtype=np.int32),
                      np.asarray(indptr,dtype=np.int64)), shape=(len(rows),nvars))


def _solve_problem_real_lp(problem):
    """Continuous relaxation of one already-built RXxRY subbranch.

    Builder convention: each inequality is row*x >= rhs.  SciPy uses
    A_ub*x <= b_ub, hence both sides are negated.  All modular/integrality
    constraints are deliberately ignored.
    """
    nv=problem['nvars']
    eq_rows=problem['eq_rows']; eq_rhs=problem['eq_rhs']
    ineq=problem['inequalities']
    Aeq=_rows_to_csr_real(eq_rows,nv) if eq_rows else None
    beq=np.asarray(eq_rhs,dtype=float) if eq_rows else None
    Aub=_rows_to_csr_real([r for r,_ in ineq],nv,negate=True) if ineq else None
    bub=np.asarray([-rhs for _,rhs in ineq],dtype=float) if ineq else None
    ub=problem.get('upper_bounds',{})
    bounds=[(0.0, None if ub.get(j) is None else float(ub[j])) for j in range(nv)]
    t0=time.perf_counter()
    options={} if TIMEOUT_MS is None else {'time_limit': float(TIMEOUT_MS)/1000.0}
    res=linprog(np.zeros(nv), A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                bounds=bounds,method='highs',options=options)
    dt=time.perf_counter()-t0
    if res.status==0: st='sat'
    elif res.status==2: st='unsat'
    else: st='unknown'
    meta={
        'scipy_status':int(res.status), 'message':str(res.message),
        'runtime_solve':dt,
        'matrix_nnz_eq':0 if Aeq is None else int(Aeq.nnz),
        'matrix_nnz_ineq':0 if Aub is None else int(Aub.nnz),
    }
    # Drop large matrices before returning to the next transverse branch.
    del Aeq,Aub,beq,bub,res
    # legacy path: no forced full GC
    return st, meta


def solve_subbranch(fixed_A, case, u, qy_case, membership_mode, timeout_ms=None):
    """Solve one transverse branch as a pure real LP (no RREF/lattice/Z3)."""
    configure_reference_branch(fixed_A, case, u, qy_case, timeout_ms=None)
    branch_start=time.perf_counter()
    print(f"        qY case={qy_case}, Y^U={membership_mode}")

    problem, build_time = run_stage_with_heartbeat(
        "        [1/2] Building joint RXxRY constraints...", build_problem,
        heartbeat_text=f"u={u},qY={qy_case},{membership_mode}: building",
    )

    xref=known_stabilizer_bucket(problem["blocks"])
    xcap=bucket_capacity(problem["blocks"], xref)
    if xcap != 2**u:
        raise AssertionError(f"Expected coarse X^U bucket capacity 2^u={2**u}, got {xcap}.")
    if problem["generator_orbit_available"] or problem["generator_comm_available"]:
        raise AssertionError("Direct X^U orbit/commutation must not descend to transverse R_Y.")

    enabled={blockwise_tau_name(p) for p in problem["enabled_blockwise_pairs"]}
    expected={
        "(000,101)", "(011,000)", "(011,101)",
        "(100,010)", "(100,111)", "(111,010)",
    }
    if enabled != expected:
        raise AssertionError(f"Expected RXxRY extra twists {sorted(expected)}, got {sorted(enabled)}")
    if problem["blockwise_parity_cases"].get(((1,1,1),(0,1,0))) != int(qy_case):
        raise AssertionError("Transverse qY three-case was not installed correctly.")

    add_transverse_membership_constraints(problem, case, qy_case, u, membership_mode)

    print(
        f"        audit: Xref-cap={xcap}; direct-X orbit/comm=NO/NO; "
        f"eq={len(problem['eq_rows'])}, ineq={len(problem['inequalities'])}; "
        f"IGNORED modular={len(problem['congruences'])}"
    )

    (status, lpmeta), solve_time = run_stage_with_heartbeat(
        "        [2/2] Solving continuous HiGHS relaxation...",
        lambda: _solve_problem_real_lp(problem),
        heartbeat_text=f"u={u},qY={qy_case},{membership_mode}: HiGHS",
    )
    elapsed=time.perf_counter()-branch_start
    print(
        f"        => {status.upper()} [solve={lpmeta['runtime_solve']:.3f}s,"
        f" total={elapsed:.3f}s, nnz(eq)={lpmeta['matrix_nnz_eq']},"
        f" nnz(ineq)={lpmeta['matrix_nnz_ineq']}]"
    )
    info={
        "u":u,"t":n-u,"qy_case":qy_case,"membership":membership_mode,
        "raw_buckets":len(problem["universe"]),"active_A_vars":problem["nvars"],
        "eq_rows":len(problem["eq_rows"]),"inequalities":len(problem["inequalities"]),
        "ignored_congruences":len(problem["congruences"]),
        "runtime_build":build_time,"runtime_solve":lpmeta["runtime_solve"],
        "runtime_total":elapsed,"scipy_status":lpmeta["scipy_status"],
        "message":lpmeta["message"],"continuous_real_lp":True,
    }
    del problem
    # legacy path: no forced full GC
    return status, info

def solve_reference_u(fixed_A, case, u, timeout_ms=None):
    """A reference shell is UNSAT only if every exhaustive qY/membership branch is UNSAT."""
    print(f"      reference u={u}, t={n-u}, A_u={fixed_A.get(u,0)}")
    branches=[]; saw_unknown=False; saw_error=False; attempted=0

    for qy_case in (1,2,3):
        if not qy_case_compatible(case, qy_case, u):
            branches.append({"status":"skipped","u":u,"qy_case":qy_case,
                             "reason":"quadratic_case_incompatible_with_known_X_reference"})
            continue
        modes=transverse_membership_modes(case,qy_case,u)
        for mode in modes:
            attempted += 1
            try:
                status, info=solve_subbranch(fixed_A,case,u,qy_case,mode,timeout_ms=timeout_ms)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                status="error"; info={"u":u,"qy_case":qy_case,"membership":mode,"error":repr(exc)}
                print(f"        => ERROR: {exc!r}")
            branches.append({"status":status, **info})
            # One SAT subbranch means this mandatory reference shell cannot exclude.
            if status=="sat":
                return "sat", {"u":u,"branches":branches,"witness_branch":info}
            if status=="unknown": saw_unknown=True
            elif status=="error": saw_error=True

    if attempted==0:
        return "error", {"u":u,"branches":branches,"reason":"no_transverse_branches"}
    if saw_error:
        overall="error"
    elif saw_unknown:
        overall="unknown"
    else:
        overall="unsat"
    return overall, {"u":u,"branches":branches}



# =====================================================================
# EXACT REAL-LP UNSAT VERIFICATION
# =====================================================================

def _z3_exact_num(v):
    """Map exact integer/rational coefficients to Z3 rationals."""
    if isinstance(v, (int, np.integer)):
        return z3.IntVal(int(v))
    if isinstance(v, Fraction):
        return z3.RealVal(f"{v.numerator}/{v.denominator}")
    if isinstance(v, sp.Rational):
        return z3.RealVal(f"{int(v.p)}/{int(v.q)}")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"non-finite exact coefficient {v}")
        if v.is_integer():
            return z3.IntVal(int(v))
        # Optimized binary builder should generate integral rows.  Keep this
        # deterministic, but callers should treat reaching here as an audit signal.
        return z3.RealVal(str(v))
    return z3.RealVal(str(v))


def _exact_sparse_expr(xs, row):
    if isinstance(row, dict):
        items = row.items()
    else:
        items = enumerate(row)
    terms = [_z3_exact_num(v) * xs[int(j)] for j, v in items if v != 0]
    return z3.Sum(terms) if terms else z3.RealVal(0)


def _rebuild_exact_inequalities_for_verification(prepared, selected_ids):
    """Rebuild the u-base only on numerical-UNSAT branches.

    This avoids keeping a second Python copy of the large inequality family in
    memory during ordinary SAT screening.  Ordering is deterministic and is
    checked against the already-built CSR evaluator.
    """
    if not selected_ids:
        return []
    rebuilt = build_problem()
    ineq = rebuilt.get('inequalities', [])
    if len(ineq) != prepared['n_ineq']:
        raise AssertionError(
            f"Exact-verifier rebuild changed inequality count: {len(ineq)} != {prepared['n_ineq']}"
        )
    rows = [ineq[int(i)] for i in selected_ids]
    del rebuilt
    return rows



def _fraction_from_exact_value(v):
    """Convert one builder coefficient to an exact Fraction.

    The optimized binary builder is intended to be integral/rational.  Floats
    are converted from their decimal spelling; reaching a genuinely
    non-rational coefficient here is an audit failure, not a reason to accept
    a numerical UNSAT.
    """
    if isinstance(v, Fraction):
        return v
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v), 1)
    if isinstance(v, sp.Integer):
        return Fraction(int(v), 1)
    if isinstance(v, sp.Rational):
        return Fraction(int(v.p), int(v.q))
    if isinstance(v, (float, np.floating)):
        fv=float(v)
        if not math.isfinite(fv):
            raise ValueError(f"non-finite Farkas coefficient {v}")
        if fv.is_integer():
            return Fraction(int(fv), 1)
        return Fraction(str(fv))
    return Fraction(v)


def _exact_row_items(row):
    if isinstance(row, dict):
        return ((int(j), _fraction_from_exact_value(v)) for j,v in row.items() if v != 0)
    return ((int(j), _fraction_from_exact_value(v)) for j,v in enumerate(row) if v != 0)


def _farkas_row_scale(row, rhs):
    vals=[abs(float(v)) for _,v in (_row_items_real(row))]
    return max([1.0, abs(float(rhs))] + vals)


def _build_farkas_primal_subsystem(base, prepared, qrows, qrhs, mrows, mrhs, selected_ids):
    """Return the exact finite primal subsystem in E x=f, C x<=d form.

    Equality multipliers in the Farkas certificate are free.  Inequality
    multipliers are nonnegative.  Original builder inequalities are >= and are
    negated here.  Variable lower/upper bounds are made explicit because they
    are part of the numerical LP that HiGHS declared infeasible.
    """
    eq_rows=list(base['eq_rows']) + list(qrows) + list(mrows)
    eq_rhs=list(base['eq_rhs']) + list(qrhs) + list(mrhs)

    le_rows=[]; le_rhs=[]; le_labels=[]
    exact_ineq=_rebuild_exact_inequalities_for_verification(prepared, selected_ids)
    for idx,(row,rhs) in zip(selected_ids, exact_ineq):
        # builder: row*x >= rhs  ->  (-row)*x <= -rhs
        if isinstance(row, dict):
            rr={int(j):-v for j,v in row.items() if v != 0}
        else:
            rr=[-v for v in row]
        le_rows.append(rr); le_rhs.append(-rhs); le_labels.append(('ineq',int(idx)))

    nv=base['nvars']
    # Same bounds as the numerical solve: x_j >= 0 and optional x_j <= ub_j.
    for j in range(nv):
        le_rows.append({j:-1}); le_rhs.append(0); le_labels.append(('lb',j))
    ub=base.get('upper_bounds',{})
    for j in range(nv):
        if ub.get(j) is not None:
            le_rows.append({j:1}); le_rhs.append(ub[j]); le_labels.append(('ub',j))

    return eq_rows,eq_rhs,le_rows,le_rhs,le_labels


def _numeric_farkas_ray(eq_rows, eq_rhs, le_rows, le_rhs, nv):
    """Use floating HiGHS only to *find* a candidate Farkas multiplier.

    We row-scale the primal constraints for conditioning.  For scaled rows
      E_i/s_i, C_k/t_k,
    certificate variables z',y' map back to the raw exact rows as
      z_i=z'_i/s_i, y_k=y'_k/t_k.
    """
    me=len(eq_rows); mi=len(le_rows)
    eq_scales=[]; efloat=[]; ffloat=[]
    for row,rhs in zip(eq_rows,eq_rhs):
        sc=_farkas_row_scale(row,rhs); eq_scales.append(sc)
        efloat.append({j:float(v)/sc for j,v in _row_items_real(row)})
        ffloat.append(float(rhs)/sc)
    ineq_scales=[]; cfloat=[]; dfloat=[]
    for row,rhs in zip(le_rows,le_rhs):
        sc=_farkas_row_scale(row,rhs); ineq_scales.append(sc)
        cfloat.append({j:float(v)/sc for j,v in _row_items_real(row)})
        dfloat.append(float(rhs)/sc)

    E=_rows_to_csr_real(efloat,nv) if me else csr_matrix((0,nv),dtype=float)
    C=_rows_to_csr_real(cfloat,nv) if mi else csr_matrix((0,nv),dtype=float)
    # Farkas equations for E x=f, C x<=d:
    #   E^T z + C^T y = 0,  f^T z + d^T y = -1,  y>=0.
    top=hstack([E.T,C.T],format='csr')
    last=csr_matrix(np.asarray(ffloat+dfloat,dtype=float).reshape(1,me+mi))
    Aeq=vstack([top,last],format='csr')
    beq=np.zeros(nv+1,dtype=float); beq[-1]=-1.0
    cbounds=[(None,None)]*me + [(0.0,None)]*mi
    t0=time.perf_counter()
    rr=linprog(np.zeros(me+mi,dtype=float), A_eq=Aeq,b_eq=beq,
               bounds=cbounds,method=FARKAS_NUMERIC_METHOD)
    dt=time.perf_counter()-t0
    meta={'farkas_numeric_runtime_s':dt,'farkas_numeric_status':int(rr.status),
          'farkas_numeric_message':str(rr.message),'farkas_cert_vars':me+mi,
          'farkas_cert_eqs':nv+1}
    if rr.status!=0 or rr.x is None:
        return None,None,None,None,meta
    vec=np.asarray(rr.x,dtype=float)
    z_scaled=vec[:me]; y_scaled=vec[me:]
    z_raw=z_scaled/np.asarray(eq_scales,dtype=float) if me else np.zeros(0)
    y_raw=y_scaled/np.asarray(ineq_scales,dtype=float) if mi else np.zeros(0)
    return z_scaled,y_scaled,z_raw,y_raw,meta


def _verify_exact_farkas(eq_rows,eq_rhs,le_rows,le_rhs,z,y,nv):
    """Independently verify a rational Farkas certificate exactly."""
    if len(z)!=len(eq_rows) or len(y)!=len(le_rows):
        return False,{'farkas_exact_reason':'certificate_length_mismatch'}
    if any(v < 0 for v in y):
        return False,{'farkas_exact_reason':'negative_inequality_multiplier'}
    residual=[Fraction(0,1) for _ in range(nv)]
    contradiction=Fraction(0,1)
    for lam,row,rhs in zip(z,eq_rows,eq_rhs):
        if not lam: continue
        for j,v in _exact_row_items(row): residual[j] += lam*v
        contradiction += lam*_fraction_from_exact_value(rhs)
    for lam,row,rhs in zip(y,le_rows,le_rhs):
        if not lam: continue
        for j,v in _exact_row_items(row): residual[j] += lam*v
        contradiction += lam*_fraction_from_exact_value(rhs)
    nz=[j for j,v in enumerate(residual) if v]
    if nz:
        return False,{'farkas_exact_reason':'nonzero_ATy','farkas_exact_residual_nnz':len(nz)}
    if not contradiction < 0:
        return False,{'farkas_exact_reason':'nonnegative_bTy','farkas_exact_bTy':str(contradiction)}
    denmax=1
    for v in list(z)+list(y): denmax=max(denmax,int(v.denominator))
    return True,{
        'farkas_exact_bTy':str(contradiction),
        'farkas_exact_max_denominator':denmax,
        'farkas_exact_bTy_num_bits':abs(int(contradiction.numerator)).bit_length(),
        'farkas_exact_bTy_den_bits':int(contradiction.denominator).bit_length(),
    }


def _rationalize_and_verify_farkas(eq_rows,eq_rhs,le_rows,le_rhs,
                                    z_scaled,y_scaled,z_raw,y_raw,nv):
    """Recover a sparse rational ray from the floating basic certificate.

    Support is read from the *scaled* certificate, not raw multipliers: a row
    with a huge coefficient scale can legitimately have a tiny raw multiplier
    (e.g. 1/2^27) while being a perfectly ordinary basic row numerically.
    """
    if z_scaled is None:
        return None,None,{'farkas_exact_reason':'no_numeric_certificate'}
    max_scaled=max(1.0,
                   float(np.max(np.abs(z_scaled))) if len(z_scaled) else 0.0,
                   float(np.max(np.abs(y_scaled))) if len(y_scaled) else 0.0)
    last_meta={}
    for reltol in FARKAS_SUPPORT_REL_TOLS:
        zmask=np.abs(z_scaled) > reltol*max_scaled
        ymask=y_scaled > reltol*max_scaled
        for maxden in FARKAS_DIRECT_MAX_DENOMINATORS:
            z=[]
            for keep,v in zip(zmask,z_raw):
                z.append(Fraction(float(v)).limit_denominator(int(maxden)) if keep else Fraction(0,1))
            y=[]
            for keep,v in zip(ymask,y_raw):
                q=Fraction(float(v)).limit_denominator(int(maxden)) if keep else Fraction(0,1)
                # Numeric y is constrained nonnegative; a tiny negative is a failed reconstruction.
                y.append(q)
            ok,vm=_verify_exact_farkas(eq_rows,eq_rhs,le_rows,le_rhs,z,y,nv)
            vm.update({'farkas_support_rel_tol':reltol,
                       'farkas_reconstruction_maxden':int(maxden),
                       'farkas_eq_support':int(sum(bool(v) for v in z)),
                       'farkas_ineq_support':int(sum(bool(v) for v in y))})
            if ok:
                return z,y,vm
            last_meta=vm
    return None,None,last_meta



def _exact_support_reconstruct_farkas(eq_rows,eq_rhs,le_rows,le_rhs,
                                       z_scaled,y_scaled,z_raw,y_raw,nv):
    """Fallback: solve the floating ray's support exactly, then verify it.

    Direct entrywise rational reconstruction is fastest but can fail on a
    degenerate basis.  Here we trust only the floating *support*.  SymPy solves
    the linear Farkas equations on that support over Q; any remaining free
    parameters are seeded with rational approximations of the floating ray.
    No result is accepted until _verify_exact_farkas succeeds independently.
    """
    if not FARKAS_EXACT_SUPPORT_RECONSTRUCTION:
        return None,None,{'farkas_support_exact_reason':'disabled'}
    max_scaled=max(1.0,
                   float(np.max(np.abs(z_scaled))) if len(z_scaled) else 0.0,
                   float(np.max(np.abs(y_scaled))) if len(y_scaled) else 0.0)
    last={}
    for reltol in FARKAS_SUPPORT_REL_TOLS:
        ze=[i for i,v in enumerate(z_scaled) if abs(float(v)) > reltol*max_scaled]
        yi=[i for i,v in enumerate(y_scaled) if float(v) > reltol*max_scaled]
        support=[('e',i) for i in ze] + [('i',i) for i in yi]
        ss=len(support)
        if not ss:
            continue
        if ss > FARKAS_EXACT_SUPPORT_MAX_VARS:
            last={'farkas_support_exact_reason':'support_too_large',
                  'farkas_support_exact_vars':ss,'farkas_support_rel_tol':reltol}
            continue
        # Sparse exact matrix M w = (0,...,0,-1).
        entries={}
        raw_approx=[]
        for c,(kind,i) in enumerate(support):
            if kind=='e':
                row=eq_rows[i]; rhs=eq_rhs[i]; raw_approx.append(float(z_raw[i]))
            else:
                row=le_rows[i]; rhs=le_rhs[i]; raw_approx.append(float(y_raw[i]))
            for j,v in _exact_row_items(row):
                if v:
                    entries[(j,c)] = sp.Rational(v.numerator,v.denominator)
            fr=_fraction_from_exact_value(rhs)
            if fr:
                entries[(nv,c)] = sp.Rational(fr.numerator,fr.denominator)
        M=sp.MutableSparseMatrix(nv+1,ss,entries)
        bvec=sp.zeros(nv+1,1); bvec[nv,0]=-1
        vars_=sp.symbols(f'_fy0:{ss}')
        t0=time.perf_counter()
        try:
            solset=sp.linsolve((M,bvec),vars_)
        except Exception as exc:
            last={'farkas_support_exact_reason':f'linsolve_exception:{type(exc).__name__}:{exc}',
                  'farkas_support_exact_vars':ss,'farkas_support_rel_tol':reltol,
                  'farkas_support_exact_runtime_s':time.perf_counter()-t0}
            continue
        solve_dt=time.perf_counter()-t0
        if solset is sp.EmptySet or not solset:
            last={'farkas_support_exact_reason':'support_inconsistent',
                  'farkas_support_exact_vars':ss,'farkas_support_rel_tol':reltol,
                  'farkas_support_exact_runtime_s':solve_dt}
            continue
        tup=next(iter(solset))
        free=set()
        for expr in tup: free.update(expr.free_symbols)
        for maxden in FARKAS_DIRECT_MAX_DENOMINATORS:
            subs={}
            for sym in free:
                # linsolve with supplied variables normally leaves a subset of
                # those variables free.  Seed it near the floating certificate.
                try:
                    ii=vars_.index(sym)
                    q=Fraction(float(raw_approx[ii])).limit_denominator(int(maxden))
                except Exception:
                    q=Fraction(0,1)
                subs[sym]=sp.Rational(q.numerator,q.denominator)
            vals=[]; ok_eval=True
            for expr in tup:
                vv=sp.cancel(expr.subs(subs))
                if vv.free_symbols or not vv.is_Rational:
                    ok_eval=False; break
                vals.append(Fraction(int(vv.p),int(vv.q)))
            if not ok_eval:
                continue
            z=[Fraction(0,1) for _ in eq_rows]
            y=[Fraction(0,1) for _ in le_rows]
            for val,(kind,i) in zip(vals,support):
                if kind=='e': z[i]=val
                else: y[i]=val
            ok,vm=_verify_exact_farkas(eq_rows,eq_rhs,le_rows,le_rhs,z,y,nv)
            vm.update({'farkas_support_exact_vars':ss,
                       'farkas_support_exact_runtime_s':solve_dt,
                       'farkas_support_rel_tol':reltol,
                       'farkas_support_exact_seed_maxden':int(maxden),
                       'farkas_eq_support':sum(bool(v) for v in z),
                       'farkas_ineq_support':sum(bool(v) for v in y)})
            if ok:
                vm['farkas_reconstruction']='exact-support-linsolve'
                return z,y,vm
            last=vm
    return None,None,last


def _farkas_verify_binary_branch(base, prepared, qrows, qrhs, mrows, mrhs, selected_ids):
    """Attempt exact real-LP UNSAT certification without SMT search."""
    t0=time.perf_counter()
    try:
        eq_rows,eq_rhs,le_rows,le_rhs,le_labels=_build_farkas_primal_subsystem(
            base,prepared,qrows,qrhs,mrows,mrhs,selected_ids)
        zsc,ysc,zraw,yraw,nmeta=_numeric_farkas_ray(
            eq_rows,eq_rhs,le_rows,le_rhs,base['nvars'])
        meta=dict(nmeta)
        if zsc is None:
            meta.update({'exact_backend':'farkas-rational','exact_runtime_s':time.perf_counter()-t0,
                         'farkas_exact_verified':False})
            return 'unknown',meta
        z,y,vmeta=_rationalize_and_verify_farkas(
            eq_rows,eq_rhs,le_rows,le_rhs,zsc,ysc,zraw,yraw,base['nvars'])
        if z is not None:
            vmeta['farkas_reconstruction']='direct-rationalization'
        elif FARKAS_EXACT_SUPPORT_RECONSTRUCTION:
            z,y,vmeta2=_exact_support_reconstruct_farkas(
                eq_rows,eq_rhs,le_rows,le_rhs,zsc,ysc,zraw,yraw,base['nvars'])
            if vmeta2: vmeta.update(vmeta2)
        meta.update(vmeta)
        meta.update({'exact_backend':'farkas-rational','exact_runtime_s':time.perf_counter()-t0,
                     'farkas_exact_verified':z is not None,
                     'exact_vars':base['nvars'],'exact_eq_rows':len(eq_rows),
                     'exact_ineq_rows':len(le_rows)})
        if z is not None:
            # Give enough structural information to audit/recognize repeated certificates
            # without dumping a large vector into the candidate database.
            meta['farkas_selected_original_ineqs']=len(selected_ids)
            return 'unsat',meta
        return 'unknown',meta
    except Exception as exc:
        return 'unknown',{
            'exact_backend':'farkas-rational','exact_runtime_s':time.perf_counter()-t0,
            'farkas_exact_verified':False,'farkas_exact_reason':f'exception:{type(exc).__name__}:{exc}'
        }


def _exact_verify_binary_branch(base, prepared, qrows, qrhs, mrows, mrhs, selected_ids):
    """Exactly certify the finite LP subsystem HiGHS called infeasible.

    Fast path: reconstruct and independently verify a rational Farkas
    certificate.  Only if that fails do we use the legacy Z3 QF_LRA solver.
    """
    farkas_meta={}
    if USE_EXACT_FARKAS_CERTIFICATE:
        fst,farkas_meta=_farkas_verify_binary_branch(
            base,prepared,qrows,qrhs,mrows,mrhs,selected_ids)
        if fst=='unsat':
            return 'unsat',farkas_meta
        if not FARKAS_FALLBACK_TO_Z3:
            return 'unknown',farkas_meta
    if z3 is None:
        out=dict(farkas_meta)
        out['exact_reason']='z3-solver not installed; Farkas reconstruction did not verify'
        return 'unavailable', out
    nv = base['nvars']
    xs = [z3.Real(f'x_{j}') for j in range(nv)]
    solver = z3.SolverFor('QF_LRA')
    if EXACT_VERIFY_TIMEOUT_MS and EXACT_VERIFY_TIMEOUT_MS > 0:
        solver.set(timeout=int(EXACT_VERIFY_TIMEOUT_MS))

    # Exact branch equalities.  Equality ordering is irrelevant mathematically.
    for row, rhs in zip(base['eq_rows'], base['eq_rhs']):
        solver.add(_exact_sparse_expr(xs, row) == _z3_exact_num(rhs))
    for row, rhs in zip(qrows, qrhs):
        solver.add(_exact_sparse_expr(xs, row) == _z3_exact_num(rhs))
    for row, rhs in zip(mrows, mrhs):
        solver.add(_exact_sparse_expr(xs, row) == _z3_exact_num(rhs))

    # A variables are nonnegative, with the same exact cell upper bounds.
    ub = base.get('upper_bounds', {})
    for j in range(nv):
        solver.add(xs[j] >= 0)
        if ub.get(j) is not None:
            solver.add(xs[j] <= _z3_exact_num(ub[j]))

    # Only the inequalities actually present in the numerical subsystem are needed.
    exact_ineq = _rebuild_exact_inequalities_for_verification(prepared, selected_ids)
    for row, rhs in exact_ineq:
        solver.add(_exact_sparse_expr(xs, row) >= _z3_exact_num(rhs))

    t0 = time.perf_counter()
    rr = solver.check()
    dt = time.perf_counter() - t0
    meta = dict(farkas_meta)
    meta.update({
        'exact_backend': 'z3-qf_lra-fallback',
        'exact_runtime_s': dt,
        'exact_vars': nv,
        'exact_eq_rows': len(base['eq_rows']) + len(qrows) + len(mrows),
        'exact_ineq_rows': len(exact_ineq),
    })
    if rr == z3.unsat:
        return 'unsat', meta
    if rr == z3.sat:
        return 'sat', meta
    meta['exact_reason'] = solver.reason_unknown()
    return 'unknown', meta


def _exact_verifier_self_test():
    if z3 is None:
        return
    xs = [z3.Real('self_x')]
    s = z3.SolverFor('QF_LRA')
    s.add(xs[0] == 0, xs[0] >= 1)
    assert s.check() == z3.unsat


# =====================================================================
# OPTIMIZED REAL-LP JOINT DRIVER (2026-09-22)
#
# Key architecture:
#   * build the expensive RXxRY u-level base problem ONCE per u;
#   * qY case and Y^U-membership branches contribute equality DELTAS only;
#   * base A_eq/A_ub CSR matrices are assembled ONCE per u;
#   * no modular/integrality constraints are generated/used in this path;
#   * no per-subbranch gc.collect(); one collection after the whole u shell.
# =====================================================================


def _configure_u_base(fixed_A, case, u, timeout_ms=None):
    global FIXED_A, GLOBAL_PARITY_CASE, T_SIZE, U_SIZE, USE_CLEANING
    global BLOCKWISE_PARITY_CASES, TIMEOUT_MS, B_INTEGRAL, A_INTEGRAL
    FIXED_A = {int(w): int(v) for w, v in fixed_A.items()}
    GLOBAL_PARITY_CASE = int(case)
    U_SIZE = int(u)
    T_SIZE = n - U_SIZE
    USE_CLEANING = (T_SIZE < d or U_SIZE < d)
    # qY parity/type is branch-specific and MUST NOT be installed in the base.
    BLOCKWISE_PARITY_CASES = {}
    A_INTEGRAL = False
    B_INTEGRAL = False
    if timeout_ms is not None:
        TIMEOUT_MS = timeout_ms


def _qy_case_delta_eqs(base, qy_case):
    """Return the two real-linear equations for the transverse (111,010) qY case."""
    pair=((1,1,1),(0,1,0))
    case=int(qy_case)
    nv=base['nvars']; blocks=base['blocks']; col_of=base['col_of']
    A_even=[0]*nv; A_odd=[0]*nv
    for bucket,col in col_of.items():
        qb=bucket_blockwise_twist_parity(blocks,bucket,pair)
        if qb is None:
            raise AssertionError('Lost qY parity visibility in optimized RXxRY base.')
        (A_even if qb==0 else A_odd)[col]+=1
    B_even=sum_sparse_rows(
        nv,
        [base['B_rows'].get(bucket,{}) for bucket in base['universe']
         if bucket_blockwise_twist_parity(blocks,bucket,pair)==0],
    )
    norm=base['norm']
    rows=[]; rhs=[]
    if case==1:
        rows.extend((A_even,B_even))
        rhs.extend((2**(n-k-1), norm*(2**(n+k-1))))
    elif case==2:
        target=2**(n-k-1)*(4**k+2**k)
        rows.extend((A_odd,B_even)); rhs.extend((0,norm*target))
    elif case==3:
        target=2**(n-k-1)*(4**k-2**k)
        rows.extend((A_odd,B_even)); rhs.extend((0,norm*target))
    else:
        raise ValueError(f'Invalid qY case {case}')
    return rows,rhs


def _delta_A_zero_on_parity(base, bits, forbidden=1):
    rows=[]; rhs=[]; nv=base['nvars']
    for bucket,col in base['col_of'].items():
        if bucket_commutation_parity(base['blocks'],bucket,bits)==forbidden:
            row=[0]*nv; row[col]=1; rows.append(row); rhs.append(0)
    return rows,rhs


def _delta_transform_zero_on_parity(base, rows_key, bits, forbidden):
    rows=[]; rhs=[]; nv=base['nvars']; source=base[rows_key]
    for bucket in base['universe']:
        if bucket_commutation_parity(base['blocks'],bucket,bits)==forbidden:
            rows.append(row_from_sparse(nv,source.get(bucket,{}))); rhs.append(0)
    return rows,rhs


def _delta_A_orbit(base, perms):
    rows=[]; rhs=[]; nv=base['nvars']; seen=set()
    for bucket in base['universe']:
        partner=bucket_orbit_map(base['blocks'],bucket,perms)
        key=tuple(sorted((repr(bucket),repr(partner))))
        if key in seen: continue
        seen.add(key)
        c1=base['col_of'].get(bucket); c2=base['col_of'].get(partner)
        if c1 is None and c2 is None: continue
        row=[0]*nv
        if c1 is not None: row[c1]+=1
        if c2 is not None: row[c2]-=1
        rows.append(row); rhs.append(0)
    return rows,rhs


def _delta_transform_orbit(base, rows_key, perms):
    rows=[]; rhs=[]; nv=base['nvars']; source=base[rows_key]; seen=set()
    for bucket in base['universe']:
        partner=bucket_orbit_map(base['blocks'],bucket,perms)
        key=tuple(sorted((repr(bucket),repr(partner))))
        if key in seen: continue
        seen.add(key)
        if bucket==partner: continue
        r1=row_from_sparse(nv,source.get(bucket,{}))
        r2=row_from_sparse(nv,source.get(partner,{}))
        rows.append(add_rows(r1,r2,-1)); rhs.append(0)
    return rows,rhs


def _membership_delta_eqs(base, u, mode):
    """Pure real-linear membership consequences; modular self-orbit parity is omitted."""
    rows=[]; rhs=[]; nv=base['nvars']; blocks=base['blocks']; norm=base['norm']
    yb=_transverse_y_bucket(blocks)
    if bucket_capacity(blocks,yb)!=1:
        raise AssertionError('Y^U must be a singleton RXxRY bucket.')
    acol=base['col_of'].get(yb)
    brow=row_from_sparse(nv,base['B_rows'].get(yb,{}))
    def append_pair(rr,bb): rows.extend(rr); rhs.extend(bb)
    if mode=='outside_normalizer':
        if acol is not None:
            r=[0]*nv; r[acol]=1; rows.append(r); rhs.append(0)
        rows.append(brow); rhs.append(0)
        return rows,rhs
    if mode not in ('normalizer_logical','normalizer_stabilizer'):
        raise ValueError(f'Unknown transverse membership mode {mode}')
    rows.append(brow); rhs.append(norm)
    perms,bits=_transverse_y_action(blocks)
    append_pair(*_delta_A_zero_on_parity(base,bits,1))
    append_pair(*_delta_transform_orbit(base,'B_rows',perms))
    if USE_SHADOW_IDENTITY and base['shadow_projected_compatible']:
        append_pair(*_delta_transform_orbit(base,'shadow_rows',perms))
    if mode=='normalizer_logical':
        if acol is not None:
            r=[0]*nv; r[acol]=1; rows.append(r); rhs.append(0)
        return rows,rhs
    # Y^U in S.
    if u%2:
        rows.append([0]*nv); rhs.append(1)
        return rows,rhs
    if acol is None:
        rows.append([0]*nv); rhs.append(1)
    else:
        r=[0]*nv; r[acol]=1; rows.append(r); rhs.append(1)
    append_pair(*_delta_transform_zero_on_parity(base,'B_rows',bits,1))
    if USE_SHADOW_IDENTITY and base['shadow_projected_compatible']:
        required=u&1
        append_pair(*_delta_transform_zero_on_parity(base,'shadow_rows',bits,1-required))
    append_pair(*_delta_A_orbit(base,perms))
    return rows,rhs


def _prepare_u_base_lp(base):
    """Assemble branch-invariant raw CSR once per u.

    Keep the original row scaling/order so full-mode HiGHS reproduces the old
    solver numerically.  The same full >= matrix is used for fast vectorized
    lazy separation; it is passed to HiGHS only in full/fallback mode.
    """
    t0=time.perf_counter(); nv=base['nvars']
    Aeq=_rows_to_csr_real(base['eq_rows'],nv) if base['eq_rows'] else None
    beq=np.asarray(base['eq_rhs'],dtype=float) if base['eq_rows'] else None
    ineq=base['inequalities']
    Age=_rows_to_csr_real([r for r,_ in ineq],nv) if ineq else None
    rhs_ge=np.asarray([rhs for _,rhs in ineq],dtype=float) if ineq else None
    ub=base.get('upper_bounds',{})
    bounds=[(0.0,None if ub.get(j) is None else float(ub[j])) for j in range(nv)]
    # qY rows originally occur immediately before cleaning equalities.
    correctable=sum(1 for block in base['blocks'] if block.size<d) if USE_CLEANING else 0
    qy_insert=len(base['eq_rows'])-2*correctable
    base['inequalities']=[]
    return {
        'Aeq':Aeq,'beq':beq,'Age':Age,'rhs_ge':rhs_ge,'bounds':bounds,
        'qy_insert':qy_insert,
        'runtime_csr':time.perf_counter()-t0,
        'nnz_eq':0 if Aeq is None else int(Aeq.nnz),
        'nnz_ineq_eval':0 if Age is None else int(Age.nnz),
        'n_ineq':0 if Age is None else int(Age.shape[0]),
        'cut_indices':[], 'cut_set':set(),
    }

def _classify_real_lp_result(res):
    msg=str(res.message)
    if res.status==0:
        return 'sat'
    if res.status==2 and 'infeasible' in msg.lower():
        return 'unsat'
    return 'unknown'


def _sat_residuals(res,Aeq,beq,Aub,bub):
    if res.status!=0 or res.x is None:
        return None,None
    meq=0.0
    if Aeq is not None:
        meq=float(np.max(np.abs(Aeq@res.x-beq))) if Aeq.shape[0] else 0.0
    mub=0.0
    if Aub is not None:
        v=np.asarray(Aub@res.x-bub)
        mub=float(max(0.0,np.max(v))) if v.size else 0.0
    return meq,mub


def _solve_prepared_branch(base,prepared,qy_case,mode,u,*,tol=1e-8):
    nv=base['nvars']; t0=time.perf_counter()
    qrows,qrhs=_qy_case_delta_eqs(base,qy_case)
    mrows,mrhs=_membership_delta_eqs(base,u,mode)
    Aq=_rows_to_csr_real(qrows,nv) if qrows else None
    bq=np.asarray(qrhs,float) if qrows else None
    Am=_rows_to_csr_real(mrows,nv) if mrows else None
    bm=np.asarray(mrhs,float) if mrows else None
    # Preserve old equality ordering: base-prefix, qY, base-cleaning-suffix, membership.
    parts=[]; bparts=[]; ins=prepared['qy_insert']; Ab=prepared['Aeq']; bb=prepared['beq']
    if Ab is not None:
        if ins>0: parts.append(Ab[:ins,:]); bparts.append(bb[:ins])
        if Aq is not None: parts.append(Aq); bparts.append(bq)
        if ins<Ab.shape[0]: parts.append(Ab[ins:,:]); bparts.append(bb[ins:])
    elif Aq is not None:
        parts.append(Aq); bparts.append(bq)
    if Am is not None: parts.append(Am); bparts.append(bm)
    Aeq=vstack(parts,format='csr') if parts else None
    beq=np.concatenate(bparts) if bparts else None
    delta_time=time.perf_counter()-t0
    options={} if TIMEOUT_MS is None else {'time_limit':float(TIMEOUT_MS)/1000.0}
    total_solve=0.0; rounds=0; used_full=False

    def run_lp(Aub,bub):
        nonlocal total_solve
        t1=time.perf_counter()
        rr=linprog(np.zeros(nv),A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                   bounds=prepared['bounds'],method=HIGHS_METHOD,options=options)
        dt=time.perf_counter()-t1; total_solve+=dt
        return rr,dt

    # Explicit full mode.
    if INEQ_MODE=='full':
        Aub=-prepared['Age'] if prepared['Age'] is not None else None
        bub=-prepared['rhs_ge'] if prepared['rhs_ge'] is not None else None
        res,dt=run_lp(Aub,bub); rounds=1; used_full=True
        status=_classify_real_lp_result(res)
        maxeq,maxub=_sat_residuals(res,Aeq,beq,Aub,bub) if status=='sat' else (None,None)
        meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
              'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
              'message':str(res.message),'max_eq_residual':maxeq,'max_ub_violation':maxub,
              'cuts':prepared['n_ineq'],'used_full':True,
              'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0])}
        if status == 'unsat':
            print('            HiGHS: NUMERICAL-UNSAT; verifying exact QF_LRA ...', flush=True)
            exst, exmeta = _exact_verify_binary_branch(
                base, prepared, qrows, qrhs, mrows, mrhs,
                list(range(prepared['n_ineq']))
            )
            meta.update(exmeta); meta['numerical_unsat'] = True
            if exst == 'unsat':
                meta['exact_verified'] = True
                status = 'exact_unsat'
                print(f"            exact {exmeta.get('exact_backend','verifier')}: UNSAT [{exmeta.get('exact_runtime_s',0):.3f}s]", flush=True)
            else:
                meta['exact_verified'] = False
                meta['numerical_false_or_unverified_unsat'] = True
                status = 'unknown'
                print(f"            exact {exmeta.get('exact_backend','verifier')}: {exst.upper()}; branch NOT excluded", flush=True)
        del Aq,Am,Aeq,beq,Aub,bub,res
        return status,meta

    while True:
        rounds+=1
        ids=prepared['cut_indices']
        if ids:
            Aub=-prepared['Age'][ids,:]; bub=-prepared['rhs_ge'][ids]
        else:
            Aub=None; bub=None
        res,dt=run_lp(Aub,bub)
        status=_classify_real_lp_result(res)
        if status=='unsat':
            meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
                  'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
                  'message':str(res.message),'cuts':len(ids),'used_full':used_full,
                  'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0]),
                  'numerical_unsat':True}
            print(f"            HiGHS: NUMERICAL-UNSAT with {len(ids)} active cuts; exact verification ...", flush=True)
            exst, exmeta = _exact_verify_binary_branch(
                base, prepared, qrows, qrhs, mrows, mrhs, list(ids)
            )
            meta.update(exmeta)
            if exst == 'unsat':
                meta['exact_verified'] = True
                final_status = 'exact_unsat'
                print(f"            exact {exmeta.get('exact_backend','verifier')}: UNSAT [{exmeta.get('exact_runtime_s',0):.3f}s]", flush=True)
            else:
                meta['exact_verified'] = False
                meta['numerical_false_or_unverified_unsat'] = True
                final_status = 'unknown'
                print(f"            exact {exmeta.get('exact_backend','verifier')}: {exst.upper()}; branch NOT excluded", flush=True)
            del Aq,Am,Aeq,beq,Aub,bub,res
            return final_status,meta
        if status!='sat' or res.x is None:
            meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
                  'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
                  'message':str(res.message),'cuts':len(ids),'used_full':used_full,
                  'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0])}
            del Aq,Am,Aeq,beq,Aub,bub,res
            return 'unknown',meta
        x=res.x
        vals=np.asarray(prepared['Age']@x).reshape(-1) if prepared['Age'] is not None else np.zeros(0)
        rhs=prepared['rhs_ge'] if prepared['rhs_ge'] is not None else np.zeros(0)
        scale=np.maximum(1.0,np.maximum(np.abs(vals),np.abs(rhs)))
        sev=(rhs-vals)/scale
        bad=np.flatnonzero(sev>tol)
        if bad.size==0:
            maxeq,maxub=_sat_residuals(res,Aeq,beq,Aub,bub)
            meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
                  'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
                  'message':str(res.message),'max_eq_residual':maxeq,'max_ub_violation':maxub,
                  'cuts':len(ids),'used_full':used_full,
                  'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0])}
            del Aq,Am,Aeq,beq,Aub,bub,res
            return 'sat',meta
        candidates=[int(i) for i in bad if int(i) not in prepared['cut_set']]
        if not candidates:
            meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
                  'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
                  'message':'active lazy cuts remain violated beyond tolerance',
                  'cuts':len(ids),'used_full':used_full,
                  'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0])}
            del Aq,Am,Aeq,beq,Aub,bub,res
            return 'unknown',meta
        candidates.sort(key=lambda i:float(sev[i]),reverse=True)
        chosen=candidates[:max(1,int(LAZY_CUTS_PER_ROUND))]
        prepared['cut_indices'].extend(chosen); prepared['cut_set'].update(chosen)
        print(f"            lazy round {rounds}: solve={dt:.3f}s, violations={bad.size}, add={len(chosen)}, max={float(sev[chosen[0]]):.3e}, cuts={len(prepared['cut_indices'])}",flush=True)
        del Aub,bub,res

        # Auto mode: lazy is excellent for easy UNSAT/small active sets, but a
        # near-SAT branch can become slower than the old full solve.  Fall back
        # to the exact same full inequality system after a modest cut budget.
        if INEQ_MODE=='auto' and (
            rounds>=LAZY_FALLBACK_ROUNDS and len(prepared['cut_indices'])>=LAZY_FALLBACK_CUTS
        ):
            print(f"            auto fallback: solving full {prepared['n_ineq']} inequalities once",flush=True)
            Aub=-prepared['Age']; bub=-prepared['rhs_ge']
            res,dt=run_lp(Aub,bub); used_full=True; rounds+=1
            status=_classify_real_lp_result(res)
            maxeq,maxub=_sat_residuals(res,Aeq,beq,Aub,bub) if status=='sat' else (None,None)
            meta={'runtime_delta':delta_time,'runtime_solve':total_solve,'rounds':rounds,
                  'delta_eq_rows':len(qrows)+len(mrows),'scipy_status':int(res.status),
                  'message':str(res.message),'max_eq_residual':maxeq,'max_ub_violation':maxub,
                  'cuts':prepared['n_ineq'],'used_full':True,
                  'eq_rows_total':0 if Aeq is None else int(Aeq.shape[0])}
            if status == 'unsat':
                meta['numerical_unsat'] = True
                print('            HiGHS full fallback: NUMERICAL-UNSAT; exact verification ...', flush=True)
                exst, exmeta = _exact_verify_binary_branch(
                    base, prepared, qrows, qrhs, mrows, mrhs,
                    list(range(prepared['n_ineq']))
                )
                meta.update(exmeta)
                if exst == 'unsat':
                    meta['exact_verified'] = True
                    status = 'exact_unsat'
                    print(f"            exact {exmeta.get('exact_backend','verifier')}: UNSAT [{exmeta.get('exact_runtime_s',0):.3f}s]", flush=True)
                else:
                    meta['exact_verified'] = False
                    meta['numerical_false_or_unverified_unsat'] = True
                    status = 'unknown'
                    print(f"            exact {exmeta.get('exact_backend','verifier')}: {exst.upper()}; branch NOT excluded", flush=True)
            del Aq,Am,Aeq,beq,Aub,bub,res
            return status,meta

def _build_u_base(fixed_A,case,u,timeout_ms=None):
    _configure_u_base(fixed_A,case,u,timeout_ms)
    t0=time.perf_counter()
    base=build_problem()
    build_time=time.perf_counter()-t0
    # Real-LP path must not carry arithmetic baggage downstream.
    ignored=len(base.get('congruences',()))
    base['congruences']=[]
    xref=known_stabilizer_bucket(base['blocks'])
    xcap=bucket_capacity(base['blocks'],xref)
    if xcap!=2**u:
        raise AssertionError(f'Expected coarse X^U bucket capacity 2^u={2**u}, got {xcap}.')
    enabled={blockwise_tau_name(p) for p in base['enabled_blockwise_pairs']}
    expected={'(000,101)','(011,000)','(011,101)','(100,010)','(100,111)','(111,010)'}
    if enabled!=expected:
        raise AssertionError(f'Expected RXxRY extra twists {sorted(expected)}, got {sorted(enabled)}')
    if base['blockwise_parity_cases']:
        raise AssertionError('Optimized u-base must contain no qY parity/type branch constraints.')
    return base,build_time,ignored


def solve_reference_u(fixed_A, case, u, timeout_ms=None):
    """Optimized: one heavy base build + one base CSR assembly per u."""
    shell0=time.perf_counter()
    print(f'      reference u={u}, t={n-u}, A_u={fixed_A.get(u,0)}')
    print('        [base] building branch-invariant RXxRY problem ONCE ...',flush=True)
    base,build_time,ignored=_build_u_base(fixed_A,case,u,timeout_ms)
    prepared=_prepare_u_base_lp(base)
    print(
        f"        [base] vars={base['nvars']}, raw={len(base['universe'])}, "
        f"eq={len(base['eq_rows'])}, ineq={prepared['n_ineq']}, "
        f"nnz(eq)={prepared['nnz_eq']}, nnz(ineq-eval)={prepared['nnz_ineq_eval']}; "
        f"build={build_time:.3f}s, csr={prepared['runtime_csr']:.3f}s; "
        f"arithmetic rows skipped={ignored}",flush=True)
    branches=[]; saw_unknown=False; saw_error=False; attempted=0
    try:
        for qy_case in (1,2,3):
            if not qy_case_compatible(case,qy_case,u):
                branches.append({'status':'skipped','u':u,'qy_case':qy_case,
                                 'reason':'quadratic_case_incompatible_with_known_X_reference'})
                continue
            for mode in transverse_membership_modes(case,qy_case,u):
                attempted+=1
                print(f'        qY case={qy_case}, Y^U={mode}',flush=True)
                try:
                    st,meta=_solve_prepared_branch(base,prepared,qy_case,mode,u,tol=1e-8)
                    info={'u':u,'t':n-u,'qy_case':qy_case,'membership':mode,
                          'runtime_base_build':build_time,'runtime_base_csr':prepared['runtime_csr'],
                          **meta,'continuous_real_lp':True}
                    print(f"          => {st.upper()} [delta={meta['runtime_delta']:.3f}s, solve={meta['runtime_solve']:.3f}s, rounds={meta.get('rounds')}, cuts={meta.get('cuts')}, eq+={meta['delta_eq_rows']}]",flush=True)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    st='error'; info={'u':u,'qy_case':qy_case,'membership':mode,'error':repr(exc)}
                    print(f'          => ERROR: {exc!r}',flush=True)
                branches.append({'status':st,**info})
                if st=='sat':
                    info['runtime_u_total']=time.perf_counter()-shell0
                    return 'sat',{'u':u,'branches':branches,'witness_branch':info,
                                  'runtime_base_build':build_time,'runtime_base_csr':prepared['runtime_csr']}
                if st=='unknown': saw_unknown=True
                elif st=='error': saw_error=True
        if attempted==0:
            return 'error',{'u':u,'branches':branches,'reason':'no_transverse_branches'}
        overall='error' if saw_error else ('unknown' if saw_unknown else 'unsat')
        return overall,{'u':u,'branches':branches,'runtime_base_build':build_time,
                        'runtime_base_csr':prepared['runtime_csr'],
                        'runtime_u_total':time.perf_counter()-shell0}
    finally:
        # Once per u shell only.  Large cyclic structures, if any, are now unreachable.
        del prepared,base
        gc.collect()


def solve_candidate_all_references(fixed_A, case, *, order=None, timeout_ms=None,
                                   min_u=None, max_u=None, max_u_tests=None):
    u_values=positive_reference_weights(fixed_A, order=order)
    if min_u is not None: u_values=[u for u in u_values if u>=min_u]
    if max_u is not None: u_values=[u for u in u_values if u<=max_u]
    if max_u_tests is not None: u_values=u_values[:max_u_tests]
    if not u_values:
        return "error", {"reason":"no_positive_stabilizer_reference_shell","tested_u":[]}

    tests=[]; saw_unknown=False; saw_error=False
    for u in u_values:
        try:
            status,info=solve_reference_u(fixed_A,case,u,timeout_ms=timeout_ms)
        except KeyboardInterrupt: raise
        except Exception as exc:
            status="error"; info={"u":u,"error":repr(exc)}; print(f"      => ERROR: {exc!r}")
        tests.append({"status":status, **info})
        if status=="unsat":
            return "unsat", {"excluded_u":u,"tested_u":u_values,"references":tests}
        if status=="unknown": saw_unknown=True
        elif status=="error": saw_error=True

    overall="error" if saw_error else ("unknown" if saw_unknown else "sat")
    return overall, {"tested_u":u_values,"references":tests}




def _safe_db_record(db, cid, status, *, runtime=None, details=None, attempts=6):
    """Update generated SLR3 DB without aborting on transient cloud-file locks."""
    try:
        db.record_test(cid,SOLVER_NAME,status,version=SOLVER_VERSION,
                       runtime=runtime,details=details,save=False)
    except TypeError:
        # Older DB runtime without save= keyword.
        db.record_test(cid,SOLVER_NAME,status,version=SOLVER_VERSION,
                       runtime=runtime,details=details)
        return True
    if not hasattr(db,'_atomic_save'):
        return True
    last=None
    for j in range(max(1,attempts)):
        try:
            db._atomic_save(); return True
        except PermissionError as exc:
            last=exc; time.sleep(min(2.0,0.15*(2**j)))
    print(f"      [DB] DEFERRED SAVE: {last}",flush=True)
    return False


def main():
    global n, k, d, TIMEOUT_MS, HIGHS_METHOD, INEQ_MODE, LAZY_CUTS_PER_ROUND, LAZY_FALLBACK_ROUNDS, LAZY_FALLBACK_CUTS, EXACT_VERIFY_TIMEOUT_MS

    parser = argparse.ArgumentParser(
        description=(
            "Binary RXxRY stabilizer-reference JOINT continuous real-LP filter."
        )
    )
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("d", type=int)
    parser.add_argument(
        "--database",
        type=str,
        default=None,
        help=(
            "persistent SLR3 candidate database; default is "
            "candidates_n{n}_k{k}_d{d}.py in the current directory"
        ),
    )
    parser.add_argument("--case", type=int, choices=(1,2,3), default=None)
    parser.add_argument("--candidate", action="append", default=None)
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument("--highs-method", choices=("highs","highs-ds","highs-ipm"), default="highs")
    parser.add_argument("--ineq-mode", choices=("auto","lazy","full"), default="auto")
    parser.add_argument("--lazy-cuts", type=int, default=256)
    parser.add_argument("--lazy-fallback-rounds", type=int, default=3)
    parser.add_argument("--lazy-fallback-cuts", type=int, default=768)
    parser.add_argument("--exact-timeout-ms", type=int, default=0, help="exact Z3 QF_LRA verification timeout; 0=no limit")
    parser.add_argument(
        "--u-order",
        choices=("ascending","descending"),
        default=REFERENCE_U_ORDER,
    )
    parser.add_argument("--min-u", type=int, default=None)
    parser.add_argument("--max-u", type=int, default=None)
    parser.add_argument(
        "--max-u-tests",
        type=int,
        default=None,
        help="diagnostic restriction; omit for a complete candidate test",
    )
    args = parser.parse_args()

    n, k, d = args.n, args.k, args.d

    if not (0 <= k <= n):
        raise ValueError("Require 0 <= k <= n.")
    if not (1 <= d <= n):
        raise ValueError("Require 1 <= d <= n.")

    if args.timeout_ms is not None:
        TIMEOUT_MS = args.timeout_ms
    HIGHS_METHOD = args.highs_method
    INEQ_MODE = args.ineq_mode
    LAZY_CUTS_PER_ROUND = args.lazy_cuts
    LAZY_FALLBACK_ROUNDS = args.lazy_fallback_rounds
    LAZY_FALLBACK_CUTS = args.lazy_fallback_cuts
    EXACT_VERIFY_TIMEOUT_MS = args.exact_timeout_ms

    _exact_verifier_self_test()
    internal_projection_self_test()
    validate_strong_configuration()

    db_path = Path(args.database) if args.database else Path.cwd() / (
        DATABASE_FILENAME.format(n=n,k=k,d=d)
        if DATABASE_FILENAME is not None
        else f"candidates_n{n}_k{k}_d{d}.py"
    )

    if not db_path.exists():
        raise FileNotFoundError(
            f"Candidate database not found: {db_path}\n"
            f"Run: python SLR3_LP_lattice.py {n} {k} {d}"
        )

    db = load_candidate_database(db_path)

    if (int(db.N), int(db.K), int(db.D)) != (n,k,d):
        raise ValueError(
            f"Database has [[{db.N},{db.K},{db.D}]], "
            f"requested [[{n},{k},{d}]]."
        )

    migrated = ensure_database_solver_key(db, db_path)
    if migrated:
        print(
            f"Database migrated in place: added refined solver key "
            f"{SOLVER_NAME!r}."
        )

    if not db.DATABASE.get("ordinary_solver",{}).get("complete",False):
        raise ValueError(
            "SLR3 candidate enumeration is incomplete; refusing heavy "
            "RXxRY refined testing."
        )

    # Start with candidates that are pending for this solver.
    if args.rerun:
        pool = []
        for candidate in db.CANDIDATES:
            dead = any(
                name != SOLVER_NAME and info.get("status") == "unsat"
                for name, info in candidate["tests"].items()
            )
            if dead and not args.include_excluded:
                continue
            pool.append(candidate)
    else:
        pool = db.pending_candidates(
            SOLVER_NAME,
            solver_version=SOLVER_VERSION,
            include_excluded=args.include_excluded,
        )

    if args.case is not None:
        pool = [
            c for c in pool
            if int(c["case"]) == args.case
        ]

    if args.candidate:
        wanted = set(args.candidate)
        known = {c["id"] for c in db.CANDIDATES}
        missing = wanted - known
        if missing:
            raise ValueError(
                "Unknown candidate id(s): "
                + ", ".join(sorted(missing))
            )
        pool = [c for c in pool if c["id"] in wanted]

    print("=" * 108)
    print("BINARY RX x RY STABILIZER-REFERENCE REAL-LP -- RUN-ALL-ALIVE JOINT FILTER")
    print(f"Code: [[{n},{k},{d}]]")
    print(f"Database: {db_path.resolve()}")
    print(f"Solver key/version: {SOLVER_NAME} / {SOLVER_VERSION}")
    print("Projection: T=R_X=I/X/(YZ), U=R_Y=I/Y/(XZ)")
    print("Actual reference: g=I^T X^U in S")
    print("Six extra joint twists:")
    print("  (000,101),(011,000),(011,101),(100,010),(100,111),(111,010)")
    print("Transverse qY cases + exhaustive Y^U membership branching: ON")
    print(f"Backend: optimized u-base/delta sparse SciPy/{HIGHS_METHOD}; inequalities={INEQ_MODE}; exact UNSAT=Farkas->Z3 fallback")
    print("Integrality / modular / lattice constraints: OMITTED")
    print("Alive-pool rule: run every candidate with NO known UNSAT")
    print("  SAT / UNKNOWN / RUNNING / NOT_TESTED all remain eligible")
    print(f"Joint RXxRY candidates selected: {len(pool)}")
    print(f"Reference order: {args.u_order}")
    if args.max_u_tests is not None:
        print(
            "WARNING: --max-u-tests makes a surviving result diagnostic only."
        )
    print("=" * 108)

    if args.dry_run:
        for candidate in pool:
            fixed_A = {
                int(w): int(v)
                for w,v in candidate["fixed_A"].items()
            }
            u_values = positive_reference_weights(
                fixed_A,
                order=args.u_order,
            )
            if args.min_u is not None:
                u_values = [u for u in u_values if u >= args.min_u]
            if args.max_u is not None:
                u_values = [u for u in u_values if u <= args.max_u]
            if args.max_u_tests is not None:
                u_values = u_values[:args.max_u_tests]

            print(
                f"{candidate['id']} Case {candidate['case']} "
                f"u={u_values} FIXED_A={fixed_A}"
            )
        return

    counts = {"sat":0,"unsat":0,"unknown":0,"error":0}

    for pos,candidate in enumerate(pool,start=1):
        cid = candidate["id"]
        case = int(candidate["case"])
        fixed_A = {
            int(w): int(v)
            for w,v in candidate["fixed_A"].items()
        }

        print()
        print("-" * 108)
        print(
            f"[{pos}/{len(pool)}] {cid}, Case {case}: "
            f"FIXED_A={fixed_A}"
        )
        print("-" * 108)

        _safe_db_record(db,cid,"running",runtime=None,details=None)

        start_time = time.perf_counter()

        try:
            status, details = solve_candidate_all_references(
                fixed_A,
                case,
                order=args.u_order,
                timeout_ms=args.timeout_ms,
                min_u=args.min_u,
                max_u=args.max_u,
                max_u_tests=args.max_u_tests,
            )
            runtime = time.perf_counter() - start_time

            restricted = (
                args.min_u is not None
                or args.max_u is not None
                or args.max_u_tests is not None
            )

            stored = status
            if restricted and status == "sat":
                stored = "unknown"
                details["reason"] = (
                    "restricted_reference_sweep_completed_without_unsat"
                )

            details["selection_rule"] = "no_known_unsat"

            _safe_db_record(db,cid,stored,runtime=runtime,details=details)
            counts[stored] += 1

        except KeyboardInterrupt:
            print(
                "Interrupted; candidate remains status=running and will "
                "be retried next time."
            )
            raise

        except Exception as exc:
            runtime = time.perf_counter() - start_time
            _safe_db_record(db,cid,"error",runtime=runtime,details={"error":repr(exc)})
            counts["error"] += 1
            print(f"ERROR: {exc!r}")
            raise

    print()
    print("=" * 108)
    print("RX x RY REAL-LP JOINT FILTER SUMMARY")
    for key in ("sat","unsat","unknown","error"):
        print(f"  {key:7s}: {counts[key]}")
    print("=" * 108)
    db.print_summary(show_status=True)


if __name__ == "__main__":
    main()
