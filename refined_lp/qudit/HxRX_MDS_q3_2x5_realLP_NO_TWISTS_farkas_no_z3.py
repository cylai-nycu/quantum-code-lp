#!/usr/bin/env python3
r"""
HxRX_MDS_q3_2x5_realLP_NO_TWISTS_farkas_no_z3.py
Version: 2026-09-27-q3-hxrx-2x5-no-twists

Continuous REAL-LP necessary-condition solver for q=3 QMDS stabilizer codes
with a mandatory stabilizer reference

    g = I^T X^U in S, |U|=u.

Projection
----------
    T : Hamming {I}/non-I  (2 cells),
    U : R_X(5)             (5 cells).

This is the q=3 2x5 HxRX baseline used for the ablation

    3x3 coarse  ->  2x5 HxRX  ->  5x5 joint R_Y x R_X.

The U block keeps the full aligned R_X character resolution, but this strict
ablation imposes only the ordinary b=0 MacWilliams family.  The two
nontrivial b=1,2 Weyl-character twisted-Fourier positivity families are off.

Continuous-real constraints
---------------------------
  * exact QMDS split A distribution;
  * projected H x R_X(5) MacWilliams transform;
  * coefficientwise B>=A and B=A below d;
  * exact QMDS split B distribution;
  * generalized split-Hamming Rains shadow precheck;
  * exact reference existence and F_3-valued commutation support;
  * additive reference orbit equalities h -> h + alpha g for A and B;
  * F_3^x scalar orbit equalities for A and B;
  * cleaning/dimension precheck;
  * exact projected-cell capacity bounds.

This is a REAL relaxation: no integrality, no modular congruences, no orbit
divisibility, and no lattice arithmetic.

NUMERICAL / EXACT POLICY
------------------------
HiGHS is only a floating-point screening backend.  Variables are scaled and
rows normalized.  A bounded primary HiGHS solve is followed by independent
HiGHS-IPM rescue on non-success.  Numerical infeasibility is NEVER accepted
by itself.  EXACT_UNSAT requires an exact rational Farkas certificate verified
against the original unscaled integer constraints.  There is NO Z3 fallback
and this file does not import z3.

Regression targets
------------------
  [[11,3,5]]_3, mandatory u=8
  [[11,5,4]]_3, mandatory u=9

Typical usage:
  python HxRX_MDS_q3_2x5_realLP_v1_farkas_no_z3.py 3 11 3 5 --u 8
  python HxRX_MDS_q3_2x5_realLP_v1_farkas_no_z3.py 3 11 5 4 --u 9

Dependencies:
  pip install numpy scipy sympy
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack as _sp_hstack, vstack as _sp_vstack

Q = 3
ALPHABET = 9
FIELD = (0, 1, 2)
NONZERO = (1, 2)
X = (1, 0)
Y = (0, 1)
Comp = Tuple[int, ...]
ExactRow = Dict[int, int]


# =============================================================================
# F_3 phase-space and exact cyclotomic arithmetic Z[zeta_3]
# =============================================================================

def fadd(a: int, b: int) -> int:
    return (int(a) + int(b)) % 3


def fmul(a: int, b: int) -> int:
    return (int(a) * int(b)) % 3


def vadd(u: Tuple[int, int], v: Tuple[int, int]) -> Tuple[int, int]:
    return fadd(u[0], v[0]), fadd(u[1], v[1])


def vscale(c: int, u: Tuple[int, int]) -> Tuple[int, int]:
    return fmul(c, u[0]), fmul(c, u[1])


def symp(u: Tuple[int, int], v: Tuple[int, int]) -> int:
    # [u,v] = b*c - a*d over F_3.
    a, b = u
    c, d = v
    return (b * c - a * d) % 3


# a+b*zeta, zeta^2+zeta+1=0.
Cyclo = Tuple[int, int]


def czero() -> Cyclo:
    return (0, 0)


def cone() -> Cyclo:
    return (1, 0)


def cadd(a: Cyclo, b: Cyclo) -> Cyclo:
    return (a[0] + b[0], a[1] + b[1])


def cscale(m: int, a: Cyclo) -> Cyclo:
    return (int(m) * a[0], int(m) * a[1])


def cmul(a: Cyclo, b: Cyclo) -> Cyclo:
    a0, a1 = a
    b0, b1 = b
    # zeta^2 = -1-zeta.
    return (a0 * b0 - a1 * b1,
            a0 * b1 + a1 * b0 - a1 * b1)


def cpow(a: Cyclo, e: int) -> Cyclo:
    out = cone()
    base = a
    e = int(e)
    while e:
        if e & 1:
            out = cmul(out, base)
        base = cmul(base, base)
        e >>= 1
    return out


def chi(x: int) -> Cyclo:
    x %= 3
    if x == 0:
        return (1, 0)
    if x == 1:
        return (0, 1)
    return (-1, -1)


PHASE = tuple((a, b) for a in FIELD for b in FIELD)


def line_cells(axis: Tuple[int, int]) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    """R_eta with 2q-1=5 cells."""
    zero = ((0, 0),)
    xi = Y if axis == X else X
    cells: List[Tuple[Tuple[int, int], ...]] = [zero]
    for c in NONZERO:
        cells.append((vscale(c, axis),))
    for r in NONZERO:
        cells.append(tuple(vadd(vscale(a, axis), vscale(r, xi)) for a in FIELD))
    flat = [v for C in cells for v in C]
    if len(flat) != 9 or len(set(flat)) != 9:
        raise AssertionError("R_eta cells do not partition F_3^2")
    return tuple(cells)


TCELLS = (
    ((0, 0),),
    tuple(v for v in PHASE if v != (0, 0)),
)
UCELLS = line_cells(X)


def projected_local_matrix(cells, twist: Optional[Tuple[int, int]] = None):
    """Exact cyclotomic projected symplectic Fourier matrix."""
    rows = []
    for C in cells:
        rep = None
        for u in C:
            tf = cone() if twist is None else chi(symp(twist, u))
            row = []
            for D in cells:
                s = czero()
                for v in D:
                    s = cadd(s, chi(symp(u, v)))
                row.append(cmul(tf, s))
            tup = tuple(row)
            if rep is None:
                rep = tup
            elif tup != rep:
                raise ValueError("Requested twist does not descend to this R_eta projection")
        rows.append(rep)
    return tuple(rows)


# =============================================================================
# Weak compositions and exact composition transforms
# =============================================================================

def weak_compositions(total: int, parts: int) -> Tuple[Comp, ...]:
    out = []
    def rec(prefix, rem, left):
        if left == 1:
            out.append(tuple(prefix + [rem]))
            return
        for x in range(rem + 1):
            rec(prefix + [x], rem - x, left - 1)
    rec([], int(total), int(parts))
    return tuple(out)


def multinomial_coeff(comp: Sequence[int]) -> int:
    rem = sum(comp)
    z = 1
    for x in comp[:-1]:
        z *= math.comb(rem, int(x))
        rem -= int(x)
    return z


@lru_cache(maxsize=None)
def multinomial_terms_cyclo(power: int, coeffs: Tuple[Cyclo, ...]):
    out = {}
    for c in weak_compositions(power, len(coeffs)):
        z = multinomial_coeff(c)
        val = (z, 0)
        for a, e in zip(coeffs, c):
            if e:
                val = cmul(val, cpow(a, e))
        if val != (0, 0):
            out[c] = val
    return out


@lru_cache(maxsize=None)
def expand_comp_cyclo(comp: Comp, local_tuple) -> Dict[Comp, Cyclo]:
    r = len(comp)
    poly: Dict[Comp, Cyclo] = {tuple([0] * r): cone()}
    for a, power in enumerate(comp):
        if power == 0:
            continue
        factor = multinomial_terms_cyclo(power, tuple(local_tuple[a]))
        new = {}
        for c1, v1 in poly.items():
            for c2, v2 in factor.items():
                oc = tuple(x + y for x, y in zip(c1, c2))
                vv = cmul(v1, v2)
                if oc in new:
                    vv = cadd(new[oc], vv)
                if vv != (0, 0):
                    new[oc] = vv
                elif oc in new:
                    del new[oc]
        poly = new
    return poly


def composition_cyclo_matrices(m: int, local, label: str):
    comps = weak_compositions(m, len(local))
    idx = {c: i for i, c in enumerate(comps)}
    data0 = []; row0 = []; col0 = []
    data1 = []; row1 = []; col1 = []
    t0 = time.perf_counter(); last = t0
    local_tuple = tuple(tuple(x for x in row) for row in local)
    maxabs = 0
    for j, c in enumerate(comps):
        ex = expand_comp_cyclo(c, local_tuple)
        for oc, vv in ex.items():
            i = idx[oc]
            if vv[0]:
                row0.append(i); col0.append(j); data0.append(int(vv[0])); maxabs=max(maxabs,abs(int(vv[0])))
            if vv[1]:
                row1.append(i); col1.append(j); data1.append(int(vv[1])); maxabs=max(maxabs,abs(int(vv[1])))
        now = time.perf_counter()
        if now - last > 30:
            print(f"      [{label}] composition transform {j+1}/{len(comps)} [{now-t0:.1f}s]", flush=True)
            last = now
    if maxabs >= 2**63:
        raise OverflowError("composition transform coefficient exceeds int64")
    M0i = csr_matrix((np.asarray(data0, dtype=np.int64),
                      (np.asarray(row0, dtype=np.int32), np.asarray(col0, dtype=np.int32))),
                     shape=(len(comps), len(comps)))
    M1i = csr_matrix((np.asarray(data1, dtype=np.int64),
                      (np.asarray(row1, dtype=np.int32), np.asarray(col1, dtype=np.int32))),
                     shape=(len(comps), len(comps)))
    M0 = M0i.astype(np.float64)
    M1 = M1i.astype(np.float64)
    return comps, idx, (M0, M1), (M0i, M1i)


# =============================================================================
# Exact Farkas certification matched to normalized primal
# =============================================================================
_FARKAS_SUPPORT_TOLS = (1e-8, 1e-10, 1e-12, 1e-14, 1e-16)


def _fx_items(row):
    return row.items() if isinstance(row, dict) else enumerate(row)


def _normalized_exact_rows(rows, rhs, var_scale_int):
    out = []; br = []; scales = []
    for row, b in zip(rows, rhs):
        nums = {int(j): int(v) * int(var_scale_int[int(j)])
                for j, v in _fx_items(row) if v}
        R = max([1, abs(int(b))] + [abs(v) for v in nums.values()])
        out.append({j: Fraction(v, R) for j, v in nums.items() if v})
        br.append(Fraction(int(b), R)); scales.append(int(R))
    return out, br, scales


def _build_normalized_exact_system(nv, eq_rows, eq_rhs, le_rows, le_rhs, bounds, var_scale_int):
    E, e, ers = _normalized_exact_rows(eq_rows, eq_rhs, var_scale_int)
    C, c, crs = _normalized_exact_rows(le_rows, le_rhs, var_scale_int)
    for j, (lo, hi) in enumerate(bounds):
        S = int(var_scale_int[j])
        if lo is not None:
            C.append({j: Fraction(-1, 1)}); c.append(Fraction(-int(lo), S)); crs.append(S)
        if hi is not None:
            C.append({j: Fraction(1, 1)}); c.append(Fraction(int(hi), S)); crs.append(S)
    return E, e, ers, C, c, crs


def _rows_fraction_to_csr(rows, nv):
    data = []; inds = []; indptr = [0]
    for row in rows:
        for j, v in sorted(row.items()):
            if v:
                data.append(float(v)); inds.append(int(j))
        indptr.append(len(data))
    return csr_matrix((np.asarray(data, float), np.asarray(inds, np.int32), np.asarray(indptr, np.int64)),
                      shape=(len(rows), nv))


def _exact_verify_transformed(nv, E, e, C, c, wp, wm, y):
    if any(v < 0 for v in wp) or any(v < 0 for v in wm) or any(v < 0 for v in y):
        return False, {"farkas_reason": "negative_exact_multiplier"}
    r = [Fraction(0, 1) for _ in range(nv)]
    obj = Fraction(0, 1)
    for ap, am, row, b in zip(wp, wm, E, e):
        lam = ap - am
        if lam:
            for j, v in row.items(): r[j] += lam * v
            obj += lam * b
    for lam, row, b in zip(y, C, c):
        if lam:
            for j, v in row.items(): r[j] += lam * v
            obj += lam * b
    if any(v != 0 for v in r):
        return False, {"farkas_reason": "nonzero_exact_stationarity",
                       "farkas_residual_nnz": sum(v != 0 for v in r)}
    mass = sum(wp, Fraction(0, 1)) + sum(wm, Fraction(0, 1)) + sum(y, Fraction(0, 1))
    if mass != 1:
        return False, {"farkas_reason": "bad_exact_normalization", "farkas_mass": str(mass)}
    if not obj < 0:
        return False, {"farkas_reason": "nonnegative_exact_objective", "farkas_bTy": str(obj)}
    den = max([1] + [v.denominator for v in list(wp) + list(wm) + list(y)])
    return True, {"farkas_bTy": str(obj), "farkas_max_denominator": int(den),
                  "farkas_eq_support": sum(v != 0 for v in wp) + sum(v != 0 for v in wm),
                  "farkas_ineq_support": sum(v != 0 for v in y)}


def _exact_verify_original(nv, eq_rows, eq_rhs, le_rows, le_rhs, bounds,
                           ers, crs, wp, wm, y):
    z = [(wp[i] - wm[i]) / Fraction(int(ers[i]), 1) for i in range(len(eq_rows))]
    nbase = len(le_rows)
    mu = [y[i] / Fraction(int(crs[i]), 1) for i in range(nbase)]
    r = [Fraction(0, 1) for _ in range(nv)]
    obj = Fraction(0, 1)
    for lam, row, b in zip(z, eq_rows, eq_rhs):
        if lam:
            for j, v in _fx_items(row):
                if v: r[int(j)] += lam * Fraction(int(v), 1)
            obj += lam * Fraction(int(b), 1)
    for lam, row, b in zip(mu, le_rows, le_rhs):
        if lam:
            for j, v in _fx_items(row):
                if v: r[int(j)] += lam * Fraction(int(v), 1)
            obj += lam * Fraction(int(b), 1)
    p = nbase
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            lam = y[p] / Fraction(int(crs[p]), 1); p += 1
            r[j] -= lam; obj += lam * Fraction(-int(lo), 1)
        if hi is not None:
            lam = y[p] / Fraction(int(crs[p]), 1); p += 1
            r[j] += lam; obj += lam * Fraction(int(hi), 1)
    if any(v != 0 for v in r):
        return False, {"farkas_reason": "original_stationarity_failed",
                       "farkas_original_residual_nnz": sum(v != 0 for v in r)}
    if not obj < 0:
        return False, {"farkas_reason": "original_objective_nonnegative",
                       "farkas_original_bTy": str(obj)}
    return True, {"farkas_original_bTy": str(obj)}


def _exact_support_reconstruct(nv, E, e, C, c, wf):
    try:
        import sympy as sp
    except Exception:
        return False, {"farkas_reason": "sympy_unavailable"}, None
    me = len(E); mi = len(C)
    wf = np.asarray(wf, float)
    mx = max(1.0, float(np.max(np.abs(wf))) if wf.size else 0.0)
    last = {"farkas_reason": "exact_support_not_recovered"}

    def get_col(idx):
        if idx < me:
            row = E[idx]; sgn = 1
        elif idx < 2 * me:
            row = E[idx - me]; sgn = -1
        else:
            row = C[idx - 2 * me]; sgn = 1
        col = [sp.Rational(0)] * (nv + 1)
        for j, v in row.items(): col[j] = sp.Rational(sgn * v.numerator, v.denominator)
        col[-1] = sp.Rational(1)
        return col

    for tol in _FARKAS_SUPPORT_TOLS:
        supp = [i for i, v in enumerate(wf) if v > tol * mx]
        ns = len(supp)
        if not supp:
            last = {"farkas_reason": "empty_support", "farkas_support_tol": tol}; continue
        if ns > nv + 12:
            last = {"farkas_reason": "support_too_large", "farkas_support_size": ns,
                    "farkas_support_tol": tol}; continue
        cols = [get_col(i) for i in supp]
        M = sp.Matrix(nv + 1, ns, lambda r, cc: cols[cc][r])
        b = sp.Matrix([0] * nv + [1])
        try:
            sol, params = M.gauss_jordan_solve(b)
        except Exception as exc:
            last = {"farkas_reason": "exact_support_inconsistent", "farkas_support_size": ns,
                    "farkas_support_tol": tol, "farkas_exact_error": type(exc).__name__}; continue
        plist = list(params)
        trials = []
        if not plist:
            trials = [{}]
        else:
            zero = {p: sp.Rational(0) for p in plist}
            x0 = sp.Matrix([sp.cancel(v.subs(zero)) for v in sol])
            N = sp.Matrix(ns, len(plist), lambda i, j: sp.cancel(sp.diff(sol[i], plist[j])))
            try:
                Nf = np.asarray(N.evalf(), dtype=float)
                x0f = np.asarray(x0.evalf(), dtype=float).reshape(-1)
                tau = np.linalg.lstsq(Nf, wf[supp] - x0f, rcond=None)[0]
            except Exception:
                tau = np.zeros(len(plist))
            for md in (10**6, 10**9, 2**31, 2**40):
                sub = {}
                for p0, v0 in zip(plist, tau):
                    fr = Fraction(float(v0)).limit_denominator(md)
                    sub[p0] = sp.Rational(fr.numerator, fr.denominator)
                trials.append(sub)
        for sub in trials:
            vals = [sp.cancel(v.subs(sub)) for v in sol]
            try:
                qv = [Fraction(int(sp.numer(v)), int(sp.denom(v))) for v in vals]
            except Exception:
                continue
            if any(v < 0 for v in qv):
                last = {"farkas_reason": "negative_exact_support_value",
                        "farkas_support_size": ns, "farkas_support_tol": tol}; continue
            wp = [Fraction(0, 1) for _ in range(me)]
            wm = [Fraction(0, 1) for _ in range(me)]
            yy = [Fraction(0, 1) for _ in range(mi)]
            for idx, v in zip(supp, qv):
                if idx < me: wp[idx] = v
                elif idx < 2 * me: wm[idx - me] = v
                else: yy[idx - 2 * me] = v
            ok, meta = _exact_verify_transformed(nv, E, e, C, c, wp, wm, yy)
            meta.update({"farkas_support_size": ns, "farkas_support_tol": tol,
                         "farkas_reconstruction": "matched-normalized-support"})
            if ok: return True, meta, (wp, wm, yy)
            last = meta
    return False, last, None


def exact_farkas_verify(nv, eq_rows, eq_rhs, le_rows, le_rhs, bounds, var_scale_int,
                        method="highs-ds", return_certificate=False):
    E, e, ers, C, c, crs = _build_normalized_exact_system(
        nv, eq_rows, eq_rhs, le_rows, le_rhs, bounds, var_scale_int)
    me = len(E); mi = len(C)
    Ef = _rows_fraction_to_csr(E, nv) if me else csr_matrix((0, nv), dtype=float)
    Cf = _rows_fraction_to_csr(C, nv) if mi else csr_matrix((0, nv), dtype=float)
    station = _sp_hstack([Ef.T, -Ef.T, Cf.T], format="csr")
    normrow = csr_matrix(np.ones((1, 2 * me + mi), dtype=float))
    Aeq = _sp_vstack([station, normrow], format="csr")
    beq = np.zeros(nv + 1, float); beq[-1] = 1.0
    rhs = np.asarray([float(v) for v in e] + [-float(v) for v in e] + [float(v) for v in c], dtype=float)
    obj_scale = max(1.0, float(np.max(np.abs(rhs))) if rhs.size else 1.0)
    cnum = rhs / obj_scale
    t0 = time.perf_counter()
    rr = linprog(cnum, A_eq=Aeq, b_eq=beq, bounds=[(0.0, None)] * (2 * me + mi),
                 method=method, options={"presolve": True})
    dt = time.perf_counter() - t0
    meta = {"farkas_numeric_runtime_s": dt, "farkas_numeric_status": int(rr.status),
            "farkas_cone_vars": 2 * me + mi}
    if rr.status != 0 or rr.x is None:
        meta["farkas_reason"] = "numeric_farkas_cone_failed"; return ("unknown", meta, None) if return_certificate else ("unknown", meta)
    obj = float(cnum @ np.asarray(rr.x, float))
    meta["farkas_numeric_objective_scaled"] = obj
    meta["farkas_numeric_support_1e10"] = int(np.sum(np.asarray(rr.x) > 1e-10))
    if not obj < 0.0:
        meta["farkas_reason"] = "no_negative_farkas_objective"; return ("unknown", meta, None) if return_certificate else ("unknown", meta)
    ok, xmeta, cert = _exact_support_reconstruct(nv, E, e, C, c, rr.x)
    xmeta.update(meta)
    if not ok or cert is None: return ("unknown", xmeta, None) if return_certificate else ("unknown", xmeta)
    wp, wm, yy = cert
    ok2, ometa = _exact_verify_original(nv, eq_rows, eq_rhs, le_rows, le_rhs,
                                        bounds, ers, crs, wp, wm, yy)
    xmeta.update(ometa)
    if not ok2: return ("unknown", xmeta, None) if return_certificate else ("unknown", xmeta)
    xmeta["exact_backend"] = "farkas-matched-normalized-exact"
    if not return_certificate:
        return "unsat", xmeta
    # Map multipliers back from the normalized/scaled cone to the exact,
    # unscaled original rows used by _exact_verify_original.
    z = [(wp[i] - wm[i]) / Fraction(int(ers[i]), 1) for i in range(len(eq_rows))]
    nbase = len(le_rows)
    mu = [yy[i] / Fraction(int(crs[i]), 1) for i in range(nbase)]
    p = nbase; lower = []; upper = []
    for j, (lo, hi) in enumerate(bounds):
        if lo is not None:
            lower.append(yy[p] / Fraction(int(crs[p]), 1)); p += 1
        else: lower.append(Fraction(0, 1))
        if hi is not None:
            upper.append(yy[p] / Fraction(int(crs[p]), 1)); p += 1
        else: upper.append(Fraction(0, 1))
    return "unsat", xmeta, {"equality": z, "inequality": mu,
                              "lower_bound": lower, "upper_bound": upper}


def _sparse_expression(row: ExactRow) -> str:
    if not row: return "0"
    return " + ".join(f"({v})*x[{j}]" for j, v in sorted(row.items()))


def write_farkas_support_report(path: str, *, n: int, k: int, d: int, u: int,
                                eq_rows, eq_rhs, eq_labels, le_rows, le_rhs,
                                le_labels, bounds, bound_labels, certificate, meta):
    """Write certificate support in original, unscaled LP coordinates only."""
    entries = []
    for i, (coef, row, rhs, label) in enumerate(zip(certificate["equality"], eq_rows, eq_rhs, eq_labels)):
        if coef:
            family, text = label
            entries.append((family, "equality", i, text, coef, int(rhs), row))
    for i, (coef, row, rhs, label) in enumerate(zip(certificate["inequality"], le_rows, le_rhs, le_labels)):
        if coef:
            family, text = label
            entries.append((family, "inequality", i, text, coef, int(rhs), row))
    for j, coef in enumerate(certificate["lower_bound"]):
        if coef:
            family, text = bound_labels[j][0]
            entries.append((family, "lower bound", j, text, coef, -int(bounds[j][0]), {j: -1}))
    for j, coef in enumerate(certificate["upper_bound"]):
        if coef:
            family, text = bound_labels[j][1]
            entries.append((family, "upper bound", j, text, coef, int(bounds[j][1]), {j: 1}))
    grouped = defaultdict(list)
    for e in entries: grouped[e[0]].append(e)
    with open(path, "w", encoding="utf-8") as f:
        f.write("EXACT RATIONAL FARKAS CERTIFICATE SUPPORT (ORIGINAL UNSCALED SYSTEM)\n")
        f.write(f"code=[[{n},{k},{d}]]_3; reference_u={u}\n")
        f.write(f"variables={len(bounds)}; equality_rows={len(eq_rows)}; inequality_rows={len(le_rows)}; lower_bounds={len(bounds)}; upper_bounds={len(bounds)}\n")
        f.write(f"nonzero_equality_duals={sum(x != 0 for x in certificate['equality'])}; nonzero_inequality_duals={sum(x != 0 for x in certificate['inequality'])}; nonzero_bound_duals={sum(x != 0 for x in certificate['lower_bound']) + sum(x != 0 for x in certificate['upper_bound'])}\n")
        f.write(f"verified_original_bTy={meta.get('farkas_original_bTy')}\n")
        f.write("twisted_b1_rows=0; twisted_b2_rows=0\n\n")
        f.write("SUPPORT GROUPED BY FAMILY\n")
        for family, xs in sorted(grouped.items(), key=lambda kv: (-sum(abs(x[4]) for x in kv[1]), kv[0])):
            f.write(f"{family}: rows={len(xs)}; sum_abs_dual={sum(abs(x[4]) for x in xs)}\n")
            f.write("  " + ", ".join(f"{x[1]}[{x[2]}] {x[3]}" for x in xs) + "\n")
        f.write("\nDETAILED NONZERO SUPPORT\n")
        for family, typ, i, text, coef, rhs, row in entries:
            f.write(f"row_index={i}; row_type={typ}; family={family}\n")
            f.write(f"label={text}\ncoefficient={coef}\nrhs={rhs}\nexpression={_sparse_expression(row)}\n\n")


# =============================================================================
# QMDS distributions, capacities, symmetry actions
# =============================================================================

def mds_exact_support(D: int, w: int) -> int:
    if w == 0: return 1
    if w < D: return 0
    return (ALPHABET - 1) * sum(
        (-1) ** s * math.comb(w - 1, s) * ALPHABET ** (w - D - s)
        for s in range(w - D + 1)
    )


def split_mds(t: int, u: int, D: int, i: int, r: int) -> int:
    return math.comb(t, i) * math.comb(u, r) * mds_exact_support(D, i + r)


def mds_distribution(n: int, D: int) -> List[int]:
    return [math.comb(n, w) * mds_exact_support(D, w) for w in range(n + 1)]


def block_capacity(comp: Comp, cells) -> int:
    z = multinomial_coeff(comp)
    for c, C in zip(comp, cells): z *= len(C) ** int(c)
    return z


def bucket_capacity(tc: Comp, uc: Comp) -> int:
    return block_capacity(tc, TCELLS) * block_capacity(uc, UCELLS)


def u_cell_pairings():
    vals = []
    for C in UCELLS:
        s = {symp(X, v) for v in C}
        if len(s) != 1: raise AssertionError("R_X cell does not retain X pairing")
        vals.append(next(iter(s)))
    return tuple(vals)


UPAIR = u_cell_pairings()


def bucket_pairing(uc: Comp) -> int:
    return sum(int(c) * int(s) for c, s in zip(uc, UPAIR)) % 3


def induced_perm(cells, fn):
    lookup = {frozenset(C): i for i, C in enumerate(cells)}
    p = []
    for C in cells:
        im = frozenset(fn(v) for v in C)
        if im not in lookup: raise AssertionError("action does not preserve projection")
        p.append(lookup[im])
    return tuple(p)


def permute_comp(comp: Comp, perm: Sequence[int]) -> Comp:
    out = [0] * len(comp)
    for old, count in enumerate(comp): out[perm[old]] += int(count)
    return tuple(out)


UADD_PERM = {a: induced_perm(UCELLS, lambda v, aa=a: vadd(v, vscale(aa, X))) for a in FIELD}
TSCALAR_PERM = {b: induced_perm(TCELLS, lambda v, bb=b: vscale(bb, v)) for b in NONZERO}
USCALAR_PERM = {b: induced_perm(UCELLS, lambda v, bb=b: vscale(bb, v)) for b in NONZERO}


def generalized_shadow_matrix(m: int):
    # Input Hamming weight i -> output weight o, substitution
    # I: (q-1)x+(q+1)y = 2x+4y, non-I: -x+y.
    rows = []
    for iw in range(m + 1):
        poly = {(0, 0): 1}
        for linear, power in [((Q - 1, Q + 1), m - iw), ((-1, 1), iw)]:
            for _ in range(power):
                nxt = {}
                for (a, b), cc in poly.items():
                    for pos, aa in enumerate(linear):
                        if aa:
                            key = (a + (pos == 0), b + (pos == 1))
                            nxt[key] = nxt.get(key, 0) + cc * aa
                poly = nxt
        rows.append(tuple(poly.get((m - ow, ow), 0) for ow in range(m + 1)))
    return tuple(tuple(rows[iw][ow] for iw in range(m + 1)) for ow in range(m + 1))


def shadow_precheck(t: int, u: int, D_A: int):
    ST = generalized_shadow_matrix(t); SU = generalized_shadow_matrix(u)
    for ot in range(t + 1):
        for ou in range(u + 1):
            z = sum(ST[ot][i] * SU[ou][r] * split_mds(t, u, D_A, i, r)
                    for i in range(t + 1) for r in range(u + 1))
            if z < 0:
                return {"reason": "fixed_split_shadow_negative", "output": (ot, ou), "value": z}
    return None


def cleaning_precheck(n: int, k: int, d: int, t: int, u: int, D_A: int):
    Tonly = sum(split_mds(t, u, D_A, i, 0) for i in range(t + 1))
    Uonly = sum(split_mds(t, u, D_A, 0, r) for r in range(u + 1))
    def ok(csize, other, corr):
        e = n - k - 2 * csize
        return other == Q ** e * corr if e >= 0 else Q ** (-e) * other == corr
    if t < d and not ok(t, Uonly, Tonly): return {"reason": "fixed_split_cleaning_T"}
    if u < d and not ok(u, Tonly, Uonly): return {"reason": "fixed_split_cleaning_U"}
    return None


# =============================================================================
# Solver helpers
# =============================================================================

def rows_to_csr(rows: Sequence[Mapping[int, float]], nvars: int):
    if not rows: return None
    data = []; inds = []; indptr = [0]
    for row in rows:
        for j, v in sorted(row.items()):
            if v: data.append(float(v)); inds.append(int(j))
        indptr.append(len(data))
    return csr_matrix((np.asarray(data, float), np.asarray(inds, np.int32), np.asarray(indptr, np.int64)),
                      shape=(len(rows), nvars))


def add_sparse(a: ExactRow, b: ExactRow, scale_b: int = 1) -> ExactRow:
    out = dict(a)
    for j, v in b.items():
        z = out.get(j, 0) + scale_b * int(v)
        if z: out[j] = z
        elif j in out: del out[j]
    return out


@dataclass
class Family:
    a: int
    b: int
    Tf: Tuple[csr_matrix, csr_matrix]
    Uf: Tuple[csr_matrix, csr_matrix]
    Ti: Tuple[csr_matrix, csr_matrix]
    Ui: Tuple[csr_matrix, csr_matrix]


def solve_u(n: int, k: int, d: int, u: int, *, lazy_cuts: int = 512,
            lazy_tol: float = 1e-8, simplex_time_limit_s: float = 3.0,
            max_active_vars: int = 30000, max_raw_cells: int = 500000,
            farkas_support_report: Optional[str] = None,
            disabled_families: Sequence[str] = ()):
    st = time.perf_counter(); t = n - u; D_A = n - d + 2; norm = Q ** (n - k)
    disabled_families = frozenset(disabled_families)
    if t < 0 or u <= 0: raise ValueError("bad u")

    tc = weak_compositions(t, 2); uc = weak_compositions(u, 5)
    raw = len(tc) * len(uc)
    if raw > max_raw_cells:
        return "unknown", {"u": u, "reason": "raw_cell_budget", "raw": raw, "max": max_raw_cells}

    # Build the Hamming T transform and the ordinary b=0 U character only.
    Tfam = {}
    loc = projected_local_matrix(TCELLS, None)
    comps, idx, Mf, Mi = composition_cyclo_matrices(t, loc, f"T Hamming/u={u}")
    tc = comps; ti = idx; Tfam[0] = (Mf, Mi)
    Ufam = {}
    loc = projected_local_matrix(UCELLS, None)
    comps, idx, Mf, Mi = composition_cyclo_matrices(u, loc, f"U b=0/u={u}")
    uc = comps; ui = idx; Ufam[0] = (Mf, Mi)
    families = {(0, 0): Family(0, 0, Tfam[0][0], Ufam[0][0], Tfam[0][1], Ufam[0][1])}

    # Raw active A buckets: exact split MDS support + exact F_3 commutation with g.
    raw_active = []; raw_meta = []; raw_index = {}
    for it, cT in enumerate(tc):
        wtT = t - cT[0]
        for iu, cU in enumerate(uc):
            wtU = u - cU[0]
            shell = split_mds(t, u, D_A, wtT, wtU)
            if shell == 0: continue
            if bucket_pairing(cU) != 0: continue
            cap = bucket_capacity(cT, cU)
            j = len(raw_active)
            raw_active.append((it, iu))
            raw_meta.append((cT, cU, wtT, wtU, shell, cap))
            raw_index[(it, iu)] = j
    if not raw_active:
        return "exact_unsat", {"u": u, "reason": "no_active_variables"}

    # -----------------------------------------------------------------
    # EXACT A-side orbit quotient.
    #
    # The old HxRX solver imposed A(h)=A(h+alpha g) and A(h)=A(beta h).
    # Instead of carrying thousands of equality rows, quotient those actions
    # before the LP.  If an orbit image lands in a pruned (hence exact-zero)
    # bucket, the entire orbit class is exact zero and is removed.
    # -----------------------------------------------------------------
    Nraw = len(raw_active)
    parent = list(range(Nraw)); rank = [0] * Nraw; forced_zero = set()

    def dfind(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def dunion(a0, b0):
        a0 = dfind(a0); b0 = dfind(b0)
        if a0 == b0: return
        if rank[a0] < rank[b0]: a0, b0 = b0, a0
        parent[b0] = a0
        if rank[a0] == rank[b0]: rank[a0] += 1

    for j, (it, iu) in enumerate(raw_active):
        cT, cU, *_ = raw_meta[j]
        for alpha in NONZERO:
            cpU = permute_comp(cU, UADD_PERM[alpha])
            jp = raw_index.get((it, ui[cpU]))
            if jp is None: forced_zero.add(j)
            else: dunion(j, jp)
        cpT = permute_comp(cT, TSCALAR_PERM[2])
        cpU = permute_comp(cU, USCALAR_PERM[2])
        jp = raw_index.get((ti[cpT], ui[cpU]))
        if jp is None: forced_zero.add(j)
        else: dunion(j, jp)

    forced_roots = {dfind(j) for j in forced_zero}
    members = defaultdict(list)
    for j in range(Nraw): members[dfind(j)].append(j)
    live_roots = [r for r in members if r not in forced_roots]
    root_to_var = {r: j for j, r in enumerate(live_roots)}

    # Bucket -> quotient variable; omitted buckets are exact zero.
    index = {}
    alive_raw = []; alive_meta = []; raw_to_var = []
    for r in live_roots:
        vj = root_to_var[r]
        for jj in members[r]:
            it, iu = raw_active[jj]
            index[(it, iu)] = vj
            alive_raw.append((it, iu))
            alive_meta.append(raw_meta[jj])
            raw_to_var.append(vj)

    nvars = len(live_roots)
    if not nvars:
        return "exact_unsat", {"u": u, "reason": "all_A_orbits_forced_zero"}
    if nvars > max_active_vars:
        return "unknown", {"u": u, "reason": "active_variable_budget", "active": nvars,
                           "raw_active": Nraw, "max": max_active_vars}

    ati = np.asarray([a for a, b in alive_raw], dtype=np.int32)
    aui = np.asarray([b for a, b in alive_raw], dtype=np.int32)
    avar = np.asarray(raw_to_var, dtype=np.int32)

    # Reference bucket: T all I; U all X[1].
    tref = (t, 0)
    uref = [0] * 5; uref[1] = u
    refcol = index.get((ti[tref], ui[tuple(uref)]))
    if refcol is None:
        return "exact_unsat", {"u": u, "reason": "reference_orbit_forced_zero"}

    # Quotient-variable upper bound is the minimum valid bound across all
    # buckets in that orbit class.
    ub_by_var = [None] * nvars
    for (cT, cU, wtT, wtU, shell, cap), vj in zip(alive_meta, raw_to_var):
        ub = min(int(shell), int(cap))
        ub_by_var[vj] = ub if ub_by_var[vj] is None else min(ub_by_var[vj], ub)
    bounds = [(0, int(ub_by_var[j])) for j in range(nvars)]
    lo, hi = bounds[refcol]; bounds[refcol] = (max(1, lo), hi)
    if hi < 1:
        return "exact_unsat", {"u": u, "reason": "reference_capacity"}
    if "nonnegativity_A" in disabled_families:
        for j, (lo, hi) in enumerate(bounds):
            if j != refcol:
                bounds[j] = (None, hi)
    # Bounds are original LP rows in Farkas verification.  Preserve the
    # quotient-variable provenance rather than pretending quotienting added
    # explicit orbit equations.
    bound_labels = []
    for j in range(nvars):
        metas = [m for m, v in zip(alive_meta, raw_to_var) if v == j]
        cap_limited = any(int(m[5]) == int(bounds[j][1]) for m in metas)
        upper_family = "A_capacity" if cap_limited else "variable_upper_bound"
        upper_text = f"{upper_family}[quotient_var={j}; upper={bounds[j][1]}]"
        lower = ("reference_existence", f"reference_existence[quotient_var={j}; x[{j}]>=1]") if j == refcol else ("nonnegativity_A", f"nonnegativity_A[quotient_var={j}; x[{j}]>=0]")
        bound_labels.append((lower, (upper_family, upper_text)))

    bad = shadow_precheck(t, u, D_A)
    if bad: return "exact_unsat", {"u": u, **bad}
    bad = cleaning_precheck(n, k, d, t, u, D_A)
    if bad: return "exact_unsat", {"u": u, **bad}

    var_scale_int = [max(1, int(hi)) for lo, hi in bounds]
    var_scale = np.asarray(var_scale_int, dtype=float)
    numeric_bounds = [(None if lo is None else float(lo) / var_scale[j], float(hi) / var_scale[j])
                      for j, (lo, hi) in enumerate(bounds)]

    # Row caches for exact transform coefficients.
    dense_row_cache = {}
    exact_row_cache = {}

    def comp_rows(Ms, oi, tag):
        key = (tag, int(oi))
        if key not in dense_row_cache:
            dense_row_cache[key] = tuple(np.asarray(M.getrow(int(oi)).toarray()).ravel().astype(np.int64) for M in Ms)
        return dense_row_cache[key]

    def exact_transform_rows(a: int, b: int, oi: int, ou: int):
        key = (a, b, int(oi), int(ou))
        if key in exact_row_cache: return exact_row_cache[key]
        F = families[(a, b)]
        tr0, tr1 = comp_rows(F.Ti, oi, ("T", a))
        ur0, ur1 = comp_rows(F.Ui, ou, ("U", b))
        # cyclotomic product coefficients at active input buckets.
        t0v = tr0[ati]; t1v = tr1[ati]; u0v = ur0[aui]; u1v = ur1[aui]
        v0 = t0v * u0v - t1v * u1v
        v1 = t0v * u1v + t1v * u0v - t1v * u1v
        r0 = {}
        for jj in np.nonzero(v0)[0]:
            vj = int(avar[jj]); z = r0.get(vj, 0) + int(v0[jj])
            if z: r0[vj] = z
            elif vj in r0: del r0[vj]
        r1 = {}
        for jj in np.nonzero(v1)[0]:
            vj = int(avar[jj]); z = r1.get(vj, 0) + int(v1[jj])
            if z: r1[vj] = z
            elif vj in r1: del r1[vj]
        exact_row_cache[key] = (r0, r1)
        return r0, r1

    def eval_family(a: int, b: int, x: np.ndarray):
        F = families[(a, b)]
        Xmat = np.zeros((len(tc), len(uc)), dtype=float)
        Xmat[ati, aui] = x[avar]
        T0, T1 = F.Tf; U0, U1 = F.Uf
        A00 = (U0.dot(T0.dot(Xmat).T)).T
        A11 = (U1.dot(T1.dot(Xmat).T)).T
        A01 = (U1.dot(T0.dot(Xmat).T)).T
        A10 = (U0.dot(T1.dot(Xmat).T)).T
        return A00 - A11, A01 + A10 - A11

    # Base exact equalities: exact split A after the A-side orbit quotient.
    base_eqE = []; base_eqR = []; base_eqL = []
    def add_base_eq(row: ExactRow, rhs: int, label):
        rr = {int(j): int(v) for j, v in row.items() if v}
        base_eqE.append(rr); base_eqR.append(int(rhs)); base_eqL.append(label)

    byshell = defaultdict(lambda: defaultdict(int))
    for (cT, cU, wtT, wtU, _shell, _cap), vj in zip(alive_meta, raw_to_var):
        byshell[(wtT, wtU)][vj] += 1
    for i in range(t + 1):
        for r in range(u + 1):
            target = split_mds(t, u, D_A, i, r)
            coeffs = byshell.get((i, r), {})
            if target and "fixed_QMDS_A_split" not in disabled_families:
                if not coeffs:
                    return "exact_unsat", {"u": u, "reason": f"positive_split_A_pruned_{i}_{r}"}
                add_base_eq(dict(coeffs), target, ("fixed_QMDS_A_split", f"fixed_QMDS_A_split[i={i},r={r}]"))

    def normalized_float_row(row: ExactRow, rhs: int):
        rf = {j: float(v) * var_scale[j] for j, v in row.items()}
        sc = max([1.0, abs(float(rhs))] + [abs(v) for v in rf.values()])
        return {j: v / sc for j, v in rf.items() if v}, float(rhs) / sc

    lazy_eqE = []; lazy_eqR = []; lazy_eqL = []; lazy_leE = []; lazy_leR = []; lazy_leL = []
    lazy_eqF = []; lazy_eqRF = []; lazy_leF = []; lazy_leRF = []
    existing = set()

    def row_label(key):
        """Stable provenance for an original (unscaled) lazily added row."""
        head = key[0]
        def cell(oi, ou):
            return f"T={tc[oi]},U={uc[ou]}"
        if head == "ordinary":
            kind = key[1]
            if kind == "cyclo": return ("ordinary_b0_MacWilliams", f"ordinary_b0_MacWilliams[imaginary; {cell(key[2], key[3])}]")
            if kind == "Bcomm": return ("reference_commutation_support", f"commutation_zero_B[{cell(key[2], key[3])}]")
            if kind == "distance": return ("distance_B_eq_A", f"distance_B_eq_A[{cell(key[2], key[3])}]")
            if kind == "contain": return ("B_ge_A", f"B_ge_A[{cell(key[2], key[3])}]")
            if kind == "Bcap": return ("B_capacity", f"B_capacity[{cell(key[2], key[3])}]")
        if head == "splitB": return ("fixed_QMDS_B_split", f"fixed_QMDS_B_split[i={key[2]},r={key[3]}]")
        if head == "Borbit":
            family = "reference_additive_orbit_B" if str(key[1]).startswith("add") else "F3_scalar_orbit_B"
            return (family, f"{family}[action={key[1]}; {cell(key[2], key[3])}]")
        return ("other", repr(key))

    def add_eq_cut(key, row, rhs=0):
        if key in existing: return False
        existing.add(key); rr = {int(j): int(v) for j, v in row.items() if v}; bb = int(rhs)
        rf, br = normalized_float_row(rr, bb)
        lazy_eqE.append(rr); lazy_eqR.append(bb); lazy_eqL.append(row_label(key)); lazy_eqF.append(rf); lazy_eqRF.append(br)
        return True

    def add_le_cut(key, row, rhs=0):
        if key in existing: return False
        existing.add(key); rr = {int(j): int(v) for j, v in row.items() if v}; bb = int(rhs)
        rf, br = normalized_float_row(rr, bb)
        lazy_leE.append(rr); lazy_leR.append(bb); lazy_leL.append(row_label(key)); lazy_leF.append(rf); lazy_leRF.append(br)
        return True

    # Base float rows.
    base_eqF = []; base_eqRF = []
    for r, b in zip(base_eqE, base_eqR):
        rf, br = normalized_float_row(r, b); base_eqF.append(rf); base_eqRF.append(br)

    build_s = time.perf_counter() - st
    print(f"  q=3 HxRX 2x5 u={u}, T={t}: raw={raw}, raw-active={Nraw}, orbit-quotient-vars={nvars}, base-eq={len(base_eqE)}, build={build_s:.3f}s", flush=True)
    print("    projection: T=Hamming(2), U=R_X(5)", flush=True)
    print("    Twisted Fourier nonzero character sectors: OFF", flush=True)
    print("    Ordinary b=0 Fourier sector: ON", flush=True)
    print("    character families: 1 total / 0 nontrivial; exact UNSAT = rational Farkas only; NO Z3", flush=True)

    # Mapping of output composition indices under actions.
    def mapped_index_pair(oi, ou, kind, param):
        cT = tc[oi]; cU = uc[ou]
        if kind == "add":
            return oi, ui[permute_comp(cU, UADD_PERM[param])]
        if kind == "scale":
            return ti[permute_comp(cT, TSCALAR_PERM[param])], ui[permute_comp(cU, USCALAR_PERM[param])]
        raise ValueError

    def shell_row0(iw, rw):
        key = ("splitBrow", iw, rw)
        if key in exact_row_cache: return exact_row_cache[key]
        acc = {}
        for oi, cT in enumerate(tc):
            if t - cT[0] != iw: continue
            for ou, cU in enumerate(uc):
                if u - cU[0] != rw: continue
                r0, _ = exact_transform_rows(0, 0, oi, ou)
                acc = add_sparse(acc, r0)
        exact_row_cache[key] = acc
        return acc

    def select_balanced(violations, limit):
        if len(violations) <= limit: return violations
        groups = defaultdict(list)
        for z in violations: groups[z[1][0:3]].append(z)
        for g in groups.values(): g.sort(key=lambda z: z[0], reverse=True)
        quota = max(1, limit // max(1, 2 * len(groups)))
        chosen = []; used = set()
        for g in groups.values():
            for z in g[:quota]: chosen.append(z); used.add(z[1])
        rest = sorted((z for z in violations if z[1] not in used), key=lambda z: z[0], reverse=True)
        chosen.extend(rest[:max(0, limit - len(chosen))])
        return chosen[:limit]

    total_solve = 0.0; roundno = 0; active_method = "highs"
    while True:
        roundno += 1
        eqF = base_eqF + lazy_eqF; eqRF = base_eqRF + lazy_eqRF
        Aeq = rows_to_csr(eqF, nvars); beq = np.asarray(eqRF, float)
        Aub = rows_to_csr(lazy_leF, nvars); bub = np.asarray(lazy_leRF, float) if lazy_leF else None
        opts = {"presolve": True}
        if active_method != "highs-ipm" and simplex_time_limit_s > 0:
            opts["time_limit"] = float(simplex_time_limit_s)
        s0 = time.perf_counter()
        res = linprog(np.zeros(nvars), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                      bounds=numeric_bounds, method=active_method, options=opts)
        dt = time.perf_counter() - s0; total_solve += dt; used_solver = active_method

        if active_method != "highs-ipm" and (res.status != 0 or res.x is None):
            why = "hit time/iteration limit" if res.status == 1 else ("reported infeasible" if res.status == 2 else f"status={res.status}")
            print(f"    round {roundno}: primary {active_method} {why} [{dt:.3f}s]; IPM rescue ...", flush=True)
            r0 = time.perf_counter()
            rescue = linprog(np.zeros(nvars), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                             bounds=numeric_bounds, method="highs-ipm", options={"presolve": True})
            rdt = time.perf_counter() - r0; total_solve += rdt
            if rescue.status == 0 and rescue.x is not None:
                yy = np.asarray(rescue.x, float)
                req = float(np.max(np.abs(Aeq @ yy - beq))) if Aeq is not None else 0.0
                rub = max(0.0, float(np.max(Aub @ yy - bub))) if Aub is not None else 0.0
                if max(req, rub) <= 1e-7:
                    print(f"      IPM RESCUED [{rdt:.3f}s]; maxeq={req:.2e}, maxub={rub:.2e}; sticky IPM", flush=True)
                    active_method = "highs-ipm"
            res = rescue; dt = rdt; used_solver = "highs-ipm(rescue)"

        msg = str(res.message).lower()
        infeas = ((res.status == 2 and "infeasible" in msg) or
                  (res.status == 4 and ("primal_status is infeasible" in msg or "primal status is infeasible" in msg)))
        if infeas:
            print(f"    round {roundno}: numerical infeasibility with eq={len(base_eqE)+len(lazy_eqE)}, le={len(lazy_leE)}; exact Farkas ...", flush=True)
            eqE = base_eqE + lazy_eqE; eqR = base_eqR + lazy_eqR
            f0 = time.perf_counter()
            if farkas_support_report:
                ex, em, cert = exact_farkas_verify(nvars, eqE, eqR, lazy_leE, lazy_leR,
                                                    bounds, var_scale_int, method="highs-ds",
                                                    return_certificate=True)
            else:
                ex, em = exact_farkas_verify(nvars, eqE, eqR, lazy_leE, lazy_leR,
                                             bounds, var_scale_int, method="highs-ds")
                cert = None
            fdt = time.perf_counter() - f0
            info = {"u": u, "rounds": roundno, "eqcuts": len(lazy_eqE), "lecuts": len(lazy_leE),
                    "active": nvars, "equality_rows": len(eqE),
                    "ordinary_inequalities": len(lazy_leE), "twisted_inequalities": 0,
                    "build_s": build_s, "solve_s": total_solve, **em}
            if ex == "unsat":
                if cert is not None:
                    write_farkas_support_report(farkas_support_report, n=n, k=k, d=d, u=u,
                                                eq_rows=eqE, eq_rhs=eqR, eq_labels=base_eqL + lazy_eqL,
                                                le_rows=lazy_leE, le_rhs=lazy_leR, le_labels=lazy_leL,
                                                bounds=bounds, bound_labels=bound_labels,
                                                certificate=cert, meta=em)
                    print(f"      wrote original-system certificate support report: {farkas_support_report}", flush=True)
                print(f"      EXACT_UNSAT by rational Farkas [{fdt:.3f}s]; support={em.get('farkas_support_size', em.get('farkas_numeric_support_1e10','?'))}; bTy={em.get('farkas_original_bTy', em.get('farkas_bTy','?'))}", flush=True)
                print(f"      constraint counts: variables={nvars}; equality rows={len(eqE)}; ordinary inequalities={len(lazy_leE)}; twisted inequalities=0", flush=True)
                return "exact_unsat", info
            print(f"      Farkas NOT verified [{fdt:.3f}s]; reason={em.get('farkas_reason','unknown')}; NO Z3 fallback", flush=True)
            return "unknown", info
        if res.status != 0 or res.x is None:
            return "unknown", {"u": u, "reason": "highs_unknown", "status": int(res.status), "message": str(res.message)}

        y = np.asarray(res.x, float); x = y * var_scale
        B0, B1 = eval_family(0, 0, x)
        violations = []  # (severity,key,type,row,rhs)

        # Ordinary rationality, containment/distance, B commutation, capacities.
        for oi, cT in enumerate(tc):
            wtT = t - cT[0]
            for ou, cU in enumerate(uc):
                wtU = u - cU[0]; w = wtT + wtU
                b0 = float(B0[oi, ou]); b1 = float(B1[oi, ou])
                col = index.get((oi, ou)); aval = norm * x[col] if col is not None else 0.0
                scale = max(1.0, abs(b0), abs(b1), abs(aval))
                key = ("ordinary", "cyclo", oi, ou)
                if key not in existing and abs(b1) / scale > lazy_tol:
                    _, r1 = exact_transform_rows(0, 0, oi, ou)
                    violations.append((abs(b1) / scale, key, "eq", r1, 0))
                if bucket_pairing(cU) != 0:
                    key = ("ordinary", "Bcomm", oi, ou)
                    if key not in existing and abs(b0) / scale > lazy_tol:
                        r0, _ = exact_transform_rows(0, 0, oi, ou)
                        violations.append((abs(b0) / scale, key, "eq", r0, 0))
                    continue
                r0 = None
                if w < d:
                    key = ("ordinary", "distance", oi, ou)
                    diff = b0 - aval
                    if "distance_B_eq_A" not in disabled_families and key not in existing and abs(diff) / max(1.0, abs(b0), abs(aval)) > lazy_tol:
                        r0, _ = exact_transform_rows(0, 0, oi, ou)
                        rr = dict(r0)
                        if col is not None: rr[col] = rr.get(col, 0) - norm
                        violations.append((abs(diff) / max(1.0, abs(b0), abs(aval)), key, "eq", rr, 0))
                else:
                    key = ("ordinary", "contain", oi, ou)
                    sev = (aval - b0) / max(1.0, abs(aval), abs(b0))
                    if "B_ge_A" not in disabled_families and key not in existing and sev > lazy_tol:
                        r0, _ = exact_transform_rows(0, 0, oi, ou)
                        # B>=norm*A  <=> norm*A-B <=0.
                        rr = {j: -v for j, v in r0.items()}
                        if col is not None: rr[col] = rr.get(col, 0) + norm
                        violations.append((sev, key, "le", rr, 0))
                key = ("ordinary", "Bcap", oi, ou)
                cap = bucket_capacity(cT, cU); rhs = norm * cap
                sev = (b0 - rhs) / max(1.0, abs(b0), abs(rhs))
                if key not in existing and sev > lazy_tol:
                    if r0 is None: r0, _ = exact_transform_rows(0, 0, oi, ou)
                    violations.append((sev, key, "le", r0, rhs))

        # Exact split B distribution.
        for iw in range(t + 1):
            for rw in range(u + 1):
                target = norm * split_mds(t, u, d, iw, rw)
                val = sum(float(B0[oi, ou])
                          for oi, cT in enumerate(tc) if t - cT[0] == iw
                          for ou, cU in enumerate(uc) if u - cU[0] == rw)
                key = ("splitB", "eq", iw, rw)
                sev = abs(val - target) / max(1.0, abs(val), abs(float(target)))
                if key not in existing and sev > lazy_tol:
                    violations.append((sev, key, "eq", shell_row0(iw, rw), target))

        # B additive and scalar orbit equalities.
        for alpha in NONZERO:
            for oi in range(len(tc)):
                for ou in range(len(uc)):
                    op, up = mapped_index_pair(oi, ou, "add", alpha)
                    if (oi, ou) >= (op, up): continue
                    key = ("Borbit", f"add{alpha}", oi, ou)
                    if key in existing: continue
                    diff = float(B0[oi, ou] - B0[op, up])
                    sev = abs(diff) / max(1.0, abs(float(B0[oi, ou])), abs(float(B0[op, up])))
                    if sev > lazy_tol:
                        r1, _ = exact_transform_rows(0, 0, oi, ou)
                        r2, _ = exact_transform_rows(0, 0, op, up)
                        violations.append((sev, key, "eq", add_sparse(r1, r2, -1), 0))
        beta = 2
        for oi in range(len(tc)):
            for ou in range(len(uc)):
                op, up = mapped_index_pair(oi, ou, "scale", beta)
                if (oi, ou) >= (op, up): continue
                key = ("Borbit", "scale2", oi, ou)
                if key in existing: continue
                diff = float(B0[oi, ou] - B0[op, up])
                sev = abs(diff) / max(1.0, abs(float(B0[oi, ou])), abs(float(B0[op, up])))
                if sev > lazy_tol:
                    r1, _ = exact_transform_rows(0, 0, oi, ou)
                    r2, _ = exact_transform_rows(0, 0, op, up)
                    violations.append((sev, key, "eq", add_sparse(r1, r2, -1), 0))

        if not violations:
            maxeq = float(np.max(np.abs(Aeq @ y - beq))) if Aeq is not None else 0.0
            maxub = max(0.0, float(np.max(Aub @ y - bub))) if Aub is not None else 0.0
            print(f"    round {roundno}: SAT via {used_solver}; eqcuts={len(lazy_eqE)}, lecuts={len(lazy_leE)}, total-solve={total_solve:.3f}s, maxeq={maxeq:.2e}, maxub={maxub:.2e}", flush=True)
            print(f"      constraint counts: variables={nvars}; equality rows={len(base_eqE)+len(lazy_eqE)}; ordinary inequalities={len(lazy_leE)}; twisted inequalities=0", flush=True)
            return "sat", {"u": u, "active": nvars, "rounds": roundno,
                           "eqcuts": len(lazy_eqE), "lecuts": len(lazy_leE),
                           "equality_rows": len(base_eqE)+len(lazy_eqE),
                           "ordinary_inequalities": len(lazy_leE), "twisted_inequalities": 0,
                           "build_s": build_s, "solve_s": total_solve,
                           "maxeq": maxeq, "maxub": maxub}

        selected = select_balanced(violations, max(1, int(lazy_cuts)))
        neq = nle = 0
        for _, key, typ, row, rhs in selected:
            if typ == "eq": neq += int(add_eq_cut(key, row, rhs))
            else: nle += int(add_le_cut(key, row, rhs))
        fams = len(set(z[1][0:2] for z in violations))
        vmax = max(z[0] for z in violations)
        print(f"    round {roundno}: feasible via {used_solver}; violations={len(violations)} across {fams} groups, add eq={neq} le={nle}, max={vmax:.2e}, totals eq/le={len(lazy_eqE)}/{len(lazy_leE)} [{dt:.3f}s]", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="q=3 QMDS Hamming(2)xR_X(5) real-LP with exact Farkas, no Z3")
    p.add_argument("q", type=int)
    p.add_argument("n", type=int); p.add_argument("k", type=int); p.add_argument("d", type=int)
    p.add_argument("--u", type=int, action="append", default=None)
    p.add_argument("--u-order", choices=("ascending", "descending"), default="ascending")
    p.add_argument("--continue-after-unsat", action="store_true")
    p.add_argument("--lazy-cuts", type=int, default=512)
    p.add_argument("--lazy-tol", type=float, default=1e-8)
    p.add_argument("--simplex-time-limit", type=float, default=3.0)
    p.add_argument("--max-active-vars", type=int, default=30000)
    p.add_argument("--max-raw-cells", type=int, default=500000)
    p.add_argument("--no-farkas-support-report", action="store_true",
                   help="do not write the original-system exact Farkas support report")
    p.add_argument("--disable-family", action="append", default=[],
                   choices=("fixed_QMDS_A_split", "distance_B_eq_A", "B_ge_A", "nonnegativity_A"),
                   help="analysis-only: remove one labeled family; baseline is unchanged when omitted")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.q != 3: raise SystemExit("This 2x5 HxRX solver is q=3-specific; pass q=3.")
    n, k, d = a.n, a.k, a.d
    expected = n - 2 * (d - 1)
    if k != expected: raise SystemExit(f"Not QMDS parameters: require k=n-2(d-1)={expected}, got {k}")
    D_A = n - d + 2
    Avec = mds_distribution(n, D_A)
    if sum(Avec) != Q ** (n - k): raise AssertionError("QMDS stabilizer enumerator cardinality mismatch")
    us = [u for u in range(1, n + 1) if Avec[u] > 0]
    if a.u is not None:
        wanted = set(a.u); us = [u for u in us if u in wanted]
    us.sort(reverse=(a.u_order == "descending"))

    print("=" * 112)
    print("q=3 QMDS HxRX Hamming(2) x R_X(5) STABILIZER-REFERENCE REAL-LP")
    print(f"Code: [[{n},{k},{d}]]_3; D_A={D_A}; mandatory u={us}")
    print("Projection: T=Hamming {I}/non-I (2), U=aligned R_X(5)")
    print("Twisted Fourier nonzero character sectors: OFF")
    print("Ordinary b=0 Fourier sector: ON")
    print("Backend: bounded HiGHS -> IPM rescue -> exact rational Farkas; NO Z3; no integrality/mod/lattice")
    print("=" * 112)
    results = []; final = "sat"
    for u in us:
        report = None if a.no_farkas_support_report else f"farkas_support_q3_{n}_{k}_{d}_u{u}.txt"
        st, info = solve_u(n, k, d, u, lazy_cuts=a.lazy_cuts, lazy_tol=a.lazy_tol,
                           simplex_time_limit_s=a.simplex_time_limit,
                           max_active_vars=a.max_active_vars, max_raw_cells=a.max_raw_cells,
                           farkas_support_report=report, disabled_families=a.disable_family)
        results.append((u, st, info))
        if st == "exact_unsat":
            final = "exact_unsat"
            if not a.continue_after_unsat: break
        elif st == "unknown" and final != "exact_unsat":
            final = "unknown"
    print("=" * 112)
    print(f"FINAL: {final.upper()}")
    for u, st, info in results: print(f"  u={u}: {st.upper()}  {info}")
    print("=" * 112)


if __name__ == "__main__":
    main()
