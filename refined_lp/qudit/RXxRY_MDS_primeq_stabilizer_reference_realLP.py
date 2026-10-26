#!/usr/bin/env python3
r"""
RXxRY_MDS_qudit_stabilizer_reference_realLP_optimized.py
Version: 2026-09-25-r13-adaptive-ipm-matched-exact

Optimized continuous real-LP necessary-condition solver for q-ary QMDS
stabilizer codes with one mandatory stabilizer reference

    g = I^T X^U in S,   |U|=u,

using the joint line projection

    T : I / X-line* / off-X,
    U : I / Y-line* / off-Y.

Supported q: 3,4,5,7.  For prime q, F_q-linearity is automatic for an additive
stabilizer.  For q=4 the default is general additive (F_2-linear); pass
--fq-linear only when restricting to the F_4-linear subclass.

This program is deliberately a REAL relaxation: no A/B integrality, no Mod,
no lattice arithmetic, no divisibility.  It keeps only rigorously valid
linear necessary conditions at this projection:

  * exact QMDS split A distribution (q^2-ary MDS enumerator);
  * exact joint projected MacWilliams transform;
  * exact split-B distribution checked algebraically, but redundant split-B
    rows are NOT inserted into HiGHS;
  * B >= A coefficientwise and B=A below distance d;
  * exact reference-bucket existence;
  * exact trace-symplectic commuting capacities for A and B because g in S;
  * optional F_q-linear commuting-capacity strengthening;
  * generalized Rains shadow positivity at split-Hamming resolution;
  * cleaning/dimension identities for correctable T/U blocks.

Optimization choices:

  * only positive QMDS A-support cells are variables; zero split shells and
    zero commuting-capacity cells are removed before any matrix is built;
  * block transform matrices are cached once per block size;
  * no exact RREF and no SymPy objects;
  * sparse rows are assembled directly to CSR;
  * no manual gc.collect() inside a solve;
  * HiGHS infeasibility is only a NUMERICAL-UNSAT screen;
  * numerical UNSAT is first certified by an exact rational Farkas certificate;
  * no automatic Z3 fallback; exact UNSAT requires a verified rational Farkas certificate;
  * SAT solutions are checked for numerical residuals.

Important scope:
The 3-class line projection is the rigorously closed q-ary joint RXxRY
projection.  It is NOT the much larger (2q-1)x(2q-1) fully character-resolved
projection.  Nonzero character-twist inequalities are therefore not invented
at this coarse resolution.

Examples:
  python RXxRY_MDS_qudit_stabilizer_reference_realLP_optimized.py 4 10 2 5
  python RXxRY_MDS_qudit_stabilizer_reference_realLP_optimized.py 3 6 2 3
  python RXxRY_MDS_qudit_stabilizer_reference_realLP_optimized.py 5 6 2 3
  python RXxRY_MDS_qudit_stabilizer_reference_realLP_optimized.py 4 10 2 5 --u 7

Dependencies:
  pip install numpy scipy z3-solver
"""

from __future__ import annotations

import argparse
import math
import time
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

try:
    import z3
except ImportError:
    z3 = None

Pair = Tuple[int, int]                 # (line*, off-line)
Cell = Tuple[int, int, int, int]       # (Tx, Toff, Uy, Uoff)
SparseRow = Dict[int, float]
ExactRow = Dict[int, int]

SUPPORTED_Q = (3, 4, 5, 7)


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    if x % 2 == 0:
        return x == 2
    r = int(math.isqrt(x))
    return all(x % p for p in range(3, r + 1, 2))


def characteristic(q: int) -> int:
    if q == 4:
        return 2
    if is_prime(q):
        return q
    raise ValueError(f"Unsupported q={q}")


def effective_fq_linear(q: int, requested: bool) -> bool:
    return is_prime(q) or bool(requested)



# =============================================================================
# Exact Farkas certification matched to the normalized primal
# =============================================================================
from fractions import Fraction
from scipy.sparse import hstack as _sp_hstack, vstack as _sp_vstack

_FARKAS_SUPPORT_TOLS = (1e-8, 1e-10, 1e-12, 1e-14, 1e-16)

def _fx_items(row):
    return row.items() if isinstance(row, dict) else enumerate(row)

def _fx_csr(rows, nv):
    data=[]; inds=[]; indptr=[0]
    for row in rows:
        for j,v in sorted((int(j),float(v)) for j,v in _fx_items(row) if v):
            data.append(v); inds.append(j)
        indptr.append(len(data))
    return csr_matrix((np.asarray(data,float),np.asarray(inds,np.int32),
                      np.asarray(indptr,np.int64)), shape=(len(rows),nv))

def _normalized_exact_rows(rows, rhs, var_scale_int):
    """Exact rational form of the SAME numerical row normalization used by primal HiGHS.

    x_j = S_j y_j, then each row is divided by
      R_i=max(1, |b_i|, max_j |a_ij S_j|).
    Returns (rows_Q, rhs_Q, row_scales R_i).
    """
    out=[]; br=[]; scales=[]
    for row,b in zip(rows,rhs):
        nums={int(j): int(v)*int(var_scale_int[int(j)]) for j,v in _fx_items(row) if v}
        R=max([1,abs(int(b))]+[abs(v) for v in nums.values()])
        out.append({j:Fraction(v,R) for j,v in nums.items() if v})
        br.append(Fraction(int(b),R))
        scales.append(int(R))
    return out,br,scales

def _build_normalized_exact_system(nv, eq_rows, eq_rhs, le_rows, le_rhs, bounds, var_scale_int):
    E,e,ers=_normalized_exact_rows(eq_rows,eq_rhs,var_scale_int)
    C,c,crs=_normalized_exact_rows(le_rows,le_rhs,var_scale_int)
    # Bounds in y_j=x_j/S_j coordinates.
    bound_meta=[]  # (kind,j,S) for mapping back to original system
    for j,(lo,hi) in enumerate(bounds):
        S=int(var_scale_int[j])
        if lo is not None:
            C.append({j:Fraction(-1,1)})
            c.append(Fraction(-int(lo),S))
            crs.append(S)
            bound_meta.append(("lo",j,S))
        if hi is not None:
            C.append({j:Fraction(1,1)})
            c.append(Fraction(int(hi),S))
            crs.append(S)
            bound_meta.append(("hi",j,S))
    return E,e,ers,C,c,crs,bound_meta

def _rows_fraction_to_csr(rows,nv):
    data=[]; inds=[]; indptr=[0]
    for row in rows:
        for j,v in sorted(row.items()):
            if v:
                data.append(float(v)); inds.append(int(j))
        indptr.append(len(data))
    return csr_matrix((np.asarray(data,float),np.asarray(inds,np.int32),
                      np.asarray(indptr,np.int64)),shape=(len(rows),nv))

def _exact_verify_transformed(nv,E,e,C,c,wp,wm,y):
    if any(v<0 for v in wp) or any(v<0 for v in wm) or any(v<0 for v in y):
        return False,{"farkas_reason":"negative_exact_multiplier"}
    r=[Fraction(0,1) for _ in range(nv)]
    obj=Fraction(0,1)
    for ap,am,row,b in zip(wp,wm,E,e):
        lam=ap-am
        if lam:
            for j,v in row.items(): r[j]+=lam*v
            obj+=lam*b
    for lam,row,b in zip(y,C,c):
        if lam:
            for j,v in row.items(): r[j]+=lam*v
            obj+=lam*b
    if any(v!=0 for v in r):
        return False,{"farkas_reason":"nonzero_exact_stationarity",
                     "farkas_residual_nnz":sum(v!=0 for v in r)}
    mass=sum(wp,Fraction(0,1))+sum(wm,Fraction(0,1))+sum(y,Fraction(0,1))
    if mass!=1:
        return False,{"farkas_reason":"bad_exact_normalization","farkas_mass":str(mass)}
    if not obj<0:
        return False,{"farkas_reason":"nonnegative_exact_objective","farkas_bTy":str(obj)}
    den=max([1]+[v.denominator for v in list(wp)+list(wm)+list(y)])
    return True,{"farkas_bTy":str(obj),"farkas_max_denominator":int(den),
                "farkas_split_eq_support":sum(v!=0 for v in wp)+sum(v!=0 for v in wm),
                "farkas_ineq_support":sum(v!=0 for v in y)}

def _exact_verify_original(nv,eq_rows,eq_rhs,le_rows,le_rhs,bounds,
                           ers,crs,wp,wm,y):
    """Map transformed multipliers back and verify the ORIGINAL exact constraints."""
    z=[(wp[i]-wm[i])/Fraction(int(ers[i]),1) for i in range(len(eq_rows))]
    nbase=len(le_rows)
    mu=[y[i]/Fraction(int(crs[i]),1) for i in range(nbase)]
    # Remaining y entries are bounds, in the same lo/hi order used below.
    r=[Fraction(0,1) for _ in range(nv)]
    obj=Fraction(0,1)
    for lam,row,b in zip(z,eq_rows,eq_rhs):
        if lam:
            for j,v in _fx_items(row):
                if v: r[int(j)]+=lam*Fraction(int(v),1)
            obj+=lam*Fraction(int(b),1)
    for lam,row,b in zip(mu,le_rows,le_rhs):
        if lam:
            for j,v in _fx_items(row):
                if v: r[int(j)]+=lam*Fraction(int(v),1)
            obj+=lam*Fraction(int(b),1)
    p=nbase
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None:
            lam=y[p]/Fraction(int(crs[p]),1); p+=1
            r[j]-=lam; obj+=lam*Fraction(-int(lo),1)
        if hi is not None:
            lam=y[p]/Fraction(int(crs[p]),1); p+=1
            r[j]+=lam; obj+=lam*Fraction(int(hi),1)
    if any(v!=0 for v in r):
        return False,{"farkas_reason":"original_stationarity_failed",
                     "farkas_original_residual_nnz":sum(v!=0 for v in r)}
    if not obj<0:
        return False,{"farkas_reason":"original_objective_nonnegative","farkas_original_bTy":str(obj)}
    return True,{"farkas_original_bTy":str(obj)}

def _exact_support_reconstruct(nv,E,e,C,c,wf):
    try:
        import sympy as sp
    except Exception:
        return False,{"farkas_reason":"sympy_unavailable"},None
    me=len(E); mi=len(C)
    wf=np.asarray(wf,float)
    mx=max(1.0,float(np.max(np.abs(wf))) if wf.size else 0.0)
    last={"farkas_reason":"exact_support_not_recovered"}

    def get_col(idx):
        if idx<me:
            row=E[idx]; sgn=1
        elif idx<2*me:
            row=E[idx-me]; sgn=-1
        else:
            row=C[idx-2*me]; sgn=1
        col=[sp.Rational(0)]*(nv+1)
        for j,v in row.items():
            col[j]=sp.Rational(sgn*v.numerator,v.denominator)
        col[-1]=sp.Rational(1)
        return col

    for tol in _FARKAS_SUPPORT_TOLS:
        supp=[i for i,v in enumerate(wf) if v>tol*mx]
        ns=len(supp)
        if not supp:
            last={"farkas_reason":"empty_support","farkas_support_tol":tol}
            continue
        if ns>nv+8:
            last={"farkas_reason":"support_too_large","farkas_support_size":ns,
                  "farkas_support_tol":tol}
            continue
        cols=[get_col(i) for i in supp]
        M=sp.Matrix(nv+1,ns,lambda r,col: cols[col][r])
        b=sp.Matrix([0]*nv+[1])
        try:
            sol,params=M.gauss_jordan_solve(b)
        except Exception as exc:
            last={"farkas_reason":"exact_support_inconsistent","farkas_support_size":ns,
                  "farkas_support_tol":tol,"farkas_exact_error":type(exc).__name__}
            continue
        plist=list(params)
        trials=[]
        if not plist:
            trials=[{}]
        else:
            zero={p:sp.Rational(0) for p in plist}
            x0=sp.Matrix([sp.cancel(v.subs(zero)) for v in sol])
            N=sp.Matrix(ns,len(plist),lambda i,j: sp.cancel(sp.diff(sol[i],plist[j])))
            try:
                Nf=np.asarray(N.evalf(),dtype=float)
                x0f=np.asarray(x0.evalf(),dtype=float).reshape(-1)
                tau=np.linalg.lstsq(Nf,wf[supp]-x0f,rcond=None)[0]
            except Exception:
                tau=np.zeros(len(plist))
            for md in (10**6,10**9,2**31,2**40):
                sub={}
                for p0,v0 in zip(plist,tau):
                    fr=Fraction(float(v0)).limit_denominator(md)
                    sub[p0]=sp.Rational(fr.numerator,fr.denominator)
                trials.append(sub)
        for sub in trials:
            vals=[sp.cancel(v.subs(sub)) for v in sol]
            try:
                qv=[Fraction(int(sp.numer(v)),int(sp.denom(v))) for v in vals]
            except Exception:
                continue
            if any(v<0 for v in qv):
                last={"farkas_reason":"negative_exact_support_value","farkas_support_size":ns,
                      "farkas_support_tol":tol}
                continue
            wp=[Fraction(0,1) for _ in range(me)]
            wm=[Fraction(0,1) for _ in range(me)]
            yy=[Fraction(0,1) for _ in range(mi)]
            for idx,v in zip(supp,qv):
                if idx<me: wp[idx]=v
                elif idx<2*me: wm[idx-me]=v
                else: yy[idx-2*me]=v
            ok,meta=_exact_verify_transformed(nv,E,e,C,c,wp,wm,yy)
            meta.update({"farkas_support_size":ns,"farkas_support_tol":tol,
                         "farkas_reconstruction":"matched-normalized-support"})
            if ok:
                return True,meta,(wp,wm,yy)
            last=meta
    return False,last,None

def exact_farkas_verify(nv,eq_rows,eq_rhs,le_rows,le_rhs,bounds,var_scale_int,method="highs"):
    """Numerical finder and exact verifier on the exact normalized primal system."""
    E,e,ers,C,c,crs,_=_build_normalized_exact_system(
        nv,eq_rows,eq_rhs,le_rows,le_rhs,bounds,var_scale_int)
    me=len(E); mi=len(C)
    Ef=_rows_fraction_to_csr(E,nv) if me else csr_matrix((0,nv),dtype=float)
    Cf=_rows_fraction_to_csr(C,nv) if mi else csr_matrix((0,nv),dtype=float)
    station=_sp_hstack([Ef.T,-Ef.T,Cf.T],format="csr")
    norm=csr_matrix(np.ones((1,2*me+mi),dtype=float))
    Aeq=_sp_vstack([station,norm],format="csr")
    beq=np.zeros(nv+1,float); beq[-1]=1.0
    rhs=np.asarray([float(v) for v in e]+[-float(v) for v in e]+[float(v) for v in c],dtype=float)
    obj_scale=max(1.0,float(np.max(np.abs(rhs))) if rhs.size else 1.0)
    cnum=rhs/obj_scale
    t0=time.perf_counter()
    rr=linprog(cnum,A_eq=Aeq,b_eq=beq,bounds=[(0.0,None)]*(2*me+mi),
               method=method,options={"presolve":True})
    dt=time.perf_counter()-t0
    meta={"farkas_numeric_runtime_s":dt,"farkas_numeric_status":int(rr.status),
          "farkas_cone_vars":2*me+mi,"farkas_objective_scale":obj_scale}
    if rr.status!=0 or rr.x is None:
        meta["farkas_reason"]="numeric_farkas_cone_failed"
        return "unknown",meta
    obj=float(cnum@np.asarray(rr.x,float))
    meta["farkas_numeric_objective_scaled"]=obj
    meta["farkas_numeric_objective_unscaled"]=obj*obj_scale
    meta["farkas_numeric_support_1e10"]=int(np.sum(np.asarray(rr.x)>1e-10))
    if not obj<0.0:
        meta["farkas_reason"]="no_negative_farkas_objective"
        return "unknown",meta
    ok,xmeta,cert=_exact_support_reconstruct(nv,E,e,C,c,rr.x)
    xmeta.update(meta)
    xmeta["exact_runtime_s"]=time.perf_counter()-t0
    if not ok or cert is None:
        return "unknown",xmeta
    wp,wm,yy=cert
    ok2,ometa=_exact_verify_original(nv,eq_rows,eq_rhs,le_rows,le_rhs,bounds,
                                     ers,crs,wp,wm,yy)
    xmeta.update(ometa)
    if not ok2:
        return "unknown",xmeta
    xmeta["exact_backend"]="farkas-matched-normalized-exact"
    return "unsat",xmeta

# =============================================================================
# QMDS / q^2-ary MDS distributions
# =============================================================================

@lru_cache(maxsize=None)
def mds_exact_support(q: int, D: int, w: int) -> int:
    """Number of MDS words with a fixed support of size w over alphabet q^2."""
    if w == 0:
        return 1
    if w < D:
        return 0
    Q2 = q * q
    return (Q2 - 1) * sum(
        (-1) ** s * math.comb(w - 1, s) * Q2 ** (w - D - s)
        for s in range(w - D + 1)
    )


def mds_distribution(q: int, n: int, D: int) -> List[int]:
    return [math.comb(n, w) * mds_exact_support(q, D, w) for w in range(n + 1)]


def split_mds(q: int, n: int, t: int, u: int, D: int, i: int, r: int) -> int:
    assert t + u == n
    return math.comb(t, i) * math.comb(u, r) * mds_exact_support(q, D, i + r)


# =============================================================================
# Joint line geometry and capacities
# =============================================================================

def pair_cells(m: int) -> Tuple[Pair, ...]:
    return tuple((a, b) for a in range(m + 1) for b in range(m - a + 1))


def raw_cells(t: int, u: int) -> Tuple[Cell, ...]:
    return tuple((a, b, j, l) for a, b in pair_cells(t) for j, l in pair_cells(u))


def multinomial3(z: int, x: int, y: int) -> int:
    if min(z, x, y) < 0:
        return 0
    return math.factorial(z + x + y) // (
        math.factorial(z) * math.factorial(x) * math.factorial(y)
    )


def split_weight(c: Cell) -> Tuple[int, int]:
    return c[0] + c[1], c[2] + c[3]


def total_weight(c: Cell) -> int:
    return sum(c)


def block_ambient_capacity(q: int, m: int, line: int, off: int) -> int:
    z = m - line - off
    return (
        multinomial3(z, line, off)
        * (q - 1) ** line
        * (q * (q - 1)) ** off
    )


def u_commuting_capacity(q: int, u: int, j: int, l: int, fq_linear: bool) -> int:
    """
    Number of U labels in a Y-line/off-Y composition whose total symplectic
    pairing with X^U has trace zero (general additive stabilizer), or is zero
    in F_q itself (F_q-linear stabilizer).

    Let N1=q-1 and N2=q(q-1).  For every nontrivial additive character,
    the character sum over Y-line* is -1 and over off-Y is 0.  Therefore

      trace-zero = [N1^j N2^l + (p-1)(-1)^j 0^l]/p,

    with p=char(F_q).  F_q-linearity upgrades p to q.
    """
    z = u - j - l
    if z < 0:
        return 0
    denom = q if fq_linear else characteristic(q)
    correction = 1 if l == 0 else 0
    phase_zero = (
        (q - 1) ** j * (q * (q - 1)) ** l
        + (denom - 1) * ((-1) ** j) * correction
    )
    if phase_zero % denom:
        raise AssertionError("Commuting capacity numerator is not divisible by field/trace size.")
    return multinomial3(z, j, l) * (phase_zero // denom)


def joint_commuting_capacity(q: int, t: int, u: int, c: Cell, fq_linear: bool) -> int:
    a, b, j, l = c
    return (
        block_ambient_capacity(q, t, a, b)
        * u_commuting_capacity(q, u, j, l, fq_linear)
    )


# =============================================================================
# Exact projected Fourier/MacWilliams matrices
# =============================================================================

def _poly_mul(poly: Mapping[Tuple[int, int, int], int], linear: Sequence[int]):
    out: Dict[Tuple[int, int, int], int] = {}
    for exp, coeff in poly.items():
        for pos, alpha in enumerate(linear):
            if alpha == 0:
                continue
            ee = list(exp)
            ee[pos] += 1
            key = tuple(ee)
            out[key] = out.get(key, 0) + coeff * int(alpha)
    return out


@lru_cache(maxsize=None)
def r_matrix(q: int, m: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Composition MacWilliams matrix for I / line* / off-line.

    Local input forms (output semantic order I,line*,off):
      I       -> y0 + (q-1)y1 + q(q-1)y2
      line*   -> y0 + (q-1)y1 - q y2
      off     -> y0 - y1
    """
    cells = pair_cells(m)
    M = [[0] * len(cells) for _ in cells]
    forms = (
        (1, q - 1, q * (q - 1)),
        (1, q - 1, -q),
        (1, -1, 0),
    )
    for ic, (line, off) in enumerate(cells):
        poly: Dict[Tuple[int, int, int], int] = {(0, 0, 0): 1}
        for L, power in (
            (forms[0], m - line - off),
            (forms[1], line),
            (forms[2], off),
        ):
            for _ in range(power):
                poly = _poly_mul(poly, L)
        for oc, (oline, ooff) in enumerate(cells):
            M[oc][ic] = poly.get((m - oline - ooff, oline, ooff), 0)
    return tuple(tuple(row) for row in M)


@lru_cache(maxsize=None)
def generalized_shadow_matrix(q: int, m: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Generalized Rains split-Hamming shadow transform used by the existing
    q-ary HxR real-LP code.  Row=output Hamming weight, col=input Hamming weight.

      I-input       -> (q-1)x + (q+1)y
      nonzero-input -> -x + y.
    """
    rows: List[Tuple[int, ...]] = []
    for iw in range(m + 1):
        poly: Dict[Tuple[int, int], int] = {(0, 0): 1}
        factors = [((q - 1, q + 1), m - iw), ((-1, 1), iw)]
        for linear, power in factors:
            for _ in range(power):
                nxt: Dict[Tuple[int, int], int] = {}
                for (e0, e1), cc in poly.items():
                    for pos, aa in enumerate(linear):
                        if aa == 0:
                            continue
                        key = (e0 + (pos == 0), e1 + (pos == 1))
                        nxt[key] = nxt.get(key, 0) + cc * aa
                poly = nxt
        col = [poly.get((m - ow, ow), 0) for ow in range(m + 1)]
        # temporary columns; transpose below
        rows.append(tuple(col))
    return tuple(tuple(rows[iw][ow] for iw in range(m + 1)) for ow in range(m + 1))


def _block_krawtchouk_from_line_matrix(q: int, s: int, out_w: int, in_w: int) -> int:
    cells = pair_cells(s)
    idx = {c: i for i, c in enumerate(cells)}
    vals = {
        sum(r_matrix(q, s)[idx[out]][idx[inc]] for out in cells if sum(out) == out_w)
        for inc in cells if sum(inc) == in_w
    }
    if len(vals) != 1:
        raise AssertionError("Line projection does not collapse to Hamming by shell.")
    return next(iter(vals))


def verify_split_b_regression(q: int, n: int, k: int, d: int, u: int) -> None:
    """Exact integer regression: fixed split-A must imply analytic split-B totals."""
    t = n - u
    D_A = n - d + 2
    norm = q ** (n - k)
    for ot in range(t + 1):
        for ou in range(u + 1):
            num = 0
            for it in range(t + 1):
                kt = _block_krawtchouk_from_line_matrix(q, t, ot, it)
                for iu in range(u + 1):
                    ku = _block_krawtchouk_from_line_matrix(q, u, ou, iu)
                    num += kt * ku * split_mds(q, n, t, u, D_A, it, iu)
            target = norm * split_mds(q, n, t, u, d, ot, ou)
            if num != target:
                raise AssertionError(
                    f"split-B regression failed at ({ot},{ou}): transform={num}, target={target}"
                )


# =============================================================================
# Sparse LP helper
# =============================================================================

class LPBuilder:
    def __init__(self, nvars: int):
        self.nvars = int(nvars)
        self.eq_rows: List[SparseRow] = []
        self.eq_rhs: List[float] = []
        self.ub_rows: List[SparseRow] = []
        self.ub_rhs: List[float] = []

    @staticmethod
    def _scaled(row: SparseRow, rhs: float):
        row = {int(j): float(v) for j, v in row.items() if v != 0}
        scale = max([1.0, abs(float(rhs))] + [abs(v) for v in row.values()])
        return {j: v / scale for j, v in row.items()}, float(rhs) / scale

    def add_eq(self, row: SparseRow, rhs: float):
        r, b = self._scaled(row, rhs)
        self.eq_rows.append(r)
        self.eq_rhs.append(b)

    def add_le(self, row: SparseRow, rhs: float):
        r, b = self._scaled(row, rhs)
        self.ub_rows.append(r)
        self.ub_rhs.append(b)

    def add_ge(self, row: SparseRow, rhs: float):
        self.add_le({j: -v for j, v in row.items()}, -rhs)

    def _csr(self, rows: Sequence[SparseRow]):
        data: List[float] = []
        indices: List[int] = []
        indptr = [0]
        for row in rows:
            for j, v in sorted(row.items()):
                data.append(float(v))
                indices.append(int(j))
            indptr.append(len(data))
        return csr_matrix(
            (np.asarray(data, dtype=float), np.asarray(indices, dtype=np.int32),
             np.asarray(indptr, dtype=np.int64)),
            shape=(len(rows), self.nvars),
        )

    def matrices(self):
        Aeq = self._csr(self.eq_rows) if self.eq_rows else None
        Aub = self._csr(self.ub_rows) if self.ub_rows else None
        beq = np.asarray(self.eq_rhs, dtype=float) if self.eq_rhs else None
        bub = np.asarray(self.ub_rhs, dtype=float) if self.ub_rhs else None
        return Aeq, beq, Aub, bub


# =============================================================================
# Exact UNSAT verification (QF_LRA over rationals)
# =============================================================================

def _z3_num(v):
    """Convert an exact Python integer/rational-like coefficient to Z3 RealVal."""
    from fractions import Fraction
    if isinstance(v, (int, np.integer)):
        return z3.IntVal(int(v))
    if isinstance(v, Fraction):
        return z3.RealVal(f"{v.numerator}/{v.denominator}")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"non-finite exact coefficient {v}")
        if v.is_integer():
            return z3.IntVal(int(v))
        # This path should not be used by the exact MDS construction, whose
        # coefficients are integers.  Keep it deterministic rather than binary-float exact.
        return z3.RealVal(str(v))
    return z3.RealVal(str(v))


def exact_lra_verify(
    nvars: int,
    eq_rows: Sequence[Mapping[int, object]],
    eq_rhs: Sequence[object],
    le_rows: Sequence[Mapping[int, object]],
    le_rhs: Sequence[object],
    bounds: Sequence[Tuple[object, Optional[object]]],
    *, timeout_ms: int = 0,
):
    """Exactly decide the finite LP subsystem currently used by HiGHS.

    Returns ("unsat"|"sat"|"unknown"|"unavailable", metadata).
    Z3's QF_LRA arithmetic is exact over rationals, so an UNSAT result is a
    rigorous certificate of emptiness of this real polyhedron.
    """
    if z3 is None:
        return "unavailable", {"reason": "z3-solver not installed"}
    xs = [z3.Real(f"x_{j}") for j in range(nvars)]
    solver = z3.SolverFor("QF_LRA")
    if timeout_ms and timeout_ms > 0:
        solver.set(timeout=int(timeout_ms))

    def lin(row):
        terms = [_z3_num(v) * xs[int(j)] for j, v in row.items() if v != 0]
        return z3.Sum(terms) if terms else z3.RealVal(0)

    for row, rhs in zip(eq_rows, eq_rhs):
        solver.add(lin(row) == _z3_num(rhs))
    for row, rhs in zip(le_rows, le_rhs):
        solver.add(lin(row) <= _z3_num(rhs))
    for j, (lb, ub) in enumerate(bounds):
        if lb is not None:
            solver.add(xs[j] >= _z3_num(lb))
        if ub is not None:
            solver.add(xs[j] <= _z3_num(ub))

    t0 = time.perf_counter()
    result = solver.check()
    dt = time.perf_counter() - t0
    meta = {
        "exact_backend": "z3-qf_lra",
        "exact_runtime_s": dt,
        "exact_eq_rows": len(eq_rows),
        "exact_le_rows": len(le_rows),
        "exact_vars": nvars,
    }
    if result == z3.unsat:
        return "unsat", meta
    if result == z3.sat:
        return "sat", meta
    meta["reason"] = solver.reason_unknown()
    return "unknown", meta


# =============================================================================
# One mandatory stabilizer-reference shell
# =============================================================================

def solve_reference_u(
    q: int, n: int, k: int, d: int, u: int, *,
    fq_linear: bool = False,
    use_shadow: bool = True,
    use_cleaning: bool = True,
    use_capacities: bool = True,
    highs_method: str = "highs",
    simplex_time_limit_s: float = 3.0,
    lazy_cuts_per_round: int = 96,
    lazy_tol: float = 1e-8,
    exact_timeout_ms: int = 0,
):
    """One mandatory reference shell with lazy B-containment/capacity separation."""
    start = time.perf_counter()
    t = n - u
    D_A = n - d + 2
    norm = q ** (n - k)
    fq_linear = effective_fq_linear(q, fq_linear)

    verify_split_b_regression(q, n, k, d, u)

    tcells = pair_cells(t)
    ucells = pair_cells(u)
    all_raw = raw_cells(t, u)

    # MDS active-support reduction.
    active: List[Cell] = []
    upper: Dict[Cell, int] = {}
    for c in all_raw:
        iw, rw = split_weight(c)
        shell = split_mds(q, n, t, u, D_A, iw, rw)
        if shell == 0:
            continue
        cap = joint_commuting_capacity(q, t, u, c, fq_linear) if use_capacities else shell
        if cap == 0:
            continue
        ub = int(min(shell, cap) if use_capacities else shell)
        if ub > 0:
            active.append(c)
            upper[c] = ub

    index = {c: j for j, c in enumerate(active)}
    nvars = len(active)
    if nvars == 0:
        return "exact_unsat", {"u": u, "reason": "no_active_A_support"}

    ref = (0, 0, 0, u)
    if ref not in index:
        return "exact_unsat", {"u": u, "reason": "reference_bucket_pruned"}
    ref_mult = q - 1 if fq_linear else 1

    bounds: List[Tuple[float, Optional[float]]] = []
    for c in active:
        lb = int(ref_mult) if c == ref else 0
        ub = int(upper[c])
        if lb > ub:
            return "exact_unsat", {"u": u, "reason": "reference_capacity", "lb": lb, "ub": ub}
        bounds.append((lb, ub))

    # Numerical variables are y_j = x_j / var_scale_j, typically in [0,1].
    # This avoids count variables of size 1e9--1e12 next to O(1) coefficients.
    var_scale_int = [max(1, int(ub)) for (lb,ub) in bounds]
    var_scale = np.asarray(var_scale_int, dtype=float)
    numeric_bounds = [
        ((None if lb is None else float(lb)/var_scale[j]),
         (None if ub is None else float(ub)/var_scale[j]))
        for j,(lb,ub) in enumerate(bounds)
    ]

    ti = {c: i for i, c in enumerate(tcells)}
    ui = {c: i for i, c in enumerate(ucells)}
    TM = r_matrix(q, t)
    UM = r_matrix(q, u)

    # Cache all B rows over the reduced active variables.
    Brow_cache: Dict[Cell, SparseRow] = {}
    def B_row(out: Cell) -> SparseRow:
        row = Brow_cache.get(out)
        if row is not None:
            return row
        oa, ob, oj, ol = out
        ot = ti[(oa, ob)]
        ou = ui[(oj, ol)]
        rr: SparseRow = {}
        for c, col in index.items():
            a, b, j, l = c
            coeff = TM[ot][ti[(a, b)]] * UM[ou][ui[(j, l)]]
            if coeff:
                rr[col] = int(coeff)
        Brow_cache[out] = rr
        return rr

    # ---------- exact constant prechecks ----------
    if use_shadow:
        ST = generalized_shadow_matrix(q, t)
        SU = generalized_shadow_matrix(q, u)
        for ot in range(t + 1):
            for ou in range(u + 1):
                fixed_value = sum(
                    ST[ot][iw] * SU[ou][rw]
                    * split_mds(q, n, t, u, D_A, iw, rw)
                    for iw in range(t + 1)
                    for rw in range(u + 1)
                )
                if fixed_value < 0:
                    return "exact_unsat", {
                        "u": u, "reason": "fixed_split_shadow_negative",
                        "output": (ot, ou), "value": fixed_value,
                    }

    if use_cleaning:
        T_only = sum(split_mds(q, n, t, u, D_A, iw, 0) for iw in range(t + 1))
        U_only = sum(split_mds(q, n, t, u, D_A, 0, rw) for rw in range(u + 1))
        def cleaning_ok(csize: int, other: int, corr: int) -> bool:
            e = n - k - 2 * csize
            return (other == (q ** e) * corr) if e >= 0 else ((q ** (-e)) * other == corr)
        if t < d and not cleaning_ok(t, U_only, T_only):
            return "exact_unsat", {"u": u, "reason": "fixed_split_cleaning_T"}
        if u < d and not cleaning_ok(u, T_only, U_only):
            return "exact_unsat", {"u": u, "reason": "fixed_split_cleaning_U"}

    # ---------- branch-invariant base LP ----------
    base_eq: List[SparseRow] = []
    base_eq_rhs: List[float] = []
    base_eq_exact: List[ExactRow] = []
    base_eq_rhs_exact: List[int] = []

    def add_base_eq(row: Mapping[int, object], rhs: object):
        exact_row: ExactRow = {}
        for j, v in row.items():
            if v == 0:
                continue
            if isinstance(v, float) and not v.is_integer():
                raise AssertionError(f"Nonintegral coefficient in exact MDS LP: {v}")
            exact_row[int(j)] = int(v)
        if isinstance(rhs, float) and not rhs.is_integer():
            raise AssertionError(f"Nonintegral RHS in exact MDS LP: {rhs}")
        exact_rhs = int(rhs)
        base_eq_exact.append(exact_row)
        base_eq_rhs_exact.append(exact_rhs)
        # Build the numerical row in normalized variables y_j=x_j/var_scale_j.
        rowf = {j: float(v) * var_scale[j] for j, v in exact_row.items()}
        scale = max([1.0, abs(float(exact_rhs))] + [abs(v) for v in rowf.values()])
        base_eq.append({j: v / scale for j, v in rowf.items()})
        base_eq_rhs.append(float(exact_rhs) / scale)

    # Fixed split A equations.
    for iw in range(t + 1):
        for rw in range(u + 1):
            target_count = split_mds(q, n, t, u, D_A, iw, rw)
            cols = [index[c] for c in active if split_weight(c) == (iw, rw)]
            if target_count == 0:
                if cols:
                    raise AssertionError("Active variables found in zero split shell")
                continue
            if not cols:
                return "exact_unsat", {"u": u, "reason": f"positive_split_shell_pruned_{iw}_{rw}"}
            add_base_eq({j: 1 for j in cols}, target_count)

    # Distance B=A below d is not a lazy inequality: impose exactly up front.
    for out in all_raw:
        if not (0 < total_weight(out) < d):
            continue
        rr = dict(B_row(out))
        acol = index.get(out)
        if acol is not None:
            rr[acol] = rr.get(acol, 0.0) - norm
        add_base_eq(rr, 0)

    # Assemble base equality CSR once.
    tmp = LPBuilder(nvars)
    tmp.eq_rows = list(base_eq)
    tmp.eq_rhs = list(base_eq_rhs)
    Aeq, beq, _, _ = tmp.matrices()
    del tmp

    build_s = time.perf_counter() - start
    print(
        f"  u={u}, T={t}: raw={len(all_raw)}, active={nvars}, "
        f"base-eq={0 if Aeq is None else Aeq.shape[0]}, "
        f"build={build_s:.3f}s; lazy B cuts ON",
        flush=True,
    )

    # Lazy inequalities are stored already row-scaled in HiGHS <= form.
    cut_rows: List[SparseRow] = []
    cut_rhs: List[float] = []
    cut_rows_exact: List[ExactRow] = []
    cut_rhs_exact: List[int] = []
    existing = set()
    round_no = 0
    total_solve = 0.0
    last_res = None
    # Start with the requested HiGHS path.  Non-IPM rounds are only a short
    # probe: structured count LPs can leave simplex grinding for a long time
    # before a false infeasibility.  On any non-success, retry the IDENTICAL
    # normalized LP with IPM.  Once IPM rescues a feasible point, stay on IPM
    # for the rest of this u-shell.
    active_method = highs_method

    def add_cut_le(key, row: Mapping[int, object], rhs: object):
        if key in existing:
            return False
        exact_row: ExactRow = {}
        for j, v in row.items():
            if v == 0:
                continue
            if isinstance(v, float) and not v.is_integer():
                raise AssertionError(f"Nonintegral lazy-cut coefficient: {v}")
            exact_row[int(j)] = int(v)
        if isinstance(rhs, float) and not rhs.is_integer():
            raise AssertionError(f"Nonintegral lazy-cut RHS: {rhs}")
        exact_rhs = int(rhs)
        cut_rows_exact.append(exact_row)
        cut_rhs_exact.append(exact_rhs)
        rowf = {j: float(v) * var_scale[j] for j, v in exact_row.items()}
        scale = max([1.0, abs(float(exact_rhs))] + [abs(v) for v in rowf.values()])
        cut_rows.append({j: v / scale for j, v in rowf.items()})
        cut_rhs.append(float(exact_rhs) / scale)
        existing.add(key)
        return True

    while True:
        round_no += 1
        # CSR only for the cuts currently active; base equality matrix is reused.
        if cut_rows:
            data=[]; inds=[]; indptr=[0]
            for row in cut_rows:
                for j,v in sorted(row.items()):
                    data.append(v); inds.append(j)
                indptr.append(len(data))
            Aub = csr_matrix(
                (np.asarray(data,float),np.asarray(inds,np.int32),np.asarray(indptr,np.int64)),
                shape=(len(cut_rows),nvars),
            )
            bub = np.asarray(cut_rhs,float)
        else:
            Aub=None; bub=None

        # Keep simplex/auto HiGHS as a cheap probe, but cap the time spent
        # there.  If it times out, declares infeasible, or otherwise fails,
        # rerun the identical normalized LP with IPM.
        primary_options={"presolve":True}
        if active_method != "highs-ipm" and simplex_time_limit_s > 0:
            primary_options["time_limit"] = float(simplex_time_limit_s)

        s0=time.perf_counter()
        res=linprog(
            c=np.zeros(nvars,dtype=float),
            A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
            bounds=numeric_bounds,method=active_method,options=primary_options,
        )
        dt=time.perf_counter()-s0
        total_solve += dt
        msg=str(res.message)
        used_solver=active_method

        # Any non-success from a non-IPM primary solve is only a numerical
        # screening result.  Before UNKNOWN or exact-UNSAT certification, try
        # the independent interior-point path on exactly the same normalized LP.
        if active_method != "highs-ipm" and (res.status != 0 or res.x is None):
            primary_status=int(res.status)
            primary_msg=msg
            if primary_status == 1:
                why="hit its time/iteration limit"
            elif primary_status == 2:
                why="reported infeasible"
            else:
                why=f"returned status={primary_status}"
            print(
                f"    round {round_no}: primary {active_method} {why} [{dt:.3f}s]; trying HiGHS-IPM rescue ...",
                flush=True,
            )
            r0=time.perf_counter()
            rescue=linprog(
                c=np.zeros(nvars,dtype=float),
                A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                bounds=numeric_bounds,method="highs-ipm",options={"presolve":True},
            )
            rdt=time.perf_counter()-r0
            total_solve += rdt

            if rescue.status==0 and rescue.x is not None:
                ry=np.asarray(rescue.x,float)
                req=0.0 if Aeq is None else float(np.max(np.abs(Aeq@ry-beq)))
                rub=0.0
                if Aub is not None:
                    rv=np.asarray(Aub@ry-bub)
                    rub=float(max(0.0,np.max(rv))) if rv.size else 0.0
                rbnd=0.0
                for jj,(lo,hi) in enumerate(numeric_bounds):
                    if lo is not None: rbnd=max(rbnd,float(lo)-ry[jj])
                    if hi is not None: rbnd=max(rbnd,ry[jj]-float(hi))
                rbnd=max(0.0,rbnd)
                if max(req,rub,rbnd) <= 1e-7:
                    print(
                        f"      HiGHS-IPM RESCUED a feasible point [{rdt:.3f}s]; "
                        f"maxeq={req:.2e}, maxub={rub:.2e}, maxbound={rbnd:.2e}",
                        flush=True,
                    )
                    # Reuse the rescue point immediately in THIS round.  Do not
                    # solve the same LP again.  Future rounds for this shell go
                    # straight to IPM, avoiding another long simplex grind.
                    res=rescue
                    msg=str(rescue.message)
                    dt=rdt
                    used_solver="highs-ipm(rescue)"
                    active_method="highs-ipm"
                    print("      switching remaining rounds for this u-shell directly to HiGHS-IPM", flush=True)
                else:
                    print(
                        f"      HiGHS-IPM returned a point but residual check failed "
                        f"[{rdt:.3f}s]; maxeq={req:.2e}, maxub={rub:.2e}, maxbound={rbnd:.2e}",
                        flush=True,
                    )
                    res=rescue
                    msg=str(rescue.message)
                    dt=rdt
                    used_solver="highs-ipm(rescue-bad-residual)"
            else:
                print(
                    f"      HiGHS-IPM rescue finished [{rdt:.3f}s], status={rescue.status}: {rescue.message}",
                    flush=True,
                )
                # From here on, the IPM result -- not the simplex result -- is
                # what controls exact certification/UNKNOWN.
                res=rescue
                msg=str(rescue.message)
                dt=rdt
                used_solver="highs-ipm(rescue)"

        # v15 control flow:
        # ANY numerical non-success after the normal rescue path triggers an
        # exact Farkas attempt on the CURRENT ACTIVE SUBSYSTEM.  Numerical
        # infeasibility is never itself a proof.
        #
        # If exact Farkas does not verify UNSAT, do NOT abandon the shell.
        # Re-solve the IDENTICAL active subsystem with no time limit using
        # dual simplex.  This distinguishes a numerical/IPM pathology from a
        # genuinely difficult active subsystem.  If that solve again fails,
        # run Farkas once more before returning UNKNOWN.
        def _force_farkas(tag):
            print(
                f"    round {round_no}: {tag}; FORCE exact Farkas on current active subsystem "
                f"(eq={len(base_eq_exact)}, le={len(cut_rows_exact)}, cuts={len(cut_rows_exact)}) ...",
                flush=True,
            )
            f0=time.perf_counter()
            exst, exmeta = exact_farkas_verify(
                nvars, base_eq_exact, base_eq_rhs_exact,
                cut_rows_exact, cut_rhs_exact, bounds, var_scale_int,
                method="highs-ds",
            )
            fdt=time.perf_counter()-f0
            if exst == "unsat":
                print(
                    f"      EXACT_UNSAT by rational Farkas "
                    f"[{fdt:.3f}s; support={exmeta.get('farkas_support_size',exmeta.get('farkas_numeric_support_1e10','?'))}; "
                    f"bTy={exmeta.get('farkas_original_bTy',exmeta.get('farkas_bTy','?'))}]",
                    flush=True,
                )
            else:
                print(
                    f"      Farkas NOT verified [{fdt:.3f}s], "
                    f"reason={exmeta.get('farkas_reason','unknown')}; "
                    f"retrying SAME active subsystem with NO time limit dual simplex",
                    flush=True,
                )
            exmeta["forced_farkas_runtime_s"] = fdt
            return exst, exmeta

        if res.status != 0 or res.x is None:
            first_status = int(res.status)
            first_msg = str(res.message)
            exst, exmeta = _force_farkas(
                f"numerical non-success status={first_status} via {used_solver}"
            )
            if exst == "unsat":
                return "exact_unsat", {
                    "u":u,"t":t,"raw_cells":len(all_raw),"active_vars":nvars,
                    "rounds":round_no,"cuts":len(cut_rows),"build_s":build_s,
                    "solve_s":total_solve,"highs_status":first_status,
                    "highs_message":first_msg,"exact_verified":True, **exmeta,
                }

            # Farkas failure is not a mathematical conclusion.  Retry the same
            # subsystem with an uncapped dual-simplex solve.
            r0=time.perf_counter()
            retry=linprog(
                c=np.zeros(nvars,dtype=float),
                A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
                bounds=numeric_bounds,method="highs-ds",
                options={"presolve":True},
            )
            rdt=time.perf_counter()-r0
            total_solve += rdt
            rmsg=str(retry.message)
            if retry.status == 0 and retry.x is not None:
                req,rub,rbnd=_residuals(retry)
                if req <= 2e-7 and rub <= 2e-7 and rbnd <= 2e-7:
                    print(
                        f"      NO-LIMIT dual simplex RESCUED feasible point [{rdt:.3f}s]; "
                        f"maxeq={req:.2e}, maxub={rub:.2e}, maxbound={rbnd:.2e}",
                        flush=True,
                    )
                    res=retry
                    msg=rmsg
                    dt=rdt
                    used_solver="highs-ds(no-limit-rescue)"
                    active_method="highs-ds"
                else:
                    print(
                        f"      NO-LIMIT dual simplex returned point with bad residuals [{rdt:.3f}s]; "
                        f"maxeq={req:.2e}, maxub={rub:.2e}, maxbound={rbnd:.2e}",
                        flush=True,
                    )
                    res=retry
            else:
                print(
                    f"      NO-LIMIT dual simplex non-success [{rdt:.3f}s], "
                    f"status={retry.status}: {retry.message}; forcing Farkas again",
                    flush=True,
                )
                res=retry
                msg=rmsg
                exst2, exmeta2 = _force_farkas(
                    f"no-limit dual-simplex non-success status={retry.status}"
                )
                if exst2 == "unsat":
                    return "exact_unsat", {
                        "u":u,"t":t,"raw_cells":len(all_raw),"active_vars":nvars,
                        "rounds":round_no,"cuts":len(cut_rows),"build_s":build_s,
                        "solve_s":total_solve,"highs_status":int(retry.status),
                        "highs_message":rmsg,"exact_verified":True, **exmeta2,
                    }
                return "unknown", {
                    "u":u,"t":t,"raw_cells":len(all_raw),"active_vars":nvars,
                    "rounds":round_no,"cuts":len(cut_rows),"build_s":build_s,
                    "solve_s":total_solve,"highs_status":int(retry.status),
                    "highs_message":rmsg,"exact_verified":False,
                    "reason":"backend_stuck_after_forced_farkas",
                    "first_farkas":exmeta,"second_farkas":exmeta2,
                }

        if res.status != 0 or res.x is None:
            # Defensive guard: every ordinary non-success should already have
            # passed through the forced-Farkas/no-limit retry block above.
            return "unknown", {
                "u":u,"t":t,"raw_cells":len(all_raw),"active_vars":nvars,
                "rounds":round_no,"cuts":len(cut_rows),"build_s":build_s,
                "solve_s":total_solve,"highs_status":int(res.status),
                "highs_message":str(res.message),
                "reason":"defensive_unresolved_after_forced_farkas",
            }

        y_num=np.asarray(res.x,float)
        x=y_num*var_scale
        violations=[]
        for out in all_raw:
            if 0 < total_weight(out) < d:
                continue  # already exact equality
            br=B_row(out)
            bval=sum(v*x[j] for j,v in br.items())
            acol=index.get(out)
            aval=(norm*x[acol]) if acol is not None else 0.0
            key=("contain",out)
            if key not in existing:
                scale=max(1.0,abs(bval),abs(aval))
                sev=(aval-bval)/scale
                if sev>lazy_tol:
                    # B-A >=0  <=>  -(B-A) <=0
                    row={j:-int(v) for j,v in br.items()}
                    if acol is not None:
                        row[acol]=row.get(acol,0)+norm
                    violations.append((sev,key,row,0))
            if use_capacities:
                key=("cap",out)
                if key not in existing:
                    cap_count=joint_commuting_capacity(q,t,u,out,fq_linear)
                    cap_num=norm*cap_count
                    scale=max(1.0,abs(bval),abs(cap_num))
                    sev=(bval-cap_num)/scale
                    if sev>lazy_tol:
                        violations.append((sev,key,{j:int(v) for j,v in br.items()},cap_num))

        if not violations:
            maxeq=0.0 if Aeq is None else float(np.max(np.abs(Aeq@y_num-beq)))
            maxub=0.0
            if Aub is not None:
                vv=np.asarray(Aub@y_num-bub)
                maxub=float(max(0.0,np.max(vv))) if vv.size else 0.0
            print(
                f"    round {round_no}: SAT via {used_solver} [{dt:.3f}s], cuts={len(cut_rows)}, "
                f"maxeq={maxeq:.2e}, maxub={maxub:.2e}",flush=True)
            return "sat", {
                "u":u,"t":t,"raw_cells":len(all_raw),"active_vars":nvars,
                "rounds":round_no,"cuts":len(cut_rows),"build_s":build_s,
                "solve_s":total_solve,"highs_status":int(res.status),"highs_message":msg,
                "max_eq_residual":maxeq,"max_ub_violation":maxub,"fq_linear":fq_linear,
            }

        violations.sort(key=lambda z:z[0],reverse=True)
        chosen=violations[:max(1,int(lazy_cuts_per_round))]
        for sev,key,row,rhs in chosen:
            add_cut_le(key,row,rhs)
        print(
            f"    round {round_no}: feasible candidate via {used_solver} [{dt:.3f}s], "
            f"violations={len(violations)}, add={len(chosen)}, "
            f"max={violations[0][0]:.3e}, total-cuts={len(cut_rows)}",
            flush=True,
        )


# =============================================================================
# Regressions / CLI
# =============================================================================

def self_test() -> None:
    # Exact-verifier regressions.
    if z3 is not None:
        st,_ = exact_lra_verify(1, [{0:1}], [0], [{0:-1}], [-1], [(None,None)])
        assert st == "unsat"  # x=0 and x>=1
        st,_ = exact_lra_verify(1, [{0:1}], [1], [], [], [(0,2)])
        assert st == "sat"
    # q=4 single-site matrix regression against the earlier engine.
    assert r_matrix(4, 1) == (
        (1, 1, 1),
        (12, 0, -4),
        (3, -1, 3),
    )
    for q in SUPPORTED_Q:
        fq = effective_fq_linear(q, False)
        for m in range(1, 4):
            amb = sum(block_ambient_capacity(q, m, a, b) for a, b in pair_cells(m))
            assert amb == (q * q) ** m
        for u in range(1, 4):
            comm = sum(u_commuting_capacity(q, u, j, l, fq) for j, l in pair_cells(u))
            denom = q if fq else characteristic(q)
            assert comm == (q * q) ** u // denom


def parse_args(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Optimized q-ary QMDS RXxRY stabilizer-reference real-LP v15 force-Farkas")
    p.add_argument("q", type=int)
    p.add_argument("n", type=int)
    p.add_argument("k", type=int)
    p.add_argument("d", type=int)
    p.add_argument("--u", type=int, action="append", default=None)
    p.add_argument("--u-order", choices=("ascending", "descending"), default="ascending")
    p.add_argument("--fq-linear", action="store_true")
    p.add_argument("--no-shadow", action="store_true")
    p.add_argument("--no-cleaning", action="store_true")
    p.add_argument("--no-capacities", action="store_true")
    p.add_argument("--continue-after-unsat", action="store_true")
    p.add_argument("--highs-method", choices=("highs", "highs-ds", "highs-ipm"), default="highs")
    p.add_argument("--simplex-time-limit", type=float, default=3.0,
                   help="seconds allowed for each non-IPM primary solve before IPM rescue; 0 disables the cap")
    p.add_argument("--lazy-cuts", type=int, default=96, help="max violated B cuts added per LP round")
    p.add_argument("--lazy-tol", type=float, default=1e-8)
    p.add_argument("--exact-timeout-ms", type=int, default=0, help="Z3 exact UNSAT verification timeout; 0=no limit")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    q, n, k, d = args.q, args.n, args.k, args.d
    if q not in SUPPORTED_Q:
        raise ValueError(f"q must be one of {SUPPORTED_Q}")
    if k != n - 2 * (d - 1):
        raise ValueError(f"Not QMDS parameters: require k=n-2(d-1)={n-2*(d-1)}, got {k}")
    if q == 4:
        fq = bool(args.fq_linear)
    else:
        fq = True
        if args.fq_linear:
            print("Prime q: --fq-linear is automatic and has no extra effect.")

    self_test()
    D_A = n - d + 2
    A = mds_distribution(q, n, D_A)
    expected_norm = q ** (n - k)
    if sum(A) != expected_norm:
        raise AssertionError(f"QMDS stabilizer MDS enumerator sums to {sum(A)}, expected {expected_norm}")
    B = mds_distribution(q, n, d)
    if sum(B) != q ** (n + k):
        raise AssertionError("QMDS dual MDS enumerator has wrong total size")

    mandatory = [u for u in range(1, n + 1) if A[u] > 0]
    if args.u is not None:
        wanted = set(args.u)
        bad = sorted(wanted - set(mandatory))
        if bad:
            raise ValueError(f"Requested u not a positive stabilizer shell: {bad}; mandatory={mandatory}")
        mandatory = [u for u in mandatory if u in wanted]
    mandatory.sort(reverse=(args.u_order == "descending"))

    print("=" * 104)
    print("OPTIMIZED QMDS q-ARY RX x RY STABILIZER-REFERENCE REAL-LP")
    print(f"Code: [[{n},{k},{d}]]_{q}; D_A={D_A}; |S|={expected_norm}")
    print("Projection: T=I/X-line*/off-X, U=I/Y-line*/off-Y")
    print(f"Linearity: {'F_q-linear' if fq else 'general additive / trace-symplectic'}")
    print(f"Mandatory u shells: {mandatory}")
    print(f"Generalized split-Hamming shadow: {'ON' if not args.no_shadow else 'OFF'}")
    print(f"Cleaning: {'ON' if not args.no_cleaning else 'OFF'}")
    print(f"Commuting capacities: {'ON' if not args.no_capacities else 'OFF'}")
    print(f"Backend: normalized-variable SciPy/{args.highs_method} + bounded primary ({args.simplex_time_limit:g}s) → IPM rescue → FORCE exact Farkas on any non-success → same-subsystem no-limit dual simplex; NO Z3 fallback")
    print("=" * 104)

    results = []
    overall = "sat"
    for u in mandatory:
        st, info = solve_reference_u(
            q, n, k, d, u,
            fq_linear=fq,
            use_shadow=not args.no_shadow,
            use_cleaning=not args.no_cleaning,
            use_capacities=not args.no_capacities,
            highs_method=args.highs_method,
            simplex_time_limit_s=args.simplex_time_limit,
            lazy_cuts_per_round=args.lazy_cuts,
            lazy_tol=args.lazy_tol,
            exact_timeout_ms=args.exact_timeout_ms,
        )
        results.append((u, st, info))
        if st == "exact_unsat":
            overall = "exact_unsat"
            if not args.continue_after_unsat:
                break
        elif st == "unknown" and overall != "exact_unsat":
            overall = "unknown"

    print("=" * 104)
    print(f"FINAL: {overall.upper()}")
    for u, st, info in results:
        print(f"  u={u}: {st.upper()}  active={info.get('active_vars')} build={info.get('build_s')} solve={info.get('solve_s')}")
    print("=" * 104)


if __name__ == "__main__":
    main()
