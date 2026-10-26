#!/usr/bin/env python3
"""Independent exact-integer checks for the [[14,3,5]] candidates 17 and 35.

Usage:
    python verify_n14_3_5.py

No external files, packages, solvers or floating-point arithmetic needed.
Candidate weight distributions are embedded exactly as supplied.
"""
from math import comb

N, K, DISTANCE = 14, 3, 5
LENGTH = 3 * N
STAB_SIZE = 1 << (N - K)

# A[j]: number of weight-j stabilizer elements, input data from refined LP.
CANDIDATES = {
    17: {0: 1, 4: 1, 6: 5, 7: 32, 8: 199, 9: 400,
         10: 350, 11: 384, 12: 463, 13: 208, 14: 5},
    35: {0: 1, 4: 1, 6: 4, 7: 34, 8: 201, 9: 394,
         10: 350, 11: 390, 12: 461, 13: 206, 14: 6},
}


def krawtchouk(j, w, n, q):
    """q-ary K_j(w;n) from (x+(q-1)y)^(n-w) (x-y)^w."""
    return sum(
        (-1)**t * (q - 1)**(j - t)
        * comb(w, t) * comb(n - w, j - t)
        for t in range(max(0, j - (n - w)), min(j, w) + 1)
    )


def macwilliams(coef, n, q, code_size):
    """Full exact MacWilliams transform of a length-n weight distribution."""
    output = []
    for j in range(n + 1):
        num = sum(a * krawtchouk(j, w, n, q)
                  for w, a in enumerate(coef) if a)
        val, rem = divmod(num, code_size)
        assert rem == 0, f'MacWilliams coefficient j={j} is nonintegral'
        assert val >= 0, f'MacWilliams coefficient j={j} is negative'
        output.append(val)
    return output


def simplex_dual_substitution(B, n):
    """Coefficients of B(x^3+y^3,x^2y+xy^2), by exact binomial expansion.

    A weight-w Pauli label contributes (x^2*y+x*y^2)^w,
    and its other n-w coordinates contribute (x^3+y^3)^(n-w).
    """
    result = [0] * (3 * n + 1)
    for w, multiplicity in enumerate(B):
        if not multiplicity:
            continue
        for a in range(n - w + 1):
            for b in range(w + 1):
                output_weight = w + 3*a + b
                result[output_weight] += (multiplicity * comb(n-w, a)
                                          * comb(w, b))
    return result


def simplex_image_enumerator(A, n, even_only=False):
    """Weights in xi(S) or xi(S_0): wt_H(xi(v)) = 2 wt(v)."""
    W = [0] * (3*n + 1)
    for w, count in enumerate(A):
        if not even_only or w % 2 == 0:
            W[2*w] += count
    return W


def fmt_small(coef, last):
    return ', '.join(f'weight {j}: {coef[j]}' for j in range(last + 1))


def verify_one(label, coeffs):
    A = [coeffs.get(j, 0) for j in range(N + 1)]
    assert sum(A) == STAB_SIZE
    assert A[0] == 1 and A[1:4] == [0, 0, 0] and A[4] == 1
    assert sum(w * a for w, a in enumerate(A)) * 2 == 21 * STAB_SIZE
    print(f'===== Candidate {label} =====')
    print(f'Input: A = {coeffs}')
    print(f'Stabilizer size |S| = {sum(A)}; A_4 = {A[4]}')

    # Quantum/quaternary MacWilliams: B(x,y)=A(x+3y,x-y)/2^(n-k)
    B = macwilliams(A, N, q=4, code_size=STAB_SIZE)
    assert B[:DISTANCE] == A[:DISTANCE]  # [[n,k,d]] distance condition
    assert sum(B) == 4**N // STAB_SIZE
    print('Quaternary MacWilliams, B_0..B_7:', B[:8])
    print('Check distance d=5: B_j == A_j for j=0,1,2,3,4: PASS')

    # Binary simplex image C, and its dual Cperp computed independently.
    W_C = simplex_image_enumerator(A, N)
    assert sum(W_C) == STAB_SIZE
    W_Cperp_from_binary_MI = macwilliams(W_C, LENGTH, q=2,
                                          code_size=STAB_SIZE)
    W_Cperp_from_simplex_lemma = simplex_dual_substitution(B, N)
    assert W_Cperp_from_simplex_lemma == W_Cperp_from_binary_MI
    W_Cperp = W_Cperp_from_simplex_lemma
    assert W_Cperp[:5] == [1, 0, 0, 14, 1]
    assert sum(W_Cperp) == 2**(LENGTH - (N-K))
    print('W_Cperp weights 0..7:', W_Cperp[:8])
    print('Full 43 coefficients: simplex dual lemma == binary MacWilliams: PASS')

    # Even-Pauli-weight stabilizer and its binary image C0.
    W_C0 = simplex_image_enumerator(A, N, even_only=True)
    size_C0 = sum(W_C0)
    assert size_C0 == STAB_SIZE // 2
    W_C0perp = macwilliams(W_C0, LENGTH, q=2,
                             code_size=size_C0)
    assert sum(W_C0perp) == 2**(LENGTH - (N-K-1))
    print(f'Even-weight subcode size |C0| = {size_C0}')

    # Independently evaluate the degree-3 Krawtchouk sum shown in the text.
    terms = [(w, A[w], krawtchouk(3, 2*w, LENGTH, q=2))
             for w in range(0, N+1, 2) if A[w]]
    numerator = sum(a * k for _, a, k in terms)
    assert numerator % size_C0 == 0
    count3_direct = numerator // size_C0
    assert count3_direct == W_C0perp[3]
    print('K_3(2w;42) terms (w, A_w, K_3):', terms)
    print(f'C0perp weight-3: {numerator}/{size_C0} = {count3_direct}')
    print('Krawtchouk sum == full binary MacWilliams: PASS')

    # T=C0perp \\ Cperp: exactly one coset since [C:C0]=2.
    W_T = [u-v for u, v in zip(W_C0perp, W_Cperp)]
    assert all(t >= 0 for t in W_T)
    assert W_T[0] == 0
    assert sum(W_T) == sum(W_Cperp)
    assert W_T[3] == W_C0perp[3] - W_Cperp[3]
    assert W_T[3] > LENGTH // 3
    print(f'T weight-3: {W_C0perp[3]} - {W_Cperp[3]} = {W_T[3]}')
    print(f'Packing upper bound: floor(42/3) = {LENGTH // 3}')
    print(f'Violation: {W_T[3]} > {LENGTH // 3}: VERIFIED')
    print('Full coset distribution nonnegative and correct total size: PASS')
    print()
    return W_T[3]


def main():
    out = {label: verify_one(label, a) for label, a in CANDIDATES.items()}
    assert out == {17: 20, 35: 19}
    print('ALL EXACT-ARITHMETIC CHECKS PASSED.')
    print('Note: This script verifies the enumerator calculations and numerical')
    print('packing-bound violation, not the separate combinatorial lemma proof')
    print('or completeness of the externally supplied LP survivor list.')


if __name__ == '__main__':
    main()
