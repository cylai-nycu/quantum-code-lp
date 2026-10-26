#!/usr/bin/env python3
"""Exact verification of the 292 SLR3 candidates for [[17,6,5]].

Usage (place both files in the same directory):
    python verify_n17_6_5.py
    python verify_n17_6_5.py candidates_n17_k6_d5.py
    python verify_n17_6_5.py candidates_n17_k6_d5.py --report verification.txt
    python verify_n17_6_5.py --verbose

Reads DATABASE['candidates'][i]['fixed_A'] from candidates_n17_k6_d5.py.
Uses AST + literal_eval; DOES NOT execute or modify the candidate database.
Does not run any LP/refined solver. Only standard-library exact integer arithmetic.
"""

import argparse
import ast
from collections import Counter
from hashlib import sha256
from math import comb
from pathlib import Path

N, K, DISTANCE = 17, 6, 5
SIZE_S = 1 << (N-K)
SIZE_DUAL = 1 << (N+K)
EXPECTED_TOTAL = 292
REASONS = (
    'R3 > 17',
    'R1 > 1',
    'R1 > 0 and R3 > 0',
)
EXPECTED_COUNTS = (258, 1, 33)


def load_database(path):
    """Extract DATABASE = {...} safely without importing/executing source."""
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    matching = [node.value for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'DATABASE'
                        for target in node.targets)]
    if len(matching) != 1:
        raise ValueError('Expected exactly one literal DATABASE assignment')
    database = ast.literal_eval(matching[0])
    if database['code'] != {'n': N, 'k': K, 'd': DISTANCE}:
        raise ValueError(f'Incorrect code metadata: {database["code"]}')
    solver = database['ordinary_solver']
    if solver['case_counts'] != {1: EXPECTED_TOTAL, 2: 0, 3: 0}:
        raise ValueError(f'Unexpected SLR3 case counts: {solver["case_counts"]}')
    if not solver['complete'] or not all(solver['case_complete'].values()):
        raise ValueError('The input database does not mark SLR3 cases complete')
    return database, sha256(source.encode('utf-8')).hexdigest()


def krawtchouk(j, w, q=4):
    """K_j^(q)(w; N), calculated as an exact integer."""
    return sum((-1)**t * (q-1)**(j-t) * comb(w, t) * comb(N-w, j-t)
               for t in range(max(0, j-(N-w)), min(j, w)+1))


def transform(A, shadow=False):
    """MacWilliams dual B_j or shadow R_j, all integer divisions exact.

    B_j = 2^-11 sum_w A_w K_j^(4)(w;17).
    R_j = 2^-11 sum_w (-1)^w A_w K_j^(4)(w;17).
    """
    result = []
    for j in range(N+1):
        numerator = sum(((-1)**w if shadow else 1) * A[w]
                        * krawtchouk(j, w) for w in range(N+1) if A[w])
        quotient, remainder = divmod(numerator, SIZE_S)
        if remainder:
            kind = 'R' if shadow else 'B'
            raise ValueError(f'Noninteger {kind}_{j}: {numerator}/{SIZE_S}')
        if quotient < 0:
            kind = 'R' if shadow else 'B'
            raise ValueError(f'Negative {kind}_{j}: {quotient}')
        result.append(quotient)
    return result


def classify(R):
    """Disjoint partition, ordered by the first violated packing bound."""
    if R[3] > N:
        return REASONS[0]
    if R[1] > 1:
        return REASONS[1]
    if R[1] > 0 and R[3] > 0:
        return REASONS[2]
    return 'NOT EXCLUDED'


def verify_candidate(entry):
    candidate_id = entry['id']
    if entry['case'] != 1:
        raise ValueError(f'{candidate_id}: expected Case 1')
    values = entry['fixed_A']
    if not isinstance(values, dict):
        raise ValueError(f'{candidate_id}: fixed_A is not a dictionary')
    if any(not isinstance(w, int) or not 0 <= w <= N or
           not isinstance(v, int) or v < 0 for w, v in values.items()):
        raise ValueError(f'{candidate_id}: invalid fixed_A entries')
    A = [values.get(w, 0) for w in range(N+1)]
    if A[0] != 1 or any(A[j] != 0 for j in range(1, DISTANCE)):
        raise ValueError(f'{candidate_id}: wrong A_0 or nonzero A_1,...,A_4')
    if sum(A) != SIZE_S:
        raise ValueError(f'{candidate_id}: sum(A) = {sum(A)}, expected {SIZE_S}')
    B = transform(A)
    if B[:DISTANCE] != A[:DISTANCE] or sum(B) != SIZE_DUAL:
        raise ValueError(f'{candidate_id}: wrong dual distance or MacWilliams total')
    R = transform(A, shadow=True)
    if R[0] != 0 or sum(R) != SIZE_DUAL:
        raise ValueError(f'{candidate_id}: wrong shadow R_0 or total')
    return {'id': candidate_id, 'A': A, 'B': B, 'R': R,
            'reason': classify(R)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database', nargs='?', type=Path,
    default=Path(__file__).resolve().parents[2] / 'candidates' / 'binary' / 'candidates_n17_k6_d5.py',
                        help='source candidate database (default: next to verifier)')
    parser.add_argument('--report', type=Path, help='save detailed verification report')
    parser.add_argument('--verbose', action='store_true', help='show all candidate results')
    args = parser.parse_args()

    database, digest = load_database(args.database)
    entries = database['candidates']
    if len(entries) != EXPECTED_TOTAL:
        raise ValueError(f'Expected 292 candidates, found {len(entries)}')

    seen_ids, seen_distributions = set(), set()
    results = []
    for entry in entries:
        if entry['id'] in seen_ids:
            raise ValueError(f'Duplicate candidate id: {entry["id"]}')
        seen_ids.add(entry['id'])
        result = verify_candidate(entry)
        distribution = tuple(result['A'])
        if distribution in seen_distributions:
            raise ValueError(f'Duplicate A distribution: {entry["id"]}')
        seen_distributions.add(distribution)
        results.append(result)

    counts = Counter(r['reason'] for r in results)
    rows = [
        '[[17,6,5]] shadow-coset exact integer verification',
        f'Input file: {args.database}',
        f'Input SHA-256: {digest}',
        f'SLR3 case counts in database: {database["ordinary_solver"]["case_counts"]}',
        f'Distinct candidates verified: {len(results)}',
        '',
        'Disjoint exclusion partition (priority: R3 > 17, R1 > 1, mixed):',
    ]
    for reason in REASONS + ('NOT EXCLUDED',):
        rows.append(f'  {reason:<23}: {counts[reason]}')
    rows.append('')
    expected_ok = ([counts[r] for r in REASONS] == list(EXPECTED_COUNTS)
                   and counts['NOT EXCLUDED'] == 0)
    rows.append('RESULT: ' + ('PASS (292/292 excluded, 258/1/33 verified)'
                              if expected_ok else 'FAIL (unexpected counts/survivors)'))
    rows.append('')
    rows.append('Per-candidate results:')
    for r in results:
        rows.append(f"{r['id']}: R1={r['R'][1]:>2}, R3={r['R'][3]:>2}, "
                    f"reason={r['reason']}, A={r['A']}")
    rows.append('')
    rows.append('Scope: checks exact MacWilliams/shadow arithmetic and packing'
                ' inequalities for the 292 supplied candidates. It does not'
                ' re-run SLR3 or independently certify its completeness.')
    report = '\n'.join(rows) + '\n'
    if args.report:
        args.report.write_text(report, encoding='utf-8')

    print('\n'.join(rows[:13]))
    if args.verbose:
        print('\n'.join(rows[14:]))
    if args.report:
        print(f'Report saved: {args.report}')
    return 0 if expected_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
