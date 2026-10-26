#!/usr/bin/env python3
"""Exact shadow-coset screening for ALL SLR3 candidates of [[14,3,5]].

Typical use (place both files in the same folder):
    python verify_n14_all47_packing_v2.py candidates_n14_k3_d5.py
    python verify_n14_all47_packing_v2.py candidates_n14_k3_d5.py --report n14_report.txt

Accepts the persistent SLR3 database structure:
    DATABASE = {'code': {'n':14, 'k':3, 'd':5},
                'candidates':[{'id':'...', 'fixed_A':{0:1,...}}, ...]}
It reads the DATABASE assignment with ast.literal_eval WITHOUT executing the file.
Run `--demo` to check only the two known candidates 17 and 35.

This tests necessary shadow-coset counting conditions, NOT HxRX/RXxRY LPs.
'NOT EXCLUDED' is never interpreted as realizable.
The completeness of the provided SLR3 database is not established by this script.
Python standard library only. No numerical approximations or external solvers.
"""
import argparse
import ast
from collections import Counter, defaultdict
from hashlib import sha256
from math import comb
from pathlib import Path

N, K, D = 14, 3, 5
L = 3*N
SIZE_S = 1 << (N-K)

DEMO = {
    '17': {0:1,4:1,6:5,7:32,8:199,9:400,10:350,11:384,12:463,13:208,14:5},
    '35': {0:1,4:1,6:4,7:34,8:201,9:394,10:350,11:390,12:461,13:206,14:6},
}


def kr(j, w, n, q):
    return sum((-1)**t * (q-1)**(j-t) * comb(w,t) * comb(n-w,j-t)
               for t in range(max(0,j-(n-w)), min(j,w)+1))

# Precompute exact transform matrices once for all 47 candidates.
K4 = [[kr(j,w,N,4) for w in range(N+1)] for j in range(N+1)]
K2 = [[kr(j,2*w,L,2) for w in range(N+1)] for j in range(L+1)]


def transform(A, rows, denom):
    out = []
    for row in rows:
        num = sum(a * x for a,x in zip(A,row))
        val, rem = divmod(num,denom)
        if rem:
            raise AssertionError(f'non-integer MacWilliams coefficient {num}/{denom}')
        out.append(val)
    return out


def load_database(path):
    source = path.read_text(encoding='utf-8-sig')
    tree = ast.parse(source, filename=str(path))
    declarations = [n.value for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='DATABASE'
                            for t in n.targets)]
    if len(declarations) != 1:
        raise ValueError('Expected exactly one literal DATABASE = {...} assignment')
    database = ast.literal_eval(declarations[0])
    meta = database.get('code',{})
    if meta != {'n':N,'k':K,'d':D}:
        raise ValueError(f'Expected [[14,3,5]], got database metadata {meta}')
    entries = database.get('candidates')
    if not isinstance(entries,list):
        raise ValueError('DATABASE["candidates"] must be a list')
    return [(str(e['id']),e['fixed_A']) for e in entries], database, sha256(source.encode()).hexdigest()


def validate(A, ident):
    assert A[0]==1 and all(v>=0 and isinstance(v,int) for v in A), ident
    # Degenerate stabilizer codes may have A_j>0 for j<d.
    assert sum(A)==SIZE_S, f'{ident}: sum A = {sum(A)} != {SIZE_S}'


def candidate(ident, fixed):
    assert isinstance(fixed,dict),f'{ident}: missing fixed_A dict'
    assert all(type(j)==int and 0<=j<=N and type(a)==int and a>=0
               for j,a in fixed.items()),f'{ident}: invalid A coefficients'
    A=[fixed.get(j,0) for j in range(N+1)]
    validate(A,ident)
    B=transform(A,K4,SIZE_S)
    assert min(B)>=0 and B[0]==1 and sum(B)==2**(N+K),f'{ident}: bad B'
    assert B[:D]==A[:D],f'{ident}: violates distance d={D}'
    # Image C has Hamming weights 2w. Apply binary MacWilliams exactly.
    WCperp=transform(A,K2,SIZE_S)
    assert min(WCperp)>=0 and WCperp[0]==1 and sum(WCperp)==2**(L-(N-K))
    even_size=sum(A[::2])
    if even_size==SIZE_S:
        return dict(id=ident,A4=A[4],D2=WCperp[2],D3=WCperp[3],
                    D4=WCperp[4],T3=None,C0perp3=None,bound=None,
                    status='NO ODD-WEIGHT STABILIZER')
    assert even_size==SIZE_S//2,f'{ident}: even part is not index two'
    # C0 dual: coefficient of weight j = 1/|C0| sum_{w even} A_w K_j(2w;42).
    Aeven=[A[w] if w%2==0 else 0 for w in range(N+1)]
    WC0perp=transform(Aeven,K2,even_size)
    assert min(WC0perp)>=0 and WC0perp[0]==1
    assert sum(WC0perp)==2**(L-(N-K-1))
    c0perp3=WC0perp[3]
    # Alternative direct signed-transform of the coset T=C0perp\Cperp.
    signed_num=sum((-1)**w*A[w]*K2[3][w] for w in range(N+1))
    assert signed_num % SIZE_S == 0
    t3=signed_num//SIZE_S
    assert c0perp3-WCperp[3]==t3 and t3>=0
    assert all(x>=y for x,y in zip(WC0perp,WCperp)), f'{ident}: invalid shadow coset'
    # When weight-4 elements in C-perp are 0 or 1, triples in any coset
    # can pack at most floor(L/3)=14 by the support-intersection lemma.
    # When m=A4>1, a general (weaker) bound follows from counting edges:
    # each triple overlaps at most m other triples (one per weight-4
    # word in C-perp), so E <= m*M/2.  With r_i the incidence counts,
    # 2E = sum_i r_i^2 - 3M >= 9M^2/L - 3M by Cauchy.
    # Hence M <= floor(L*(m+3)/9).
    m=WCperp[4]
    if WCperp[2]>0:
        return dict(id=ident,A4=A[4],D2=WCperp[2],D3=WCperp[3],
                    D4=m,T3=t3,C0perp3=c0perp3,bound=None,
                    status='WEIGHT-TWO IN Cperp')
    if m<=1:
        bound=L//3
        rule='unique-or-zero weight-four packing'
    else:
        bound=L*(m+3)//9
        rule='general overlap-count bound'
    status='EXCLUDED' if t3>bound else 'NOT EXCLUDED'
    return dict(id=ident,A4=A[4],D2=WCperp[2],D3=WCperp[3],D4=m,
                T3=t3,C0perp3=c0perp3,bound=bound,status=status,rule=rule)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('database',nargs='?',type=Path,
    default=Path(__file__).resolve().parents[2] / 'candidates' / 'binary' / 'candidates_n14_k3_d5.py')
    p.add_argument('--demo',action='store_true',help='verify only embedded candidates 17 and 35')
    p.add_argument('--report',type=Path,help='save complete per-candidate report')
    p.add_argument('--require-47',action='store_true',help='fail unless 47 candidates supplied')
    args=p.parse_args()
    if args.demo:
        entries=list(DEMO.items()); label='DEMO (only 17 and 35)';digest='embedded'
    else:
        if not args.database.exists():
            p.error(f'Cannot find {args.database}. Supply the actual 47-candidate database, or use --demo.')
        entries,db,digest=load_database(args.database)
        label=f'DATABASE: {args.database}'
    if args.require_47 and len(entries)!=47:
        p.error(f'Expected 47 candidates, found {len(entries)}')
    seen=set(); results=[]
    for ident,fixed in entries:
        if ident in seen:raise AssertionError(f'duplicate candidate id {ident}')
        seen.add(ident)
        results.append(candidate(ident,fixed))
    counts=Counter(r['status'] for r in results)
    a4counts=Counter(r['A4'] for r in results)
    lines=[f'[[14,3,5]] shadow-coset screening: {label}',
           f'SHA256 input: {digest}',
           f'Candidates checked: {len(results)}',
           f'Excluded by counting: {counts["EXCLUDED"]}',
           f'Not excluded by this counting: {counts["NOT EXCLUDED"]}',
           f'Not applicable (all even): {counts["NO ODD-WEIGHT STABILIZER"]}',
           f'Not applicable (Cperp has weight-two): {counts["WEIGHT-TWO IN Cperp"]}',
           f'A4 distribution: {dict(sorted(a4counts.items()))}',
           '', 'Candidate        A4   D2  D3  D4   C0perp_3   T_3   bound   result']
    for r in results:
        def fmt(x):return '-' if x is None else str(x)
        lines.append(f'{r["id"]:<17}{r["A4"]:<5}{r["D2"]:<4}{r["D3"]:<4}{r["D4"]:<5}'
                     f'{fmt(r["C0perp3"]):<11}{fmt(r["T3"]):<6}'
                     f'{fmt(r["bound"]):<8}{r["status"]}')
    lines+=['', 'Methods:',
            'C0perp_3 = (1/|C0|) sum_even_w A_w K_3^(2)(2w;42).',
            'T_3 = C0perp_3 - Cperp_3 (Cperp_3 can vary).',
            'm=Cperp_4 is the number of weight-four codewords in Cperp.',
            'For candidates with A1=A2=A3=0, Cperp_3=14, Cperp_4=A4.',
            'The packing bounds require Cperp_2=0 and an index-two C0.',
            'If m<=1, any coset has at most 14 weight-three words.',
            'If m>1, the proven but weaker bound is floor(42*(m+3)/9).',
            'This screen does not test HxRX or RXxRY or prove SLR3 completeness.',
            'NOT EXCLUDED means INCONCLUSIVE, not that a code exists.']
    report='\n'.join(lines)+'\n'
    print('\n'.join(lines[:8])); print('\n'.join(lines[8:]))
    if args.report:
        args.report.write_text(report,encoding='utf-8')
        print(f'Wrote report: {args.report}')

if __name__=='__main__':
    main()
