# Computational Status and Verification Scope

This note records the state represented by the candidate databases and the checks performed when this repository was organized. A candidate database is both data and an executable status viewer. Refined solvers may update its embedded database in place; preserve a copy before running long candidate sweeps if the original snapshot is needed.

## Stage distinctions

1. **SLR enumeration** searches the exact-integer Shor–Laflamme–Rains constraints for ordinary weight distributions. A database marked complete records completion of that program's configured search, not a structural proof that an associated code exists.
2. **Refined LP tests** add projected or reference-based necessary constraints to fixed ordinary distributions. `SAT` means only that the tested relaxation is feasible. Real-LP solvers use floating-point HiGHS for screening; only their documented exact rational Farkas or Z3 verification can promote a numerical infeasibility to an exact exclusion.
3. **Structural exclusion** applies separate combinatorial or exact-integer bounds to candidates. Its scope is limited to the candidates and inequalities its verifier checks.

Do not combine a feasible result from one stage with an exclusion status from another as though they were one solver run. Do not interpret a numerical infeasibility by itself as a nonexistence certificate.

## Stored candidate counts

Counts and SLR completion flags were read from each literal `DATABASE` object in the files below. `complete` refers to the stored ordinary SLR enumeration flag.

| Database | Candidate count | SLR enumeration complete |
|---|---:|---|
| `candidates/binary/candidates_n13_k5_d4.py` | 30 | Yes |
| `candidates/binary/candidates_n14_k3_d5.py` | 47 | Yes |
| `candidates/binary/candidates_n16_k5_d5.py` | 95 | Yes |
| `candidates/binary/candidates_n17_k3_d6.py` | 7 | Yes |
| `candidates/binary/candidates_n17_k6_d5.py` | 292 | Yes |
| `candidates/binary/candidates_n18_k4_d6.py` | 2,868 | Yes |
| `candidates/binary/candidates_n23_k1_d9.py` | 1 | Yes |
| `candidates/qudit/candidates_q3_n11_k3_d5.py` | 1 | Yes |

The incomplete `[[17,5,5]]` candidate database is not distributed in this repository. Other stored databases include unfinished, unknown, and running tests; several refined LP entries are `UNKNOWN`, `RUNNING`, or `NOT TESTED`. H × RX and RX × RY sweeps must not be described as complete based on these files.

## Structural results

### `[[14,3,5]]`

`structural_obstructions/pauli_shadow/verify_n14_all47_packing_v2.py` safely reads the literal candidate database using AST parsing and `ast.literal_eval`; it does not execute the candidate file. The run on the supplied database checked 47 entries, excluded 15 by counting, returned 30 as not excluded by that counting test, and marked 2 not applicable because the necessary dual weight-two condition failed. The tool's own documentation warns that it does not establish completeness of the SLR3 list.

`structural_obstructions/simplex/verify_n14_3_5.py` independently checks two embedded distributions, labels 17 and 35, with exact integer arithmetic. The recorded output is preserved in `structural_obstructions/simplex/verify_n14_3_5_results.txt`. It verifies the enumerator calculations and the resulting packing-bound violations for those two distributions. It does not establish completeness of the LP survivor list or prove the separate combinatorial lemma.

The manuscript's `[[14,3,5]]` nonexistence argument therefore relies on its structural reasoning in addition to these scoped computational checks. The checked scripts should not be presented as independently excluding all 47 input distributions.

### `[[17,6,5]]`

`structural_obstructions/verification/verify_n17_6_5.py` uses AST parsing and exact integer arithmetic. On the supplied database it checked 292 distinct candidates and reported all 292 excluded: 258 by `R3 > 17`, 1 by `R1 > 1`, and 33 by the mixed case. The verifier requires the database's ordinary SLR stage to be marked complete. Its result is a check of the stored candidate list and the implemented counting inequalities; interpretation as a code nonexistence proof also uses the manuscript's structural bound.

## Reproduction commands

Run from the repository root with Python 3.10 or later:

```bash
python structural_obstructions/pauli_shadow/verify_n14_all47_packing_v2.py --require-47
python structural_obstructions/simplex/verify_n14_3_5.py
python structural_obstructions/verification/verify_n17_6_5.py
```

All three use only the Python standard library and exact integer or rational arithmetic. They do not rerun an SLR or refined LP sweep.
