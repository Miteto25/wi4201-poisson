# WI4201 Poisson Computer Project

Finite differences for -Delta u = f on the unit square, with manufactured
solution u(x,y)=sin(xy). Includes stationary SSOR, CG, SSOR-PCG and one-level
overlapping additive Schwarz-PCG. Boundary nodes remain algebraic unknowns;
assembly preserves symmetry and positive definiteness.

## Setup (Windows PowerShell)

Python 3.10+ is required; the complete workflow was checked using Python 3.13.
Using explicit executable paths avoids selecting a different IDE interpreter.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe reproduce.py
```

On Linux/macOS use `python3 -m venv .venv` and `.venv/bin/python` in place
of the Windows executable. Run commands from the project root.
Required packages are NumPy, SciPy and Matplotlib; pytest is the development
dependency. The exact versions used are recorded in `results/environment.json`.
Installation needs package access; experiments run locally without network access.

## Reproduce individual experiments

After installation, use your environment's Python for these commands.
The final report step also requires Tectonic or pdfLaTeX (see below):

```text
python -m pytest -q -p no:cacheprovider
python experiments/accuracy.py
python experiments/ssor_study.py
python experiments/cg_study.py
python experiments/schwarz_study.py
python experiments/schwarz_parameter_study.py
python report/build_report.py
```

All commands run without prompts. `reproduce.py` runs them in order with the
noninteractive Matplotlib Agg backend. Individual scripts save figures rather
than opening windows. Reproduction overwrites generated results and the report.
CSV files contain numerical results; PNG files contain plots. Timings vary by
machine and load; the mathematical results should agree to floating-point accuracy.

## Assignment map

| Question | Implementation | Main output |
|---|---|---|
| Coding 1 | `src/wi4201/discretization.py`, `problem.py` | sparse A and rhs |
| Coding 2 | `experiments/accuracy.py` | `discretization_error.*` |
| Coding 3-5 | `experiments/ssor_study.py` | `ssor_*` |
| Coding 6 | `experiments/cg_study.py` | `cg_*` |
| Coding 7 | `experiments/schwarz_study.py` | `schwarz_q7_*` |
| Coding 8 | `experiments/schwarz_parameter_study.py` | `schwarz_q8_*` |
| Theory 1-6 and numerical discussion | `report/report.pdf`, `report/report.md` | title + 8 content pages |

## Numerical conventions and timing

Grid index (i,j) maps to i+j*(N+1): x varies fastest. All iterative solves start
from zero and stop at the true unpreconditioned relative residual <= 1e-10.
One SSOR iteration is a forward-backward sweep; omega=1.5. Physical boundary
unknowns have identity equations and are removed from interior couplings.

For Schwarz, internal interface columns belong to the strip on the right;
the final strip includes column N. Each strip extends by ell columns on each
available side, clipped at the boundary. delta=ell/N is the ONE-sided extension.
Local factorizations are created once and reused; overlapping corrections add.

All timings use `time.process_time`. Solver timers exclude matrix assembly,
matrix splitting and preconditioner construction, and include all iteration
operations, preconditioner applications and true-residual checks. Short solves
are repeated until >=0.25 accumulated iteration CPU seconds (cap 5000 calls).
Schwarz setup includes index construction, extraction and factorization; batched
setup measurements accumulate >=0.25 CPU seconds for Q7 and >=0.15 for Q8
(cap 4096 repetitions per batch). Serial implementation; no parallel speedup
is claimed. Separate experiment runs may give different timing means.

## Own algorithms and library routines

Project code implements the manufactured source/boundary functions, sparse
assembly and symmetric boundary handling, SSOR sweep equations, CG/PCG update
algorithms, strip ownership/overlap construction, Schwarz restriction and
accumulation, residual checks, experiment design and report generation.

NumPy supplies array operations, norms, dot products and line fitting. SciPy
supplies sparse storage/construction, Kronecker products, triangular solves
(`spsolve_triangular`), sparse direct solves (`spsolve`) and sparse LU
factorizations (`splu`). We do not implement those library kernels or use
SciPy's CG routine. Matplotlib supplies plots; LaTeX typesets the report;
pytest supplies test execution. No explicit local inverse is formed.

## Submission and group review

`report/report.tex` is the authoritative editable report source.
`report/build_report.py` refreshes CSV-backed tables, compiles `report/report.pdf`,
and builds `report-latex.zip` (LaTeX plus figures),
the environment record, and `submission.zip`. The archive includes only source,
tests, documentation, report and results; it excludes environments, caches,
temporary PDF tooling, local IDE settings and build metadata.

The report is currently nameless at the user's request. Before submission, add
the required group number and all three names, and have all three members review
the mathematical answers and conclusions. The PDF has a title page and eight
content pages. Submit the PDF and source files through Brightspace according
to the assignment. Nothing is uploaded automatically.

For LaTeX editing, upload `report-latex.zip` to Overleaf (main file `report.tex`),
or extract it and run `tectonic report.tex` or `pdflatex report.tex` twice in
the extracted folder. The standalone bundle includes the generated tables
and figures, so no Python run is required to edit and compile it.

For full local reproduction, install Tectonic or pdfLaTeX and put it on PATH.
The build script also recognises the portable Windows compiler at
`.tools/tectonic/tectonic.exe`; compiler binaries are excluded from submission.
Tectonic may download TeX packages and fonts on first use. Within this project,
compile from `report/` or run `python report/build_report.py` from the root.
The main source is never overwritten; only `report/tables/*.tex` is regenerated.
Check pagination after editing to retain the eight-content-page limit.
