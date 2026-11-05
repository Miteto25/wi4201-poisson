# Assignment audit

Scope: WI4201_ComputerProject.pdf supplied by the user, coding questions 1-8,
theory questions 1-6, reproducibility and submission requirements.

## Implementation findings

- Assembly uses a 1D operator and Kronecker products; retains boundary unknowns,
  transfers their contributions to the RHS, and clears rows AND columns.
- Exact source and Dirichlet values match u=sin(xy).
- SSOR uses a forward-backward sweep, omega=1.5 and zero initial guesses.
- CG/PCG are implemented in project code. Stopping and histories use the true
  residual, while CG algebra uses the recurred residual.
- SSOR preconditioning applies the factored inverse with the correct scaling.
- Schwarz ownership covers all unknowns exactly once at zero overlap. Interface
  columns belong to the right strip. Extended indices use the specified layers.
- Local matrices are principal submatrices, factored once with sparse LU and
  reused. Local corrections add; no explicit inverse is constructed.
- Q8 fixes N=64, uses divisors p=2,4,8,16 and ell=0,1,2,4. All requested
  parameters, setup time and iteration time are saved.

## Corrections made during audit

- Q8 previously timed the entire solver call; it now averages the solver's
  iteration-only CPU timer, consistently excluding solver setup.
- SSOR and Q7 now repeat short timings to reduce Windows timer quantisation.
- Shared timing helpers avoid separate definitions for iteration averaging.
- Added a regression test verifying iteration CPU averaging semantics.
- Replaced an import-time print-only discretization check with assertions
  for stencil coefficients and symmetric boundary elimination.
- README now documents installation, all commands, conventions and library use.
- Added a generated report and a whitelisted submission archive.

## Evidence and practical limits

Run `python reproduce.py` for tests, all five experiments, regenerated report,
environment versions and submission archive. Numerical assertions verify
convergence and direct-solution agreement. Discretization order approaches two;
SSOR asymptotic factors approach one with refinement. Timing ranks reflect this
serial Python/SciPy implementation and are not universal algorithm rankings.
Independent measurements can vary even on the same machine.

The report is a nameless draft requiring group review and addition of group
details before submission. Editable LaTeX and a figure bundle are included;
The PDF is now typeset directly from the editable LaTeX source; generated
tables use the current CSV results. The compiler is a local build tool,
excluded from both archives. User edits to report.tex survive rebuilding.
The assignment requires justified answers and implementation choices; comments
may support those explanations rather than replacing report discussion.
Code comments cover conventions, setup, iterations, residual checks and Schwarz
ownership and accumulation. No external submission or upload is performed. The course PDF
is not redistributed in the source archive.
