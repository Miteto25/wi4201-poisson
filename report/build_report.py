"""Refresh CSV-backed LaTeX tables, compile the editable report, package files.

report.tex is the authoritative source and is NEVER overwritten by this script.
"""

import csv
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report'
RESULTS = ROOT / 'results'


def read(name):
    with (RESULTS / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def tabular(name, headers, rows, spec=None):
    """Write a booktabs table; mathematical headers are intentional LaTeX."""
    folder = OUT / 'tables'
    folder.mkdir(exist_ok=True)
    lines = [r'\begin{tabular}{' + (spec or 'r'*len(headers)) + '}',
             r'\toprule', ' & '.join(headers) + r' \\', r'\midrule']
    lines += [' & '.join(str(v) for v in row) + r' \\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular}']
    (folder / (name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def sci(value):
    mantissa, exponent = f'{float(value):.3e}'.split('e')
    return rf'${mantissa}\times10^{{{int(exponent)}}}$'


def main():
    a = read('discretization_error.csv')
    s = read('ssor_summary.csv')
    f = read('ssor_asymptotic_factors.csv')
    c = read('cg_summary.csv')
    q7 = read('schwarz_q7_summary.csv')[0]
    q8 = read('schwarz_q8_parameters.csv')
    comparison = read('schwarz_q8_comparison.csv')
    assert len(a)==5 and len(s)==4 and len(c)==12 and len(q8)==16
    for rows,key in ((s,'final_relative_residual'),(c,'final_relative_residual'),
                     (q8,'final_relative_residual'),(comparison,'true_relative_residual')):
        assert all(0 <= float(row[key]) <= 1e-10 for row in rows)
    tabular('accuracy', [r'$N$',r'$n$',r'$h$',r'$E_h$',r'Order'], [
        [r['N'],r['n'],f"$1/{r['N']}$",sci(r['error_inf']),
         '--' if r['order']=='nan' else f"{float(r['order']):.4f}"] for r in a])
    tabular('ssor', [r'$N$',r'$n$',r'$K$',r'CPU (s)',r'CPU$/nK$'], [
        [r['N'],r['n'],r['iterations'],f"{float(r['cpu_time_seconds']):.5f}",
         sci(float(r['cpu_time_seconds'])/(int(r['n'])*int(r['iterations'])))] for r in s])
    tabular('factors',[r'$N$',r'$q_{K-4}$',r'$q_{K-3}$',r'$q_{K-2}$',r'$q_{K-1}$',r'$q_K$',r'$\widehat\rho$'],[
        [r['N'],*[f"{float(r[k]):.6f}" for k in ('ratio_K_minus_4','ratio_K_minus_3',
          'ratio_K_minus_2','ratio_K_minus_1','ratio_K','estimated_factor')]] for r in f])
    tabular('cg',[r'$N$',r'Method',r'$K$',r'CPU (s)',r'True residual'],[
        [r['N'],r['method'],r['iterations'],f"{float(r['cpu_time_seconds']):.6f}",sci(r['final_relative_residual'])] for r in c], 'rlrrr')
    tabular('parameters',[r'$p$',r'$\ell$',r'$\delta$',r'$K$',r'Setup (s)',r'PCG (s)',r'Total (s)',r'$n_{\max}$'],[
        [r['p'],r['overlap'],f"{float(r['delta']):.6f}",r['iterations'],
         *[f"{float(r[k]):.5f}" for k in ('setup_cpu_seconds','pcg_cpu_seconds','combined_cpu_seconds')],
         r['largest_local_size']] for r in q8])
    tabular('comparison',[r'Method',r'$K$',r'CPU (s)',r'True residual'],[
        [r['method'],r['iterations'],f"{float(r['iteration_cpu_seconds']):.6f}",sci(r['true_relative_residual'])] for r in comparison], 'lrrr')
    best = min(q8,key=lambda r:float(r['combined_cpu_seconds']))
    macros = {
        'QsevenIterations':q7['iterations'], 'QsevenResidual':sci(q7['final_true_relative_residual']),
        'QsevenDifference':sci(q7['difference_from_direct_inf']),
        'QsevenSetup':f"{float(q7['setup_cpu_seconds']):.6f}",
        'QsevenSolve':f"{float(q7['pcg_cpu_seconds']):.6f}",
        'BestP':best['p'],'BestOverlap':best['overlap'],
        'BestTime':f"{float(best['combined_cpu_seconds']):.5f}"}
    (OUT/'tables'/'values.tex').write_text('\n'.join(
        rf'\newcommand{{\{name}}}{{{value}}}' for name,value in macros.items()),encoding='utf-8')
    compiler = shutil.which('tectonic') or ROOT/'.tools'/'tectonic'/'tectonic.exe'
    if Path(compiler).is_file() or shutil.which(str(compiler)):
        subprocess.run([str(compiler),'report.tex','--keep-logs'],cwd=OUT,check=True)
    elif shutil.which('pdflatex'):
        for _ in range(2):
            subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=OUT,check=True)
    else:
        raise RuntimeError('Install Tectonic or pdfLaTeX to compile report/report.tex.')
    environment = {'python':platform.python_version(),'platform':platform.platform(),
                   'packages':{p:metadata.version(p) for p in ('numpy','scipy','matplotlib','pytest')}}
    (RESULTS/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    with zipfile.ZipFile(ROOT/'report-latex.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for p in OUT.rglob('*.tex'):
            bundle.write(p,p.relative_to(OUT))
        for p in RESULTS.glob('*.png'):
            bundle.write(p,'figures/'+p.name)
    with zipfile.ZipFile(ROOT/'submission.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        files = [ROOT/p for p in ('README.md','pyproject.toml','reproduce.py','AUDIT.md')]
        for folder in ('src/wi4201','experiments','tests','report','results'):
            files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and not p.name.startswith('.')
                         and p.suffix in ('.py','.tex','.pdf','.png','.csv','.json'))
        for p in files:
            bundle.write(p,p.relative_to(ROOT))
    print('Compiled report/report.pdf; refreshed submission.zip and report-latex.zip')


if __name__ == '__main__':
    main()
