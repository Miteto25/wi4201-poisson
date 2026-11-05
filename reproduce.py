"""Regenerate tests, experiments, report, audit and submission archive."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    commands = [[sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider']]
    commands += [[sys.executable, f'experiments/{name}.py'] for name in (
        'accuracy', 'ssor_study', 'cg_study', 'schwarz_study', 'schwarz_parameter_study'
    )]
    commands.append([sys.executable, 'report/build_report.py'])
    for command in commands:
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
