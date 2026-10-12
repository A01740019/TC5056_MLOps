"""Run five MLflow experiments plus a reproducibility check."""
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
CONFIGS = [
    ('E1_baseline', '1.0', 'lbfgs', '1000'),
    ('E2_regularizacion_fuerte', '0.01', 'lbfgs', '1000'),
    ('E3_regularizacion_debil', '10.0', 'lbfgs', '1000'),
    ('E4_solver_newton_cg', '1.0', 'newton-cg', '1000'),
    ('E5_solver_saga', '1.0', 'saga', '2000'),
]

for name, C, solver, max_iter in CONFIGS:
    cmd = [sys.executable, str(PROJECT / 'train.py'), '--run-name', name,
           '--C', C, '--solver', solver, '--max-iter', max_iter, '--seed', '17']
    print(f'\n>>> {name}', flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT)

# Repetition of identical configuration: sixth run; metrics should be identical.
subprocess.run([sys.executable, str(PROJECT / 'train.py'),
                '--run-name', 'E1_baseline_repeticion', '--C', '1.0',
                '--solver', 'lbfgs', '--max-iter', '1000', '--seed', '17'],
               check=True, cwd=PROJECT)
