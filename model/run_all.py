"""Run the whole pipeline in order: eda -> tune -> final model -> ablation -> error analysis -> report -> docx."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for step in ["eda.py", "tune.py", "final_model.py", "ablation.py", "error_analysis.py", "main_results.py", "report.py", "export_docx.py"]:
    print(f"=== {step}", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", str(HERE / step)], check=True, cwd=HERE.parent)
