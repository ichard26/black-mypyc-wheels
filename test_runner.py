import subprocess
import sys

_, good_analysis = sys.argv
base = [sys.executable, "-m"]

print("===== [START PYTEST] =====")
cmd = [*base, "pytest", "tests/", "-k", "not incompatible_with_mypyc", "--color=yes", "-v"]
subprocess.run(cmd, check=True)

print("===== [START DIFF-SHADES - ANALYZE NEWLY BUILT WHEEL]")
cmd = [
    *base, "diff_shades", "analyze", ".analysis.json",
    "--repeat-projects-from", good_analysis, "--work-dir", ".cache", "-v"
]
subprocess.run(cmd, check=True)

print("===== [START DIFF-SHADES - CHECK FOR FAILED FILES]")
subprocess.run([*base, "diff_shades", "show-failed", ".analysis.json", "--show-log"])

print("===== [START DIFF-SHADES - COMPARE PURE PYTHON & COMPILED RESULTS]")
cmd = [
    *base, "diff_shades", "compare",
    good_analysis, ".analysis.json", "--diff", "--check"
]
subprocess.run(cmd, check=True)
