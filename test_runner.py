import subprocess
import sys

import click

base = [sys.executable, "-m"]


@click.command()
@click.argument("good-analysis", type=click.Path(resolve_path=True, exists=True))
def main(good_analysis: str) -> None:
    print("===== [PYTEST] =====", flush=True)
    cmd = [
        *base, "pytest", "tests/",
        "-k", "not incompatible_with_mypyc", "--color=yes", "-v"
    ]
    subprocess.run(cmd, check=True)
    if sys.version_info >= (3, 7):
        sys.stdout.flush()
        print("===== [DIFF-SHADES - ANALYZE NEWLY BUILT WHEEL] ===== ", flush=True)
        cmd = [
            *base, "diff_shades", "analyze", ".analysis.json",
            "--repeat-projects-from", good_analysis, "--work-dir", ".cache", "-v"
        ]
        subprocess.run(cmd, check=True)

        print("===== [DIFF-SHADES - CHECK FOR FAILED FILES] =====", flush=True)
        subprocess.run([*base, "diff_shades", "show-failed", ".analysis.json", "--show-log"])

        print("===== [DIFF-SHADES - COMPARE PURE PYTHON & COMPILED RESULTS] ===== ", flush=True)
        cmd = [
            *base, "diff_shades", "compare",
            good_analysis, ".analysis.json", "--diff", "--check"
        ]
        subprocess.run(cmd, check=True)


main()
