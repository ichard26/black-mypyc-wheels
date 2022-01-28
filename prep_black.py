# This exists to avoid the mess that build isolation causes. Originally
# the workflow disabled build isolation so setup.py could import mypyc
# ... which worked until I added diff-shades. diff-shades needs flit_core
# to build so unfortunately disabling diff-shades meant diff-shades was
# uninstallable. I tried flipping the PIP_NO_BUILD_ISOLATION envvar
# temporarily to enable build isolation, but I didn't want to even try
# figuring out what the cross-platform syntax is.

from pathlib import Path

import tomli
import tomli_w

mypyc_req_lines = Path(".mypyc-support/mypyc-requirements.txt").read_text("utf-8").splitlines()
mypyc_reqs = [l for l in mypyc_req_lines if l and not l.strip().startswith("#")]
pyproject = tomli.loads(Path("pyproject.toml").read_text("utf-8"))

pyproject["build-system"]["requires"].extend(mypyc_reqs)

new_contents = tomli_w.dumps(pyproject, multiline_strings=True)
Path("pyproject.toml").write_text(new_contents, "utf-8")
