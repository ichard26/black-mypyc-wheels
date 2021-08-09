from __future__ import annotations

import contextlib
import dataclasses
import difflib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections import defaultdict
from contextlib import nullcontext, redirect_stdout, redirect_stderr
from dataclasses import replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (TYPE_CHECKING, Any, ContextManager, Dict, Iterator, List, Optional,
                    Tuple)
from unittest.mock import patch

if sys.version_info >= (3, 8):
    from typing import Final
else:
    from typing_extensions import Final

import click
from click import secho, style

if TYPE_CHECKING:
    import black

GIT_BIN: Final = shutil.which("git")
FILE_RESULTS_COLOUR_KEY: Final = {
    "reformatted": "cyan",
    "nothing-changed": "magenta",
    "failed": "red"
}
TERMINAL_SIZE: Final = shutil.get_terminal_size()


# ===========================
# > Utilities
# ===========================


@dataclasses.dataclass
class MsgHelper:
    force_colors: Optional[bool] = None
    indent_size: int = 2

    def prep(self, msg: str, *, indent: int = 0, **kwargs: Any) -> str:
        return style(textwrap.indent(msg, " " * self.indent_size * indent), **kwargs)

    def emit(self, msg: str = "", *, indent: int = 0, **kwargs: Any) -> None:
        """Emit a message with indentation and colour control handled for you.

        Additional keywords arguments are forwarded to click.secho.
        """
        formatted = self.prep(msg, indent=indent)
        click.secho(formatted, color=self.force_colors, **kwargs)

    __call__ = emit

    def newline(self) -> None:
        click.echo()


emit_msg: Final = MsgHelper()
prep_msg: Final = emit_msg.prep
num: Final = partial(click.style, fg="cyan", bold=True)
run_cmd: Final = partial(
    subprocess.run, check=True, encoding="utf8", stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)


def clone_repo(url: str, *, to: Path, sha: Optional[str] = None) -> None:
    assert GIT_BIN
    if sha:
        if not to.exists():
            to.mkdir()
        run_cmd(["git", "init"], cwd=to)
        run_cmd(["git", "fetch", url, sha], cwd=to)
        run_cmd(["git", "checkout", sha], cwd=to)
    else:
        run_cmd([GIT_BIN, "clone", url, "--depth", "1", str(to)])


CommitMsg = str
CommitSHA = str


def get_commit(repo: Path) -> Tuple[CommitSHA, CommitMsg]:
    assert GIT_BIN
    proc = run_cmd([GIT_BIN, "log", "--format=%H:%s", "-n1"], cwd=repo)
    output = proc.stdout.strip()
    sha, _, msg = output.partition(":")
    return sha, msg


# NOTE: These two functions were copied straight from black.output :P


def unified_diff(a: str, b: str, a_name: str, b_name: str) -> str:
    """Return a unified diff string between strings `a` and `b`."""
    a_lines = [line for line in a.splitlines(keepends=True)]
    b_lines = [line for line in b.splitlines(keepends=True)]
    diff_lines = []
    for line in difflib.unified_diff(
        a_lines, b_lines, fromfile=a_name, tofile=b_name, n=5
    ):
        # Work around https://bugs.python.org/issue2142. See also:
        # https://www.gnu.org/software/diffutils/manual/html_node/Incomplete-Lines.html
        if line[-1] == "\n":
            diff_lines.append(line)
        else:
            diff_lines.append(line + "\n")
            diff_lines.append("\\ No newline at end of file\n")
    return "".join(diff_lines)


def colour_diff(contents: str) -> str:
    """Inject ANSI colour codes to the diff."""
    lines = contents.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("+++") or line.startswith("---"):
            line = "\033[1;37m" + line + "\033[0m"  # bold white, reset
        elif line.startswith("@@"):
            line = "\033[36m" + line + "\033[0m"  # cyan, reset
        elif line.startswith("+"):
            line = "\033[32m" + line + "\033[0m"  # green, reset
        elif line.startswith("-"):
            line = "\033[31m" + line + "\033[0m"  # red, reset
        lines[i] = line
    return "\n".join(lines)


@contextlib.contextmanager
def suppress_output() -> Iterator:
    with open(os.devnull, "w", encoding="utf-8") as blackhole:
        with redirect_stdout(blackhole), redirect_stderr(blackhole):
            yield


# ============================
# > Project definition & setup
# ============================


@dataclasses.dataclass
class Project:
    name: str
    url: str
    custom_arguments: List[str] = dataclasses.field(default_factory=list)
    python_requires: Optional[str] = None
    commit: Optional[str] = None


PROJECTS: Final = [
    Project("aioexabgp", "https://github.com/cooperlees/aioexabgp.git"),
    Project("attrs", "https://github.com/python-attrs/attrs.git"),
    Project("bandersnatch", "https://github.com/pypa/bandersnatch.git"),
    Project("blackbench", "https://github.com/ichard26/blackbench.git"),
    Project("channel", "https://github.com/django/channels.git"),
    Project(
        "django", "https://github.com/django/django.git",
        custom_arguments=[
            "--skip-string-normalization",
            "--extend-exclude",
            "/((docs|scripts)/|django/forms/models.py|tests/gis_tests/test_spatialrefsys.py|tests/test_runner_apps/tagged/tests_syntax_error.py)"
        ],
        python_requires=">=3.8"
    ),
    Project("flake8-bugbear", "https://github.com/PyCQA/flake8-bugbear.git"),
    Project("hypothesis", "https://github.com/HypothesisWorks/hypothesis.git"),
    Project("pandas", "https://github.com/pandas-dev/pandas.git"),
    Project("pillow", "https://github.com/python-pillow/Pillow.git"),
    Project("poetry", "https://github.com/python-poetry/poetry.git"),
    Project("pyanalyze", "https://github.com/quora/pyanalyze.git"),
    Project("pyramid", "https://github.com/Pylons/pyramid.git"),
    Project("ptr", "https://github.com/facebookincubator/ptr.git"),
    Project("pytest", "https://github.com/pytest-dev/pytest.git"),
    Project("scikit-lego", "https://github.com/koaning/scikit-lego"),
    Project("sqlalchemy", "https://github.com/sqlalchemy/sqlalchemy.git"),
    Project("tox", "https://github.com/tox-dev/tox.git"),
    Project("typeshed", "https://github.com/python/typeshed.git"),
    Project("virtualenv", "https://github.com/pypa/virtualenv.git"),
    Project("warehouse", "https://github.com/pypa/warehouse.git")
]
for p in PROJECTS:
    if p.custom_arguments is None:
        p.custom_arguments = ["--experimental-string-processing"]
    else:
        p.custom_arguments.append("--experimental-string-processing")


def setup_projects(projects: List[Project], workdir: Path) -> List[Tuple[Project, Path]]:
    ready = []
    for proj in projects:
        target = Path(workdir, proj.name)
        can_reuse = False
        if target.exists():
            if proj.commit is None:
                can_reuse = True
            else:
                sha, _ = get_commit(target)
                can_reuse = proj.commit == sha

        if can_reuse:
            emit_msg(f"Using pre-existing clone of {proj.name}", indent=1)
        else:
            emit_msg(f"Cloning {proj.name} (url: {proj.url}) ... ", indent=1, nl=False)
            clone_repo(proj.url, to=target, sha=proj.commit)
            emit_msg("done")

        commit_sha, commit_msg = get_commit(target)
        emit_msg(f"commit: {commit_msg}", indent=2, dim=True)
        emit_msg(f"commit-sha: {commit_sha}", indent=2, dim=True)
        proj = replace(proj, commit=commit_sha)
        ready.append((proj, target))

    return ready


# HACK: I know this is hacky but the benefit is I don't need to copy and
# paste a bunch of black's argument parsing, file discovery, and
# configuration code. I also get to keep the pretty output since I can
# directly invoke black.format_file_contents :D


def get_project_files_and_mode(
    project: Project, path: Path
) -> Tuple[List[Path], black.FileMode]:
    import black

    files = []
    mode = None

    def shim(sources: List[Path], *args: Any, **kwargs: Any) -> None:
        nonlocal files, mode
        files.extend(sources)
        mode = kwargs["mode"]

    with suppress_output(), patch("black.reformat_many", new=shim):
        black.main([str(path), *project.custom_arguments, "--check"], standalone_mode=False)

    assert files and isinstance(mode, black.FileMode)
    return sorted(files), mode


# ============================
# > Collect shade information (ie. checking black's behaviour)
# =========================


# TODO: make this a little more reasonable + helpful
FileResults = Dict[str, str]  # AKA JSON


def _check_file(path: Path, *, mode: Optional[black.FileMode] = None) -> FileResults:
    # TODO: record log files if available
    import black

    mode = mode or black.FileMode()
    if path.suffix == ".pyi":
        mode = replace(mode, is_pyi=True)

    src = path.read_text("utf8")
    try:
        dst = black.format_file_contents(src, fast=False, mode=mode)
    except black.NothingChanged:
        return {"type": "nothing-changed"}

    except Exception as err:
        return {"type": "failed", "error": err.__class__.__name__, "message": str(err)}

    return {"type": "reformatted", "dst": dst}


def check_file(
    file: Path,
    project_path: Path,
    mode: black.FileMode,
    progress_queue: multiprocessing.Queue,
) -> Tuple[str, FileResults]:
    # Duties of this function include:
    # - calling _check_file & adding the results to the queue
    # - getting progress percentage from the progress queue
    # - emitting a pretty status line once file has been process
    result = _check_file(file, mode=mode)
    normalized_path = file.relative_to(project_path).as_posix()
    pretty_result = click.style(result["type"], fg=FILE_RESULTS_COLOUR_KEY[result["type"]])
    percentage = progress_queue.get(timeout=0.2)
    width = (
        # The -6 is to account for the indentation (-2), characters between the
        # string format expressions below (-3), and a single space from the terminal
        # end (so things look less squished).
        TERMINAL_SIZE.columns - len(normalized_path) - len(result["type"]) - 6
    )
    pretty_percentage = click.style(f"[{percentage}%]".rjust(width), dim=True)
    emit_msg(f"{normalized_path}: {pretty_result} {pretty_percentage}", indent=1)

    return (normalized_path, result)


def analyze_projects(projects: List[Tuple[Project, Path]]) -> Dict[str, Any]:
    # Duties of this function include:
    # - setting up and managing the multiprocessing logic
    # - acquiring all of the data needed to check each project
    # - feeding the workers & collecting results
    # - emitting pretty status messages at per-project analysis start / end
    files_and_modes = [get_project_files_and_mode(proj, path) for proj, path in projects]
    file_count = sum(len(files) for files, _ in files_and_modes)

    def check_project_files(
        files: List[Path], project_path: Path, *, mode: black.FileMode
    ) -> Dict[str, FileResults]:
        data_packets = [(file_path, project_path, mode, progress_queue) for file_path in files]
        outputs = pool.starmap(check_file, data_packets)
        return {path: data for (path, data) in outputs}

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    for index in range(1, file_count + 1):
        percentage = round(index / file_count * 100, 1)
        progress_queue.put(percentage)

    with multiprocessing.Pool() as pool:
        results: Dict[str, Any] = {}
        for (project, path), (files, mode) in zip(projects, files_and_modes):
            emit_msg(f"\nChecking {project.name} ({len(files)} files) ...", bold=True)
            t0 = time.perf_counter()
            results[project.name] = {}
            results[project.name]["metadata"] = dataclasses.asdict(project)
            results[project.name]["files"] = check_project_files(files, path, mode=mode)
            elapsed = time.perf_counter() - t0
            emit_msg(f"{project.name} finished (took {elapsed:.3f} seconds)", bold=True)

    return results


# ==================================
# > Data representation & helpers
# ================================


JSON = Dict[str, Any]


@dataclasses.dataclass(frozen=True)
class ProjectResults:
    name: str
    data: JSON

    @property
    def metadata(self) -> JSON:
        return self.data["metadata"]

    @property
    def commit(self) -> str:
        return self.metadata["commit"]

    @property
    def files(self) -> Dict[str, FileResults]:
        return {name: results for name, results in self.data["files"].items()}

    def files_by_result_type(self, result_type: str) -> Dict[str, FileResults]:
        result_type = result_type.casefold().strip()
        return {name: results for name, results in self.files.items() if results["type"] == result_type}

    @property
    def failed(self) -> bool:
        return any(f["type"] == "failed" for f in self.files.values())

    @property
    def reformatted(self) -> bool:
        return any(f["type"] == "reformatted" for f in self.files.values())

    @property
    def nothing_changed(self) -> bool:
        return all(f["type"] == "nothing-changed" for f in self.files.values())


@dataclasses.dataclass(frozen=True)
class AnalysisData:
    data: JSON

    @property
    def projects(self) -> Dict[str, ProjectResults]:
        return {name: ProjectResults(name, data) for name, data in self.data["projects"].items()}

    @property
    def total_files(self) -> int:
        return sum(len(proj.files) for proj in self.projects.values())

    def files_by_result_type(self, result_type: str) -> Dict[str, FileResults]:
        files = {}
        for p in self.projects.values():
            files.update(p.files_by_result_type(result_type))
        return files


# =============================
# > Command implementations
# ============================


@click.group()
@click.option("--color/--no-color", default=None, help="Force the use/ban of colors in the emitted output.")
def main(color: Optional[bool]):
    """
    The Black shade analyser and comparsion tool.

    AKA Richard's personal take at a better black-primer (by stealing
    ideas from mypy-primer) :p

    Basically runs Black over millions of lines of code from various
    open source projects. Why? So any changes to Black can be gauged
    on their relative impact.

    \b
    Features include:
     - Simple but readable diffing capabilities
     - Repeatable analyses via --repeat-projects-from
     - Structured JSON output
     - Oh and of course, pretty output!

    \b
    Potential feature additions:
     - jupyter notebook support
     - per-project python_requires support
     - even more helpful output
     - stronger diffing abilities
     - better UX (particularly when things go wrong)
    """
    emit_msg.force_colors = color


@main.command()
@click.argument(
    "result-filepath", metavar="result-filepath",
    type=click.Path(resolve_path=True, path_type=Path)
)
@click.option(
    "-p", "--project",
    multiple=True,
    help="Select projects from the main list."
)
@click.option(
    "-e", "--exclude",
    multiple=True,
    help="Exclude projects from running."
)
@click.option(
    "-w", "--work-dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True, path_type=Path),
    help=(
        "Directory where project clones are used / stored. By default a"
        "temporary directory is used which will be cleaned up at exit."
        " Use this option to reuse or cache projects."
    )
)
@click.option(
    "--repeat-projects-from",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, resolve_path=True, path_type=Path),
    help=(
        "Use the same projects (and commits!) used during another anaylsis."
        " This is similar to --work-dir but for when you don't have the"
        " checkouts available."
    )
)
def analyze(
    result_filepath: Path,
    project: List[str],
    exclude: List[str],
    work_dir: Optional[Path],
    repeat_projects_from: Optional[Path]
) -> None:
    """Gather behaviour info for the current install of Black."""

    try:
        import black
    except ImportError as err:
        emit_msg(f"Couldn't import black: {err}.", fg="red", bold=True)
        emit_msg("This command requires an installation of Black.", indent=1)
        sys.exit(1)

    if GIT_BIN is None:
        emit_msg("Couldn't find a Git executable.", fg="red", bold=True)
        emit_msg("This command requires git sadly enough.", indent=1)
        sys.exit(1)

    if repeat_projects_from:
        data = json.loads(repeat_projects_from.read_text("utf-8"))
        projects = [Project(**proj.metadata) for proj in AnalysisData(data).projects.values()]
    else:
        projects = PROJECTS
    if exclude:
        excluders = [e.casefold().strip() for e in exclude]
        projects = [p for p in projects if p.name not in excluders]
    if project:
        selectors = [p.casefold().strip() for p in project]
        projects = [p for p in projects if p.name in selectors]

    workdir_provider: ContextManager
    if work_dir:
        workdir_provider = nullcontext(work_dir)
    else:
        workdir_provider = TemporaryDirectory(prefix="diff-shades-")
    with workdir_provider as _work_dir:
        emit_msg("Setting up projects ...", bold=True)
        t0 = time.perf_counter()
        prepped_projects = setup_projects(projects, Path(_work_dir))
        elapsed = time.perf_counter() - t0
        emit_msg(f"Setup finished (took {elapsed:.3f} seconds)", bold=True)

        results = {}
        results["projects"] = analyze_projects(prepped_projects)
        results["black-version"] = black.__version__

    with open(result_filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, separators=(",", ":"), ensure_ascii=False)
        f.write("\n")

    data = AnalysisData(results)
    no_changes = len(data.files_by_result_type("nothing-changed"))
    reformatted = len(data.files_by_result_type("reformatted"))
    failed = len(data.files_by_result_type("failed"))
    proj_no_changes = sum(proj.nothing_changed for proj in data.projects.values())
    proj_reformatted = sum(proj.reformatted for proj in data.projects.values())
    proj_failed = sum(proj.failed for proj in data.projects.values())
    emit_msg.newline()
    emit_msg("Summary:", bold=True)
    emit_msg(f" * ran Black over {num(len(prepped_projects))} projects & {num(data.total_files)} files")
    emit_msg(
        f" * file-wise, {num(no_changes)} were untouched, {num(reformatted)} were reformatted,"
        f" and {num(failed)} errored out"
    )
    emit_msg(
        f" * project-wise, {num(proj_no_changes)} were totally untouched,"
        f" {num(proj_reformatted)} were reformatted, and {num(proj_failed)} upsetted Black"
    )
    emit_msg.newline()
    emit_msg("I'm done looking at the painted code, have a good one!", bold=True, fg=(255,142,143))


def explain_project_differences(
    base_project_results: ProjectResults,
    other_project_results: ProjectResults,
    base_filename: str,
    other_filename: str,
    *, verbose: int
) -> Tuple[bool, str]:
    project_differed = False
    lines: List[str] = []
    differences: Dict[Tuple[str, str], Dict[str, Tuple[FileResults, FileResults]]] = defaultdict(dict)
    for path, base_results in base_project_results.files.items():
        other_results = other_project_results.files[path]
        if base_results != other_results:
            project_differed = True
            key = (base_results["type"], other_results["type"])
            differences[key][path] = (base_results, other_results)

    for kind, files in differences.items():
        header = prep_msg(f"{kind[0]} -> {kind[1]}", indent=1, fg="yellow", bold=True)
        if kind[0] == kind[1]:
            header += style(" (differing)", fg="yellow", bold=True)
        if not verbose:
            header += f": {num(len(files))} files"
        lines.append(header)
        if verbose:
            body = []
            if kind == ("reformatted", "reformatted"):
                for path, (base_results, other_results) in files.items():
                    diff = unified_diff(
                        base_results["dst"], other_results["dst"],
                        f"{path}::{base_filename}", f"{path}::{other_filename}"
                    )
                    diff = prep_msg(colour_diff(diff), indent=2)
                    body.extend(diff.splitlines())
            elif kind.count("reformatted") == 1 or kind.count("failed") == 1:
                for idx, path in enumerate(files):
                    body.append(prep_msg(f"{idx}. {path}", indent=2))
            elif kind == ("failed", "failed"):
                for path, (base, other) in files.items():
                    body.append(prep_msg(f"{path}:", indent=2))
                    body.append(prep_msg(
                        f"{base_filename}: {base['error']} - {base['message']}",
                        indent=3, dim=True
                    ))
                    body.append(prep_msg(
                        f"{other_filename}: {other['error']} - {other['message']}",
                        indent=3, dim=True
                    ))
            else:
                assert False, f"oh! there's no logic handling for {kind}"
            lines.extend(body + [""])

    if project_differed:
        lines.insert(0, "")
    else:
        lines.insert(0, style("everything is the same", fg="magenta"))
    return project_differed, "\n".join(lines)


@main.command()
@click.argument("file-one", type=click.File(mode="r", encoding="utf-8"))
@click.argument("file-two", type=click.File(mode="r", encoding="utf-8"))
@click.option("-v", "--verbose", count=True, help="Emit more details.")
def compare(file_one, file_two, verbose: int) -> None:
    """Compare two analyses for behavioural differences."""
    secho("Loading files ... ", nl=False, bold=True)
    base = AnalysisData(json.load(file_one))
    other = AnalysisData(json.load(file_two))
    secho("done\n")
    if set(base.projects) ^ set(other.projects):
        secho("ERROR: the two files don't contain the same projects:", fg="red", bold=True)
        secho(f"  {file_one.name}: {', '.join(base.data['projects'].keys())}")
        secho(f"  {file_two.name}: {', '.join(other.data['projects'].keys())}")
        sys.exit(1)

    for name, base_project in base.projects.items():
        other_project = other.projects[name]
        assert not set(base_project.files) ^ set(other_project.files), name
        secho(f"{name} ", bold=True, nl=False)
        emit_msg(f"({len(base_project.files)} files): ", nl=False)
        if base_project.metadata != other_project.metadata:
            secho("SKIPPED (due to mismatched project configuration)", dim=True)
            continue

        project_differed, report = explain_project_differences(
            base_project, other_project,
            file_one.name, file_two.name,
            verbose=verbose
        )
        emit_msg(report)

    emit_msg.newline()
    secho("As usual, we're done here, have a wonderful day/night! ", nl=False, bold=True, fg="green")
    secho("Cya! <3", bold=True, fg=(255,142,143))


if __name__ == "__main__":
    main()
