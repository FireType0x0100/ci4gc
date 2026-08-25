"""
ci4gc

Usage:

>>> import ci4gc
>>> ci4gc.install()

For more details, check the docstrings for ``install_from_url()``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import re
import sys
import shutil
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from subprocess import run, PIPE, STDOUT
from urllib.parse import urlparse
from urllib.request import urlopen

from IPython import get_ipython

try:
    import google.colab  # noqa
except ImportError:
    raise RuntimeError("This module must ONLY run as part of a Colab notebook!")


__version__ = "0.0.1"

PREFIX = "/usr/local"

MINICONDA_REPO_URL = "https://repo.anaconda.com/miniconda/"
MINIFORGE_BUILD_REPO_URL = "https://github.com/conda-forge/miniforge"


def _chunked_sha256(path: str | Path, chunksize: int = 1_048_576) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunksize):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_python_version_info() -> tuple[int, int, int, str, int]:
    # colab_python = ".".join(map(str, sys.version_info[:2]))
    colab_python_info = sys.version_info
    return colab_python_info


def _check_git() -> None:
    assert shutil.which("git"), "💥💔💥 Git was not found!"


def install_from_url(
    installer_url: str,
    prefix: str | Path = PREFIX,
    env: dict[str, str] | None = None,
    run_checks: bool = True,
    sha256: str | None = None,
) -> None:
    """
    Download and run a constructor-like installer, patching
    the necessary bits so it works on Colab right away.

    This will restart your kernel as a result!

    Parameters
    ----------
    installer_url
        URL pointing to a ``constructor``-like installer, such
        as Miniconda or Mambaforge
    prefix
        Target location for the installation
    env
        Environment variables to inject in the kernel restart.
        We *need* to inject ``LD_LIBRARY_PATH`` so ``{PREFIX}/lib``
        is first, but you can also add more if you need it. Take
        into account that no quote handling is done, so you need
        to add those yourself in the raw string. They will
        end up added to a line like ``exec env VAR=VALUE python3...``.
        For example, a value with spaces should be passed as::

            env={"VAR": '"a value with spaces"'}
    run_checks
        Run checks to see if installation was run previously.
        Change to False to ignore checks and always attempt
        to run the installation.
    sha256
        Expected SHA256 checksum of the installer. Optional.
    """
    if run_checks:
        try:  # run checks to see if it this was run already
            return check(prefix)
        except AssertionError:
            pass  # just install

    t0 = datetime.now()
    print(f"⏬ Downloading {installer_url}...")
    installer_fn = "__installer__.sh"
    with urlopen(installer_url) as response, open(installer_fn, "wb") as out:
        shutil.copyfileobj(response, out)

    if sha256 is not None:
        digest = _chunked_sha256(installer_fn)
        assert digest == sha256, (
            f"💥💔💥 Checksum failed! Expected {sha256}, got {digest}"
        )

    print("📦 Installing...")
    task = run(
        ["bash", installer_fn, "-bfp", str(prefix)],
        check=False,
        stdout=PIPE,
        stderr=STDOUT,
        text=True,
    )
    os.unlink(installer_fn)
    with open("ci4gc_install.log", "w") as f:
        f.write(task.stdout)
    assert task.returncode == 0, (
        "💥💔💥 The installation failed! Logs are available at `/content/ci4gc_install.log`."
    )

    print("📌 Adjusting configuration...")
    cuda_version = os.environ.get("CUDA_VERSION", "*.*.*").split(".")[:2]
    prefix = Path(prefix)
    condameta = prefix / "conda-meta"
    condameta.mkdir(parents=True, exist_ok=True)
    pymaj, pymin = sys.version_info[:2]

    if cuda_version[0] == "11":
        cuda_pin = f"cudatoolkit {cuda_version[0]}.{cuda_version[1]}.*"
    else:
        # Assume forward compatibility on major version
        cuda_pin = f"cuda-version {cuda_version[0]}.*"

    with open(condameta / "pinned", "a") as f:
        f.write(f"python {pymaj}.{pymin}.*\n")
        f.write(f"python_abi {pymaj}.{pymin}.* *cp{pymaj}{pymin}*\n")
        f.write(f"{cuda_pin}\n")

    with open(prefix / ".condarc", "a") as f:
        f.write("always_yes: true\n")

    with open("/etc/ipython/ipython_config.py", "a") as f:
        f.write(
            f"""\nc.InteractiveShellApp.exec_lines = [
                    "import sys",
                    "sp = f'{prefix}/lib/python{pymaj}.{pymin}/site-packages'",
                    "if sp not in sys.path:",
                    "    sys.path.insert(0, sp)",
                ]
            """
        )
    sitepackages = f"{prefix}/lib/python{pymaj}.{pymin}/site-packages"
    if sitepackages not in sys.path:
        sys.path.insert(0, sitepackages)

    print("🩹 Patching environment...")
    env = env or {}
    bin_path = f"{prefix}/bin"
    if bin_path not in os.environ.get("PATH", "").split(":"):
        env["PATH"] = f"{bin_path}:{os.environ.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{prefix}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    os.rename(sys.executable, f"{sys.executable}.real")
    with open(sys.executable, "w") as f:
        f.write("#!/bin/bash\n")
        envstr = " ".join(f"{k}={v}" for k, v in env.items())
        f.write(f"exec env {envstr} {sys.executable}.real -x $@\n")
    run(["chmod", "+x", sys.executable])

    taken = timedelta(seconds=round((datetime.now() - t0).total_seconds(), 0))
    print(f"⏲ Done in {taken}")

    print("🔁 Restarting kernel...")
    get_ipython().kernel.do_shutdown(True)


def _parse_miniforge_installer_url(python_version_info: tuple[int, int, int, str, int]) -> str:
    _check_git()
    
    @contextlib.contextmanager
    def browse_git_repository(repository_url):
        previous_directory = os.getcwd()
        git_repository_name = PurePosixPath(urlparse(repository_url).path).stem
    
        run(["git", "clone", repository_url])
        os.chdir(git_repository_name)
        
        try:
            yield
        
        finally:
            os.chdir(previous_directory)
            shutil.rmtree(git_repository_name)
    
    version_number = ""
    
    with browse_git_repository(MINIFORGE_BUILD_REPO_URL):
        construct_yaml_path = Path("Miniforge3", "construct.yaml")
        with open(construct_yaml_path, "r") as construct_yaml:
            construct_yaml_lines = construct_yaml.readlines()
        
        python_specification_regex_string = r"^  - python ([0-9,.=<>*]+)$"
        python_specification_regex = re.compile(python_specification_regex_string)
        
        yaml_match_checkpoints = [False, False]
        
        for line_number, line in enumerate(construct_yaml_lines, start=1):
            if line.rstrip("\r\n") == "specs:":
                yaml_match_checkpoints[0] = True
                continue
            if yaml_match_checkpoints[0]:
                assert line.startswith("  - "), "💥💔💥 Invalid specification data in `Miniforge3/construct.yaml`!"
                if python_specification_regex.match(line):
                    yaml_match_checkpoints[1] = True
                    break
        
        assert yaml_match_checkpoints[0] and yaml_match_checkpoints[1], "💥💔💥 No Python version specification found in `Miniforge3/construct.yaml`!"
        
        result = run(["git", "log", "-L", f"{line_number},+1:{str(construct_yaml_path)}"], stdout=PIPE)
        
        git_output = result.stdout.decode("utf-8")
        
        git_commit_regex = re.compile(r"^commit ([0-9a-f]{40})$")
        git_python_specification_regex_string = "^([-+])" + python_specification_regex_string[1:]
        git_python_specification_regex = re.compile(git_python_specification_regex_string)
        
        git_python_context = ["", False, 0]
        
        for git_line in git_output.splitlines():
            if commit_match := re.search(git_commit_regex, git_line):
                git_python_context[0] = commit_match.group(1)
                continue
            
            if specification_match := re.search(git_python_specification_regex, git_line):
                if (
                    specification_match.group(2).startswith(".".join(map(str, python_version_info[:2])))
                    or specification_match.group(2).endswith(f"<{'.'.join(map(str, (python_version_info[0], python_version_info[1]+1)))}")
                ):
                    git_python_context[1] = True
                    
                    if specification_match.group(1) == "-":
                        git_python_context[2] = -1
                    elif specification_match.group(1) == "+":
                        git_python_context[2] = 1
                    
                break
                
        assert git_python_context[0] and git_python_context[1] and git_python_context[2] != 0, (
            "💥💔💥 The specification of the given Python version was not found in Miniforge logs!"
        )
                
        if git_python_context[2] == 1:
            tag_result = run(["git", "tag", "--contains", git_python_context[0], "--sort=-version:refname"], stdout=PIPE)
            assert tag_result.stdout.splitlines() != [], "💥💔💥 No Miniforge version number was found containing given (newer) Python version!"
            version_number = tag_result.stdout.splitlines()[0]
        elif git_python_context[2] == -1:
            tag_result = run(["git", "describe", "--tags", "--abbrev=0", f"{git_python_context[0]}^"], stdout=PIPE)
            assert tag_result.stdout != "", "💥💔💥 No Miniforge version number was found containing given (older) Python version!"
            version_number = tag_result.stdout.rstrip(b"\r\n")
    
    version_number = version_number.decode("utf-8") # The subprocess module returns 'bytes' objects in stdout
    
    assert version_number, "💥💔💥 No Miniforge version number was calculated!"
    
    machine_duplet = f"{platform.system()}-{platform.machine()}" # Python uses duplets, not triplets
    installer_name = f"Miniforge3-{version_number}-{machine_duplet}.sh"
    installer_url = f"{MINIFORGE_BUILD_REPO_URL}/{str(PurePosixPath('releases', 'download', version_number, installer_name))}"
    
    with urlopen(f"{installer_url}.sha256") as checksum_file:
        installer_checksum = checksum_file.read().rstrip(b"\r\n").decode("utf-8").split()[0]
    
    installer_info = (installer_url, installer_checksum)
    return installer_info


def install_miniforge(
    prefix: str | Path = PREFIX,
    env: dict[str, str] | None = None,
    run_checks: bool = True,
) -> None:
    """
    Install the latest version of Miniforge made for the current Python version.

    Miniforge consists of a Miniconda-like distribution optimized
    and preconfigured for conda-forge packages.

    Parameters
    ----------
    prefix
        Target location for the installation
    env
        Environment variables to inject in the kernel restart.
        We *need* to inject ``LD_LIBRARY_PATH`` so ``{PREFIX}/lib``
        is first, but you can also add more if you need it. Take
        into account that no quote handling is done, so you need
        to add those yourself in the raw string. They will
        end up added to a line like ``exec env VAR=VALUE python3...``.
        For example, a value with spaces should be passed as::

            env={"VAR": '"a value with spaces"'}
    run_checks
        Run checks to see if installation was run previously.
        Change to False to ignore checks and always attempt
        to run the installation.
    """
    installer_url, checksum = _parse_miniforge_installer_url(_get_python_version_info())
    install_from_url(
        installer_url, prefix=prefix, env=env, run_checks=run_checks, sha256=checksum
    )


# Make mambaforge the default
install = install_miniforge


def install_mambaforge(*args, **kwargs):
    print(
        "Mambaforge has been sunset. It is now identical to Miniforge. Installing Miniforge...",
        file=sys.stderr,
    )
    install_miniforge(*args, **kwargs)


def _parse_miniconda_installer_url(python_version_info: tuple[int, int, int, str, int]) -> tuple[str, str]:
    pypi_notation_python_version = f"py{''.join(map(str, python_version_info[:2]))}"
    machine_duplet = f"{platform.system()}-{platform.machine()}" # Python uses duplets, not triplets
    installer_regex_string = f"^Miniconda3-{pypi_notation_python_version}_\\d+\\.\\d+\\.\\d+(-\\d+)?-{machine_duplet}\\.sh$"
    installer_regex = re.compile(installer_regex_string)
    installer_info = None
    
    with urlopen(MINICONDA_REPO_URL) as page:
        parsed_soup = BeautifulSoup(page, "lxml")
        
        download_table = parsed_soup.select_one("table", recursive=False)
        
        for download_entry in download_table.find_all("tr", recursive=False):
            download_entry_array = download_entry.find_all("td", recursive=False)
            
            if len(download_entry_array) != 4:
                continue
            
            installer_name = download_entry_array[0].get_text()
            
            if not installer_regex.match(installer_name):
                continue
            
            installer_checksum = download_entry_array[3].get_text()
            installer_url = f"{MINICONDA_REPO_URL}/{installer_name}"
            
            installer_info = (installer_url, installer_checksum)
            
            break
    
    return installer_info


def install_miniconda(
    prefix: str | Path = PREFIX,
    env: dict[str, str] | None = None,
    run_checks: bool = True,
) -> None:
    """
    Install the latest Miniconda version available for the current Python version.

    Compatible installers may be available at https://repo.anaconda.com/miniconda/.

    Parameters
    ----------
    prefix
        Target location for the installation
    env
        Environment variables to inject in the kernel restart.
        We *need* to inject ``LD_LIBRARY_PATH`` so ``{PREFIX}/lib``
        is first, but you can also add more if you need it. Take
        into account that no quote handling is done, so you need
        to add those yourself in the raw string. They will
        end up added to a line like ``exec env VAR=VALUE python3...``.
        For example, a value with spaces should be passed as::

            env={"VAR": '"a value with spaces"'}
    run_checks
        Run checks to see if installation was run previously.
        Change to False to ignore checks and always attempt
        to run the installation.
    """
    installer_url, checksum = _parse_miniconda_installer_url(_get_python_version_info())
    print(
        "Miniconda is subject to terms of service:",
        "https://anaconda.com/legal/terms/terms-of-service",
        file=sys.stderr,
    )
    install_from_url(
        installer_url, prefix=prefix, env=env, run_checks=run_checks, sha256=checksum
    )


def install_anaconda(
    prefix: str | Path = PREFIX,
    env: dict[str, str] | None = None,
    run_checks: bool = True,
) -> None:
    """
    **Unsupported**: Anaconda distribution installation support
    has been removed as no efficient method of obtaining
    such distribution suitable for a specific Python version has been found.
    
    Install the latest Anaconda version built
    for the current Python version.

    Compatible installers may be available at https://repo.anaconda.com/archive/

    Parameters
    ----------
    prefix
        Target location for the installation
    env
        Environment variables to inject in the kernel restart.
        We *need* to inject ``LD_LIBRARY_PATH`` so ``{PREFIX}/lib``
        is first, but you can also add more if you need it. Take
        into account that no quote handling is done, so you need
        to add those yourself in the raw string. They will
        end up added to a line like ``exec env VAR=VALUE python3...``.
        For example, a value with spaces should be passed as::

            env={"VAR": '"a value with spaces"'}
    run_checks
        Run checks to see if installation was run previously.
        Change to False to ignore checks and always attempt
        to run the installation.
    """
    print(
        "Anaconda Distribution is subject to terms of service:",
        "https://anaconda.com/legal/terms/terms-of-service",
        file=sys.stderr,
    )
    print(
        "Anaconda distribution support has been deprecated. In the name of compatibility, Miniconda will be installed instead. Installing Miniconda...",
        file=sys.stderr,
    )
    install_miniconda(
        prefix=prefix, env=env, run_checks=run_checks
    )


def check(prefix: str | Path = PREFIX, verbose: bool = True) -> None:
    """
    Run some basic checks to ensure that ``conda`` has been installed
    correctly

    Parameters
    ----------
    prefix
        Location where ``conda`` was installed (should match the one
        provided for ``install()``.
    verbose
        Print success message if True
    """
    assert shutil.which("conda"), "💥💔💥 Conda not found!"

    pymaj, pymin = sys.version_info[:2]
    sitepackages = f"{prefix}/lib/python{pymaj}.{pymin}/site-packages"
    assert sitepackages in sys.path, (
        f"💥💔💥 PYTHONPATH was not patched! Value: {sys.path}"
    )
    assert f"{prefix}/bin" in os.environ["PATH"], (
        f"💥💔💥 PATH was not patched! Value: {os.environ['PATH']}"
    )
    assert f"{prefix}/lib" in os.environ["LD_LIBRARY_PATH"], (
        f"💥💔💥 LD_LIBRARY_PATH was not patched! Value: {os.environ['LD_LIBRARY_PATH']}"
    )
    if verbose:
        print("✨🍰✨ Everything looks OK!")


__all__ = [
    "install",
    "install_from_url",
    "install_mambaforge",
    "install_miniforge",
    "install_miniconda",
    "install_anaconda",
    "check",
    "PREFIX",
]
