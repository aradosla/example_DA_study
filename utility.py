import subprocess
import os

# from (https://github.com/broadinstitute/catch/blob/
#       2fedca15f921116f580de8b2ae7ac9972932e59e/catch/utils/version.py#L43)
def get_version_from_git_describe(path=None):
    """Determine a version according to git.
    This calls 'git describe', if git is available.
    Returns:
    version from git describe--tags-always-dirty' if git is
    available; otherwise, None
    """
    if path == None:
        path = os.getcwd()

    cwd = os.getcwd()

    try:
        os.chdir(path)
        cmd = ['git', 'describe', '--tags', '--always', '--dirty']
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        if not isinstance(out, str):
            out = out.decode('utf-8')
            ver = out.strip()
    except Exception:
        ver = None
        os.chdir(cwd)
    return ver

# from (https://stackoverflow.com/questions/31304041/
#       how-to-retrieve-pip-requirements-freeze-within-python)
def get_pip_freeze():
    try:
        from pip._internal.operations import freeze
    except ImportError:  # pip < 10.0
        from pip.operations import freeze
    x = freeze.freeze()
    return [ii for ii in x]
