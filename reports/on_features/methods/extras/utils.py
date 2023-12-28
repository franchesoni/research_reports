import subprocess

def get_current_git_commit():
    try:
        # Run the git command to get the current commit hash
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
        # Decode from bytes to a string
        return commit_hash.decode('utf-8')
    except subprocess.CalledProcessError:
        # Handle the case where the command fails (e.g., not a Git repository)
        print("An error occurred while trying to retrieve the git commit hash.")
        return None

def clean_dir(dirname):
    """Removes all directories in dirname that don't have a done.txt file"""
    import shutil
    import datetime
    from pathlib import Path

    dstdir = Path(dirname)
    for f in dstdir.iterdir():
        # if the directory doesn't have a done.txt file remove it
        if f.is_dir() and not (f / "done.txt").exists():
            shutil.rmtree(f)

