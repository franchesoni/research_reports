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