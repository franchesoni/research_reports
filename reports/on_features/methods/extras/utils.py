import subprocess
import shutil
from pathlib import Path
import torch
from PIL import Image

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
    dstdir = Path(dirname)
    for f in dstdir.iterdir():
        # if the directory doesn't have a done.txt file remove it
        if f.is_dir() and not (f / "done.txt").exists():
            shutil.rmtree(f)

def save_tensor_as_image(tensor, dstfile, global_step):
    dstfile = Path(dstfile)
    dstfile = (dstfile.parent / (dstfile.stem + "_" + str(global_step))).with_suffix(
        ".jpg"
    )
    save(tensor, str(dstfile))

def save(tensor, name, channel_offset=0):

    def minmaxnorm(x):
        return (x - x.min()) / (x.max() - x.min())
    
    tensor = minmaxnorm(tensor)
    tensor = (tensor * 255).to(torch.uint8)
    tensor = tensor.squeeze()  # C, H*W
    tensor = tensor.reshape(-1, 224, 224)  # C, H, W
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.shape[0] == 2:
        tensor = torch.stack([tensor[0], torch.zeros_like(tensor[0]), tensor[1]], dim=0)
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[0] >= 3:
        tensor = tensor[channel_offset:channel_offset+3]
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    Image.fromarray(tensor).save(name)

