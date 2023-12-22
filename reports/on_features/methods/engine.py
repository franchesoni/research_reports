print("importing basics...")
from pathlib import Path
import tqdm
from PIL import Image
import numpy as np

print("importing torch...")
from torchvision.transforms.functional import to_tensor
import torch

print("loading custom...")
from data import pad_resized_img, get_train_val_ds, custom_collate
from losses import losses_dict
from trainer import Trainer, TrainableModule
from network import get_network

print("firing...")


def seed_everything(seed=0):
    print("seeding")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_from_ckpt(trainable_module, ckpt_path):
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        trainable_module.load_state_dict(ckpt)
    return trainable_module


def train(
    datadir,
    loss_fn_name,
    comment,
    output_channels=3,
    ckpt_path=None,
    batch_size=8,
    epochs=9999,
    dev=False,
    overfit=0,
    val_check_interval=None,
    train_size=10000,
    val_size=10,
    dummy_decoder=False,
):
    print("getting model")
    net = get_network(output_channels=output_channels, dummy=dummy_decoder)

    train_ds, val_ds = get_train_val_ds(datadir)
    train_ds.sample_paths = train_ds.sample_paths[:train_size]
    val_ds.sample_paths = val_ds.sample_paths[:val_size]

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=overfit == 0,
        drop_last=True,
        num_workers=8 if not dev else 0,
        pin_memory=True,
        persistent_workers=not dev,
        collate_fn=custom_collate,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8 if not dev else 0,
        pin_memory=False,
        collate_fn=custom_collate,
    )

    print("initializing model and trainer")
    loss_fn = losses_dict[loss_fn_name]
    trainable = TrainableModule(net, loss_fn=loss_fn, comment=comment)
    trainable = load_from_ckpt(trainable, ckpt_path)

    trainer = Trainer(
        max_epochs=epochs,
        fast_dev_run=dev,
        overfit_batches=overfit,
        val_check_interval=None if overfit else val_check_interval,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("training")
    trainer.fit(trainable, train_dataloaders=train_dl, val_dataloaders=val_dl)


def inference(ckpt_path, test_img_dir_path, dstdir="vis", output_channels=3):
    assert Path(dstdir).exists()
    ckpt_path = Path(ckpt_path)
    net = get_network(output_channels=output_channels)
    plmodel = TrainableModule(net, loss_fn=None)
    plmodel = load_from_ckpt(plmodel, ckpt_path)
    plmodel.eval()

    # glob png or jpg
    test_img_paths = sorted(Path(test_img_dir_path).glob("*.png")) + sorted(
        Path(test_img_dir_path).glob("*.jpg")
    )

    for ind, test_img_path in tqdm.tqdm(enumerate(test_img_paths)):
        test_img = Image.open(test_img_path).convert("RGB")
        test_img = pad_resized_img(test_img)
        test_img.save(f"input{ind}.png")
        test_img = to_tensor(test_img)
        test_img = test_img / test_img.max()
        with torch.no_grad():
            output = plmodel(test_img[None])[0]
            output = (output.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(output[..., :3]).save(f"out{ind}.png")
            for c in range(4):
                Image.fromarray(output[..., c]).save(f"colorout{ind}_c{c}.png")


if __name__ == "__main__":
    seed_everything()
    print("handling args")
    from fire import Fire

    Fire({"train": train, "inference": inference})
