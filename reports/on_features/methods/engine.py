print("importing external...")
from pathlib import Path
import tqdm
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import torch

print("importing internal...")
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
    val_check_interval=None,
    train_size=int(1e9),
    val_size=10,
    dummy_decoder=False,
    max_lr=1e-2,
    weight_decay=1e-5,
    sing=False,
    gpu_number=0,
    model='vitregs',
    overfit=False,
    clean=False,
):
    print("getting model")
    net = get_network(output_channels=output_channels, dummy=dummy_decoder, model=model)

    print("getting dataloaders")
    train_ds, val_ds = get_train_val_ds(datadir)

    if overfit:
        train_batch = custom_collate([train_ds[i] for i in range(batch_size)])
        val_batch = custom_collate([val_ds[i] for i in range(batch_size)])
    else:
        train_ds.sample_paths = train_ds.sample_paths[:train_size]
        val_ds.sample_paths = val_ds.sample_paths[:val_size]
        train_dl = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
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
            drop_last=False,
        )



    print("initializing model and trainer")
    total_steps = epochs if overfit else len(train_dl) * epochs
    loss_fn = losses_dict[loss_fn_name]
    trainable = TrainableModule(
        net,
        loss_fn=loss_fn,
        max_lr=max_lr,
        weight_decay=weight_decay,
        total_steps=total_steps,
        sing=sing,
        )
    trainable = load_from_ckpt(trainable, ckpt_path)

    extra_hparams = dict(
        dummy_decoder=dummy_decoder,
        batch_size=batch_size,
        output_channels=output_channels,
        comment=comment,
        max_lr=max_lr,
        weight_decay=weight_decay,
        total_steps=epochs,
        overfit=overfit,
    )
    if not overfit:
        extra_hparams = extra_hparams | dict(
            train_size=train_size,
            val_size=val_size,
        )
    trainer = Trainer(
        max_epochs=epochs,
        fast_dev_run=dev,
        val_check_interval=val_check_interval,
        device=f"cuda:{gpu_number}" if torch.cuda.is_available() else "cpu",
        comment=comment,
        extra_hparams=extra_hparams,
        clean=clean,
    )
    print("training")
    if overfit:
        trainer.overfit(trainable, train_batch, val_batch)
    else:
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
