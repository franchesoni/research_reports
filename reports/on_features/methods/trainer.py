from pathlib import Path
import os
import torch
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from extras.utils import get_current_git_commit, clean_dir


from losses import losses_dict
from extras.sing import SING
# from lr_schedule import FileBasedLRScheduler, file_path as lr_file_path


def check_for_nan(loss, model, batch):
    try:
        assert torch.isnan(loss) == False
    except Exception as e:
        # print things useful to debug
        # does the batch contain nan?
        print("img batch contains nan?", torch.isnan(batch[0]).any())
        print("mask batch contains nan?", torch.isnan(batch[1]).any())
        # does the model weights contain nan?
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(name, "contains nan")
        # does the output contain nan?
        print("output contains nan?", torch.isnan(model(batch[0])).any())
        # now raise the error
        raise e


def minmax(tensor):
    """min max norm of tensor, return 0.5 if constant"""
    if tensor.max() == tensor.min():
        return tensor.new_tensor(0.5)
    else:
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def save_tensor_as_image(dstfile, tensor, global_step):
    tensor = tensor.detach().cpu().numpy()  # torch -> numpy
    tensor = tensor.transpose(1, 2, 0)  # CHW -> HWC
    tensor = minmax(tensor)
    tensor = tensor * 255.0
    tensor = tensor.clip(0, 255).astype(np.uint8)
    tensor = tensor if tensor.shape[2] > 1 else tensor[..., 0]
    image = Image.fromarray(tensor[..., :3])
    dstfile = Path(dstfile)
    dstfile = (dstfile.parent / (dstfile.stem + "_" + str(global_step))).with_suffix(
        ".jpg"
    )
    image.save(dstfile)

def log_input_output(name, x, y_hat, global_step, img_dstdir, out_dstdir):
    for i in range(len(x)):
        save_tensor_as_image(
            img_dstdir / f"input_{name}_{str(i).zfill(2)}",
            x[i],
            global_step=global_step,
        )
        for c in range(y_hat.shape[1]):
            save_tensor_as_image(
                out_dstdir / f"pred_channel_{name}_{str(i).zfill(2)}_{c}",
                y_hat[i, c : c + 1],
                global_step=global_step,
            )
        # log color image
        save_tensor_as_image(
            out_dstdir / f"pred_3channels_{name}_{str(i).zfill(2)}",
            y_hat[i][:3],
            global_step=global_step,
        )


class TrainableModule(torch.nn.Module):
    def __init__(
        self,
        model,
        loss_fn,
        max_lr=1e-2,
        weight_decay=5e-5,
        total_steps=1000,
        sing=False,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.sing = sing

    def forward(self, x):
        return self.model(x)

    def training_step(
        self, batch, batch_idx, global_step, return_input_output_for_logging=True
    ):
        x, y = batch
        y_hat = self.model(x)
        # check if 'image' is an argument name in self.loss_fn
        if hasattr(self.loss_fn, 'forward') and "image" in self.loss_fn.forward.__code__.co_varnames:
            loss = self.loss_fn(y_hat, y, image=x)
        else:
            loss = self.loss_fn(y_hat, y)
        if return_input_output_for_logging:
            return loss, x, y_hat
        return loss, None, None

    def validation_step(
        self,
        batch,
        batch_idx,
        global_step,
        return_input_output_for_logging=True,
        return_many_losses=True,
    ):
        # step
        x, y = batch
        y_hat = self.model(x)
        # check if 'image' is an argument name in self.loss_fn
        if hasattr(self.loss_fn, 'forward') and "image" in self.loss_fn.forward.__code__.co_varnames:
            loss = self.loss_fn(y_hat, y, image=x)
        else:
            loss = self.loss_fn(y_hat, y)

        # [optional] compute other losses
        many_losses = {}
        if return_many_losses:
            for loss_name, loss_fn in losses_dict.items():
                # check if 'image' is an argument name in self.loss_fn
                if hasattr(loss_fn, 'forward') and "image" in loss_fn.forward.__code__.co_varnames:
                    many_losses[loss_name] = loss_fn(y_hat, y, image=x)
                else:
                    many_losses[loss_name] = loss_fn(y_hat, y)

        # [optional] return input and output for logging
        if return_input_output_for_logging:
            return loss, many_losses, x, y_hat
        return loss, many_losses, None, None

    def configure_optimizers(self):
        optim_class = SING if self.sing else torch.optim.AdamW
        optim = optim_class(
            self.parameters(), lr=self.max_lr // 10, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=self.max_lr,
            total_steps=self.total_steps,
            verbose=False,
            pct_start=0.05,
            final_div_factor=1e12,
        )


 
        return optim, scheduler


class Trainer:
    def __init__(
        self,
        max_epochs=24,
        fast_dev_run=False,
        val_check_interval=None,
        device=None,
        comment="",
        extra_hparams={},
        clean=False,
    ):
        # training variables
        self.max_epochs = max_epochs
        self.fast_dev_run = fast_dev_run
        self.val_check_interval = val_check_interval
        self.device = device
        self.best_val_loss = float("inf")
        self.global_step = 0
        # logging variables
        if clean:
            clean_dir('runs')
        self.hparams = extra_hparams
        self.logger = SummaryWriter(comment=comment)
        self.img_dstdir = Path(self.logger.get_logdir()) / "images"
        self.img_dstdir.mkdir(parents=True, exist_ok=True)
        self.out_dstdir = self.img_dstdir.parent / "outputs"
        self.out_dstdir.mkdir(parents=True, exist_ok=True)

    def log(self, name, value, step):
        self.logger.add_scalar(name, value, step)

    def prefit(self, model, dllen, compile=False):
        model.to(self.device)
        if compile:
            print("compiling...")
            model.model = torch.compile(model.model)
            print("compiled.")
        optimizer, scheduler = model.configure_optimizers()

        # save hparams
        hparams = {
            "model": str(model.model.__class__.__name__),
            "loss_fn": str(model.loss_fn),
            "max_epochs": self.max_epochs,
            "fast_dev_run": self.fast_dev_run,
            "val_check_interval": self.val_check_interval,
            "git_commit": get_current_git_commit(),
        } | self.hparams
        print("hparams:", hparams)
        with open(os.path.join(self.logger.log_dir, "hparams.txt"), "w") as f:
            json.dump(hparams, f, indent=4)

        # logging variables
        batch_idx_width = len(str(dllen))
        epoch_width = len(str(self.max_epochs))
        return model, optimizer, scheduler, hparams, batch_idx_width, epoch_width

    def update_step(self, model, optimizer, scheduler, batch, log_input_output=False):
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, self.global_step)
        loss, x, y_hat = model.training_step(
            batch,
            batch_idx=None,
            global_step=self.global_step,
            return_input_output_for_logging=log_input_output
        )
        self.log("train_loss", loss, self.global_step)
        check_for_nan(loss, model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        self.global_step += 1

        if x is not None and y_hat is not None:
            log_input_output("train", x, y_hat, self.global_step, self.img_dstdir, self.out_dstdir)

        return loss, current_lr
        

    def fit(self, model, train_dataloaders, val_dataloaders, compile=False):
        dllen = len(train_dataloaders)
        model, optimizer, scheduler, hparams, batch_idx_width, epoch_width = self.prefit(model, dllen, compile)


        for epoch in range(self.max_epochs):
            # Training Phase
            model.train()
            for batch_idx, batch in enumerate(train_dataloaders):
                batch = [item.to(self.device) for item in batch]
                loss, current_lr = self.update_step(model, optimizer, scheduler, batch, log_input_output=batch_idx==0 or epoch % (self.max_epochs // 10) == 0 or epoch == self.max_epochs - 1)

                print(
                    f"{batch_idx / dllen:.4%}".ljust(8)
                    + f"of epoch {epoch:{epoch_width}}".ljust(15)
                    + f", Batch: {batch_idx:{batch_idx_width}} / {dllen},".ljust(20)
                    + f"Train loss: {loss.item():.3e}"
                    + f", lr: {current_lr:.3e}",
                    end="\r",
                )

                if self.fast_dev_run:
                    print("fast_dev_run, one batch only")
                    break


                # Validation Check at Specified Batch Interval
                if (
                    self.val_check_interval
                    and self.global_step % self.val_check_interval == 0
                    or self.global_step == 1
                ):
                    self._validate(
                        model,
                        val_dataloaders,
                    )
                    model.train()

            # End of Epoch Validation Check if Interval Not Specified
            if self.val_check_interval is None:
                self._validate(
                    model,
                    val_dataloaders,
                )

            if self.fast_dev_run:
                print("fast_dev_run, one validation only")
                break

        # log hparams
        self.logger.add_hparams(hparams, {})
        with open(os.path.join(self.logger.log_dir, "done.txt"), "w") as f:
            f.write("done")


    def overfit(self, model, train_batch, val_batch, compile=False):
        dllen = len(train_batch)
        model, optimizer, scheduler, hparams, batch_idx_width, epoch_width = self.prefit(model, dllen, compile)

        batch = [item.to(self.device) for item in train_batch]
        val_batch = [item.to(self.device) for item in val_batch]

        model.train()
        for epoch in range(self.max_epochs):
            loss, current_lr = self.update_step(model, optimizer, scheduler, batch)

            print(
                f"{epoch / self.max_epochs:.4%}".ljust(8)
                + f"of training at step {epoch:{epoch_width}}, ".ljust(15)
                + f"Train loss: {loss.item():.3e}"
                + f", lr: {current_lr:.3e}",
                end="\r",
            )

            # Validation Check at Specified Batch Interval
            if (
                self.val_check_interval
                and self.global_step % self.val_check_interval == 0
                or self.global_step == 1
            ):
                self._validate(
                    model,
                    val_batch,
                )
                model.train()


        self._validate(
            model,
            val_batch,
        )

        # log hparams
        self.logger.add_hparams(hparams, {})


    def _validate(self, model: TrainableModule, val_data):
        # check if val_data is dataloader or batch
        use_dataloader = isinstance(val_data, torch.utils.data.DataLoader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            if use_dataloader:
                losses_placeholder = {"main_loss": 0} | {
                    loss_name: 0 for loss_name in losses_dict.keys()
                }
                for batch_idx, batch in enumerate(val_data):
                    batch = [item.to(self.device) for item in batch]
                    loss, losses, x, y_hat = model.validation_step(
                        batch,
                        batch_idx,
                        global_step=self.global_step,
                        return_input_output_for_logging=batch_idx == 0,
                    )
                    check_for_nan(loss, model, batch)
                    losses_placeholder["main_loss"] += loss
                    for loss_name, loss_val in losses.items():
                        losses_placeholder[loss_name] += loss_val
                    # optionally log input output
                    if x is not None and y_hat is not None:
                        log_input_output("val", x, y_hat, self.global_step, self.img_dstdir, self.out_dstdir)
            else:
                loss, losses, x, y_hat = model.validation_step(
                    val_data,  # should be on device
                    batch_idx=None,
                    global_step=self.global_step,
                    return_input_output_for_logging=True,
                )
                # optionally log input output
                if x is not None and y_hat is not None:
                    log_input_output("val", x, y_hat, self.global_step, self.img_dstdir, self.out_dstdir)

        losses_placeholder = {
            loss_name: loss_val / len(val_data)
            for loss_name, loss_val in losses_placeholder.items()
        } if use_dataloader else losses | {'main_loss': loss}
        for loss_name, loss_val in losses_placeholder.items():
            self.log(f"val/{loss_name}", loss_val, self.global_step)

        print(" " * 79, end="\r")
        print(f"Global Step: {self.global_step}, Validation Loss: {val_loss:.4e}")

        if use_dataloader:  # not overfitting
            # Checkpointing Last Validated Model
            last_validated_model_path = os.path.join(
                self.logger.log_dir, "last_validated_model.pth"
            )
            torch.save(model.state_dict(), last_validated_model_path)

            # Checkpointing Best Model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_path = os.path.join(self.logger.log_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)




