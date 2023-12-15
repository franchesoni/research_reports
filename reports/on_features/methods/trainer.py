from pathlib import Path
import os
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from losses import losses_dict


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


class TrainableModule(torch.nn.Module):
    def __init__(self, model, loss_fn, comment=''):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.logger = SummaryWriter(comment=comment)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, global_step):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, global_step)
        if batch_idx == 0:
            for i in range(len(x)):
                self.logger.add_image(
                    f"train_img_{str(i).zfill(2)}", x[i], global_step=global_step
                )
                for c in range(y_hat.shape[1]):
                    self.logger.add_image(
                        f"train_pred_{str(i).zfill(2)}_{c}",
                        y_hat[i, c : c + 1],
                        global_step=global_step,
                    )
        return loss

    def validation_step(self, batch, batch_idx, global_step):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        self.log("val/main_loss", loss, global_step)

        for loss_name, loss_fn in losses_dict.items():
            self.log(
                f"val/{loss_name}",
                loss_fn(y_hat, y),
                global_step,
            )

        # save image
        if batch_idx == 0:
            for i in range(len(x)):
                self.logger.add_image(
                    f"val_img_{str(i).zfill(2)}", x[i], global_step=global_step
                )
                for c in range(y_hat.shape[1]):
                    self.logger.add_image(
                        f"val_pred_{str(i).zfill(2)}_{c}",
                        y_hat[i, c : c + 1],
                        global_step=global_step,
                    )
                # log color image
                self.logger.add_image(
                    f"val_img_{str(i).zfill(2)}_color",
                    y_hat[i][:3],
                    global_step=global_step,
                )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, factor=0.5, verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim, T_0=10, T_mult=2, eta_min=1e-8, verbose=False
        )
        return optim, scheduler

    def log(self, name, value, step):
        self.logger.add_scalar(name, value, step)


class Trainer:
    def __init__(
        self,
        max_epochs=24,
        fast_dev_run=False,
        overfit_batches=0,
        val_check_interval=None,
        device=None,
    ):
        self.max_epochs = max_epochs
        self.fast_dev_run = fast_dev_run
        self.overfit_batches = overfit_batches
        self.val_check_interval = val_check_interval
        self.device = device
        self.best_val_loss = float("inf")
        self.global_step = 0

    def fit(self, model, train_dataloaders, val_dataloaders, compile=False):
        model.to(self.device)
        if compile:
            print("compiling...")
            model.model = torch.compile(model.model)
            print("compiled.")
        optimizer, scheduler = model.configure_optimizers()
        # save hparams
        hparams = {
            "max_epochs": self.max_epochs,
            "fast_dev_run": self.fast_dev_run,
            "overfit_batches": self.overfit_batches,
            "val_check_interval": self.val_check_interval,
            "model": str(model.model.__class__.__name__),
            "loss_fn": str(model.loss_fn),
        }
        print("hparams:", hparams)
        model.logger.add_hparams(hparams, {})
        with open(os.path.join(model.logger.log_dir, "hparams.txt"), "w") as f:
            json.dump(hparams, f, indent=4)

        dllen = len(train_dataloaders)
        epoch_width = len(str(self.max_epochs))
        batch_idx_width = len(str(dllen))

        for epoch in range(self.max_epochs):
            # Training Phase
            for batch_idx, batch in enumerate(train_dataloaders):
                model.train()
                if self.overfit_batches > 0 and batch_idx >= self.overfit_batches:
                    break

                batch = [item.to(self.device) for item in batch]
                current_lr = optimizer.param_groups[0]["lr"]
                model.logger.add_scalar("lr", current_lr, self.global_step)
                loss = model.training_step(
                    batch, batch_idx, global_step=self.global_step
                )
                check_for_nan(loss, model, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.global_step += 1

                if self.fast_dev_run:
                    print("fast_dev_run, one batch only")
                    break

                # print(' ' * 100, end='\r')
                print(
                    f"{batch_idx / dllen:.4%}".ljust(8)
                    + f"of epoch {epoch:{epoch_width}}".ljust(15)
                    + f", Batch: {batch_idx:{batch_idx_width}} / {dllen},".ljust(20)
                    + f"Train loss: {loss.item():.4f}",
                    end="\r",
                )
                # Validation Check at Specified Batch Interval
                if (
                    self.val_check_interval
                    and self.global_step % self.val_check_interval == 0
                    or self.global_step == 1
                ):
                    self._validate(
                        model, val_dataloaders, small_val=self.overfit_batches > 0
                    )

            # End of Epoch Validation Check if Interval Not Specified
            if self.val_check_interval is None:
                self._validate(
                    model, val_dataloaders, small_val=self.overfit_batches > 0
                )

            if self.fast_dev_run:
                print("fast_dev_run, one validation only")
                break

    def _validate(self, model, val_dataloader, small_val):
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                batch = [item.to(self.device) for item in batch]
                loss = model.validation_step(
                    batch, batch_idx, global_step=self.global_step
                )
                check_for_nan(loss, model, batch)
                val_loss += loss.item()
                if small_val and batch_idx == 10:
                    break

        val_loss = val_loss / 10 if small_val else val_loss / len(val_dataloader)
        print(" " * 100, end="\r")
        print(f"Global Step: {self.global_step}, Validation Loss: {val_loss:.4f}")

        # Directory where logs are written
        log_dir = model.logger.log_dir

        # Checkpointing Last Validated Model
        last_validated_model_path = os.path.join(log_dir, "last_validated_model.pth")
        torch.save(model.state_dict(), last_validated_model_path)

        # Checkpointing Best Model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
