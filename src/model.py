from lightning import LightningModule
from torch import optim


class BaseModel(LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        x, adj, mask, target = batch
        preds = self(x, adj, mask)
        loss = self.loss_fn(preds, target)
        self.train_loss.update(preds, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        train_mae = self.train_loss.compute()
        self.log("train_mae_epoch", train_mae)
        self.train_loss.reset()

    def validation_step(self, batch, batch_idx):
        x, adj, mask, target = batch
        preds = self(x, adj, mask)
        loss = self.loss_fn(preds, target)
        self.val_loss.update(preds, target)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        val_mae = self.val_loss.compute()
        self.log("val_mae_epoch", val_mae, prog_bar=True)
        self.val_loss.reset()

    def test_step(self, batch, batch_idx):
        x, adj, mask, target = batch
        preds = self(x, adj, mask)
        loss = self.loss_fn(preds, target)
        self.test_loss.update(preds, target)
        self.log("test_loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        test_mae = self.test_loss.compute()
        self.log("test_mae_epoch", test_mae)
        self.test_loss.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
