import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryFBetaScore
import torch

# Threshold for the risks expressed in logarithmic scale
THRESHOLD = -6


class RiskTrainer(LightningModule):
    def __init__(self, model,lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.save_hyperparameters(ignore=['model'])
        self.mse = torch.nn.MSELoss()
        self.f2 = BinaryFBetaScore(beta=2.0)

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.evaluate(y, y_hat)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        print("\ngt:",y,"\npred:",y_hat)
        return y_hat

    # def validation_epoch_end(self):
    #     pass
    
    # def training_epoch_end(self):
    #     pass

    # def test_epoch_end(self):
    #     pass
    
    def evaluate(self, gt_risk, pred_risk):
        """Evaluate the model performance.

        Ground truth and prediction risk arrays must be expressed in logarithmic scale.

        Args:
            gt_risk: ground truth risk values as a PyTorch tensor of shape (N,)
            pred_risk: predicted risk values as a PyTorch tensor of shape (N,)

        Returns:
            The evaluation score according to the ESA challenge.
        """
        # gt_mask = gt_risk >= THRESHOLD
        # pred_mask = pred_risk >= THRESHOLD
        
        mse_loss = self.mse(gt_risk, pred_risk)
        
        # # logic to handle devide by zero error
        # if torch.all(~gt_mask):
        #     if torch.all(~pred_mask):
        #         return mse_loss
        #     else:
        #         f2_score = 1#e-1
        # elif torch.all(~pred_mask):
        #         f2_score = 1#e-1
        # else:
        #     mse_loss = self.mse(gt_risk[gt_mask], pred_risk[gt_mask])
        #     f2_score = self.f2(gt_mask, pred_mask)
        

        return mse_loss#/f2_score

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    