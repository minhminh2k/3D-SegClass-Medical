from typing import Any, Dict, Tuple, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric
from torchmetrics import F1Score
from src.models.losses.dice_loss import DiceLoss
from src.models.losses.focal_loss import FocalLoss
from src.models.losses.iou_loss import IoULoss
import torch.nn.functional as F

class SegFormer3DLitModule(LightningModule):
    """Example of a `LightningModule` for Fine-tuning SAM.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion_1: torch.nn.Module,
        criterion_2: torch.nn.Module,
        criterion_3: torch.nn.Module,
        compile: bool,
        warmup_steps: int,
        steps: Tuple[int, int],
        decay_factor: int,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion_1", "criterion_2", "criterion_3"])

        self.net = net

        # loss function
        self.criterion_1 = criterion_1
        self.criterion_2 = criterion_2
        self.criterion_3 = criterion_3
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_metric_1 = JaccardIndex(task="binary", threshold=0.5, num_classes=2, average="micro")
        self.val_metric_1 = JaccardIndex(task="binary", threshold=0.5, num_classes=2, average="micro")
        self.test_metric_1 = JaccardIndex(task="binary", threshold=0.5, num_classes=2, average="micro")
        
        self.train_metric_2 = F1Score(task="binary", threshold=0.5, average='micro', num_classes=2)
        self.val_metric_2 = F1Score(task="binary", threshold=0.5, average='micro', num_classes=2)
        self.test_metric_2 = F1Score(task="binary", threshold=0.5, average='micro', num_classes=2)
        
        # for averaging loss across batches
        self.train_loss_1 = MeanMetric()
        self.val_loss_1 = MeanMetric()
        self.test_loss_1 = MeanMetric()
        
        self.train_loss_2 = MeanMetric()
        self.val_loss_2 = MeanMetric()
        self.test_loss_2 = MeanMetric()
        
        self.train_loss_3 = MeanMetric()
        self.val_loss_3 = MeanMetric()
        self.test_loss_3 = MeanMetric()
        
        self.train_loss_total = MeanMetric()
        self.val_loss_total = MeanMetric()
        self.test_loss_total = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best_1 = MaxMetric()
        self.val_metric_best_2 = MaxMetric()


    def forward(self, x: torch.Tensor, x_bbox: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, x_bbox)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss_1.reset()
        self.val_loss_2.reset()
        self.val_loss_3.reset()
        self.val_loss_total.reset()
        self.val_metric_1.reset()
        self.val_metric_2.reset()
        self.val_metric_best_1.reset()
        self.val_metric_best_2.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        images, bboxes, gt_masks = batch
        batch_size = images.size(0)
        
        pred_masks, iou_predictions = self.forward(images, bboxes)
        num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        
        loss_focal = torch.tensor(0., device=self.device)
        loss_dice = torch.tensor(0., device=self.device)
        loss_iou = torch.tensor(0., device=self.device)
        for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
            loss_focal += self.criterion_1(pred_mask, gt_mask, num_masks)
            loss_dice += self.criterion_2(pred_mask, gt_mask, num_masks)
            loss_iou += self.criterion_3(pred_mask, gt_mask, iou_prediction, num_masks)

        # loss_focal = self.criterion_1(pred_masks, gt_masks, num_masks)
        # loss_dice = self.criterion_2(pred_masks, gt_masks, num_masks)
        # loss_iou = self.criterion_3(pred_masks, gt_masks, iou_predictions, num_masks)
        
        total_loss = 20. * loss_focal + loss_dice + loss_iou
        return loss_focal, loss_dice, loss_iou, total_loss, \
                pred_masks, gt_masks

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_focal, loss_dice, loss_iou, total_loss, pred_masks, gt_masks = self.model_step(batch)

        # update and log metrics
        self.train_loss_1(loss_focal)
        self.train_loss_2(loss_dice)
        self.train_loss_3(loss_iou)
        self.train_loss_total(total_loss)
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            self.train_metric_1(pred_mask, gt_mask.int())
            self.train_metric_2(pred_mask, gt_mask.int())
        
        self.log("train/focal_loss", self.train_loss_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice_loss", self.train_loss_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/iou_loss", self.train_loss_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss_total, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("train/jaccard", self.train_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        

        # return loss or backpropagation will fail
        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_focal, loss_dice, loss_iou, total_loss, pred_masks, gt_masks = self.model_step(batch)
        
        # update and log metrics
        self.val_loss_1(loss_focal)
        self.val_loss_2(loss_dice)
        self.val_loss_3(loss_iou)
        self.val_loss_total(total_loss)
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            self.val_metric_1(pred_mask, gt_mask.int())
            self.val_metric_2(pred_mask, gt_mask.int())
        
        self.log("val/focal_loss", self.val_loss_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice_loss", self.val_loss_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/iou_loss", self.val_loss_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss", self.val_loss_total, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("val/jaccard", self.val_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        
        # return {"loss": loss, "preds": preds, "targets": targets}
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        jaccard = self.val_metric_1.compute()  # get current val acc
        f1 = self.val_metric_2.compute()  # get current val acc
        
        self.val_metric_best_1(jaccard)  # update best so far val acc
        self.val_metric_best_2(f1)
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/jaccard_best", self.val_metric_best_1.compute(), sync_dist=True, prog_bar=True)
        self.log("val/f1_best", self.val_metric_best_2.compute(), sync_dist=True, prog_bar=True)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_focal, loss_dice, loss_iou, total_loss, pred_masks, gt_masks = self.model_step(batch)

        # update and log metrics
        self.test_loss_1(loss_focal)
        self.test_loss_2(loss_dice)
        self.test_loss_3(loss_iou)
        self.test_loss_total(total_loss)
        
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            self.test_metric_1(pred_mask, gt_mask.int())
            self.test_metric_2(pred_mask, gt_mask.int())
        
        self.log("test/focal_loss", self.test_loss_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/dice_loss", self.test_loss_2, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/iou_loss", self.test_loss_3, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss_total, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("test/jaccard", self.test_metric_1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_metric_2, on_step=False, on_epoch=True, prog_bar=True)
        

        # return {"loss": loss, "preds": preds, "targets": targets}
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(
                optimizer=optimizer, 
                lr_lambda=lambda step: lr_lambda(step, self.hparams.warmup_steps, self.hparams.steps, self.hparams.decay_factor)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import rootutils
    from omegaconf import DictConfig, OmegaConf

    # find paths
    rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = str(path / "configs")
    print(f"project-root: {path}")
    print(f"config path: {config_path}")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(f"config: \n {OmegaConf.to_yaml(cfg.model, resolve=True)}")

        model = hydra.utils.instantiate(cfg.model)
        batch = torch.rand(1, 3, 1024, 1024)
        output = model(batch)

        print(f"output shape: {output.shape}")  # [1, 1, 256, 256]

    main()
