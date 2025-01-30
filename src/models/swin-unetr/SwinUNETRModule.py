from typing import Any, Dict, Tuple, List

import torch
import logging
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import Dice, JaccardIndex, MaxMetric, MeanMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.data import decollate_batch
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from functools import partial

# References: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class SwinUNETRModule(LightningModule):
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
        criterion: torch.nn.Module,
        compile: bool,
        roi_size: Tuple[int, int, int] = [128, 128, 128],
        sw_batch_size: int = 4,
        infer_overlap: float = 0.5,
        num_classes: int = 4,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net", "criterion"])

        self.net = net

        # loss function
        self.criterion = criterion
        
        # Post processing each class
        self.post_sigmoid = Activations(sigmoid=True)
        self.post_pred = AsDiscrete(argmax=False, threshold=0.5)
        
        # Post processing for many classes
        # self.post_pred = AsDiscrete(argmax=True, to_onehot=self.hparams.num_classes)
        # self.post_label = AsDiscrete(to_onehot=self.hparams.num_classes)
        
        # Not use
        # self.model_inferer = partial(
        #     sliding_window_inference,
        #     roi_size=[self.hparams.roi_size[0], self.hparams.roi_size[1], self.hparams.roi_size[2]],
        #     sw_batch_size=self.hparams.sw_batch_size,
        #     predictor=self.forward,
        #     overlap=self.hparams.infer_overlap,
        # )
        
        # metric objects for calculating and averaging accuracy across batches
        self.train_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.val_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.test_metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        
        # self.train_metric = Dice(average='macro', num_classes=self.hparams.num_classes, ignore_index=0) # ignore background
        # self.val_metric = Dice(average='macro', num_classes=self.hparams.num_classes, ignore_index=0)
        # self.test_metric = Dice(average='macro', num_classes=self.hparams.num_classes, ignore_index=0)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best = MaxMetric()
        self.test_metric_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric.reset()
        self.val_metric_best.reset()

    def model_step(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        image, target = batch["image"], batch["label"]
        
        # Before forward
        logging.info(f"Image shape: {image.shape}, Target shape: {target.shape}")
        logging.info(f"Max Image: {torch.max(image)}, Min Image: {torch.min(image)}")
        logging.info(f"Max Target: {torch.max(target)}, Min Target: {torch.min(target)}")
        
        logits = self.forward(image)
        
        # After forward
        logging.info(f"Logits shape: {logits.shape}")
        logging.info(f"Max Logits: {torch.max(target)}, Min Logits: {torch.min(target)}")
        
        loss = self.criterion(logits, target)
        return loss, logits, target

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, pred_masks, gt_masks = self.model_step(batch)
        
        logging.info(f"Before Decollate: Pred masks: {pred_masks.shape}, GT masks: {gt_masks.shape}")
        
        labels_list = decollate_batch(gt_masks)
        outputs_list = decollate_batch(pred_masks)
        outputs_convert =  [self.post_pred(self.post_sigmoid(pred_tensor)) for pred_tensor in outputs_list]     
        
        # outputs_convert =  [self.post_pred(pred_tensor) for pred_tensor in outputs_list]
        # label_convert = [self.post_label(i) for i in labels_list]
        
        # update and log metrics
        self.train_loss(loss)
        train_dice = self.train_metric(y_pred=outputs_convert, y=labels_list)
        
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/dice", train_dice[0, 1], on_step=False, on_epoch=True, prog_bar=True) # Whole tumor

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        acc1, _ = self.train_metric.aggregate()  # get current val acc
        dice_tc, dice_wt, dice_et = acc1[0:3]
        
        self.train_metric.reset()
        
        self.log("train/dice_epoch", dice_wt, sync_dist=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred_masks, labels = self.model_step(batch)
        logging.info(f"Validation step: Image shape: {pred_masks.shape}, Target shape: {labels.shape}")

        # images, labels = batch["image"], batch["label"]
        # pred_masks = sliding_window_inference(
        #     images,
        #     [self.hparams.roi_size[0], self.hparams.roi_size[1], self.hparams.roi_size[2]],
        #     self.hparams.sw_batch_size,
        #     self.forward,
        #     overlap=self.hparams.infer_overlap
        # )
        # pred_masks = self.model_inferer(images)
        
        loss = self.criterion(pred_masks, labels)
        
        labels_list = decollate_batch(labels)
        outputs_list = decollate_batch(pred_masks)
        
        outputs_convert =  [self.post_pred(self.post_sigmoid(pred_tensor)) for pred_tensor in outputs_list]
       
        # outputs_convert =  [self.post_pred(pred_tensor) for pred_tensor in outputs_list]
        # label_convert = [self.post_label(i) for i in labels_list]
        
        # update and log metrics
        self.val_loss(loss) 
        val_dice = self.val_metric(y_pred=outputs_convert, y=labels_list) # label_convert
        logging.info(f"Val Dice Step {val_dice}")
        
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dice", val_dice[0, 1], on_step=False, on_epoch=True, prog_bar=True)
        
        return {"loss": loss, "preds": pred_masks, "targets": labels}
        
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc1, _ = self.val_metric.aggregate()  # get current val acc
        dice_tc, dice_wt, dice_et = acc1[0:3]
        self.val_metric.reset()
        
        self.val_metric_best(dice_wt)  # update best so far val acc

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/dice_best", self.val_metric_best.compute(), sync_dist=True, prog_bar=True)
        

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, pred_masks, labels = self.model_step(batch)
        
        # images, labels = batch["image"], batch["label"]
        # pred_masks = sliding_window_inference(
        #     images,
        #     [self.hparams.roi_size[0], self.hparams.roi_size[1], self.hparams.roi_size[2]],
        #     self.hparams.sw_batch_size,
        #     self.forward,
        #     overlap=self.hparams.infer_overlap
        # )
        # pred_masks = self.model_inferer(images)
        
        loss = self.criterion(pred_masks, labels)
        
        labels_list = decollate_batch(labels)
        outputs_list = decollate_batch(pred_masks)
        outputs_convert =  [self.post_pred(self.post_sigmoid(pred_tensor)) for pred_tensor in outputs_list]
        
        # outputs_convert =  [self.post_pred(pred_tensor) for pred_tensor in outputs_list]
        # label_convert = [self.post_label(i) for i in labels_list]

        # update and log metrics
        self.test_loss(loss)
        test_dice = self.test_metric(y_pred=outputs_convert, y=labels_list)
        
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log("test/dice", test_dice[0, 1], on_step=False, on_epoch=True, prog_bar=True) # whole tumor
        
        return {"loss": loss, "preds": pred_masks, "targets": labels}
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        acc1, _ = self.test_metric.aggregate()  # get current val acc
        dice_tc, dice_wt, dice_et = acc1[0:3]

        self.test_metric.reset()
        
        self.test_metric_best(dice_wt)
        
        self.log("test/dice_best", self.test_metric_best.compute(), sync_dist=True, prog_bar=True)

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
            scheduler = self.hparams.scheduler(optimizer=optimizer)
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

        print(f"output shape: {output.shape}")

    main()
