# coding=utf-8
import logging
from mxnet import gluon
from mxnet import nd as F
import numpy as np

logger = logging.getLogger(__name__)


def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    n_sample = output.shape[0]
    output = output.asnumpy().reshape((-1,))
    target = target.asnumpy().reshape((-1,))
    predict = np.where(output > 0.7, np.ones_like(output), np.zeros_like(output))

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    acc = 1.0 * pixel_correct / (pixel_labeled + 1e-5)
    return acc


class DiceLoss_with_OHEM(gluon.loss.Loss):
    def __init__(
        self, lam=0.7, weight=None, batch_axis=0, debug=False, num_kernels=7, **kwargs
    ):
        super(DiceLoss_with_OHEM, self).__init__(
            weight=weight, batch_axis=batch_axis, **kwargs
        )
        self.lam = lam
        self.kernel_loss = 0.0
        self.C_loss = 0.0
        self.pixel_acc = None
        self.debug = debug
        self.num_kernels = num_kernels
        self.kernel_acc = None

    def _ohem_single(self, score_gt, score_pred, training_masks):
        if self.debug:
            print(
                "score_gt_shape:",
                score_gt.shape,
                "score_pred_shape:",
                score_pred.shape,
                "train_mask_shape:",
                training_masks.shape,
            )
        pos_gt_thres = F.where(
            score_gt > 0.5, F.ones_like(score_gt), F.zeros_like(score_gt)
        )
        pos_num = F.sum(pos_gt_thres) - F.sum(pos_gt_thres * training_masks)

        if pos_num == 0:
            selected_mask = training_masks
            return selected_mask

        neg_lt_thres = F.where(
            score_gt <= 0.5, F.ones_like(score_gt), F.zeros_like(score_gt)
        )
        neg_num = F.sum(neg_lt_thres)
        neg_num = min(pos_num * 3, neg_num)

        if neg_num == 0:
            selected_mask = training_masks
            return training_masks
        neg_score = neg_lt_thres * score_pred
        neg_score_sorted = F.sort(neg_score.reshape(-1), is_ascend=0, axis=None)
        threshold = neg_score_sorted[neg_num - 1]
        score_gt_thres = F.where(
            score_pred >= threshold, F.ones_like(score_pred), F.zeros_like(score_pred)
        )

        trained_sample_mask = F.logical_or(score_gt_thres, pos_gt_thres)
        selected_mask = F.logical_and(trained_sample_mask, training_masks)

        return selected_mask

    def hybrid_forward(
        self, F, score_gt, kernel_gt, score_pred, training_masks, *args, **kwargs
    ):

        # import pdb; pdb.set_trace()
        # cal ohem mask
        selected_masks = []
        for i in range(score_gt.shape[0]):
            # cal for text region
            selected_mask = self._ohem_single(
                score_gt[i : i + 1], score_pred[i : i + 1, 0, :, :], training_masks[i : i + 1]
            )
            selected_masks.append(selected_mask)
        selected_masks = F.concat(*selected_masks, dim=0)

        C_pred = score_pred[:, 0, :, :]
        self.pixel_acc = batch_pix_accuracy(C_pred, score_gt)
        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * C_pred * selected_masks, axis=(1, 2))
        union = (
            F.sum(selected_masks * score_gt * score_gt, axis=(1, 2))
            + F.sum(selected_masks * C_pred * C_pred, axis=(1, 2))
            + eps
        )

        C_dice_loss = 1.0 - (2 * intersection) / (union)

        # loss for kernel
        kernel_mask = F.where(
            training_masks * C_pred > 0.5, F.ones_like(C_pred), F.zeros_like(C_pred)
        )
        kernel_mask = F.expand_dims(kernel_mask, axis=1)
        kernel_mask = F.repeat(kernel_mask, repeats=self.num_kernels - 1, axis=1)
        self.kernel_acc = batch_pix_accuracy(
            score_pred[:, 1, :, :] * score_gt, kernel_gt[:, 0, :, :]
        )
        kernel_intersection = F.sum(
            kernel_gt * score_pred[:, 1:, :, :] * kernel_mask, axis=(2, 3)
        )
        kernel_union = (
            F.sum(kernel_gt * kernel_gt * kernel_mask, axis=(2, 3))
            + F.sum(
                score_pred[:, 1:, :, :] * score_pred[:, 1:, :, :] * kernel_mask,
                axis=(2, 3),
            )
            + eps
        )
        kernel_dice = 1.0 - (2 * kernel_intersection) / kernel_union

        kernel_dice_loss = F.mean(kernel_dice, axis=1)
        self.C_loss = C_dice_loss
        self.kernel_loss = kernel_dice_loss

        loss = self.lam * C_dice_loss + (1.0 - self.lam) * kernel_dice_loss

        return loss


class DiceLoss(gluon.loss.Loss):
    def __init__(self, lam=0.7, weight=None, batch_axis=0, num_kernels=7, **kwargs):
        super(DiceLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self.lam = lam
        self.kernel_loss = 0.0
        self.C_loss = 0.0
        self.pixel_acc = None
        self.kernel_acc = None
        self.num_kernels = num_kernels

    def hybrid_forward(
        self, F, score_gt, kernel_gt, score_pred, training_masks, *args, **kwargs
    ):
        """
        kernels map's order: [1, ..., 0.5]
        """
        C_pred = score_pred[:, 0, :, :]
        self.pixel_acc = batch_pix_accuracy(C_pred, score_gt)
        # classification loss
        eps = 1e-5
        intersection = F.sum(score_gt * C_pred * training_masks, axis=(1, 2))
        union = (
            F.sum(training_masks * score_gt * score_gt, axis=(1, 2))
            + F.sum(training_masks * C_pred * C_pred, axis=(1, 2))
            + eps
        )

        C_dice_loss = 1.0 - (2 * intersection) / (union)

        # loss for kernel
        kernel_mask = F.where(
            training_masks * C_pred > 0.5, F.ones_like(C_pred), F.zeros_like(C_pred)
        )
        kernel_mask = F.expand_dims(kernel_mask, axis=1)
        kernel_mask = F.repeat(kernel_mask, repeats=self.num_kernels - 1, axis=1)
        self.kernel_acc = batch_pix_accuracy(
            score_pred[:, 1, :, :] * score_gt, kernel_gt[:, 0, :, :]
        )
        kernel_intersection = F.sum(
            kernel_gt * score_pred[:, 1:, :, :] * kernel_mask, axis=(2, 3)
        )
        kernel_union = (
            F.sum(kernel_gt * kernel_gt * kernel_mask, axis=(2, 3))
            + F.sum(
                score_pred[:, 1:, :, :] * score_pred[:, 1:, :, :] * kernel_mask,
                axis=(2, 3),
            )
            + eps
        )
        kernel_dice = 1.0 - (2 * kernel_intersection) / kernel_union

        kernel_dice_loss = F.mean(kernel_dice, axis=1)
        self.C_loss = C_dice_loss
        self.kernel_loss = kernel_dice_loss

        loss = self.lam * C_dice_loss + (1.0 - self.lam) * kernel_dice_loss

        return loss


if __name__ == '__main__':
    import numpy as np
    from mxnet import autograd

    np.random.seed(29999)
    # loss = DiceLoss_with_OHEM(lam=0.7, debug=True, num_kernels=7)
    loss = DiceLoss(lam=0.7, num_kernels=7)
    for i in range(1):
        score_gt = F.array(np.random.uniform(0, 1, size=(7, 128, 128)))
        x = F.array(np.random.uniform(0, 1, size=(7, 6, 128, 128)))
        x.attach_grad()
        x_pred = F.array(np.random.uniform(0, 1, size=(7, 7, 128, 128)))
        mask = F.ones(shape=(7, 128, 128))
        with autograd.record():
            tmp_loss = loss.forward(score_gt, x, x_pred, mask)
            # tmp_loss.backward()
        print(tmp_loss, loss.C_loss, loss.kernel_loss, loss.pixel_acc)
