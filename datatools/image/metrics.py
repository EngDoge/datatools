import numpy as np
from typing import Optional, List


class RunningMetrics:
    INDEICES = ['accuracy', 'accuracy_cls', 'mIoU', 'IoU']

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        self.class_id = [f"Class#{i}" for i in range(num_classes)]
        self.epsilon = np.finfo(np.float32).eps

    def update(self, pred: np.ndarray, target: np.ndarray):
        assert pred.shape == target.shape, f"pred shape {pred.shape} != target shape {target.shape}"
        pred = pred.flatten()
        target = target.flatten()
        self.confusion_matrix += self._fast_hist(pred, target, self.num_classes)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    @property
    def accuracy(self):
        return np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.epsilon)

    @property
    def recall(self):
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + self.epsilon)

    @property
    def accuracy_cls(self):
        return np.nanmean(self.recall)

    @property
    def _IoU(self):
        return np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) +
                                                 self.confusion_matrix.sum(axis=0) -
                                                 np.diag(self.confusion_matrix) +
                                                 self.epsilon)

    @property
    def mIoU(self):
        return np.nanmean(self._IoU)

    @property
    def mIoU_non_zero(self):
        return np.nanmean(self._IoU[self._IoU != 0])

    @property
    def IoU(self):
        return dict(zip(self.class_id, self._IoU))

    def get_metrics(self, indices: Optional[List] = None):
        if indices is None:
            return {index: getattr(self, index) for index in self.INDEICES}
        return {index: getattr(self, index) for index in indices}

    @staticmethod
    def _fast_hist(pred, label, n_class):
        mask = (label >= 0) & (label < n_class)
        hist = np.bincount(n_class * label[mask].astype(int) + pred[mask],
                           minlength=n_class ** 2).reshape(n_class, n_class)
        return hist


if __name__ == '__main__':
    from datatools import SingleImage, ImageConvertor
    from datatools.image import AVICompDataMapper

    running_metrics = RunningMetrics(num_classes=len(AVICompDataMapper.classes))
    img = SingleImage(r"\data\dataset\Workshop\luopengcheng\Test\20231019_test_compseg\compnet0925\Ref\1695548438432_NG9_9_003_4_ref_mask.png")
    img2 = SingleImage(
        r"\data\dataset\Workshop\luopengcheng\Test\20231019_test_compseg\compnet0925\Ref\1695548438432_NG9_9_003_4_ref_mask_old.png")
    id_img = ImageConvertor.color2idx(img=img.image, color_map=AVICompDataMapper.bgr_to_idx)
    id_img_2 = ImageConvertor.color2idx(img=img2.image, color_map=AVICompDataMapper.bgr_to_idx)
    running_metrics.update(id_img, id_img_2)
    print(running_metrics.mIoU)

