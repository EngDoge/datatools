import cv2
import numpy as np


def show(image: np.ndarray, name: str = 'image', destroy: bool = False, wait_key: bool = True):
    cv2.imshow(name, image)
    if wait_key:
        cv2.waitKey(0)
        if destroy:
            cv2.destroyAllWindows()
