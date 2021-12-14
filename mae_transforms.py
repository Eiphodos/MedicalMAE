import numpy as np
from PIL import Image


class ClipCTIntensity(object):
    def __init__(self, ct_min, ct_max):
        self.ct_min = ct_min
        self.ct_max = ct_max

    def __call__(self, img):
        npimg = np.array(img).astype(np.int32)
        # Convert from 16-bit image not already done.
        if np.min(npimg) >= 32768:
            npimg = npimg - 32768
        windowed_npimg = np.minimum(255, np.maximum(0, (npimg-self.ct_min)/(self.ct_max-self.ct_min)*255))
        windowed_npimg = windowed_npimg.astype(np.uint8)
        windowed_img = Image.fromarray(windowed_npimg)
        return windowed_img.convert('L')

    def __repr__(self):
        return self.__class__.__name__ + '(min={}, max={})'.format(self.ct_min, self.ct_max)