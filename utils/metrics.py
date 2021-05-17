import numpy as np
import lpips
import torch

from impt import preset_import
import pytorch_ssim


def rgb2lum(im):
    """Converts RGB to relative luminance (if input is linear RGB) or luma
    (if input is gamma-corrected RGB).
    Args:
        im (numpy.ndarray): RGB array of shape ``(..., 3)``.
    Returns:
        numpy.ndarray: Relative luminance or luma array.
    """
    assert im.shape[-1] == 3, "Input's last dimension must hold RGB"

    lum = 0.2126 * im[..., 0] + 0.7152 * im[..., 1] + 0.0722 * im[..., 2]

    return lum

def compute_ci(data, level=0.95):
    r"""Computes confidence interval.
    Args:
        data (list(float)): Samples.
        level (float, optional): Confidence level. Defaults to :math:`0.95`.
    Returns:
        float: One-sided interval (i.e., mean :math:`\pm` this number).
    """
    from scipy import stats

    data = np.array(data).astype(float)
    n = len(data)
    se = stats.sem(data)

    return se * stats.t.ppf((1 + level) / 2., n - 1)


class Base():
    """The base metric.
    Attributes:
        dtype (numpy.dtype): Data type, with which data dynamic range is
            derived.
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.
    """

    def __init__(self, dtype):
        """
        Args:
            dtype (str or numpy.dtype): Data type, from which dynamic range will
                be derived.
        """
        self.dtype = np.dtype(dtype)
        if self.dtype.kind == 'f':
            self.drange = 1.
            print("Input type is float, so assuming dynamic range to be 1")
        elif self.dtype.kind == 'u':
            iinfo = np.iinfo(self.dtype)
            self.drange = float(iinfo.max - iinfo.min)
        else:
            raise NotImplementedError(self.dtype.kind)

    def _assert_type(self, im):
        assert im.dtype == self.dtype, (
            "Input data type ({in_dtype}) different from what was "
            "specified ({dtype})"
        ).format(in_dtype=im.dtype, dtype=self.dtype)

    def _assert_drange(self, im):
        actual = im.max() - im.min()
        assert self.drange >= actual, (
            "The actual dynamic range ({actual}) is larger than what was "
            "derived from the data type ({derived})"
        ).format(actual=actual, derived=self.drange)

    @staticmethod
    def _assert_same_shape(im1, im2):
        assert im1.shape == im2.shape, \
            "The two images are not even of the same shape"

    @staticmethod
    def _ensure_3d(im):
        if im.ndim == 2:
            return np.expand_dims(im, -1)
        if im.ndim == 3:
            assert im.shape[2] in (1, 3), (
                "If 3D, input must have either 1 or 3 channels, but has %d"
            ) % im.shape[2]
            return im
        raise ValueError(
            "Input must be 2D (H-by-W) or 3D (H-by-W-by-C), but is %dD"
            % im.ndim)

    def __call__(self, im1, im2, **kwargs):
        """
        Args:
            im1 (numpy.ndarray): An image of shape H-by-W, H-by-W-by-1,
                or H-by-W-by-3.
            im2
        Returns:
            float: The metric computed.
        """
        raise NotImplementedError


class PSNR(Base):
    """Peak Signal-to-Noise Ratio (PSNR) in dB (higher is better).
    If the inputs are RGB, they are first converted to luma (or relative
    luminance, if the inputs are not gamma-corrected). PSNR is computed
    on the luma.
    """

    def __call__(self, im1, im2, mask=None):
        """
        Args:
            im1
            im2
            mask (numpy.ndarray, optional): An H-by-W logical array indicating
                pixels that contribute to the computation.
        Returns:
            float: PSNR in dB.
        """
        self._assert_type(im1)
        self._assert_type(im2)
        im1 = im1.astype(float)  # must be cast to an unbounded type
        im2 = im2.astype(float)
        im1 = self._ensure_3d(im1)
        im2 = self._ensure_3d(im2)
        self._assert_same_shape(im1, im2)
        self._assert_drange(im1)
        self._assert_drange(im2)
        # To luma
        if im1.shape[2] == 3:
            im1 = np.expand_dims(rgb2lum(im1), -1)
            im2 = np.expand_dims(rgb2lum(im2), -1)
        # Inputs guaranteed to be HxWx1 now
        if mask is None:
            mask = np.ones(im1.shape)
        elif mask.ndim == 2:
            mask = np.expand_dims(mask, -1)
        # Mask guaranteed to be 3D
        assert mask.shape == im1.shape, (
            "Mask must be of shape {input_shape}, but is of shape "
            "{mask_shape}"
        ).format(input_shape=im1.shape, mask_shape=mask.shape)
        # Mask guaranteed to be HxWx1 now
        mask = mask.astype(bool)  # in case it's not logical yet
        se = np.square(im1[mask] - im2[mask])
        mse = np.sum(se) / np.sum(mask)
        psnr = 10 * np.log10((self.drange ** 2) / mse)  # dB
        return psnr

class SSIM(Base):
    r"""The (multi-scale) Structural Similarity Index (SSIM) :math:`\in [0,1]`
    (higher is better).
    If the inputs are RGB, they are first converted to luma (or relative
    luminance, if the inputs are not gamma-corrected). SSIM is computed
    on the luma.
    """

    def __init__(self, dtype):
        super().__init__(dtype)

    def __call__(self, im1, im2, multiscale=False):
        self._assert_type(im1)
        self._assert_type(im2)
        im1 = im1.astype(float)  # must be cast to an unbounded type
        im2 = im2.astype(float)
        im1 = self._ensure_3d(im1)
        im2 = self._ensure_3d(im2)
        self._assert_same_shape(im1, im2)
        self._assert_drange(im1)
        self._assert_drange(im2)
        # To luma
        if im1.shape[2] == 3:
            im1 = np.expand_dims(rgb2lum(im1), -1)
            im2 = np.expand_dims(rgb2lum(im2), -1)
        # Guaranteed to be 1x1xHxW now
        im1 = torch.tensor(im1).unsqueeze(dim=0).permute(0, 3, 1, 2)
        im2 = torch.tensor(im2).unsqueeze(dim=0).permute(0, 3, 1, 2)
        similarity = pytorch_ssim.ssim(im1, im2)
        similarity = similarity.numpy()
        return similarity


class LPIPS(Base):
    r"""The Learned Perceptual Image Patch Similarity (LPIPS) metric (lower is
    better).
    Project page: https://richzhang.github.io/PerceptualSimilarity/
    Note:
        This implementation assumes the minimum value allowed is :math:`0`, so
        data dynamic range becomes the maximum value allowed.
    Attributes:
        dtype (numpy.dtype): Data type, with which data dynamic range is
            derived.
        drange (float): Dynamic range, i.e., difference between the maximum and
            minimum allowed.
    """

    def __init__(self, dtype, weight_pb=None):
        """
        Args:
            dtype (str or numpy.dtype): Data type, from which maximum allowed
                will be derived.
            weight_pb (str, optional): Path to the network weight protobuf.
                Defaults to the bundled ``net-lin_alex_v0.1.pb``.
        """
        self.lpips = lpips.LPIPS(net='alex')

    def __call__(self, im1, im2):
        return self.lpips(im1, im2)
