import attr
import numpy as np


""" The default pixel coordinate type
"""
@attr.s
class PixelCoord:
    x: int = -1
    y: int = -1

    @property
    def row(self):
        return self.y

    @property
    def col(self):
        return self.x

    def is_valid(self) -> bool:
        return self.x >= 0 and self.y >= 0


# The group of method used for bounding box processing.
# Mainly used to rectify the bounding box
def first_nonzero_idx(binary_array: np.ndarray, reversed: bool) -> int:
    """
    Get the index of the first element in an array that is not zero
    reversed means whether the binary_array should be reversed
    :param binary_array: A 1-D numpy array
    :param reversed:
    :return: The index to the first non-zero element
    """
    start = 0
    end = binary_array.size
    step = 1
    if reversed:
        start = binary_array.size - 1
        end = -1
        step = -1

    # The iteration
    for i in range(start, end, step):
        if binary_array[i] > 0:
            return i

    # Everything is zero
    return None


def mask2bbox(mask_img: np.ndarray) -> (PixelCoord, PixelCoord):
    """
    Given an object binary mask, get the tight object bounding box
    as a tuple contains top_left and bottom_right pixel coord
    :param mask_img: (height, width, 3) mask image
    :return: A tuple contains top_left and bottom_right pixel coord
    """
    if len(mask_img.shape) == 2:
        binary_mask = mask_img
    else:
        binary_mask = mask_img.max(axis=2)
    n_rows, n_cols = binary_mask.shape
    # Compute sum over the row and compute the left and right
    mask_rowsum = np.sum(binary_mask, axis=0, keepdims=False)
    assert mask_rowsum.size == n_cols
    left = first_nonzero_idx(mask_rowsum, False)
    right = first_nonzero_idx(mask_rowsum, True)

    # Compute sum over the col and compute the top and bottom
    mask_colsum = np.sum(binary_mask, axis=1)
    assert mask_colsum.size == n_rows
    top = first_nonzero_idx(mask_colsum, False)
    bottom = first_nonzero_idx(mask_colsum, True)

    # Ok
    top_left = PixelCoord()
    top_left.x = left
    top_left.y = top
    bottom_right = PixelCoord()
    bottom_right.x = right
    bottom_right.y = bottom
    return top_left, bottom_right
