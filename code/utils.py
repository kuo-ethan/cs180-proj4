import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import skimage.io as skio
import cv2


# ===== Helpers ======


# Appends a one to each coordinate
def homogeneous_coords(coords):
    ones = np.ones((coords.shape[0], 1))
    return np.hstack([coords, ones])


# Samples the color for each coordinate using bilinear interpolation. Coords outside the image default to 0.
def bilinear_interpolation(img, coords):
    height, width = img.shape[:2]

    # Create the grid for the image
    x = np.arange(width)
    y = np.arange(height)

    if len(img.shape) == 2:  # Grayscale image
        interpolator = RegularGridInterpolator(
            (y, x), img, bounds_error=False, fill_value=0
        )
        interpolated_colors = interpolator(coords)
    else:  # RGB image
        interpolated_colors = []
        for c in range(3):
            interpolator = RegularGridInterpolator(
                (y, x), img[:, :, c], bounds_error=False, fill_value=0
            )
            interpolated_colors.append(interpolator(coords))
        interpolated_colors = np.stack(interpolated_colors, axis=-1)

    return interpolated_colors


def display_img_with_keypoints(img, keypoints):
    plt.imshow(img)
    x_coords, y_coords = zip(*keypoints)
    plt.scatter(x_coords, y_coords, color="red", marker="o", s=10)
    plt.show()


# ===== Algorithms =====


# Returns the homography matrix that maps im1 to im2 using given correspondances
def compute_homography(im1_pts, im2_pts):
    n = len(im1_pts)
    A = []
    y = []

    for i in range(n):
        x1, y1 = im1_pts[i]
        x2, y2 = im2_pts[i]
        A.append([x1, y1, 1, 0, 0, 0, -x1 * x2, -y1 * x2])
        A.append([0, 0, 0, x1, y1, 1, -x1 * y2, -y1 * y2])
        y.extend([x2, y2])

    A = np.array(A)
    b = np.array(y)

    # Solve for h using least squares
    h, _, _, _ = np.linalg.lstsq(A, b)

    # Add the last element 1 to form the homography matrix
    H = np.append(h, 1).reshape(3, 3)
    return H


# Returns the warped image after applying a homography and a shift vector to bring it to its "true" coordinates
def warp_image(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]])
    corners = H @ np.reshape(corners, (4, 3, 1))  # 4 column vectors
    corners = np.array(
        [corner / corner[2] for corner in corners]
    )  # account for scaling factor w

    corners = np.squeeze(corners)
    x_values = corners[:, 0]
    y_values = corners[:, 1]
    min_x = int(np.min(x_values))
    max_x = int(np.max(x_values))
    min_y = int(np.min(y_values))
    max_y = int(np.max(y_values))

    x_range = np.arange(min_x, max_x)
    y_range = np.arange(min_y, max_y)

    x, y = np.meshgrid(x_range, y_range)
    im_coords = np.vstack([x.ravel(), y.ravel(), np.ones(x.size)])

    xs, ys, ws = np.linalg.inv(H) @ im_coords
    xs = xs / ws
    ys = ys / ws

    colors = bilinear_interpolation(im, np.column_stack((ys, xs)))
    colors = np.reshape(colors, (max_y - min_y, max_x - min_x, 3))

    return colors.astype(np.uint8), min_x, min_y


# Return a warped version of the image with a "straightened out" rectangle
def rectify_image(im, im_pts, rect_pts):
    H = compute_homography(np.array(im_pts), np.array(rect_pts))
    ret, _, _ = warp_image(im, H)
    return ret


# Returns a mosaic of two images, where the second image is projected onto the first
def build_mosaic(im1, im2, im1_pts, im2_pts):
    H = compute_homography(np.array(im2_pts), np.array(im1_pts))
    warped_im2, dx, dy = warp_image(im2, H)

    im1_min_x, im1_max_x = 0, im1.shape[1] - 1
    im1_min_y, im1_max_y = 0, im1.shape[0] - 1
    warped_im2_min_x, warped_im2_max_x = dx, dx + warped_im2.shape[1] - 1
    warped_im2_min_y, warped_im2_max_y = dy, dy + warped_im2.shape[0] - 1

    min_x = min(im1_min_x, warped_im2_min_x)
    max_x = max(im1_max_x, warped_im2_max_x)
    min_y = min(im1_min_y, warped_im2_min_y)
    max_y = max(im1_max_y, warped_im2_max_y)

    ret = np.zeros((max_y - min_y + 1, max_x - min_x + 1, 3))
    shift_x, shift_y = -min_x, -min_y

    # Shift and place im1 into ret
    ret[shift_y : shift_y + im1.shape[0], shift_x : shift_x + im1.shape[1]] = im1

    # Shift and place warped_im2 into the correct position in ret
    warped_y_start = warped_im2_min_y + shift_y
    warped_y_end = warped_y_start + warped_im2.shape[0]
    warped_x_start = warped_im2_min_x + shift_x
    warped_x_end = warped_x_start + warped_im2.shape[1]

    # Extract the region where warped_im2 will be placed
    ret_region = ret[warped_y_start:warped_y_end, warped_x_start:warped_x_end]

    # Find non-zero pixels in ret_region (overlapping pixels)
    overlap_mask = ret_region != 0

    # Average overlapping pixels
    ret_region[overlap_mask] = (ret_region[overlap_mask] + warped_im2[overlap_mask]) / 2

    # Assign non-overlapping pixels directly
    ret_region[~overlap_mask] = warped_im2[~overlap_mask]

    return ret.astype(np.uint8)
