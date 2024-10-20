import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ref import warp_image_kiran

import cv2

# ===== Helper functions ======

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

    if len(img.shape) == 2: # Grayscale image
        interpolator = RegularGridInterpolator((y, x), img, bounds_error=False, fill_value=0)
        interpolated_colors = interpolator(coords)
    else: # RGB image
        interpolated_colors = []
        for c in range(3):
            interpolator = RegularGridInterpolator((y, x), img[:, :, c], bounds_error=False, fill_value=0)
            interpolated_colors.append(interpolator(coords))
        interpolated_colors = np.stack(interpolated_colors, axis=-1)

    return interpolated_colors


# ===== Algorithms =====

# Returns the homography matrix that maps im1 to im2 using given correspondances
def compute_homography(im1_pts, im2_pts):
    n = len(im1_pts)
    A = []
    y = []

    for i in range(n):
        y1, x1 = im1_pts[i]
        y2, x2 = im2_pts[i]
        A.append([y1, x1, 1, 0, 0, 0, -y1 * y2, -x1 * y2])
        A.append([0, 0, 0, y1, x1, 1, -y1 * x2, -x1 * x2])
        y.extend([y2, x2])

    A = np.array(A)
    b = np.array(y)

    # Solve for h using least squares
    h, _, _, _ = np.linalg.lstsq(A, b)

    # Add the last element 1 to form the homography matrix
    H = np.append(h, 1).reshape(3, 3)
    return H

# Returns the warped image after applying the given homography transformation
def warp_image(im, H):
    h, w = im.shape[:2]

    # Warp each corner of the image
    src_corners = [(0, 0, 1),  (h-1, 0, 1), (h-1, w-1, 1), (0, w-1, 1)]
    target_corners = [H @ np.array(corner) for corner in src_corners]

    # Scale target corners into (y, x) coordinates
    scaled_target_corners = np.array([(corner / corner[2])[:2] for corner in target_corners])

    # Compute the bounding box (potentially in negative space)
    y_max = int(np.ceil(max([y for y, _ in scaled_target_corners])))
    y_min = int(np.floor(min([y for y, _ in scaled_target_corners])))
    x_max = int(np.ceil(max([x for _, x in scaled_target_corners])))
    x_min = int(np.floor(min([x for _, x in scaled_target_corners])))

    # Get all points in target image (may include negative coordinates)
    target_coords = np.array([(y, x) for y in np.arange(y_min, y_max + 1) for x in np.arange(x_min, x_max + 1)])

    # Find the preimage for each of these coordinates
    H_inv = np.linalg.inv(H)    
    target_coords_mat = homogeneous_coords(target_coords).T
    ys, xs, ws = H_inv @ target_coords_mat
    ys /= ws
    xs /= ws

    # Find out which preimages are actually in the original image
    bounded_indices = np.where((0 <= xs) & (xs < w) & (0 <= ys) & (ys < h))
    valid_ys = ys[bounded_indices].astype(int)
    valid_xs = xs[bounded_indices].astype(int)

    # Construct warped image using nearest neighbor interpolation
    y_shift = -y_min
    x_shift = -x_min
    warped_im = np.zeros((y_max + y_shift + 1, x_max + x_shift + 1, 3)).astype(np.uint8)
    valid_target_coords = target_coords[bounded_indices]
    shifted_valid_target_coords = valid_target_coords + np.array((y_shift, x_shift))
    warped_im[shifted_valid_target_coords[:, 0], shifted_valid_target_coords[:, 1]] = im[valid_ys, valid_xs]
    return warped_im


# Return a warped version of the image with a "straightened out" rectangle
def rectify_image(im, im_pts, rect_pts):
    H = compute_homography(np.array(im_pts), np.array(rect_pts))
    return warp_image(im, H)