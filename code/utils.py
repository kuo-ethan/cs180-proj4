import numpy as np
from scipy.interpolate import RegularGridInterpolator

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

# Returns the warped image after applying the given homography transformation
def warp_image(im, H):
    h, w = im.shape[:2]
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [w-1, h-1, 1], [0, h-1, 1]])
    corners = (H @ np.reshape(corners, (4, 3, 1))) # 4 column vectors
    corners = np.array([corner / corner[2] for corner in corners]) # account for scaling factor w

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

    colors = bilinear_interpolation(im, np.column_stack((xs, ys)))
    colors = np.reshape(colors, (max_y - min_y, max_x - min_x, 3))

    return colors.astype(np.uint8)


# Return a warped version of the image with a "straightened out" rectangle
def rectify_image(im, im_pts, rect_pts):
    H = compute_homography(np.array(im_pts), np.array(rect_pts))
    return warp_image(im, H)