import numpy as np
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import cv2
from scipy.signal import convolve2d
import skimage.io as skio
import numpy as np
from skimage.feature import corner_harris, peak_local_max


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


# Given the shape of an RGB image and corner coordinates, return 3 channel distance transform
def distance_transform(im_shape, x_values=None, y_values=None):
    if x_values is None:
        mask = np.ones((im_shape[0], im_shape[1]), dtype=np.uint8)

        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

    else:
        mask = np.zeros((im_shape[0], im_shape[1]), dtype=np.uint8)
        corners = np.stack((x_values, y_values), axis=-1).astype(np.int32)
        corners = corners.reshape((-1, 1, 2))

        cv2.fillPoly(mask, [corners], 1)

    distance = distance_transform_edt(mask)
    distance /= np.max(distance)
    distance_rgb = np.stack([distance] * 3, axis=-1)

    return distance_rgb


def gaussian_blur(image, ksize=5, sigma=1):
    gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d.T)

    if len(image.shape) == 2:  # Grayscale image
        blurred_image = convolve2d(image, gaussian_2d, mode="same")

    elif len(image.shape) == 3:  # RGB image
        blurred_image = []
        for i in range(3):
            blurred_image.append(convolve2d(image[:, :, i], gaussian_2d, mode="same"))
        blurred_image = np.dstack([channel for channel in blurred_image])

    return blurred_image


def pad_image(image, start_y, start_x, shape):
    padded_image = np.zeros(shape, dtype=image.dtype)
    end_y = start_y + image.shape[0]
    end_x = start_x + image.shape[1]

    padded_image[start_y:end_y, start_x:end_x, :] = image
    return padded_image


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, "Data dimension does not match dimension of centers"

    return (
        (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T
        + np.ones((ndata, 1)) * np.sum((c**2).T, axis=0)
        - 2 * np.inner(x, c)
    )


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


# Returns the warped image and its corners after applying a homography and a shift vector to bring it to its "true" coordinates
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

    return colors.astype(np.uint8), x_values - min_x, y_values - min_y, min_x, min_y


# Return a warped version of the image with a "straightened out" rectangle
def rectify_image(im, im_pts, rect_pts):
    H = compute_homography(np.array(im_pts), np.array(rect_pts))
    ret = warp_image(im, H)[0]
    return ret


# Returns a mosaic of two images, where the second image is projected onto the first
def build_mosaic(im1, im2, im1_pts, im2_pts):
    H = compute_homography(np.array(im2_pts), np.array(im1_pts))
    im2, x_values, y_values, dx, dy = warp_image(im2, H)

    im1_min_x, im1_max_x = 0, im1.shape[1] - 1
    im1_min_y, im1_max_y = 0, im1.shape[0] - 1
    im2_min_x, im2_max_x = dx, dx + im2.shape[1] - 1
    im2_min_y, im2_max_y = dy, dy + im2.shape[0] - 1

    min_x = min(im1_min_x, im2_min_x)
    max_x = max(im1_max_x, im2_max_x)
    min_y = min(im1_min_y, im2_min_y)
    max_y = max(im1_max_y, im2_max_y)

    ret = np.zeros((max_y - min_y + 1, max_x - min_x + 1, 3), dtype=np.float64)
    shift_x, shift_y = -min_x, -min_y

    # Create all needed padded data structures and overlap mask
    im1_padded = pad_image(im1, shift_y, shift_x, ret.shape)
    im2_padded = pad_image(im2, shift_y + dy, shift_x + dx, ret.shape)

    dt1_padded = pad_image(distance_transform(im1.shape), shift_y, shift_x, ret.shape)
    dt2_padded = pad_image(
        distance_transform(im2.shape, x_values, y_values),
        shift_y + dy,
        shift_x + dx,
        ret.shape,
    )

    im1_low_freq = gaussian_blur(im1)
    im2_low_freq = gaussian_blur(im2)

    im1_low_freq_padded = pad_image(im1_low_freq, shift_y, shift_x, ret.shape)
    im2_low_freq_padded = pad_image(im2_low_freq, shift_y + dy, shift_x + dx, ret.shape)
    im1_high_freq_padded = im1_padded - im1_low_freq_padded
    im2_high_freq_padded = im2_padded - im2_low_freq_padded

    mask1 = np.sum(im1_padded, axis=-1) > 0
    mask2 = np.sum(im2_padded, axis=-1) > 0
    overlap_mask = np.logical_and(mask1, mask2)

    # Fill in non-overlap regions
    ret += im1_padded
    ret += im2_padded
    ret[overlap_mask] = 0

    # For low frequencies of overlap, take weighted average based on distance transform
    epsilon = 1e-10  # A small value to prevent division by zero
    total_dt = dt1_padded + dt2_padded + epsilon
    weight1 = dt1_padded / total_dt
    weight2 = dt2_padded / total_dt
    ret[overlap_mask] = (
        im1_low_freq_padded[overlap_mask] * weight1[overlap_mask]
        + im2_low_freq_padded[overlap_mask] * weight2[overlap_mask]
    )

    # For high frequencies of overlap, use the image with greater distance transform value
    high_freq_mask = dt1_padded > dt2_padded
    high_freq_result = np.zeros_like(ret)
    high_freq_result[high_freq_mask] = im1_high_freq_padded[high_freq_mask]
    high_freq_result[~high_freq_mask] = im2_high_freq_padded[~high_freq_mask]
    ret[overlap_mask] += high_freq_result[overlap_mask]

    return np.clip(ret, 0, 255).astype(np.uint8)


def get_harris_corners(im, edge_discard=20):
    """
    This function takes am RGB image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # convert to grayscale image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # find harris corners
    h = corner_harris(im, method="eps", sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (
        (coords[:, 0] > edge)
        & (coords[:, 0] < im.shape[0] - edge)
        & (coords[:, 1] > edge)
        & (coords[:, 1] < im.shape[1] - edge)
    )
    coords = coords[mask].T

    # display_img_with_keypoints(im, list(zip(coords[1], coords[0])))  # TEST

    return h, coords


# Filters a list of candidate corners into K good, spatially distributed corners
def anms(h, corners, k=100, c_robust=0.9):
    _, n = corners.shape
    ys, xs = corners

    r = np.full(n, np.inf)

    all_coords = np.stack((xs, ys), axis=-1)

    for i in range(n):
        condition = h[ys[i], xs[i]] < h[ys, xs] * c_robust
        better_corners = all_coords[condition]

        if better_corners.size > 0:
            X = all_coords[i].reshape(1, -1)
            distances = dist2(X, better_corners)
            r[i] = distances.min()

    top_k_indices = np.argpartition(-r, k)[:k]
    return corners[:, top_k_indices]


# Returns the feature patch for a keypoint
def extract_feature(keypoint, im):
    pass


# Returns tuples of indices that represent matchings between feature patches
def match_features(features1, features2, c_robust=0.65):
    pass


# Runs RANSAC algorithm to filter out some matchings
def ransac(matchings, iterations=1000, c_robust=3):
    pass


# Automatically derive and return correspondances between 2 images (feature matching)
def automatic_feature_matching(im1, im2):
    # [1] Harris corner detection
    h1, corners1 = get_harris_corners(im1)
    h2, corners2 = get_harris_corners(im2)

    # [2] Adaptive Non-Maximal Supression (ANMS)
    corners1 = anms(h1, corners1)
    corners2 = anms(h2, corners2)

    display_img_with_keypoints(im1, list(zip(corners1[1], corners1[0])))  # TEST
    display_img_with_keypoints(im2, list(zip(corners2[1], corners2[0])))  # TEST

    # # [3] Feature descriptor extraction
    # features1 = [extract_feature(corner, im1) for corner in corners1]
    # features2 = [extract_feature(corner, im2) for corner in corners2]

    # # [4] Feature matching (with Lowe's technique)
    # matchings = [
    #     (corners1[i], corners2[j]) for i, j in match_features(features1, features2)
    # ]

    # # [5] Random Sample Consensus (RANSAC)
    # matchings = ransac(matchings)

    # pts1 = [match[0] for match in matchings]
    # pts2 = [match[1] for match in matchings]
    # return pts1, pts2
