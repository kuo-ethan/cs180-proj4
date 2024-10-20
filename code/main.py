import skimage.io as skio
from utils import rectify_image

# Image rectification
ART_KEYPOINTS = [[87, 98], [138, 128], [259, 127], [197, 97]]
ART_RECTANGLE = [[0, 0], [50, 0], [50, 50], [0, 50]]
im = skio.imread(f'../data/art.jpg')
rectified_im = rectify_image(im, ART_KEYPOINTS, ART_RECTANGLE)
skio.imsave(f'../images/rectified_art.jpg', rectified_im)

CARDS_KEYPOINTS = [[218, 489], [382, 351], [589, 583], [417, 733]]
CARDS_RECTANGLE = [[0, 0], [250, 0], [250, 350], [0, 350]]
im = skio.imread(f'../data/cards.jpg')
rectified_im = rectify_image(im, CARDS_KEYPOINTS, CARDS_RECTANGLE)
skio.imsave(f'../images/rectified_cards.jpg', rectified_im)

IPAD_KEYPOINTS = [(40, 188), (172, 503), (462, 363), (264, 135)]
IPAD_RECTANGLE = [(0, 0), (281 * 2, 0), (281 * 2, 215 * 2), (0, 215 * 2)]
im = skio.imread(f'../data/ipad.jpg')
rectified_im = rectify_image(im, IPAD_KEYPOINTS, IPAD_RECTANGLE)
skio.imsave(f'../images/rectified_ipad.jpg', rectified_im)