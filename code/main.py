import skimage.io as skio
from utils import rectify_image, build_mosaic, automatic_feature_matching


# Drivers
def image_rectification():
    ART_KEYPOINTS = [[87, 98], [138, 128], [259, 127], [197, 97]]
    ART_RECTANGLE = [[0, 0], [0, 200], [200, 200], [200, 0]]
    im = skio.imread(f"../data/art.jpg")
    rectified_im = rectify_image(im, ART_KEYPOINTS, ART_RECTANGLE)
    skio.imsave(f"../images/rectified_art.jpg", rectified_im)

    IPAD_KEYPOINTS = [(40, 188), (172, 503), (462, 363), (264, 135)]
    IPAD_RECTANGLE = [(0, 0), (0, 281 * 2), (215 * 2, 281 * 2), (215 * 2, 0)]
    im = skio.imread(f"../data/ipad.jpg")
    rectified_im = rectify_image(im, IPAD_KEYPOINTS, IPAD_RECTANGLE)
    skio.imsave(f"../images/rectified_ipad.jpg", rectified_im)


def mosaic_blending(auto=False):
    im1 = skio.imread(f"../data/balcony1.JPG")
    im2 = skio.imread(f"../data/balcony2.JPG")
    balcony_keypoints1 = [
        (443, 465),
        (477, 467),
        (478, 560),
        (460, 437),
        (355, 376),
        (472, 378),
        (497, 416),
        (700, 425),
        (877, 572),
        (798, 250),
        (739, 278),
        (967, 241),
    ]

    balcony_keypoints2 = [
        (204, 479),
        (254, 478),
        (253, 587),
        (233, 442),
        (56, 367),
        (250, 373),
        (282, 417),
        (490, 420),
        (608, 527),
        (563, 275),
        (522, 292),
        (660, 284),
    ]
    if auto:
        balcony_keypoints1, balcony_keypoints2 = automatic_feature_matching(
            im1, im2, c_lowes=0.75
        )

    mosaic = build_mosaic(im1, im2, balcony_keypoints1, balcony_keypoints2)
    filename = f"../images/balcony_mosaic{'_auto' if auto else ''}.jpg"
    skio.imsave(filename, mosaic)

    # im1 = skio.imread(f"../data/livingroom1.JPG")
    # im2 = skio.imread(f"../data/livingroom2.JPG")

    # livingroom_keypoints1 = [
    #     (298, 355),
    #     (268, 210),
    #     (255, 142),
    #     (438, 174),
    #     (442, 244),
    #     (447, 321),
    #     (699, 216),
    #     (688, 288),
    #     (169, 343),
    #     (222, 376),
    #     (243, 263),
    #     (295, 109),
    #     (665, 158),
    #     (35, 44),
    # ]
    # livingroom_keypoints2 = [
    #     (276, 719),
    #     (272, 558),
    #     (271, 492),
    #     (443, 521),
    #     (443, 592),
    #     (444, 679),
    #     (703, 569),
    #     (702, 644),
    #     (140, 705),
    #     (189, 746),
    #     (237, 614),
    #     (314, 463),
    #     (661, 510),
    #     (87, 407),
    # ]
    # if auto:
    #     livingroom_keypoints1, livingroom_keypoints2 = automatic_feature_matching(
    #         im1, im2
    #     )

    # # display_img_with_keypoints(im1, livingroom_keypoints1)
    # # display_img_with_keypoints(im2, livingroom_keypoints2)
    # mosaic = build_mosaic(im1, im2, livingroom_keypoints1, livingroom_keypoints2)
    # filename = f"../images/livingroom_mosaic{'_auto' if auto else ''}.jpg"
    # skio.imsave(filename, mosaic)

    # im1 = skio.imread(f"../data/shoes1.JPG")
    # im2 = skio.imread(f"../data/shoes2.JPG")

    # shoes_keypoints1 = [
    #     (184, 428),
    #     (557, 429),
    #     (201, 612),
    #     (540, 612),
    #     (213, 764),
    #     (528, 766),
    #     (225, 894),
    #     (516, 896),
    # ]
    # shoes_keypoints2 = [
    #     (179, 242),
    #     (570, 239),
    #     (210, 434),
    #     (542, 431),
    #     (233, 571),
    #     (522, 571),
    #     (249, 681),
    #     (508, 681),
    # ]

    # if auto:
    #     shoes_keypoints1, shoes_keypoints2 = automatic_feature_matching(im1, im2)

    # # display_img_with_keypoints(im1, shoes_keypoints1)
    # # display_img_with_keypoints(im2, shoes_keypoints2)
    # mosaic = build_mosaic(im1, im2, shoes_keypoints1, shoes_keypoints2)
    # filename = f"../images/shoes_mosaic{'_auto' if auto else ''}.jpg"
    # skio.imsave(filename, mosaic)

    # if auto:
    #     im1 = skio.imread(f"../data/painting1.JPG")
    #     im2 = skio.imread(f"../data/painting2.JPG")

    #     painting_keypoints1, painting_keypoints2 = automatic_feature_matching(
    #         im1, im2, c_lowes=0.75
    #     )

    #     # display_img_with_keypoints(im1, door_keypoints1)
    #     # display_img_with_keypoints(im2, door_keypoints2)
    #     mosaic = build_mosaic(im1, im2, painting_keypoints1, painting_keypoints2)
    #     filename = f"../images/painting_mosaic_auto.jpg"
    #     skio.imsave(filename, mosaic)

    #     im1 = skio.imread(f"../data/kitchen1.JPG")
    #     im2 = skio.imread(f"../data/kitchen2.JPG")

    #     kitchen_keypoints1, kitchen_keypoints2 = automatic_feature_matching(im1, im2)

    #     # display_img_with_keypoints(im1, door_keypoints1)
    #     # display_img_with_keypoints(im2, door_keypoints2)
    #     mosaic = build_mosaic(im1, im2, kitchen_keypoints1, kitchen_keypoints2)
    #     filename = f"../images/kitchen_mosaic_auto.jpg"
    #     skio.imsave(filename, mosaic)


# Project 4A
# image_rectification()
# mosaic_blending()

# Project 4B
mosaic_blending(auto=True)
