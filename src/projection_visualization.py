import napari


def viewer(images):

    viewer, image_layer = napari.imshow(images)
    napari.run()
