from PIL import ImageDraw


def display(image, objects: list, show: bool = True):
    img = image.copy()
    img1 = ImageDraw.Draw(img)
    for obj in objects:
        img1.rectangle(obj, outline="red", width=5)
    if show:
        img.show()
    return img
