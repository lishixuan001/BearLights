import io
import cv2
import os
import sys
from google.cloud import vision
import numpy as np

def detect_properties(path):
    """Detects image properties in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.image_properties(image=image)
    props = response.image_properties_annotation
    # print('Properties:')

    # for color in props.dominant_colors.colors:
    #     print('fraction: {}'.format(color.pixel_fraction))
    #     print('\tr: {}'.format(color.color.red))
    #     print('\tg: {}'.format(color.color.green))
    #     print('\tb: {}'.format(color.color.blue))
    #     print('\ta: {}'.format(color.color.alpha))

    return [[color.pixel_fraction, color.color.red, color.color.green, color.color.blue] for color in props.dominant_colors.colors]

def get_grade(path1, path2):
    """
        Grades the image at path2 based on the differences in properties of the image at path1
    """
    color_properties_1 = np.array(max(detect_properties(path1), key=lambda x: x[0]))
    color_properties_2 = np.array(max(detect_properties(path2), key=lambda x: x[0]))
    # while len(color_properties_2) != len(color_properties_1):
    #     if len(color_properties_1) > len(color_properties_2):
    #         color_properties_2 += [0, 0, 0, 0]
    #     elif len(color_properties_2) > len(color_properties_1):
    #         color_properties_1 += [0, 0, 0, 0]
    return 1 - np.linalg.norm(color_properties_1 - color_properties_2)/441.672956

if __name__ == "__main__":
    white = "/users/andrewlou/Desktop/white-square.png"
    black = "/users/andrewlou/Desktop/black-square.jpg"
    print("White square with black square:" + str(get_grade(white, black)))
    print("White square with white square:" + str(get_grade(white, white)))
    print("Black square with black square:" + str(get_grade(black, black)))