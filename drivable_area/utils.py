"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import numpy as np

def heat_map(img_size, points, sigma):
    """produce a heat map(gray scale) according the points
    Args:
        img_size: img height and width
        points: ndarray or list, represents the coordinate of points. (x, y)
        sigma: control the heap point range
    return:
        a heap map with the shape (h, w)
    Example:
        aa = heat_map(img_size=(224, 224), points=[[50, 50], [100, 100]], sigma=2)
        cv2.imshow('test', aa)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    x = np.arange(0, img_size[1], 1)
    y = np.arange(0, img_size[0], 1)
    z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)
    heat_map = np.array([gauss_2d(z, point, sigma=2) for point in points])
    heat_map = np.sum(heat_map, axis=0)
    return heat_map/np.max(heat_map)


def gauss_2d(point, mu, sigma):
    """2d gauss func
    Args:
        point: 2d point coordinate
        mu: 2d mean value
        sigma: ...
    Return:
        a img with the shape (h, w)
    Example:
        x = np.arange(0, 224, 1)
        y = np.arange(0, 224, 1)
        z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)

        z = gauss_2d(z, mu=(100,100), sigma=2)
        cv2.imshow('test', z)
        cv2.waitKey()
        cv2.destroyAllWindows()
    """
    h = point.shape[0]
    w = point.shape[1]
    point = np.reshape(point, newshape=[-1, 2])
    score = np.exp(-(np.sum(np.square(np.array(point)-np.array(mu)), axis=-1))/(2*sigma**2))

    return np.reshape(score, newshape=[h, w])
