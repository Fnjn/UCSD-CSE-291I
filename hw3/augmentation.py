#!/usr/bin/env python3

import numpy as np

def augmentation(x):
    r = np.random.random()
    if r < 0.03:
        return jitter(x)
    elif r < 0.06:
        return rotation(x)
    elif r < 0.1:
        return jitter(rotation(x))
    else:
        return x


def rotation(x):
    rad = 2 * np.pi * np.random.random()

    #Rx = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    Ry = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    #Rz = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])

    return np.matmul(x, Ry)

def jitter(x, mu=0, sigma=0.02):
    shift = np.random.normal(mu, sigma, x.shape[1:])
    return x + shift
