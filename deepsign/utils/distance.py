import numpy as np

def cosine(u,v):
    return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)

def cosine_distance(u,v):
    return 1 - cosine(u,v)