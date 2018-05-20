from glob import glob
import pickle
from cmath import log
import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
from heapq import nsmallest


# euclidian distance between 2 vecs
def euclid(v, w):
    sum = 0
    for i in range(len(v)):
        sum += (v[i] - w[i])**2
    return sum**0.5


def clustered_descriptors(image_names):
    # helping list of indexes, e.g. if it is [0, 5, 8], so descriptors 0-5 is from first image, 5-8 is from second
    ranges = [0]
    sift = cv2.xfeatures2d.SIFT_create()
    # matrix where each row is a descriptor
    descriptors = np.ndarray((0, 128))
    for imname in image_names:
        img = cv2.imread(imname, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cur_des = sift.detectAndCompute(gray, None)[1]
        # add new descriptors to matrix
        descriptors = np.concatenate((descriptors, cur_des))
        # add new index to helping list
        ranges.append(ranges[-1] + len(cur_des))
    num_clusters = round(descriptors.shape[0] ** 0.5)
    # kmeans2 returns list where elem number i contains id of cluster which contains descriptor number i
    labels = kmeans2(descriptors, num_clusters)[1]
    return ranges, labels, num_clusters


def build_bag(ranges, labels, num_images, num_clusters):
    # "bag" of tf-idf vectors
    tf = []
    # document frequency vector
    df = [0 for _ in range(num_clusters)]
    for i in range(num_images):
        # term frequency vector
        words = [0 for _ in range(num_clusters)]
        start, stop = ranges[i], ranges[i + 1]
        # words that current image contains
        clusters = set()
        for j in range(start, stop):
            cluster = labels[j]
            # add new occurence of word
            words[cluster] += 1
            clusters.add(cluster)
        for cluster in clusters:
            # modify document frequency
            df[cluster] += 1
        for k in range(num_clusters):
            # normalize vector by number of descriptors
            words[k] /= (stop - start)
        # add tf vector to "bag"
        tf.append(words)
    # make from tf vectors tf-idf vectors by multiplying each term freq on idf of this term
    for vector in tf:
        for i in range(num_clusters):
            if df[i]:
                vector[i] *= log(num_images / df[i]).real
            else:
                vector[i] = 0
    return tf


def search(bag, n, id):
    candidates = [i for i in range(len(bag))]
    # find n closest candidates
    return nsmallest(n, candidates, lambda v: euclid(bag[v], bag[id]))


if __name__ == '__main__':
    ims = glob('ims/*.jpg')
    ranges, labels, num = clustered_descriptors(ims)
    bag = build_bag(ranges, labels, 15, num)
    for j in range(15):
        for i in search(bag, 3,  j):
            print(ims[i])
        print()
