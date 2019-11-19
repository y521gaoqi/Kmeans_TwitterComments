import numpy as np
import random
import re
import matplotlib.pyplot as plt
import prettytable
from prettytable import from_csv
from prettytable import PrettyTable



dataSet = np.genfromtxt(
    'usnewshealth.txt', delimiter='\t', dtype=str, comments=None)
length = len(dataSet)
for i in range(length):
    s = dataSet[i]
    a = s.split('|')
    c = re.sub(r'http\S+', '', a[2])
    d = ' '.join(word for word in c.split(' ') if not word.startswith('@'))
    e = d.replace('#', '')
    f = e.lower()
    dataSet[i] = f


a = dataSet[100].split(' ')
b = dataSet[1].split(' ')


def jaccard(list1, list2):
    s1 = set(list1.split(' '))
    s2 = set(list2.split(' '))
    return (1-len(s1.intersection(s2)) / len(s1.union(s2)))


def _rand_center(data, k):
    """Generate k center within the range of data set."""
    n = len(data)  # features
    rarray = np.random.random(size=k)
    rarray = np.floor(rarray*n)
    rarray = rarray.astype(int)
    # print('random index', rarray)
    center = data[rarray]
    return center


def _converged(centroids1, centroids2):
    # if centroids not changed, we say 'converged'
    set1 = set([tuple(c) for c in centroids1])
    set2 = set([tuple(c) for c in centroids2])
    return (set1 == set2)


n = len(dataSet)  # number of entries

table=PrettyTable()
table.field_names = ["Value of K", "SSE", "Size of each cluster"]


# print(centroids)
for k in range(5,12):
    centroids = _rand_center(dataSet, k)
    label = np.zeros(n, dtype=np.int)  # track the nearest centroid
    converged = False

    times_converged = 0
    while not converged:
        old_centroids = np.copy(centroids)
        for i in range(n):
            # determine the nearest centroid and track it with label
            min_dist, min_index = np.inf, -1
            for j in range(k):
                dist = jaccard(dataSet[i], centroids[j])
                if dist < min_dist:
                    min_dist, min_index = dist, j
                    label[i] = j
        # update centroid
        for m in range(k):
            index_centroid = 0
            min_dist = np.inf
            for a in range(n):
                sum_dist = 0
                if label[a] == m:
                    for p in range(n):
                        if label[p] == m:
                            dist = jaccard(dataSet[p], dataSet[a])
                            sum_dist = sum_dist+dist
                    if sum_dist < min_dist:
                        min_dist = sum_dist
                        index_centroid = a
            centroids[m] = dataSet[index_centroid]
        converged = _converged(old_centroids, centroids)
    error_total=0
    size=" "
    for m in range(k):
        error=0.0
        cluster=dataSet[label==m]
        for b in range(len(cluster)):
            error=jaccard(cluster[b],centroids[m])+error
            #print("error     ",jaccard(cluster[b],centroids[m]))
        error=error**2
        error_total=error_total+error
        #print("size of each cluster ",m,": ",len(cluster))
        size=str(m+1)+": "+str(len(cluster))+" tweets\n"+size
    #print ("total error for value of k ",error_total)
    table.add_row([k,error_total, size])
    table.add_row(["-------------","-------------------", "----------------------"])

print(table)