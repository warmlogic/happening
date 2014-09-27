#!/usr/bin/env python
"""
happening.py
author: Matt Mollison
Find social activity around you above baseline levels
"""

# import json
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import select_data as sd
# import jsonOpen, figSetup
# from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# import rauth
# import urllib2
# import configparser

from nltk.corpus import stopwords
from nltk import FreqDist
# import nltk
# nltk.download() # get the stopwords corpus
import string

import pdb

# class struct():
#     pass

def findHotspots(activity_now, activity_then, nbins=[100, 100],\
    n_top_hotspots=5, diffthresh=30):
    '''Compare now vs then
    '''
    
    ############
    # get difference between events
    ############

    show_plot=False
    # plt = sd.make_hist(df,nbins,show_plot)
    Hnow, xedges, yedges = sd.make_hist(activity_now,nbins,show_plot)
    Hthen, xedges, yedges = sd.make_hist(activity_then,nbins,show_plot)

    Hdiff = Hnow - Hthen

    # return n_top_hotspots values, sorted; ascend=biggest first
    morevals,moreind = sd.choose_n_sorted(Hdiff, n=n_top_hotspots, min_val=diffthresh, srt='max', return_order='ascend')
    lessvals,lessind = sd.choose_n_sorted(Hdiff, n=n_top_hotspots, min_val=diffthresh, srt='min', return_order='ascend')

    diffmore_lon = xedges[moreind[:,1]]
    diffmore_lat = yedges[moreind[:,0]]
    diffless_lon = xedges[lessind[:,1]]
    diffless_lat = yedges[lessind[:,0]]
    print 'At threshold %d, found %d "events" that have more activity than previous time' % (diffthresh,len(morevals))
    print 'At threshold %d, found %d "events" that have less activity than previous time' % (diffthresh,len(lessvals))
    return diffmore_lon, diffmore_lat

def clusterActivity(activity_now, diffmore_lon, diffmore_lat, nbins=[100, 100],\
    min_nclusters=1, max_nclusters=100, eps=0.025, min_samples=100,\
    centerData=True, plotData=False):
    '''min_samples is minimum number of samples per hour on average
    '''

    ############
    # Find cluster shapes
    ############

    X = np.vstack((activity_now.longitude, activity_now.latitude)).T
    if centerData:
        scaler = StandardScaler(copy=True)
        X_centered = scaler.fit(X).transform(X)
    else:
        X_centered = X

    n_clusters_db = 0

    if len(diffmore_lon) > 0:
        n_tries_db = 0
        orig_eps = eps
        while n_clusters_db < min_nclusters:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_centered)
            # db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            n_tries_db += 1
            print 'try number %d' % n_tries_db

            labels = db.labels_
            n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
            if n_tries_db < 3:
                eps += orig_eps
                print 'increasing eps to %.3f' % eps
            else:
                print 'found %d clusters' % n_clusters_db
            if n_tries_db >= 3 and n_tries_db <= 7:
                if min_samples > 15.0:
                    min_samples = int(np.ceil(min_samples * 0.67))
                    print 'decreasing min_samples to %d' % min_samples
                else:
                    break
            elif n_tries_db > 7 and n_tries_db <= 8:
                eps += orig_eps
                print 'increasing eps to %.3f' % eps
            elif n_tries_db > 8:
                break

        print 'Estimated number of clusters: %d (eps=%f, min_samples=%d)' % (n_clusters_db,eps,min_samples)
    else:
        print 'no differences found using %d x %d bins' % (nbins[0], nbins[1])

    if n_clusters_db > 0:
        nbins_combo = int(np.floor(np.prod(nbins) / 100))
        print 'real nbins_combo %d' % nbins_combo
        if nbins_combo < 5:
            nbins_combo = 5
            print 'modified nbins_combo %d' % nbins_combo
        core_samples = db.core_sample_indices_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        unique_labels = np.unique(labels)

        # go through the found clusters
        keepClus = []
        cluster_centers = []
        clusterNums = np.repeat(-1,activity_now.shape[0])
        # for k, col in zip(unique_labels, colors):
        for k in unique_labels:
            if k != -1 and k < max_nclusters:
                # if in a cluster, set a mask for this cluster
                class_member_mask = (labels == k)

                # get the lat and long for this cluster
                cluster_lon = X[class_member_mask,0]
                cluster_lat = X[class_member_mask,1]

                # default setting for keeping the cluster
                keepThisClus = False

                # keep clusters that contain a hist2d hotspot
                # print 'len diffmore_lon: %d' % len(diffmore_lon)
                for i in range(len(diffmore_lon)):
                    binscale = 0.001
                    n_tries_bin = 0
                    while keepThisClus is False:
                        n_tries_bin += 1
                        if diffmore_lon[i] > (min(cluster_lon) - nbins_combo*binscale) and diffmore_lon[i] < (max(cluster_lon) + nbins_combo*binscale) and diffmore_lat[i] > (min(cluster_lat) - nbins_combo*binscale) and diffmore_lat[i] < (max(cluster_lat) + nbins_combo*binscale):
                            print 'keeping this cluster'
                            keepThisClus = True
                            break
                        else:
                            binscale += 0.001
                            print 'increasing binscale to %.4f' % binscale
                        if n_tries_bin > 3:
                            print 'this cluster did not contain a hotspot'
                            break
                    if keepThisClus:
                        break

                keepClus.append(keepThisClus)
                if keepThisClus:
                    # fill in the cluster lable vector
                    clusterNums[class_member_mask] = k

                    # set the mean latitude and longitude
                    mean_lon = np.mean(X[core_samples_mask & class_member_mask,0])
                    mean_lat = np.mean(X[core_samples_mask & class_member_mask,1])
                    cluster_centers.append([mean_lon,mean_lat,int(k)])

            # else:
            #     keepClus.append(False)
            #     # Black used for noise.
            #     # col = 'k'
        activity_now['clusterNum'] = clusterNums
        n_clusters_real = sum(keepClus)

        if plotData:
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    # Black used for noise.
                    col = 'k'

                class_member_mask = (labels == k)

                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=2)

                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=14)

            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
            plt.title('Estimated number of clusters: %d' % n_clusters_)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            # plt.show()

            # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
            # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
            # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
            # print("Adjusted Rand Index: %0.3f"
            #       % metrics.adjusted_rand_score(labels_true, labels))
            # print("Adjusted Mutual Information: %0.3f"
            #       % metrics.adjusted_mutual_info_score(labels_true, labels))
            # print("Silhouette Coefficient: %0.3f"
            #       % metrics.silhouette_score(X_centered, labels))
    else:
        n_clusters_real = 0
        cluster_centers = []

    if len(cluster_centers) > 0:
        message = 'found clusters, hoooray!'
        success = True
    else:
        message = 'all clusters rejected!'
        success = False

    return activity_now, n_clusters_real, cluster_centers, message, success

def cleanTextGetWordFrequency(activity):
    # for removing punctuation (via translate)
    table = string.maketrans("","")
    clean_text = []
    # for removing stop words
    stop = stopwords.words('english')
    stop.extend(stopwords.words('spanish'))
    tokens = []
    # stop.append('AT_USER')
    # stop.append('URL')
    stop.append('')
    stop.append('unicode_only')
    stop.append('u')
    stop.append('w')
    stop.append('im')
    stop.append('amp')
    stop.append('rt')
    stop.append('ca')
    stop.append('sf')
    stop.append('#sf')
    stop.append('san')
    stop.append('francisco')
    stop.append('sanfrancisco')
    stop.append('#ca')
    stop.append('#san')
    stop.append('#francisco')
    stop.append('#sanfrancisco')

    for txt in activity['text'].values:
        txt = sd.processTweet(txt)
        nopunct = txt.translate(table, string.punctuation)
        #Remove additional white spaces
        # nopunct = re.sub('[\s]+', ' ', nopunct)
        # if nopunct is not '':
        clean_text.append(nopunct)
        # split it and remove stop words
        txt = sd.getFeatureVector(txt,stop)
        tokens.extend([t for t in txt])
    freq_dist = FreqDist(tokens)
    return tokens, freq_dist, clean_text

    # stop = stopwords.words('english')

    # tokens = []
    # for txt in activity_clustered['text'].values:
    #     txt = sd.processTweet(txt)
    #     txt = sd.getFeatureVector(txt,stop)
    #     tokens.extend([t for t in txt])
    # freq_dist = FreqDist(tokens)
    # return tokens, freq_dist, clean_text

# modified from here: http://alexdavies.net/twitter-sentiment-analysis/
def readSentimentList(file_name='./data/twitter_sentiment_list_cleaned.csv'):
    ifile = open(file_name, 'r')
    happy_log_probs = {}
    sad_log_probs = {}
    ifile.readline() #Ignore title row
    
    for line in ifile:
        tokens = line[:-1].split(',')
        happy_log_probs[tokens[0]] = float(tokens[1])
        sad_log_probs[tokens[0]] = float(tokens[2])

    return happy_log_probs, sad_log_probs

def classifySentiment(words, happy_log_probs, sad_log_probs):
    # Get the log-probability of each word under each sentiment
    happy_probs = [happy_log_probs[word] for word in words if word in happy_log_probs]
    sad_probs = [sad_log_probs[word] for word in words if word in sad_log_probs]

    # Sum all the log-probabilities for each sentiment to get a log-probability for the whole tweet
    tweet_happy_log_prob = np.sum(happy_probs)
    tweet_sad_log_prob = np.sum(sad_probs)

    # Calculate the probability of the tweet belonging to each sentiment
    prob_happy = np.reciprocal(np.exp(tweet_sad_log_prob - tweet_happy_log_prob) + 1)
    prob_sad = 1 - prob_happy

    return prob_happy, prob_sad

# def clusterSentiment(activity, n_clusters, happy_log_probs, sad_log_probs):
#     cluster_happy_sentiment = []
#     for clusNum in range(n_clusters):
#         activity_thisclus = activity.loc[activity['clusterNum'] == clusNum]
#         tokens, freq_dist, clean_text = cleanTextGetWordFrequency(activity_thisclus)
#         happy_probs = []
#         for tweet in clean_text:
#             prob_happy, prob_sad = classifySentiment(tweet.split(), happy_log_probs, sad_log_probs)
#             happy_probs.append(prob_happy)
#         cluster_happy_sentiment.append(sum(np.array(happy_probs) > .5) / float(len(happy_probs)))
#     return cluster_happy_sentiment
