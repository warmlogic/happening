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
# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
# import rauth
# import urllib2
# import configparser

from nltk.corpus import stopwords
from nltk import FreqDist
# import nltk
# nltk.download() # get the stopwords corpus
import string

import pdb

class struct():
    pass

def whatsHappening(activity_now, activity_then, nbins=[100, 100],\
    n_top_hotspots=5, min_nclusters=1, max_nclusters=100,\
    diffthresh=30, eps=0.025, min_samples=100):
    '''
    min_samples is minimum number of samples per hour on average
    '''
    # ############
    # # Read the data
    # ############

    # latlong = open("./data/latlong_userdategeo_combined.csv")

    # print 'Reading locations...'
    # df = pd.read_csv(latlong,header=None,parse_dates=[2],dtype={'url': str},\
    #     names=['user_id','tweet_id','datetime','longitude','latitude','text','url'],index_col='datetime')
    # print 'Done.'
    # latlong.close()
    # # df.fillna('', inplace=True)

    # # twitter times are in UTC
    # df = df.tz_localize('UTC').tz_convert(tz)

    # set the bounding box for the requested area
    # this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    # for our loc, just set the average
    # user_lon = np.mean(this_lon)
    # user_lat = np.mean(this_lat)

    # geo_activity = sd.selectSpaceBB(df,this_lon,this_lat)

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

    activity_now_clustered, n_clusters, cluster_centers =  sd.clusterThose(activity_now=activity_now,\
        nbins=nbins,diffmore_lon=diffmore_lon,diffmore_lat=diffmore_lat,\
        min_nclusters=min_nclusters,max_nclusters=max_nclusters,eps=eps,min_samples=min_samples)
    if len(cluster_centers) > 0:
        message = 'found clusters, hoooray!'
        success = True
    else:
        message = 'all clusters rejected!'
        success = False

    return activity_now_clustered, n_clusters, cluster_centers, message, success

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

def main():
    print 'not ready yet'
    return

if __name__ == '__main__':
    main()

