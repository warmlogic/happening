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
# from sklearn import metrics
# from sklearn.preprocessing import StandardScaler
# import rauth
# import urllib2
# import configparser
import pdb

class struct():
    pass

# def whatsHappening(this_lon, this_lat, area_str='apple_flint_center', tz='US/Pacific'):
def whatsHappening(area_str='apple_flint_center',\
    time_now=['2014-09-09 08:00:00', '2014-09-09 15:00:00'],\
    time_then=['2014-09-08 08:00:00', '2014-09-08 15:00:00'],\
    nbins=100,nclusters=5,\
    tz='US/Pacific'):
    ############
    # Read the data
    ############

    latlong = open("./data/latlong_userdategeo_combined.csv")

    print 'Reading locations...'
    df = pd.read_csv(latlong,header=None,parse_dates=[2],\
        names=['user_id','tweet_id','datetime','longitude','latitude','text'],index_col='datetime')
    print 'Done.'
    latlong.close()

    # twitter times are in UTC
    df = df.tz_localize('UTC').tz_convert(tz)

    # set the bounding box for the requested area
    this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    # for our loc, just set the average
    user_lon = np.mean(this_lon)
    user_lat = np.mean(this_lat)

    geo_activity = sd.selectSpaceBB(df,this_lon,this_lat)

    activity_now = geo_activity.ix[time_now[0]:time_now[1]]
    activity_then = geo_activity.ix[time_then[0]:time_then[1]]

    print 'Now: Selecting %d entries from %s to %s' % (activity_now.shape[0],time_now[0],time_now[1])
    print 'Then: Selecting %d entries from %s to %s' % (activity_then.shape[0],time_then[0],time_then[1])

    ############
    # get difference between events
    ############

    show_plot=False
    # plt = sd.make_hist(df,nbins,show_plot)
    Hnow, xedges, yedges = sd.make_hist(activity_now,nbins,show_plot)
    Hprev, xedges, yedges = sd.make_hist(activity_then,nbins,show_plot)

    Hdiff = Hnow - Hprev

    # return the top nclusters values, sorted; ascend=biggest first
    diffthresh = nbins * 0.75
    morevals,moreind = sd.choose_n_sorted(Hdiff, n=nclusters, min_val=diffthresh, srt='max', return_order='ascend')
    lessvals,lessind = sd.choose_n_sorted(Hdiff, n=nclusters, min_val=diffthresh, srt='min', return_order='ascend')

    diffmore_lon = xedges[moreind[:,0]]
    diffmore_lat = yedges[moreind[:,1]]
    diffless_lon = xedges[lessind[:,0]]
    diffless_lat = yedges[lessind[:,1]]
    print 'At threshold %d, found %d "events" that have more activity than previous time' % (diffthresh,len(morevals))
    print 'At threshold %d, found %d "events" that have less activity than previous time' % (diffthresh,len(lessvals))

    activity_clustered, n_clusters =  sd.clusterThose(activity_now,nbins,diffmore_lon,diffmore_lat)
    print n_clusters

    return activity_clustered, n_clusters, user_lon, user_lat
    # return activity_now, n_clusters, user_lon, user_lat, diffmore_lon, diffmore_lat

def main():
    print 'not ready yet'
    return


if __name__ == '__main__':
    main()

