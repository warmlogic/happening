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
def whatsHappening(area_str='apple_flint_center', tz='US/Pacific'):
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

    # # apple keynote
    # sinceDatetime_now = '2014-09-09 08:00:00'
    # untilDatetime_now = '2014-09-09 15:00:00'
    # activity_now = geo_activity.ix[sinceDatetime_now:untilDatetime_now]
    # sinceDatetime_base = '2014-09-08 08:00:00'
    # untilDatetime_base = '2014-09-08 15:00:00'
    # activity_base = geo_activity.ix[sinceDatetime_base:untilDatetime_base]

    # giants vs diamondbacks
    sinceDatetime_now = '2014-09-09 17:00:00'
    untilDatetime_now = '2014-09-09 23:30:00'
    activity_now = geo_activity.ix[sinceDatetime_now:untilDatetime_now]
    sinceDatetime_base = '2014-09-08 17:00:00'
    untilDatetime_base = '2014-09-08 23:30:00'
    activity_base = geo_activity.ix[sinceDatetime_base:untilDatetime_base]

    ############
    # get difference between events
    ############

    nbins = 50
    show_plot=False
    # plt = sd.make_hist(df,nbins,show_plot)
    Hnow, xedges, yedges = sd.make_hist(activity_now,nbins,show_plot)
    Hprev, xedges, yedges = sd.make_hist(activity_base,nbins,show_plot)

    Hdiff = Hnow - Hprev

    # return the top n values, sorted; ascend=biggest first
    n = 5
    diffthresh = 100
    morevals,moreind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='max', return_order='ascend')
    lessvals,lessind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='min', return_order='ascend')

    diffmore_lon = xedges[moreind[:,0]]
    diffmore_lat = yedges[moreind[:,1]]
    diffless_lon = xedges[lessind[:,0]]
    diffless_lat = yedges[lessind[:,1]]
    print 'At threshold %d, found %d "events" that have more activity than previous time' % (diffthresh,len(morevals))
    print 'At threshold %d, found %d "events" that have less activity than previous time' % (diffthresh,len(lessvals))

    # # collect tweets from dataframe within radius X of lon,lat
    # unit = 'meters'
    # radius = 200
    # radius_increment = 50
    # radius_max = 1000
    # min_activity = 200
    # # for point in range(len(diffmore_lon)):
    # point = 0
    # print 'getting tweets from near: %.6f,%.6f' % (diffmore_lat[point],diffmore_lon[point])
    # now_nearby = sd.selectActivityFromPoint(activity_now,diffmore_lon[point],diffmore_lat[point],unit,radius,radius_increment,radius_max,min_activity)
    # # pdb.set_trace()

    return activity_now, diffmore_lon, diffmore_lat, user_lon, user_lat

def main():
    print 'not ready yet'
    return
    # center = struct()
    # center.lat = 37.786382
    # center.long = -122.432883
    # clusters = foodGroups(center.lat, center.long )
    # X=clusters['X']
    # n_clusters_ = clusters['n_clusters']
    # labels = clusters['labels']
    # core_samples_mask = clusters['core_samples_mask']

    # fig,ax = figSetup.figSetup()
    # # Black removed and is used for noise instead.
    # unique_labels = set(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = 'k'

    #     class_member_mask = (labels == k)
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
        
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=6)
                 
    #     plt.plot(center.long,center.lat,'*g',markersize=8)
    
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plt.show()



if __name__ == '__main__':
    main()

