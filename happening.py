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
import rauth
import urllib2
import configparser
import pdb

class struct():
    pass

############
# Read the data
############

latlong = open("data/latlong_userdategeo_combined.csv")

print 'Reading locations...'
# df = pd.read_csv(latlong,header=None,names=['longitude', 'latitude'])
# df = pd.read_csv(latlong,header=None,names=['tweet_id','datestr', 'longitude','latitude','text'])
df = pd.read_csv(latlong,header=None,parse_dates=[2],\
    names=['user_id','tweet_id','datetime','longitude','latitude','text'],index_col='datetime')
print 'Done.'
latlong.close()

# twitter times are in UTC
# df.datetime = df.datetime.apply(lambda x: x.tz_localize('UTC'), convert_dtype=False)
df = df.tz_localize('UTC').tz_convert('US/Pacific')

# df['point'] = df.apply(lambda row: pd.Point(df['latitude'], df['longitude']))

############
# User
############

# df = sd.selectUser(df,rmnull=True)

############
# Space
############

# this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
# this_lon, this_lat = sd.set_get_boundBox(area_str='sf')
# this_lon, this_lat = sd.set_get_boundBox(area_str='fishwharf')
# this_lon, this_lat = sd.set_get_boundBox(area_str='embarc')
# this_lon, this_lat = sd.set_get_boundBox(area_str='att48')
# this_lon, this_lat = sd.set_get_boundBox(area_str='pier48')
# this_lon, this_lat = sd.set_get_boundBox(area_str='attpark')
# this_lon, this_lat = sd.set_get_boundBox(area_str='mission')
# this_lon, this_lat = sd.set_get_boundBox(area_str='sf_concerts')
# this_lon, this_lat = sd.set_get_boundBox(area_str='nobhill')
# this_lon, this_lat = sd.set_get_boundBox(area_str='mtview_caltrain')
this_lon, this_lat = sd.set_get_boundBox(area_str='apple_flint_center')
# this_lon, this_lat = sd.set_get_boundBox(area_str='levisstadium')

geo_activity = sd.selectSpaceBB(df,this_lon,this_lat)



def main():

    center = struct()
    center.lat = 37.786382
    center.long = -122.432883
    clusters = foodGroups(center.lat, center.long )
    X=clusters['X']
    n_clusters_ = clusters['n_clusters']
    labels = clusters['labels']
    core_samples_mask = clusters['core_samples_mask']

    fig,ax = figSetup.figSetup()
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
        
        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 1], xy[:, 0], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
                 
        plt.plot(center.long,center.lat,'*g',markersize=8)
    
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()



if __name__ == '__main__':
    main()

