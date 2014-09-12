'''
check out the data
'''

import pandas as pd
import select_data as sd
import matplotlib.pyplot as plt
import numpy as np
import pdb
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from scipy.spatial.distance import pdist

%load_ext autoreload
%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import Image
plt.rcParams['figure.figsize'] = 12, 8 # plotsize

############
# Read the data
############

latlong = open("./data/latlong_userdategeo_combined.csv")

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

# choose only coordinates in our bounding box of interest

# this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
# this_lon, this_lat = sd.set_get_boundBox(area_str='sf')
# this_lon, this_lat = sd.set_get_boundBox(area_str='fishwharf')
# this_lon, this_lat = sd.set_get_boundBox(area_str='embarc')
# this_lon, this_lat = sd.set_get_boundBox(area_str='att48')
# this_lon, this_lat = sd.set_get_boundBox(area_str='pier48')
this_lon, this_lat = sd.set_get_boundBox(area_str='attpark')
# this_lon, this_lat = sd.set_get_boundBox(area_str='mission')
# this_lon, this_lat = sd.set_get_boundBox(area_str='sf_concerts')
# this_lon, this_lat = sd.set_get_boundBox(area_str='nobhill')
# this_lon, this_lat = sd.set_get_boundBox(area_str='mtview_caltrain')
# this_lon, this_lat = sd.set_get_boundBox(area_str='apple_flint_center')
# this_lon, this_lat = sd.set_get_boundBox(area_str='levisstadium')

geo_activity = sd.selectSpaceBB(df,this_lon,this_lat)

############
# Time
############

# # set the start and end datetimes

# sinceDatetime = '2014-09-05 09:00:00'
# untilDatetime = '2014-09-05 17:00:00'
# sinceDatetime = ''
# untilDatetime = ''

# tz = 'US/Pacific'
# geo_activity = sd.selectTime(geo_activity,tz=tz,sinceDatetime=sinceDatetime,untilDatetime=untilDatetime)

# # night life
# time_now = ['2014-09-05 17:00:00', '2014-09-06 05:00:00']
# time_then = ['2014-09-08 17:00:00', '2014-09-09 05:00:00']

# # apple keynote
# time_now = ['2014-09-09 08:00:00', '2014-09-09 15:00:00']
# time_then = ['2014-09-08 08:00:00', '2014-09-08 15:00:00']

# giants vs diamondbacks
time_now = ['2014-09-09 17:00:00', '2014-09-09 23:30:00']
time_then = ['2014-09-08 17:00:00', '2014-09-08 23:30:00']

activity_now = geo_activity.ix[time_now[0]:time_now[1]]
activity_then = geo_activity.ix[time_then[0]:time_then[1]]

print 'Now: Selecting %d entries from %s to %s' % (activity_now.shape[0],time_now[0],time_now[1])
print 'Then: Selecting %d entries from %s to %s' % (activity_then.shape[0],time_then[0],time_then[1])

###########
# plot over time
###########

# tweetlocs = df.ix[:, ['longitude','latitude']]
tweetlocs_now = activity_now.ix[:, ['longitude','latitude']].resample('60min', how='count')
tweetlocs_then = activity_then.ix[:, ['longitude','latitude']].resample('60min', how='count')

# volume = df.resample('60min', how='count')
fig, ax = plt.subplots()
tweetlocs_now.plot(kind='line',style='b')
tweetlocs_then.plot(kind='line',style='r')
fig.autofmt_xdate()
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # ax.set_xlim(['17:00:00','05:00:00'])


############
# Plot heat map and difference
############

nbins = 50
show_plot=True
savefig = False
# plt = sd.make_hist(df,nbins,show_plot)
Hnow, xedges, yedges = sd.make_hist(activity_now,nbins,show_plot)
Hprev, xedges, yedges = sd.make_hist(activity_then,nbins,show_plot)
Hdiff = Hnow - Hprev

if show_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hdiff)
    ax.set_title('Difference in tweets (%d bins)' % nbins)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    cb = plt.colorbar()
    cb.set_label('Count')

    if savefig:
        figname = 'data/latlong_plot.png'
        print 'saving figure to ' + figname
        plt.savefig(figname, bbox_inches='tight')
    # plt.show()


# # diffthresh = 15
# diffthresh = 100
# diffmore = np.column_stack(np.where(Hdiff > diffthresh))
# diffless = np.column_stack(np.where(Hdiff < -diffthresh))


# return the top n values, sorted; ascend=biggest first
n = 5
diffthresh = 100
morevals,moreind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='max', return_order='ascend')
lessvals,lessind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='min', return_order='ascend')

# bigcoord = zip(xedges[bigdiff[:,0]], yedges[bigdiff[:,1]])
# diffmore_lon = xedges[diffmore[:,0]]
# diffmore_lat = yedges[diffmore[:,1]]
# diffless_lon = xedges[diffless[:,0]]
# diffless_lat = yedges[diffless[:,1]]
diffmore_lon = xedges[moreind[:,0]]
diffmore_lat = yedges[moreind[:,1]]
diffless_lon = xedges[lessind[:,0]]
diffless_lat = yedges[lessind[:,1]]
print 'At threshold %d, found %d "events" that have more activity than previous time' % (diffthresh,len(morevals))
print 'At threshold %d, found %d "events" that have less activity than previous time' % (diffthresh,len(lessvals))



# collect tweets from dataframe within radius X of lon,lat
unit = 'meters'
radius = 200
radius_increment = 50
radius_max = 1000
min_activity = 200
events = []
for i in range(len(diffmore_lon)):
    print 'getting tweets from near: %.6f,%.6f' % (diffmore_lat[i],diffmore_lon[i])
    now_nearby = sd.selectActivityFromPoint(activity_now,diffmore_lon[i],diffmore_lat[i],unit,radius,radius_increment,radius_max,min_activity)
    if now_nearby.shape[0] > 0:
        # events.append(dict(lat=now_nearby['latitude'][0], long=now_nearby['longitude'][0], clusterid=i, tweet=now_nearby['text'][0]))
        for j in range(now_nearby.shape[0]):
            events.append(dict(lat=now_nearby['latitude'][j], long=now_nearby['longitude'][j], clusterid=i, tweet=now_nearby['text'][j]))

# difftweets_now = sd.selectSpaceFromPoint(activity_now,diffmore_lon,diffmore_lat)
# difftweets_then = sd.selectSpaceFromPoint(activity_then,diffless_lon,diffless_lat)




############
# Distance
############

# # calculate distance
# sf_center = [-122.4167,37.7833]
# # castro_muni = [-122.43533,37.76263]
# distance_to_user = sd.compute_distance_from_point(sf_center[0], sf_center[1], df.longitude, df.latitude, 'meters')

# distance_to_user = sd.compute_distance_from_point(df.longitude, df.latitude, df.longitude, df.latitude, 'meters')

############
# Distance matrix
############

# X = np.vstack((df.longitude, df.latitude)).T
# dist_matrix=(sd.spherical_dist_matrix(np.hstack([X,]*X.shape[0]).reshape(-1,2).T, np.vstack([X,]*X.shape[0]).T)).reshape(X.shape[0],X.shape[0])

# X = np.vstack((df.longitude, df.latitude)).T
# dist_matrix = (pdist(X,'euclidean')).reshape(X.shape[0],X.shape[0])


############
# Cluster
############

# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics

# concert lat/lon coordinates to UTM?

# Use DBSCAN

# TODO

# def clusterThose(activity_now):

X = np.vstack((activity_now.longitude, activity_now.latitude)).T
# X = np.vstack((mon_pm.longitude, mon_pm.latitude)).T
# X = np.vstack((fri_pm.longitude, fri_pm.latitude)).T
# X = StandardScaler().fit_transform(X)
scaler = StandardScaler(copy=True)
X_centered = scaler.fit(X).transform(X)

# xx, yy = zip(*X)
# scatter(xx,yy)
# show()

db = DBSCAN(eps=0.0001, min_samples=50).fit(X)
core_samples = db.core_sample_indices_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

# X = scaler.inverse_transform(X_centered)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    this_lon = X[class_member_mask]


# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, labels))

###############
# Plot clusters
###############

# Plot result
# unique_labels = set(labels)
# colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
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



# n = 1e5
# x = y = np.linspace(-5, 5, 100)

# x = np.array(df.latitude)
# y = np.array(df.longitude)

# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 2, 2, 0, 0)
# Z2 = mlab.bivariate_normal(X, Y, 4, 1, 1, 1)
# ZD = Z2 - Z1
# x = X.ravel()
# y = Y.ravel()
# z = ZD.ravel()
# gridsize=30
# plt.subplot(111)

# # if 'bins=None', then color of each hexagon corresponds directly to its count
# # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# # the result is a pure 2D histogram 

# plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=1000)
# plt.axis([x.min(), x.max(), y.min(), y.max()])

# cb = plt.colorbar()
# cb.set_label('mean value')
# plt.show()
