'''
check out the data
'''

import pandas as pd
import select_data as sd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
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

# latlong = open("data/latlong_combined.csv")
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

# choose only coordinates in our bounding box of interest

bayarea_lon = [-122.53,-121.8]
bayarea_lat = [36.94,38.0]

sf_lon = [-122.5686,-122.375]
sf_lat = [37.6681,37.8258]

attpark_lon = [-122.3977,-122.3838]
attpark_lat = [37.7733,37.7840]

fishwharf_lon = [-122.4231,-122.4076]
fishwharf_lat = [37.8040,37.8116]

sf_concerts_lon = [-122.4258,-122.4000]
sf_concerts_lat = [37.7693,37.7926]

nobhill_lon = [-122.4322,-122.3976]
nobhill_lat = [37.7845,37.8042]

mtview_caltrain_lon = [-122.0832,-122.0743]
mtview_caltrain_lat = [37.3897,37.3953]

# this_lon = bayarea_lon;
# this_lat = bayarea_lat;

# this_lon = sf_lon
# this_lat = sf_lat

# this_lon = attpark_lon
# this_lat = attpark_lat

# this_lon = fishwharf_lon
# this_lat = fishwharf_lat

this_lon = sf_concerts_lon
this_lat = sf_concerts_lat

# this_lon = nobhill_lon
# this_lat = nobhill_lat

# this_lon = mtview_caltrain_lon
# this_lat = mtview_caltrain_lat

df = sd.selectSpace(df,this_lon,this_lat)

############
# Time
############

# # set the start and end datetimes

# sinceDatetime = '2014-09-05 09:00:00'
# untilDatetime = '2014-09-05 17:00:00'
# sinceDatetime = ''
# untilDatetime = ''

# tz = 'US/Pacific'
# df = sd.selectTime(df,tz=tz,sinceDatetime=sinceDatetime,untilDatetime=untilDatetime)

sinceDatetime = '2014-09-05 17:00:00'
untilDatetime = '2014-09-06 05:00:00'
fri_pm = df.ix[sinceDatetime:untilDatetime]

sinceDatetime = '2014-09-08 17:00:00'
untilDatetime = '2014-09-09 05:00:00'
mon_pm = df.ix[sinceDatetime:untilDatetime]



############
# Plot
############

nbins = 25
# plt = sd.plot_hist(df,nbins)
plt = sd.plot_hist(fri_pm,nbins)
plt = sd.plot_hist(mon_pm,nbins)

savefig = True
if savefig:
    figname = 'data/latlong_plot.png'
    print 'saving figure to ' + figname
    plt.savefig(figname, bbox_inches='tight')
# plt.show()

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

###########
# plot over time
###########

# tweetlocs = df.ix[:, ['longitude','latitude']]
tweetlocs_fri = fri_pm.ix[:, ['longitude','latitude']].resample('60min', how='count')
tweetlocs_mon = mon_pm.ix[:, ['longitude','latitude']].resample('60min', how='count')

# volume = df.resample('60min', how='count')
fig, ax = plt.subplots()
tweetlocs_fri.plot(kind='line',style='b')
tweetlocs_mon.plot(kind='line',style='r')
fig.autofmt_xdate()
# ax.set_xlim(['17:00:00','05:00:00'])

fig, ax = subplots()
volume.plot(ax=ax, kind='bar', legend=False)
fig.autofmt_xdate()
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

############
# Cluster
############

# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics

# concert lat/lon coordinates to UTM?

# Use DBSCAN

X = np.vstack((df.longitude, df.latitude)).T
# X = StandardScaler().fit_transform(X)

# xx, yy = zip(*X)
# scatter(xx,yy)
# show()

db = DBSCAN(eps=0.0002, min_samples=100).fit(X)
core_samples = db.core_sample_indices_
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)

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
unique_labels = set(labels)
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
