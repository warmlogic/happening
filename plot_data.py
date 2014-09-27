'''
check out the data
'''

import pandas as pd
import select_data as sd
import happening as hap
import matplotlib.pyplot as plt
import numpy as np
import pymysql as mdb
from authent import dbauth as authsql

import pdb
# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from scipy.spatial.distance import pdist

# from nltk.corpus import stopwords
# from nltk import FreqDist
# import string
# import re

%load_ext autoreload
%autoreload 2

%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import Image
plt.rcParams['figure.figsize'] = 12, 8 # plotsize

tz = 'US/Pacific'
this_lon, this_lat = sd.set_get_boundBox(area_str='sf')

# # # giants diamondbacks
time_now = ['2014-09-09 17:00:00', '2014-09-09 21:00:00']
time_then = ['2014-09-08 13:00:00', '2014-09-08 17:00:00']

# # # night life
# # time_now = ['2014-09-05 17:00:00', '2014-09-06 05:00:00']
# # time_then = ['2014-09-08 17:00:00', '2014-09-09 05:00:00']

# # # apple keynote
# # time_now = ['2014-09-09 08:00:00', '2014-09-09 15:00:00']
# # time_then = ['2014-09-08 08:00:00', '2014-09-08 15:00:00']

if 'port' in authsql:
    con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'],port=authsql['port'])
else:    
    con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'])

activity_now = sd.selectFromSQL(con,time_now,this_lon,this_lat,tz)
activity_then = sd.selectFromSQL(con,time_then,this_lon,this_lat,tz)

con.close()

print 'Now: Selecting %d entries from %s to %s' % (activity_now.shape[0],time_now[0],time_now[1])
print 'Then: Selecting %d entries from %s to %s' % (activity_then.shape[0],time_then[0],time_then[1])

# ############
# # Read the data
# ############

# # latlong = open("./data/latlong_userdategeo_combined.csv")

# # print 'Reading locations...'
# # df = pd.read_csv(latlong,header=None,parse_dates=[2],dtype={'url': str},\
# #     names=['user_id','tweet_id','datetime','longitude','latitude','text','url'],index_col='datetime')
# # print 'Done.'
# # latlong.close()

# # # twitter times are in UTC
# # # df.datetime = df.datetime.apply(lambda x: x.tz_localize('UTC'), convert_dtype=False)
# # df = df.tz_localize('UTC').tz_convert('US/Pacific')

# # df['point'] = df.apply(lambda row: pd.Point(df['latitude'], df['longitude']))

# ############
# # User
# ############

# # df = sd.selectUser(df,rmnull=True)

# ############
# # Space
# ############

# # choose only coordinates in our bounding box of interest

# # area_str='attpark'
# # area_str='apple_flint_center'
# # area_str='bayarea'
# area_str='sf'
# # area_str='fishwharf'
# # area_str='embarc'
# # area_str='att48'
# # area_str='pier48'
# # area_str='mission'
# # area_str='sf_concerts'
# # area_str='nobhill'
# # area_str='mtview_caltrain'
# # area_str='levisstadium'

# this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

# geo_activity = sd.selectSpaceBB(df,this_lon,this_lat)

# ############
# # Time
# ############

# # # set the start and end datetimes

# # sinceDatetime = '2014-09-05 09:00:00'
# # untilDatetime = '2014-09-05 17:00:00'
# # sinceDatetime = ''
# # untilDatetime = ''

# # tz = 'US/Pacific'
# # geo_activity = sd.selectTime(geo_activity,tz=tz,sinceDatetime=sinceDatetime,untilDatetime=untilDatetime)

# # # night life
# # time_now = ['2014-09-05 17:00:00', '2014-09-06 05:00:00']
# # time_then = ['2014-09-08 17:00:00', '2014-09-09 05:00:00']

# # # apple keynote
# # time_now = ['2014-09-09 08:00:00', '2014-09-09 15:00:00']
# # time_then = ['2014-09-08 08:00:00', '2014-09-08 15:00:00']

# # giants vs diamondbacks
# # time_now = ['2014-09-09 17:00:00', '2014-09-09 23:30:00']
# # time_then = ['2014-09-08 17:00:00', '2014-09-08 23:30:00']
# time_now = ['2014-09-09 17:00:00', '2014-09-09 21:00:00']
# time_then = ['2014-09-08 13:00:00', '2014-09-08 17:00:00']

# activity_now = geo_activity.ix[time_now[0]:time_now[1]]
# activity_then = geo_activity.ix[time_then[0]:time_then[1]]

# print 'Now: Selecting %d entries from %s to %s' % (activity_now.shape[0],time_now[0],time_now[1])
# print 'Then: Selecting %d entries from %s to %s' % (activity_then.shape[0],time_then[0],time_then[1])

###########
# plot over time
###########

show_plot=True
savefig = False
if show_plot:
    # tweetlocs = df.ix[:, ['longitude','latitude']]
    tweetlocs_now = activity_now.ix[:, ['longitude','latitude']].resample('60min', how='count')
    tweetlocs_then = activity_then.ix[:, ['longitude','latitude']].resample('60min', how='count')

    # volume = df.resample('60min', how='count')
    fig, ax = plt.subplots()
    tweetlocs_now.plot(kind='line',style='b')
    tweetlocs_then.plot(kind='line',style='r')
    fig.autofmt_xdate()
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if savefig:
        figname = 'data/activity_over_time_' + area_str + '.png'
        print 'saving figure to ' + figname
        plt.savefig(figname, bbox_inches='tight')
    # plt.show()


# # ax.set_xlim(['17:00:00','05:00:00'])


############
# Plot heat map and difference
############

# 0.003 makes bins about the size of AT&T park
bin_scaler = 0.009
nbins_lon = int(np.floor(float(np.diff(this_lon)) / bin_scaler))
nbins_lat = int(np.floor(float(np.diff(this_lat)) / bin_scaler))
if nbins_lon == 0:
    nbins_lon = 1
if nbins_lat == 0:
    nbins_lat = 1

nbins = [nbins_lon, nbins_lat]
show_plot=True
savefig = False
# plt = sd.make_hist(df,nbins,show_plot)
Hnow, xedges, yedges = sd.make_hist(activity_now,nbins,show_plot,savefig,'latlong_now')
Hprev, xedges, yedges = sd.make_hist(activity_then,nbins,show_plot,savefig,'latlong_then')
Hdiff = Hnow - Hprev

# Hweight = Hnow ./ Hprev

if show_plot:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.pcolormesh(xedges,yedges,Hdiff)
    ax.set_title('Difference in tweets (%d bins x %d bins)' % (nbins[0], nbins[1]))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    cb = plt.colorbar()
    cb.set_label('Count')

    if savefig:
        figname = 'data/latlong_diff_apple.png'
        print 'saving figure to ' + figname
        plt.savefig(figname, bbox_inches='tight')
    # plt.show()


# return the top n values, sorted; ascend=biggest first
n = 5
# diffthresh = int(np.floor(nbins * 0.75))
# diffthresh = int(np.floor(np.prod(nbins) / 100))
diffthresh = 30
morevals,moreind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='max', return_order='ascend')
lessvals,lessind = sd.choose_n_sorted(Hdiff, n=n, min_val=diffthresh, srt='min', return_order='ascend')

# bigcoord = zip(xedges[bigdiff[:,0]], yedges[bigdiff[:,1]])
# diffmore_lon = xedges[diffmore[:,0]]
# diffmore_lat = yedges[diffmore[:,1]]
# diffless_lon = xedges[diffless[:,0]]
# diffless_lat = yedges[diffless[:,1]]
diffmore_lon = xedges[moreind[:,1]]
diffmore_lat = yedges[moreind[:,0]]
diffless_lon = xedges[lessind[:,1]]
diffless_lat = yedges[lessind[:,0]]
print 'At threshold %d, found %d "events" that have more activity than previous time' % (diffthresh,len(morevals))
print 'At threshold %d, found %d "events" that have less activity than previous time' % (diffthresh,len(lessvals))


############
# Cluster
############

# return activity with cluster numbers after clustering with DBSCAN
# activity_clustered, n_clusters, cluster_centers =  sd.clusterThose(activity_now,nbins,diffmore_lon,diffmore_lat,centerData=True,plotData=True)
eps = 0.025
min_samples=100
plotData = True
activity_now_clustered, n_clusters, cluster_centers =  sd.clusterThose(activity_now=activity_now,nbins=nbins,diffmore_lon=diffmore_lon,diffmore_lat=diffmore_lat,eps=eps,min_samples=min_samples,plotData=plotData)


############
# NLP
############

# # for removing punctuation (via translate)
# table = string.maketrans("","")
# clean_text = []
# # for removing stop words
# stop = stopwords.words('english')
# tokens = []
# # stop.append('AT_USER')
# # stop.append('URL')
# stop.append('unicode_only')
# stop.append('w')
# stop.append('im')
# stop.append('')
# for txt in activity_clustered['text'].values:
#     txt = sd.processTweet(txt)
#     nopunct = txt.translate(table, string.punctuation)
#     #Remove additional white spaces
#     # nopunct = re.sub('[\s]+', ' ', nopunct)
#     # if nopunct is not '':
#     clean_text.append(nopunct)
#     # split it and remove stop words
#     txt = sd.getFeatureVector(txt,stop)
#     tokens.extend([t for t in txt])
# freq_dist = FreqDist(tokens)

# tokens, freq_dist, clean_text = hap.cleanTextGetWordFrequency(activity_clustered)

# happy_log_probs, sad_log_probs = hap.readSentimentList()

# cluster_happy_sentiment = hap.clusterSentiment(activity_clustered, happy_log_probs, sad_log_probs)

word_freq = []
# top_nWords = 20
# top_words = []
# cluster_happy_sentiment = []
for clusNum in range(n_clusters):
    activity_thisclus = activity_clustered.loc[activity_clustered['clusterNum'] == clusNum]
    tokens, freq_dist, clean_text = hap.cleanTextGetWordFrequency(activity_thisclus)
    word_freq.append(freq_dist)
    # top_w = sorted(freq_dist, key=freq_dist.get, reverse=True)
    # try:
    #     top_words.append(top_w[:top_nWords])
    # except:
    #     top_words.append(top_w)
    # happy_probs = []
    # for tweet in clean_text:
    #     prob_happy, prob_sad = hap.classifySentiment(tweet.split(), happy_log_probs, sad_log_probs)
    #     happy_probs.append(prob_happy)
    #     print tweet
    #     print prob_happy
    # cluster_happy_sentiment.append(sum(np.array(happy_probs) > .5) / float(len(happy_probs)))

freq_dist.plot(50)
freq_dist.keys()[:50]





############
# Select activity using radius from point
############

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
# Calculate distance from a given point
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


