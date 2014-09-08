'''
search for geographic tweets
'''

import pandas as pd
import numpy as np
import cPickle
# # import MySQLdb as mdb
# import pymysql as mdb
import time
import twitter_tools
# from authent import dbauth as authsql

import pdb

# load beer names with >500 ratings

# sql='''
#     SELECT beers.beername, beers.id
#     FROM beers
#     JOIN revstats ON beers.id=revstats.id
#     WHERE revstats.nreviews>500;
#     '''

# con=mdb.connect(**authsql)
# print 'Loading neighborhoods'
# df=pd.io.sql.read_frame(sql,con)
# beers=list(df['neighborhoods'])
# ids=list(df['id'])
# totalnum=len(beers)
# print 'Found %i beers'%totalnum

# # NB: tweets seem to come in from outside bounding box
# bayArea_bb_twit = [-122.75,36.8,-121.75,37.8] # from twitter's dev site
# bayArea_bb_me = [-122.53,36.94,-121.8,38.0] # I made this one

query = "since:2014-09-02 until:2014-09-03"
sf_center = "37.75,-122.44,4mi"

# count = 100
# results = twitter_tools.TwitSearchGeoOld(query,sf_center,count,twitter_tools.twitAPI)

count = 100
max_tweets = 1000
results = twitter_tools.TwitSearchGeo(query,sf_center,count,max_tweets,twitter_tools.twitAPI)

if len(results) > 0:
    pdb.set_trace()

# # search twitter for beers and save out to dataframe
# count=0
# tweetholder=[]
# for bn in beers:
#     searchstr='"'+bn+'"'
#     print 'On %i of %i'%(count+1,totalnum)
#     results = twittertools.TwitSearch(searchstr,twittertools.twitAPI)
#     tweetholder.append(results)
#     count+=1

print('Done.')
# save
# timeint = np.int(time.time())
# cPickle.dump(tweetholder,open('tweetsearch_%i.cpk'%timeint,'w'))
