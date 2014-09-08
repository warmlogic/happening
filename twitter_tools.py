"""
twitter tools
"""

import sys
import tweepy
# from twython import Twython
from authent import twitauth
import json
from HTMLParser import HTMLParser
import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import pdb


# get your key and secret here: https://dev.twitter.com/apps

# Keep the "API secret" a secret!
apikey = twitauth['apikey']
apikeysecret = twitauth['apikeysecret']

# This access token can be used to make API requests on your own account's behalf.
# Do not share your access token secret with anyone
accesstoken = twitauth['accesstoken']
accesstokensecret = twitauth['accesstokensecret']

# tweepy: authenticate to API
auth = tweepy.OAuthHandler(apikey, apikeysecret)
# tweepy: use the access token
auth.set_access_token(accesstoken, accesstokensecret)

twitAPI = tweepy.API(auth)

# http://stackoverflow.com/questions/25224692/getting-the-location-using-tweepy

# mongodb: http://stackoverflow.com/questions/17213991/how-can-i-consume-tweets-from-twitters-streaming-api-and-store-them-in-mongodb

# saving
# http://stackoverflow.com/questions/23531608/how-do-i-save-streaming-tweets-in-json-via-tweepy

# consider investigating trends http://tweepy.readthedocs.org/en/v2.3.0/api.html#API.trends_location

# POI could be found with http://tweepy.readthedocs.org/en/v2.3.0/api.html#API.reverse_geocode

def parseTweet(tweet):
    # get rid of unicode characters, newlines, commas, trailing whitespace
    parsedTweet = tweet.text.encode('ascii','ignore').replace('\n',' ').replace(',','').strip()

    print tweet.user.screen_name + ' ' + tweet.text
    if len(parsedTweet) > 0:
        print tweet.user.screen_name + ' ' + parsedTweet
    else:
        # if we lost everything
        print tweet.user.screen_name + ' ' + '\tAll unicode removed, no text remaining'
        parsedTweet = 'unicode_only'
    pdb.set_trace()
    # parse user info, time, location, text from tweet into dict
    #if tweet.created_at.utcoffset() == None:
    if tweet.created_at.strftime("%z") == '':
        created_at = tweet.created_at.strftime("%a %b %d %T") + ' +0000 ' + tweet.created_at.strftime("%Y")
    else:
        created_at = tweet.created_at.strftime("%a %b %d %T %z %Y")
    pt= {'user_id':tweet.user.id,'user_name':tweet.user.screen_name,\
    'tweet_id':tweet.id_str,'datetime':created_at,'text_full':tweet.text,'text':parsedTweet,\
    'latitude':tweet.coordinates['coordinates'][0],'longitude':tweet.coordinates['coordinates'][1]}

    return pt

# def TwitSearchGeo(keywords,geo,API=twitAPI,searchopts={'lang':'en'}):
def TwitSearchGeo(keywords,geo,API=twitAPI,searchopts={}):
    # search twitter for keywords through full timeline available
    searchresult = API.search(q=keywords,geocode=geo,count=100,**searchopts)
    parsedresults=[parseTweet(x) for x in searchresult if x.coordinates]
    print 'Searching for %s'%keywords
    if len(parsedresults)<100: # not even 100 results, so return
        print 'Found %i results'%len(parsedresults)
        if len(parsedresults)>0:
            print 'Last tweet at %s'%parsedresults[-1]['time']
        time.sleep(5.1)
        return parsedresults
    else:
        maxdepth=1000
        
        while True:
            try:
                nextresults=searchresult.next_results
                print nextresults
                kwargs= dict([kv.split('=') for kv in nextresults[1:].split('&')])
                print 'still digging...'
            except:
                print 'Out of results'
                time.sleep(5.1)
                break
            # update keyword arguments for next round of searches
            
            searchresult = API.search(**kwargs)
            print 'Found %i more results'%len(searchresult)
            parsedresults+=[parseTweet(x) for x in searchresult]
            if len(parsedresults)>maxdepth:
                break
            time.sleep(5.1)
        print 'Oldest tweet at %s'%parsedresults[-1]['time']
        print 'Found %i results'%len(parsedresults)
        return parsedresults

class StreamLogger(tweepy.StreamListener):
    def __init__(self, fileToWrite):
        self.fileToWrite = fileToWrite
        # self.dbcon = dbcon
        super(tweepy.StreamListener, self).__init__()

    def on_status(self, status):
        print status.text
        if status.coordinates:
            print 'coords:', status.coordinates
        if status.place:
            print 'place:', status.place.full_name

        return True

    def on_data(self, data):
        data = json.loads(HTMLParser().unescape(data))
        if data['coordinates']:
            # if we have latitude and longitude, parse it
            pt = self.parseStreamTweet(data)
            # write it to disk
            self.fileToWrite.write('%d,%s,%s,%.6f,%.6f,%s\n' % (pt['user_id'],pt['tweet_id'],pt['datetime'],pt['latitude'],pt['longitude'],pt['text']))
        return True

    #on_event = on_status
    on_event = on_data

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

    def parseStreamTweet(self, data):
        # get rid of unicode characters, newlines, commas, trailing whitespace
        parsedTweet = data['text'].encode('ascii','ignore').replace('\n',' ').replace(',','').strip()

        print data['user']['screen_name'] + ' ' + data['text']
        if len(parsedTweet) > 0:
            print data['user']['screen_name'] + ' ' + parsedTweet
        else:
            # if we lost everything
            print data['user']['screen_name'] + ' ' + '\tAll unicode removed, no text remaining'
            parsedTweet = 'unicode_only'

        # parse user info, time, location, text from tweet into dict
        pt= {'user_id':data['user']['id'],'user_name':data['user']['screen_name'],\
        'tweet_id':data['id_str'],'datetime':data['created_at'],'text_full':data['text'],'text':parsedTweet,\
        'latitude':data['coordinates']['coordinates'][0],'longitude':data['coordinates']['coordinates'][1]}

        return pt

# def TwitStreamGeo(boundingBox,dbcon,creds=auth):
def TwitStreamGeo(boundingBox,save_file,creds=auth):
    # append to file where we want to save tweets
    fileToWrite = open(save_file, 'a')

    # get data from streaming api
    # listener = StreamLogger(dbcon)
    listener = StreamLogger(fileToWrite)
    print listener
    stream = tweepy.streaming.Stream(creds, listener)    
    print 'Starting stream, ctrl-c to exit'
    stream.filter(locations=boundingBox)

if __name__ == '__main__':
    main()
