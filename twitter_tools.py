'''
twitter tools
'''

import sys
import tweepy
# from twython import Twython
from authent import twitauth
import json
from HTMLParser import HTMLParser
import time
import datetime
import dateutil.parser as parser
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import pdb


# get your key and secret here: https://dev.twitter.com/apps

# Keep the API secret a secret!
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

    # parse user info, time, location, text from tweet into dict

    # dates are UTC; store in ISO 8601 format
    created_at = tweet.created_at.isoformat()
    if tweet.created_at.utcoffset() is None:
        created_at += '+00:00'

    # print created_at + ' ' + tweet.user.screen_name + ' ' + + tweet.text
    if len(parsedTweet) > 0:
        print created_at + ' ' + tweet.user.screen_name + ' ' + parsedTweet
    else:
        # if we lost everything
        print created_at + ' ' + tweet.user.screen_name + ' ' + '\tAll unicode removed, no text remaining'
        parsedTweet = 'unicode_only'
    
    pt= {'user_id':tweet.user.id,'user_name':tweet.user.screen_name,\
    'tweet_id':tweet.id_str,'tweettime':created_at,'text_full':tweet.text,'text':parsedTweet,\
    'longitude':tweet.coordinates['coordinates'][0],'latitude':tweet.coordinates['coordinates'][1]}

    return pt

# def TwitSearchGeo(keywords,geo,count=count,API=twitAPI,searchopts={'lang':'en'}):
def TwitSearchGeoOld(keywords,geo,count,API=twitAPI,searchopts={}):
    # search twitter for keywords through full timeline available
    # searchresult = API.search(q=keywords,geocode=geo,count=count,**searchopts)
    searchresult = API.search(q=keywords,geocode=geo,count=count)
    parsedresults = [parseTweet(x) for x in searchresult if x.coordinates]
    print 'Searching for %s %s' % (keywords,geo)
    if len(parsedresults) < 100: # not even 100 results, so return
        print 'Found %i results' % len(parsedresults)
        if len(parsedresults) > 0:
            print 'Last tweet at %s' % (parsedresults[-1]['tweettime'])
        time.sleep(5.1)
        return parsedresults
    else:
        maxdepth = 1000
        
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
            print 'Found %i more results' % len(searchresult)
            parsedresults+=[parseTweet(x) for x in searchresult if x.coordinates]
            if len(parsedresults)>maxdepth:
                break
            time.sleep(5.1)
        print 'Oldest tweet at %s' % parsedresults[-1]['tweettime']
        print 'Found %i results' % len(parsedresults)
        return parsedresults

def TwitSearchGeo(keywords,geo,count,max_tweets,API=twitAPI,searchopts={}):
    'http://stackoverflow.com/questions/22469713/managing-tweepy-api-search'
    # only allowed to make 180 searches per user in a 15-minute sliding window
    searchCount = 0
    searchLimit = 180
    searchLimitMin = 15
    searchTimes = []

    parsedresults = []
    # searched_tweets = []
    last_id = -1
    while len(parsedresults) < max_tweets:
        # # will only return 100 tweets, I don't know why the example did this
        # count = max_tweets - len(parsedresults)
        if len(searchTimes) >= searchLimit:
            elapsed = searchTimes[-1] - searchTimes[0]
            print 'first search in mem ' + str(searchTimes[0])
            print 'last search in mem  ' + str(searchTimes[-1])
            print 'elapsed ' + str(elapsed)
            if elapsed < datetime.timedelta(minutes=searchLimitMin):
                timeToSleep = datetime.timedelta(minutes=searchLimitMin) - elapsed
                print 'Made %d requests in last %dmin %dsec, sleeping for %dmin %dsec' % (len(searchTimes),elapsed.seconds / 60, elapsed.seconds % 60,timeToSleep.seconds / 60, timeToSleep.seconds % 60)
                time.sleep(timeToSleep.seconds)
            searchTimes.pop(0)
        try:
            searchCount += 1
            searchTimes.append(datetime.datetime.now())
            new_tweets = API.search(q=keywords, geocode=geo, count=count, max_id=str(last_id - 1))
            print '%d searches' % searchCount
            if not new_tweets:
                break
            # searched_tweets.extend(new_tweets)
            parsedresults_new = [parseTweet(x) for x in new_tweets if x.coordinates]
            parsedresults.extend(parsedresults_new)
            last_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            pdb.set_trace()
            # depending on TweepError.code, one may want to retry or wait
            # to keep things simple, we will give up on an error
            break
    # parsedresults=[parseTweet(x) for x in searched_tweets if x.coordinates]
    print 'Found %i results' % len(parsedresults)
    if len(parsedresults) > 1:
        print 'Newest tweet at %s' % parsedresults[0]['tweettime']
    if len(parsedresults) > 2:
        print 'Oldest tweet at %s' % parsedresults[-1]['tweettime']
    else:
        print 'Only one tweet found!'
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
            self.fileToWrite.write('%d,%s,%s,%.6f,%.6f,%s\n' %\
                (pt['user_id'],pt['tweet_id'],pt['tweettime'],pt['longitude'],pt['latitude'],pt['text']))
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

        created_at = parser.parse(data['created_at']).isoformat()
        # print created_at + ' ' + data['user']['screen_name'] + ' ' + data['text']
        if len(parsedTweet) > 0:
            print created_at + ' ' + data['user']['screen_name'] + ' ' + parsedTweet
        else:
            # if we lost everything
            print created_at + ' ' + data['user']['screen_name'] + ' ' +\
            '\tAll unicode removed, no text remaining'
            parsedTweet = 'unicode_only'

        # parse user info, time, location, text from tweet into dict
        pt= {'user_id':data['user']['id'],'user_name':data['user']['screen_name'],\
        'tweet_id':data['id_str'],'tweettime':created_at,'text_full':data['text'],'text':parsedTweet,\
        'longitude':data['coordinates']['coordinates'][0],'latitude':data['coordinates']['coordinates'][1]}

        return pt

# def TwitStreamGeo(boundingBox,dbcon,creds=auth):
def TwitStreamGeo(boundingBox,save_file,creds=auth):
    # append to file where we want to save tweets
    fileToWrite = open(save_file, 'a')

    # get data from streaming api
    # listener = StreamLogger(dbcon)
    listener = StreamLogger(fileToWrite)
    stream = tweepy.streaming.Stream(creds, listener)    
    print 'Starting stream, ctrl-c to exit'
    stream.filter(locations=boundingBox)

# if __name__ == '__main__':
#     import streamFromGeo_twitter
