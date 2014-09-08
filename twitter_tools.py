"""
twitter tools
"""

import sys
import tweepy
# from twython import Twython
from authent import twitauth
import json
from HTMLParser import HTMLParser
# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

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


class CustomStreamListener(tweepy.StreamListener):
    def __init__(self, api):
        self.api = api
        super(tweepy.StreamListener, self).__init__()

        #self.list_of_tweets = []

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
            pt = self.parseTweet(data)
            # write it to disk
            save_file.write('%d,%s,%s,%.6f,%.6f,%s\n' % (pt['user_id'],pt['tweet_id'],pt['datetime'],pt['latitude'],pt['longitude'],pt['text']))
        return True

    #on_event = on_status
    on_event = on_data

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

    def parseTweet(self, data):
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

# append to file where we want to save tweets
save_file = open('latlong_user_geodate.csv', 'a')

# NB: tweets seem to come in from outside bounding box
bayArea_bb_twit = [-122.75,36.8,-121.75,37.8]
bayArea_bb_me = [-122.53,36.94,-121.8,38.0]

# get data from streaming api
sapi = tweepy.streaming.Stream(auth, CustomStreamListener(save_file))    
sapi.filter(locations=bayArea_bb_me)

if __name__ == '__main__':
    main()
