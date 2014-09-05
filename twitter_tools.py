"""
twitter tools
"""

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

# # Twython: authenticate to API
# twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
# # Twython: Obtain an OAuth 2 Access Token (only request once and save it somewhere)
# ACCESS_TOKEN = twitter.obtain_access_token()
#
# # Twython: Use the ACCESS_TOKEN
# twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

# http://stackoverflow.com/questions/25224692/getting-the-location-using-tweepy

# mongodb: http://stackoverflow.com/questions/17213991/how-can-i-consume-tweets-from-twitters-streaming-api-and-store-them-in-mongodb

# saving
# http://stackoverflow.com/questions/23531608/how-do-i-save-streaming-tweets-in-json-via-tweepy

# initialize blank list to contain latitude and longitude
#latlong = []
save_file = open('latlong.csv', 'a')

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
        # print the tweet and all metadata
        #print data
        # print the tweet
        print data['text']
        if data['coordinates']:
            # print data['coordinates']
            print data['coordinates']['coordinates']
            # self.latlong.append(data['coordinates']['coordinates'])
            save_file.write('%.6f,%.6f\n' % (data['coordinates']['coordinates'][0],data['coordinates']['coordinates'][1]))
        # if data.get('place'):
        #     print data['place']['full_name']
        return True
        # return data['coordinates']['coordinates']

    #on_event = on_status
    on_event = on_data

    def on_error(self, status_code):
        print >> sys.stderr, 'Encountered error with status code:', status_code
        return True # Don't kill the stream

    def on_timeout(self):
        print >> sys.stderr, 'Timeout...'
        return True # Don't kill the stream

# get data from streaming api
sapi = tweepy.streaming.Stream(auth, CustomStreamListener(save_file))    
sapi.filter(locations=[-122.75,36.8,-121.75,37.8]) # SF bounding box lat,long

# consider investigating trends http://tweepy.readthedocs.org/en/v2.3.0/api.html#API.trends_location

# POI could be found with http://tweepy.readthedocs.org/en/v2.3.0/api.html#API.reverse_geocode


if __name__ == '__main__':
  main()
