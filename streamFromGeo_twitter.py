'''
collect tweets from a geo bounding box, will eventually send to mysql db
'''

import pymysql as mdb
import twitter_tools
from authent import dbauth as authsql

# Create table to hold the tweets before processing

# con=mdb.connect(**authsql)
con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'])
with con:
    cur=con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS happening.tweet_template(
        userid BIGINT UNSIGNED DEFAULT '0',
        tweetid BIGINT UNSIGNED,
        tweettime DATETIME,
        tweetlon DECIMAL(9,6),
        tweetlat DECIMAL(9,6),
        tweettext VARCHAR(140),
        picurl VARCHAR(140) DEFAULT ""
        );
        """)
    cur.execute("""CREATE TABLE IF NOT EXISTS happening.tweet_table LIKE happening.tweet_template;""")

# sentiment VARCHAR(20)
# note: connection closed, need to reopen for writing

# NB: tweets seem to come in from outside bounding box
bayArea_bb_twit = [-122.75,36.8,-121.75,37.8] # from twitter's dev site
bayArea_bb_me = [-122.53,36.94,-121.8,38.0] # I made this one

# save_file = 'latlong_user_geodate.csv'

print_debug = False
twitter_tools.TwitStreamGeo(bayArea_bb_me,con,print_debug)
# twitter_tools.TwitStreamGeo(bayArea_bb_me,save_file,print_debug)
