'''
collect tweets from a geo bounding box, will eventually send to mysql db
'''

# import pymysql as mdb
import twitter_tools
# from authent import dbauth as authsql

# # Create table to hold the tweets before processing
#
# con=mdb.connect(**authsql)
# with con:
#     cur=con.cursor()
#     #cur.execute("""DROP TABLE IF EXISTS rawtweets;""")
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS rawtweets(
#         userid BIGINT NOT NULL PRIMARY KEY,
#         username VARCHAR(15),
#         tweetid BIGINT, 
#         tweettime DATETIME, 
#         tweettext VARCHAR(140),
#         tweetlon DECIMAL(9,6),
#         tweetlat DECIMAL(9,6),
#         picurl TEXT,
#         sentiment VARCHAR(20)
#         """)
#
# # note: connection closed, need to reopen for writing

# NB: tweets seem to come in from outside bounding box
bayArea_bb_twit = [-122.75,36.8,-121.75,37.8] # from twitter's dev site
bayArea_bb_me = [-122.53,36.94,-121.8,38.0] # I made this one

save_file = 'latlong_user_geodate.csv'

# twitter_tools.TwitStreamGeo(bayArea_bb_me,con)
twitter_tools.TwitStreamGeo(bayArea_bb_me,save_file)
