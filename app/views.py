from app import app
from flask import render_template, request
import pymysql as mdb
import happening
import jinja2
# import app.helpers.maps as maps
import select_data as sd
import pdb

# db = mdb.connect(user="username",host="localhost",passwd="secretsecret",db="world_innodb",
#     charset='utf8')

# @app.route('/')
# @app.route('/index')
# def splash():
#     return render_template('splash.html')
        
# @app.route('/db')
# def cities_page():
# 	with db: 
# 		cur = db.cursor()
# 		cur.execute("SELECT Name FROM city LIMIT 15;")
# 		query_results = cur.fetchall()
# 	cities = ""
# 	for result in query_results:
# 		cities += result[0]
# 		cities += "<br>"
# 	return cities
	
# @app.route("/db_fancy")
# def cities_page_fancy():
# 	with db:
# 		cur = db.cursor()
# 		cur.execute("SELECT Name, CountryCode, Population FROM city ORDER BY Population LIMIT 15;")
# 
# 		query_results = cur.fetchall()
# 	cities = []
# 	for result in query_results:
# 		cities.append(dict(name=result[0], country=result[1], population=result[2]))
# 	return render_template('cities.html', cities=cities)
	
# @app.route("/happening",methods=['GET'])
@app.route('/')
@app.route('/index')
def happening_page():

    # # request location from user
    # user_location = request.args.get("origin")
    # lat,lon,full_add,data = maps.geocode(user_location)

    # # night life
    # area_str='sf_concerts'
    # time_now = ['2014-09-05 17:00:00', '2014-09-06 05:00:00']
    # time_then = ['2014-09-08 17:00:00', '2014-09-09 05:00:00']

    # apple keynote
    area_str='apple_flint_center'
    time_now = ['2014-09-09 08:00:00', '2014-09-09 15:00:00']
    time_then = ['2014-09-08 08:00:00', '2014-09-08 15:00:00']

    # # giants vs diamondbacks
    # area_str='attpark'
    # time_now = ['2014-09-09 17:00:00', '2014-09-09 23:30:00']
    # time_then = ['2014-09-08 17:00:00', '2014-09-08 23:30:00']

    tz = 'US/Pacific'

    nbins = 50
    nclusters = 5

    activity, n_clusters, user_lon, user_lat = happening.whatsHappening(area_str=area_str,\
        nbins=nbins,nclusters=nclusters,\
        time_now=time_now, time_then=time_then, tz=tz)

    # activity, n_clusters, user_lon, user_lat, diff_lon, diff_lat = happening.whatsHappening(area_str=area_str,\
    #     nbins=nbins,nclusters=nclusters,\
    #     time_now=time_now, time_then=time_then, tz=tz)

    # restaurants = []
    # for i in range(len(clusters['X'])):
    #     restaurants.append(dict(lat=clusters['X'][i][0], long=clusters['X'][i][1], clusterid=clusters['labels'][i]))
    # return render_template('index.html',results=restaurants,user_lat = lat, user_long = lon, faddress = full_add, ncluster = clusters['n_clusters'])

    # collect tweets from dataframe within radius X of lon,lat
    # unit = 'meters'
    # radius = 200
    # radius_increment = 50
    # radius_max = 1000
    # min_activity = 200
    # clusCount = 0

    events = []
    heatmap = True

    for i in range(activity.shape[0]):
        # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
        events.append(dict(lat=activity['latitude'][i], long=activity['longitude'][i], clusterid=activity['clusterNum'][i], tweet=activity['text'][i]))
    return render_template('index.html',results=events,user_lat = user_lat, user_lon = user_lon, ncluster=n_clusters, heatmap=heatmap)


    # for i in range(len(diff_lon)):
    #     print 'getting tweets from near: %.6f,%.6f' % (diff_lat[i],diff_lon[i])
    #     nearby = sd.selectActivityFromPoint(activity,diff_lon[i],diff_lat[i],unit,radius,radius_increment,radius_max,min_activity)
    #     if nearby.shape[0] > 0:
    #         clusCount += 1
    #         print 'clusCount: %d' % (clusCount)
    #         # # just pass in the first tweet for now
    #         # events.append(dict(lat=nearby['latitude'][0], long=nearby['longitude'][0], clusterid=i, tweet=nearby['text'][0]))
    #         for j in range(nearby.shape[0]):
    #             # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
    #             events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=nearby['clusterNum'][j], tweet=nearby['text'][j]))
    # return render_template('index.html',results=events,user_lat = user_lat, user_lon = user_lon, ncluster=len(diff_lon), heatmap=heatmap)
    
@app.route("/testmap")
def test_maps_page():
    return render_template('testmap.html')
    
@app.route("/testmapcal")
def test_maps_cal_page():
    return render_template('testmapcal.html')

if __name__ == "__main__":
    app.run(debug=True)
