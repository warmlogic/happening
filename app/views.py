from app import app
from flask import render_template, request
import pymysql as mdb
import happening
import jinja2
import pdb
import app.helpers.maps as maps
import select_data as sd

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

    # this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='sf')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='fishwharf')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='embarc')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='att48')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='pier48')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='attpark')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='mission')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='sf_concerts')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='nobhill')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='mtview_caltrain')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='apple_flint_center')
    # this_lon, this_lat = sd.set_get_boundBox(area_str='levisstadium')

    area_str='attpark'
    # area_str='apple_flint_center'
    activity, diff_lon, diff_lat, user_lon, user_lat = happening.whatsHappening(area_str=area_str, tz='US/Pacific')
    # activity, diff_lon, diff_lat, user_lon, user_lat = happening.whatsHappening(this_lon, this_lat, area_str='apple_flint_center', tz='US/Pacific')

    # restaurants = []
    # for i in range(len(clusters['X'])):
    #     restaurants.append(dict(lat=clusters['X'][i][0], long=clusters['X'][i][1], clusterid=clusters['labels'][i]))
    # return render_template('results.html',results=restaurants,user_lat = lat, user_long = lon, faddress = full_add, ncluster = clusters['n_clusters'])

    # collect tweets from dataframe within radius X of lon,lat
    unit = 'meters'
    radius = 200
    radius_increment = 50
    radius_max = 1000
    min_activity = 200
    events = []
    for i in range(len(diff_lon)):
        print 'getting tweets from near: %.6f,%.6f' % (diff_lat[i],diff_lon[i])
        nearby = sd.selectActivityFromPoint(activity,diff_lon[i],diff_lat[i],unit,radius,radius_increment,radius_max,min_activity)
        # just pass in the first tweet for now
        if nearby.shape[0] > 0:
            events.append(dict(lat=nearby['latitude'][0], long=nearby['longitude'][0], clusterid=i, tweet=nearby['text'][0]))
            # pdb.set_trace()
    # return render_template('results.html',results=events,user_lat = lat, user_lon = lon, faddress = full_add, ncluster = clusters['n_clusters'])
    return render_template('results.html',results=events,user_lat = user_lat, user_lon = user_lon)
    
@app.route("/testmap")
def test_maps_page():
    return render_template('testmap.html')
