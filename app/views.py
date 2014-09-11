from app import app
from flask import render_template, request
import pymysql as mdb
import foodgroups
import jinja2
import pdb
import app.helpers.maps as maps

# db = mdb.connect(user="username",host="localhost",passwd="secretsecret",db="world_innodb",
#     charset='utf8')

@app.route('/')
@app.route('/index')
def splash():
    return render_template('splash.html')
        
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
	
@app.route("/foodgroups",methods=['GET'])
def food_groups_page():
    user_location = request.args.get("origin")
    lat,lon,full_add,data = maps.geocode(user_location)
    clusters = foodgroups.foodGroups(lat,lon)
    restaurants = []
    for i in range(len(clusters['X'])):
        restaurants.append(dict(lat=clusters['X'][i][0], long=clusters['X'][i][1], clusterid=clusters['labels'][i]))
    return render_template('results.html',results=restaurants,user_lat = lat, user_long = lon, faddress = full_add, ncluster = clusters['n_clusters'])
    
@app.route("/testmap")
def test_maps_page():
    return render_template('testmap.html')
