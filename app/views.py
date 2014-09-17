from app import app
from flask import render_template, request
import pymysql as mdb
import happening as hap
import jinja2
import app.helpers.maps as maps
import select_data as sd
import numpy as np
import pdb

from nltk.corpus import stopwords
from nltk import FreqDist
# import nltk
# nltk.download() # get the stopwords corpus
import string

# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def happening_page():
    # Renders index.html.
    # # request location from user
    # user_location = request.args.get("origin")
    # lat,lon,full_add,data = maps.geocode(user_location)
    
    events = []
    this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
    # for our loc, just set the average
    user_lon = np.mean(this_lon)
    user_lat = np.mean(this_lat)

    return render_template('index.html',results=events,user_lat = user_lat, user_lon = user_lon)


@app.route("/results",methods=['GET'])
# @app.route('/results/<entered>', methods=['GET', 'POST'])
def results():
    # pdb.set_trace()
    user_location = request.args.get("origin")
    lat,lon,full_add,data = maps.geocode(user_location)
    # print lat
    # print lon
    # print full_add
    # print data

    # entered = json.loads(entered)
    # start_address, categories, yelp_perc  = sanitize_input(entered)
    # start_lat,start_long = hopper.get_coordinates(start_address)
    # if not hopper.in_bay_area(start_lat, start_long):
    #     raise InvalidUsage("Starting Location Not in Bay Area")
    # try:
    #     locations = hopper.get_path(start_lat, start_long, yelp_perc, tuple(categories))
    # except Exception as e:
    #     raise InvalidUsage(e)
    # suggestion = hopper.get_recommended(locations)
    # return render_template('results.html', locations = locations, start = (start_lat,start_long), suggestion=suggestion)

    # Renders index.html.
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

    activity, n_clusters, cluster_centers, user_lon, user_lat = hap.whatsHappening(area_str=area_str,\
        nbins=nbins,nclusters=nclusters,\
        time_now=time_now, time_then=time_then, tz=tz)

    # tokens, freq_dist = hap.getWordFrequency(activity_clustered)

    # for removing punctuation (via translate)
    table = string.maketrans("","")
    clean_text = []
    # for removing stop words
    stop = stopwords.words('english')
    tokens = []
    # stop.append('AT_USER')
    # stop.append('URL')
    stop.append('unicode_only')
    stop.append('w')
    stop.append('im')
    stop.append('')

    for txt in activity['text'].values:
        txt = sd.processTweet(txt)
        nopunct = txt.translate(table, string.punctuation)
        #Remove additional white spaces
        # nopunct = re.sub('[\s]+', ' ', nopunct)
        # if nopunct is not '':
        clean_text.append(nopunct)
        # split it and remove stop words
        txt = sd.getFeatureVector(txt,stop)
        tokens.extend([t for t in txt])
    freq_dist = FreqDist(tokens)
    top_words = freq_dist.keys()[:20]
    
    events = []
    clus_centers = []
    heatmap = True
    for i in range(activity.shape[0]):
        # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
        events.append(dict(lat=activity['latitude'][i], long=activity['longitude'][i], clusterid=activity['clusterNum'][i], tweet=activity['text'][i]))
    for clus in cluster_centers:
        clus_centers.append(dict(lat=clus[1], long=clus[0], clusterid=int(clus[2])))
    return render_template('results.html', results=events,\
        top_words=top_words,\
        ncluster=n_clusters, clus_centers=clus_centers,\
        user_lat = user_lat, user_lon = user_lon, heatmap=heatmap)



@app.route('/author')
def contact():
    # Renders author.html.
    return render_template('author.html')

# @app.route('/slides')
# def about():
#     # Renders slides.html.
#     return render_template('slides.html')

# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('404.html'), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return render_template('500.html'), 500

