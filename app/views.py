from app import app
from flask import render_template, request, redirect, url_for
import pymysql as mdb
import happening as hap
import jinja2
import app.helpers.maps as maps
import select_data as sd
import numpy as np
import pandas as pd
import pdb


#############
# Helpers
#############


# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def happening_page():
    # Renders index.html.
    # # request location from user
    # user_location = request.args.get("origin")
    # lat,lon,full_add,data = maps.geocode(user_location)
    
    # TODO: pass in a page title


    # TODO: pass in an examples variable to populate the pulldown form on the html page.
    # can have id and name for easy passing

    # can include map and form on base.html in {% block body %}

    events = []
    this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
    # for our loc, just set the average
    user_lon = np.mean(this_lon)
    user_lat = np.mean(this_lat)

    return render_template('index.html', results=events, examples=examples, user_lat=user_lat, user_lon=user_lon)


@app.route("/results_location",methods=['POST'])
def results_procLocation():
    user_location = request.form.get("location")
    lat,lon,full_add,data = maps.geocode(user_location)

    # set the bounding box for the requested area
    res = data['results'][0]
    lng_sw = res['geometry']['bounds']['southwest']['lng']
    lng_ne = res['geometry']['bounds']['northeast']['lng']
    lat_sw = res['geometry']['bounds']['southwest']['lat']
    lat_ne = res['geometry']['bounds']['northeast']['lat']

    # get the times
    hoursOffset = 1
    # endTime = pd.datetime.isoformat(pd.datetime.now())
    endTime = pd.datetime.replace(pd.datetime.now(), microsecond=0)
    startTime = pd.datetime.isoformat(endTime - pd.tseries.offsets.Hour(hoursOffset))
    endTime = pd.datetime.isoformat(endTime)
    # time_now = [startTime, endTime]

    # pdb.set_trace()

    return redirect(url_for('.results', lng_sw=lng_sw, lng_ne=lng_ne, lat_sw=lat_sw, lat_ne=lat_ne, startTime=startTime, endTime=endTime))

@app.route("/results_predef",methods=['POST'])
def results_procPredef():
    area_str = request.form.get("event_id")

    # set the bounding box for the requested area
    this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    # get the pre-defined time period
    startTime = [dct["startTime"] for dct in examples if dct["id"] == area_str][0]
    endTime = [dct["endTime"] for dct in examples if dct["id"] == area_str][0]
    # time_now = [startTime, endTime]

    # pdb.set_trace()

    # get bounding box from this area_str
    return redirect(url_for('.results', lng_sw=this_lon[0], lng_ne=this_lon[1], lat_sw=this_lat[0], lat_ne=this_lat[1], startTime=startTime, endTime=endTime))
    # return redirect(url_for('.results', this_lat=this_lat, this_lon=this_lon, time_now=time_now))



# @app.route("/results",methods=['GET', 'POST'])
# @app.route('/results/<entered>', methods=['GET', 'POST'])
@app.route("/results",methods=['GET'])
def results():
    this_lon = [float(request.args.get('lng_sw')), float(request.args.get('lng_ne'))]
    this_lat = [float(request.args.get('lat_sw')), float(request.args.get('lat_ne'))]
    time_now = [request.args.get('startTime'), request.args.get('endTime')]

    # compare to the day before
    daysOffset = 1
    startTime_then = pd.datetime.isoformat(pd.datetools.parse(time_now[0]) - pd.tseries.offsets.Day(daysOffset))
    endTime_then = pd.datetime.isoformat(pd.datetools.parse(time_now[1]) - pd.tseries.offsets.Day(daysOffset))
    time_then = [startTime_then, endTime_then]

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
    # area_str='apple_flint_center'
    # time_now = ['2014-09-09 08:00:00', '2014-09-09 15:00:00']
    # time_then = ['2014-09-08 08:00:00', '2014-09-08 15:00:00']

    # # giants vs diamondbacks
    # area_str='attpark'
    # time_now = ['2014-09-09 17:00:00', '2014-09-09 23:30:00']
    # time_then = ['2014-09-08 17:00:00', '2014-09-08 23:30:00']

    tz = 'US/Pacific'

    nbins = 50
    nclusters = 5

    # activity, n_clusters, cluster_centers, user_lon, user_lat = hap.whatsHappening(area_str=area_str,\
    activity, n_clusters, cluster_centers, user_lon, user_lat = hap.whatsHappening(this_lon=this_lon,this_lat=this_lat,\
        nbins=nbins,nclusters=nclusters,\
        time_now=time_now, time_then=time_then, tz=tz)


    # # for removing punctuation (via translate)
    # table = string.maketrans("","")
    # clean_text = []
    # # for removing stop words
    # stop = stopwords.words('english')
    # tokens = []
    # # stop.append('AT_USER')
    # # stop.append('URL')
    # stop.append('unicode_only')
    # stop.append('w')
    # stop.append('im')
    # stop.append('')

    # for txt in activity['text'].values:
    #     txt = sd.processTweet(txt)
    #     nopunct = txt.translate(table, string.punctuation)
    #     #Remove additional white spaces
    #     # nopunct = re.sub('[\s]+', ' ', nopunct)
    #     # if nopunct is not '':
    #     clean_text.append(nopunct)
    #     # split it and remove stop words
    #     txt = sd.getFeatureVector(txt,stop)
    #     tokens.extend([t for t in txt])
    # freq_dist = FreqDist(tokens)

    # do this for each cluster?
    # tokens, freq_dist, clean_text = hap.cleanTextGetWordFrequency(activity)

    # top_words = freq_dist.keys()[:20]

    # happy_log_probs, sad_log_probs = hap.readSentimentList()

    word_freq = []
    # top_nWords = 20
    # top_words = []
    # cluster_happy_sentiment = []
    for clusNum in range(n_clusters):
        activity_thisclus = activity.loc[activity['clusterNum'] == clusNum]
        tokens, freq_dist, clean_text = hap.cleanTextGetWordFrequency(activity_thisclus)
        word_freq.append(freq_dist)
        # top_w = sorted(freq_dist, key=freq_dist.get, reverse=True)
        # try:
        #     top_words.append(top_w[:top_nWords])
        # except:
        #     top_words.append(top_w)
        # happy_probs = []
        # for tweet in clean_text:
        #     prob_happy, prob_sad = hap.classifySentiment(tweet.split(), happy_log_probs, sad_log_probs)
        #     happy_probs.append(prob_happy)
        #     print tweet
        #     print prob_happy
        # cluster_happy_sentiment.append(sum(np.array(happy_probs) > .5) / float(len(happy_probs)))

    # sentiment analysis
    #
    # short sentences:
    # https://github.com/abromberg/sentiment_analysis_python
    # http://andybromberg.com/sentiment-analysis-python/
    #
    # bag of words:
    # https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107
    # http://jeffreybreen.wordpress.com/2011/07/04/twitter-text-mining-r-slides/
    
    # tried this but it doesn't work well for my data
    # http://alexdavies.net/twitter-sentiment-analysis/


    # create a word cloud array
    # https://github.com/lucaong/jQCloud

    events = []
    clus_centers = []
    for i in range(activity.shape[0]):
        # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
        events.append(dict(lat=activity['latitude'][i], long=activity['longitude'][i], clusterid=activity['clusterNum'][i], tweet=activity['text'][i]))
    for clus in cluster_centers:
        clus_centers.append(dict(lat=clus[1], long=clus[0], clusterid=int(clus[2])))

    heatmap = True
    return render_template('results.html', results=events,\
        examples=examples,\
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

examples = [{"id": "apple_flint_center", "name": "Apple Keynote - Sep 9, 2014", "startTime": "2014-09-09 08:00:00", "endTime": "2014-09-09 15:00:00"},
            {"id": "attpark", "name": "Diamondbacks at Giants - Sep 9, 2014", "startTime": "2014-09-09 17:00:00", "endTime": "2014-09-09 23:30:00"},
            {"id": "3", "name": "Event 3", "startTime": "", "endTime": ""},
            {"id": "4", "name": "Event 4", "startTime": "", "endTime": ""}
            ]
