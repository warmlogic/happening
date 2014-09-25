from app import app
from flask import render_template, request, redirect, url_for
import json
import pymysql as mdb
import happening as hap
import jinja2
import app.helpers.maps as maps
import select_data as sd
import numpy as np
import pandas as pd
from authent import instaauth
from authent import dbauth as authsql
import pdb

#############
# ROUTING/VIEW FUNCTIONS
#############

@app.route('/')
@app.route('/index')
def happening_page():
    # Renders index.html
    
    # TODO: pass in a page title

    # TODO: can include map and form on base.html in {% block body %}

    #############
    # read the data
    #############

    # latlong = open("./data/latlong_userdategeo_combined.csv")

    # print 'Reading locations...'
    # df = pd.read_csv(latlong,header=None,parse_dates=[2],\
    #     names=['user_id','tweet_id','datetime','longitude','latitude','text','url'],index_col='datetime')
    # print 'Done.'
    # latlong.close()

    events = []
    this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
    # for our loc, just set the average
    user_lon = np.mean(this_lon)
    user_lat = np.mean(this_lat)

    # latlng_sw = [this_lat[0], this_lon[0]]
    # latlng_ne = [this_lat[1], this_lon[1]]
    latlng_sw = [this_lat[1], this_lon[1]]
    latlng_ne = [this_lat[0], this_lon[0]]
    # pdb.set_trace()

    return render_template('index.html', results=events, examples=examples,\
        user_lat=user_lat, user_lon=user_lon,\
        latlng_sw=latlng_sw, latlng_ne=latlng_ne)

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
    endTime = pd.datetime.replace(pd.datetime.now(), microsecond=0)
    startTime = pd.datetime.isoformat(endTime - pd.tseries.offsets.Hour(hoursOffset))
    endTime = pd.datetime.isoformat(endTime)

    return redirect(url_for('.results', lng_sw=lng_sw, lng_ne=lng_ne, lat_sw=lat_sw, lat_ne=lat_ne, startTime=startTime, endTime=endTime))

@app.route("/results_predef",methods=['POST'])
def results_procPredef():
    event_id = request.form.get("event_id")

    # get the pre-defined time period
    endTime = [dct["endTime"] for dct in examples if dct["id"] == event_id][0]
    startTime = pd.datetime.isoformat(pd.to_datetime(endTime) - pd.tseries.offsets.Hour(hoursOffset))

    area_str = [dct["area_str"] for dct in examples if dct["id"] == event_id][0]

    # set the bounding box for the requested area
    this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    # get bounding box from this area_str
    return redirect(url_for('.results', lng_sw=this_lon[0], lng_ne=this_lon[1], lat_sw=this_lat[0], lat_ne=this_lat[1], startTime=startTime, endTime=endTime, selected=request.form.get("event_id")))

@app.route("/results",methods=['GET'])
def results():

    # get the selected event
    selected = request.args.get('selected')
    if selected == None:
        selected = "1"

    this_lon = [float(request.args.get('lng_sw')), float(request.args.get('lng_ne'))]
    this_lat = [float(request.args.get('lat_sw')), float(request.args.get('lat_ne'))]

    user_lon = np.mean(this_lon)
    user_lat = np.mean(this_lat)

    latlng_sw = [float(request.args.get('lat_sw')), float(request.args.get('lng_sw'))]
    latlng_ne = [float(request.args.get('lat_ne')), float(request.args.get('lng_ne'))]

    startTime_now_UTC = pd.to_datetime(request.args.get('startTime')).tz_localize(tz).tz_convert('UTC').isoformat()
    endTime_now_UTC = pd.to_datetime(request.args.get('endTime')).tz_localize(tz).tz_convert('UTC').isoformat()
    time_now = [startTime_now_UTC, endTime_now_UTC]

    nhours = np.round((pd.datetools.parse(time_now[1]) - pd.datetools.parse(time_now[0])).seconds / 60.0 / 60.0)

    # # compare to the day before
    # daysOffset = 1
    # startTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[0]) - pd.tseries.offsets.Day(daysOffset))
    # endTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[1]) - pd.tseries.offsets.Day(daysOffset))
    # time_then = [startTime_then_UTC, endTime_then_UTC]

    # compare to the previous X hours
    startTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[0]) - pd.tseries.offsets.Hour(hoursOffset))
    endTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[1]) - pd.tseries.offsets.Hour(hoursOffset))
    time_then = [startTime_then_UTC, endTime_then_UTC]

    # open connection to database
    if 'port' in authsql:
        con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'],port=authsql['port'])
    else:    
        con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'])

    # query the database
    activity_now = sd.selectFromSQL(con,time_now,this_lon,this_lat,tz)
    print 'Now: Selected %d entries from now' % (activity_now.shape[0])
    activity_then = sd.selectFromSQL(con,time_then,this_lon,this_lat,tz)
    print 'Then: Selected %d entries from then' % (activity_then.shape[0])

    # close connection to database
    con.close()

    # 0.003 makes bins about the size of AT&T park
    bin_scaler = 0.003
    nbins_lon = int(np.ceil(float(np.diff(this_lon)) / bin_scaler))
    nbins_lat = int(np.ceil(float(np.diff(this_lat)) / bin_scaler))
    # print 'nbins_lon: %d' % nbins_lon
    # print 'nbins_lat: %d' % nbins_lat
    n_top_hotspots = 5
    min_nclusters = 2
    max_nclusters = 5
    diffthresh = 15 * nhours
    # diffthresh = int(np.floor((nbins[0] * nbins[1] / 100) * 0.75))
    # diffthresh = int(np.floor(np.prod(nbins) / 100))
    # print 'diffthresh: %d' % diffthresh
    # eps = 0.1
    eps = 0.075
    # eps = 0.05
    # eps = 0.025
    # min_samples = 30 * nhours
    min_samples = 15 * nhours

    if activity_now.shape[0] > 0 and activity_then.shape[0] > 0:
        activity, n_clusters, cluster_centers, message, success = hap.whatsHappening(\
            activity_now=activity_now, activity_then=activity_then,\
            nbins=[nbins_lon, nbins_lat],\
            min_nclusters=min_nclusters, max_nclusters=max_nclusters,\
            n_top_hotspots=n_top_hotspots,\
            diffthresh=diffthresh, eps=eps, min_samples=min_samples)
    elif len(activity_now) > 0 and len(activity_then) == 0:
        n_clusters = 0
        cluster_centers = []
        message = 'Sorry, no activity found during the baseline time!'
        success = False
    elif len(activity_now) == 0 and len(activity_then) > 0:
        n_clusters = 0
        cluster_centers = []
        message = 'Sorry, no activity found during this time!'
        success = False
    else:
        n_clusters = 0
        cluster_centers = []
        message = 'Sorry, no activity found in this region!'
        success = False

    print 'message: ' + message

    if success is False:
        # TODO: set redirect to failure page
        events = []
        return render_template('no_events.html', results=events, examples=examples,\
            user_lat=user_lat, user_lon=user_lon,\
            latlng_sw=latlng_sw, latlng_ne=latlng_ne,\
            selected=selected)

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

        # top_words.append(freq_dist.most_common(top_nWords))
        # top_w = sorted(freq_dist, key=freq_dist.get, reverse=True)
        # print top_w[:top_nWords]
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
    for i in range(activity.shape[0]):
        # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
        events.append(dict(lat=activity['latitude'][i], long=activity['longitude'][i], clusterid=activity['clusterNum'][i], tweet=activity['text'][i]))

    clus_centers = []
    word_array = []
    top_nWords = 20
    for i, clus in enumerate(cluster_centers):
        clus_centers.append(dict(lat=clus[1], long=clus[0], clusterid=int(clus[2])))
        this_array = []
        for word in word_freq[i].most_common(top_nWords):
            this_array.append({'text': word[0], 'weight': word[1]})
            # this_array.append({'text': word[0], 'weight': word[1], 'style': 'color:' + clusterColor[i] + ';'})
            # this_array.append({'text': word[0], 'weight': word[1], 'class': str(i)})
        word_array.append(this_array)

    resample_activity_overtime = '15min'
    grouper = pd.TimeGrouper(resample_activity_overtime)
    activity_resamp = activity.groupby(grouper).apply(lambda x: x['clusterNum'].value_counts()).unstack()
    colNums = np.array(activity_resamp.columns)
    colNums = colNums[colNums[:] >= 0]

    plotdata = [];
    for i in range(activity_resamp.shape[0]):
        thisRow = []
        thisRow.extend([activity_resamp.index[i].isoformat()])
        for clusNum in colNums:
            thisCluster = activity_resamp[clusNum]
            thisCluster.fillna(value=0, inplace=True)
            thisRow.extend([thisCluster[i]])
        plotdata.append(thisRow)
    
    returnObject = {
        'visData': plotdata
    }
    # returnObject = {
    #     'visData': plotdata,
    #     'photoURLs': photo_urls
    # }

    insta_access_token = instaauth['accesstoken']
    return render_template('results.html', results=events,\
        examples=examples,\
        ncluster=n_clusters, clus_centers=clus_centers,\
        user_lat = user_lat, user_lon = user_lon,\
        latlng_sw = latlng_sw, latlng_ne = latlng_ne,\
        heatmap=True,\
        word_array=word_array,\
        message = message,\
        plotdata=plotdata,\
        selected=selected,\
        clusterColor=clusterColor,\
        insta_access_token=insta_access_token,\
        cluster_centers=cluster_centers,\
        time_now_start=activity.index[0].value // 10**9,\
        time_now_end=activity.index[-1].value // 10**9)
        # clus1_lat=cluster_centers[0][1],\
        # clus1_lon=cluster_centers[0][0],\

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


#############
# set up some info that we'll use
#############
tz = 'US/Pacific'
hoursOffset = 3
# hoursOffset = 24

# ["gray","orange","yellow","green","blue","purple"]
clusterColor = ["D1D1E0","FF9933","FFFF66","00CC00","0066FF","CC0099"]
# FFFF66 # yellow
# CC0099 # purple
# E78AC3 # pink
# 8DA0CB # purplish

# ["gray","turquoise","orangy","purplish","pink","limey"]
# clusterColor = ["D1D1E0","66C2A5","FC8D62","8DA0CB","E78AC3","A6D854"]

#############
# set up some examples
#############

examples = [{"id": "1", "area_str": "apple_flint_center", "name": "Tue Sep 9, 2014, 12 PM - Cupertino", "endTime": "2014-09-09T12:00:00"},
            {"id": "2", "area_str": "apple_flint_center", "name": "Tue Sep 9, 2014, 3 PM - Cupertino", "endTime": "2014-09-09T15:00:00"},
            {"id": "3", "area_str": "sf", "name": "Tue Sep 9, 2014, 9 PM - SF", "endTime": "2014-09-09T21:00:00"},
            {"id": "4", "area_str": "sf", "name": "Fri Sep 19, 2014, 9 PM - SF", "endTime": "2014-09-19T21:00:00"},
            {"id": "5", "area_str": "mtview_caltrain", "name": "Sun Sep 21, 2014, 12 PM - MtnView", "endTime": "2014-09-21T08:00:00"}
            ]
