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
import pdb


#############
# Helpers
#############


# ROUTING/VIEW FUNCTIONS
@app.route('/')
@app.route('/index')
def happening_page():
    # Renders index.html
    
    # TODO: pass in a page title

    # TODO: can include map and form on base.html in {% block body %}

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
    hoursOffset = 1
    endTime = pd.datetime.replace(pd.datetime.now(), microsecond=0)
    startTime = pd.datetime.isoformat(endTime - pd.tseries.offsets.Hour(hoursOffset))
    endTime = pd.datetime.isoformat(endTime)

    return redirect(url_for('.results', lng_sw=lng_sw, lng_ne=lng_ne, lat_sw=lat_sw, lat_ne=lat_ne, startTime=startTime, endTime=endTime))

@app.route("/results_predef",methods=['POST'])
def results_procPredef():
    area_str = request.form.get("event_id")

    # set the bounding box for the requested area
    this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    # get the pre-defined time period
    startTime = [dct["startTime"] for dct in examples if dct["id"] == area_str][0]
    endTime = [dct["endTime"] for dct in examples if dct["id"] == area_str][0]

    # get bounding box from this area_str
    return redirect(url_for('.results', lng_sw=this_lon[0], lng_ne=this_lon[1], lat_sw=this_lat[0], lat_ne=this_lat[1], startTime=startTime, endTime=endTime, selected=request.form.get("event_id")))

@app.route("/results",methods=['GET'])
def results():
    # get the selected event
    selected = request.args.get('selected')
    if selected == None:
        selected = "apple_flint_center"

    this_lon = [float(request.args.get('lng_sw')), float(request.args.get('lng_ne'))]
    this_lat = [float(request.args.get('lat_sw')), float(request.args.get('lat_ne'))]
    time_now = [request.args.get('startTime'), request.args.get('endTime')]

    # compare to the day before
    daysOffset = 1
    startTime_then = pd.datetime.isoformat(pd.datetools.parse(time_now[0]) - pd.tseries.offsets.Day(daysOffset))
    endTime_then = pd.datetime.isoformat(pd.datetools.parse(time_now[1]) - pd.tseries.offsets.Day(daysOffset))
    time_then = [startTime_then, endTime_then]

    # # night life
    # area_str='sf_concerts'
    # time_now = ['2014-09-05 17:00:00', '2014-09-06 05:00:00']
    # time_then = ['2014-09-08 17:00:00', '2014-09-09 05:00:00']

    tz = 'US/Pacific'

    nbins = 50
    nclusters = 5

    activity, n_clusters, cluster_centers, user_lon, user_lat, message, success = hap.whatsHappening(\
        this_lon=this_lon,this_lat=this_lat,\
        nbins=nbins,nclusters=nclusters,\
        time_now=time_now, time_then=time_then, tz=tz)
    if success:
        print 'message: found clusters, hoooray!'
    else:
        print 'message: ' + message
        # TODO: set redirect to failure page

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
    clus_centers = []
    top_nWords = 20
    for i in range(activity.shape[0]):
        # events.append(dict(lat=nearby['latitude'][j], long=nearby['longitude'][j], clusterid=clusCount, tweet=nearby['text'][j]))
        events.append(dict(lat=activity['latitude'][i], long=activity['longitude'][i], clusterid=activity['clusterNum'][i], tweet=activity['text'][i]))

    word_array = []
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

    plotdata = [];
    for i in range(activity_resamp.shape[0]):
        thisRow = []
        thisRow.extend([activity_resamp.index[i].isoformat()])
        for clusNum in range(n_clusters):
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

    latlng_sw = [float(request.args.get('lat_sw')), float(request.args.get('lng_sw'))]
    latlng_ne = [float(request.args.get('lat_ne')), float(request.args.get('lng_ne'))]

    insta_access_token = instaauth['accesstoken']
    print cluster_centers
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
# read the data
#############

# latlong = open("./data/latlong_userdategeo_combined.csv")

# print 'Reading locations...'
# df = pd.read_csv(latlong,header=None,parse_dates=[2],\
#     names=['user_id','tweet_id','datetime','longitude','latitude','text','url'],index_col='datetime')
# print 'Done.'
# latlong.close()

#############
# set up some examples
#############

# ["gray","orange","yellow","green","blue","purple"]
clusterColor = ["D1D1E0","FF9933","FFFF66","00CC00","0066FF","CC0099"]

examples = [{"id": "apple_flint_center", "name": "Apple Keynote - Sep 9, 2014", "startTime": "2014-09-09T08:00:00", "endTime": "2014-09-09 15:00:00"},
            {"id": "attpark", "name": "Diamondbacks at Giants - Sep 9, 2014", "startTime": "2014-09-09T17:00:00", "endTime": "2014-09-09 23:30:00"},
            {"id": "3", "name": "Event 3", "startTime": "", "endTime": ""},
            {"id": "4", "name": "Event 4", "startTime": "", "endTime": ""}
            ]
