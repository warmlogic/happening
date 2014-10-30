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
import urllib
import pdb
# import time

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"

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

    latlng_sw = [this_lat[1], this_lon[1]]
    latlng_ne = [this_lat[0], this_lon[0]]

    selected = "1"
    return render_template('index.html', results=events, examples=examples,\
        user_lat=user_lat, user_lon=user_lon,\
        latlng_sw=latlng_sw, latlng_ne=latlng_ne,\
        selected=selected)

@app.route("/results_location",methods=['POST'])
def results_procLocation():
    user_location = request.form.get("location")
    if user_location == '':
        user_location = 'San Francisco, CA, United States'

    lat,lon,full_add,data = maps.geocode(user_location)

    # set the bounding box for the requested area
    res = data['results'][0]
    lng_sw = res['geometry']['viewport']['southwest']['lng']
    lng_ne = res['geometry']['viewport']['northeast']['lng']
    lat_sw = res['geometry']['viewport']['southwest']['lat']
    lat_ne = res['geometry']['viewport']['northeast']['lat']

    if lat_sw >= 36.94 and lat_ne <= 38.0 and lng_ne >= -122.53 and lng_sw <= -121.8:
        # get the times
        endTime = pd.datetime.replace(pd.datetime.now(), microsecond=0)
        startTime = pd.datetime.isoformat(endTime - pd.tseries.offsets.Hour(timeWindow_hours))
        endTime = pd.datetime.isoformat(endTime)

        return redirect(url_for('.results', lng_sw=lng_sw, lng_ne=lng_ne, lat_sw=lat_sw, lat_ne=lat_ne, startTime=startTime, endTime=endTime, city=full_add))
    else:
        this_lon, this_lat = sd.set_get_boundBox(area_str='bayarea')
        # for our loc, just set the average
        user_lon = np.mean(this_lon)
        user_lat = np.mean(this_lat)
        latlng_sw = [this_lat[1], this_lon[1]]
        latlng_ne = [this_lat[0], this_lon[0]]
        selected = "1"
        return render_template('out_of_bounds.html', examples=examples,\
            user_lat=user_lat, user_lon=user_lon,\
            latlng_sw=latlng_sw, latlng_ne=latlng_ne,\
            selected=selected)

@app.route("/results_predef",methods=['POST'])
def results_procPredef():
    event_id = request.form.get("event_id")

    # get the pre-defined time period
    endTime = [dct["endTime"] for dct in examples if dct["id"] == event_id][0]
    startTime = pd.datetime.isoformat(pd.to_datetime(endTime) - pd.tseries.offsets.Hour(timeWindow_hours))

    area_str = [dct["area_str"] for dct in examples if dct["id"] == event_id][0]

    # set the bounding box for the requested area
    this_lon, this_lat = sd.set_get_boundBox(area_str=area_str)

    city = [dct["city"] for dct in examples if dct["id"] == event_id][0]
    # get bounding box from this area_str
    return redirect(url_for('.results', lng_sw=this_lon[0], lng_ne=this_lon[1], lat_sw=this_lat[0], lat_ne=this_lat[1], startTime=startTime, endTime=endTime, city=city, selected=request.form.get("event_id")))

@app.route("/results",methods=['GET'])
def results():

    # get the selected event
    selected = request.args.get('selected')
    if selected == None:
        selected = "1"

    # get the city/neighborhood
    city = request.args.get('city')
    if city == None:
        city = "San Francisco, CA"
    elif city == "Embarcadero, San Francisco, CA, USA":
        city = "The Embarcadero, San Francisco"
    elif city == "South of Market, San Francisco, CA, USA":
        city = "SoMa, San Francisco"
    elif city == "Mission District, San Francisco, CA, USA":
        city = "Mission, San Francisco"
    elif city == "Nob Hill, San Francisco, CA, USA":
        city = city = city.replace(', CA','')
    elif city == "Dogpatch, San Francisco, CA, USA":
        city = city = city.replace(', CA','')
    elif city == "Golden Gate Park, San Francisco, CA, USA":
        city = city = city.replace(', CA','')
    city = city.replace(', USA','')

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

    # 0.003 makes bins about the size of AT&T park
    # bin_scaler = 0.006
    bin_scaler = 0.009
    nbins_lon = int(np.floor(float(np.diff(this_lon)) / bin_scaler))
    nbins_lat = int(np.floor(float(np.diff(this_lat)) / bin_scaler))
    binDiffThresh = 5
    if nbins_lon > binDiffThresh and nbins_lat < binDiffThresh:
        nbins_lon = int(np.floor(nbins_lon / 2))
    if nbins_lat > binDiffThresh and nbins_lon < binDiffThresh:
        nbins_lat = int(np.floor(nbins_lat / 2))
    if nbins_lon <= 3:
        nbins_lon = 1
    if nbins_lat <= 3:
        nbins_lat = 1
    print 'nbins_lon: %d' % nbins_lon
    print 'nbins_lat: %d' % nbins_lat
    n_top_hotspots = 5
    min_nclusters = 2
    max_nclusters = 5

    # eps = 0.1
    # eps = 0.15
    eps = 0.08
    # eps = 0.075 # good
    # eps = 0.05
    # eps = 0.025
    # min_samples = 30 * nhours

    onlyUnique = True

    if onlyUnique:
        diffthresh = 10 * nhours
        # diffthresh = 1 * nhours
        min_samples = 12 * nhours
    else:
        diffthresh = 10 * nhours
        min_samples = 20 * nhours
    # diffthresh = 15 * nhours
    # diffthresh = int(np.floor((nbins[0] * nbins[1] / 100) * 0.75))
    # diffthresh = int(np.floor(np.prod(nbins) / 100))
    # print 'diffthresh: %d' % diffthresh
    # open connection to database

    if 'port' in authsql:
        con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'],port=authsql['port'])
    else:    
        con=mdb.connect(host=authsql['host'],user=authsql['user'],passwd=authsql['word'],database=authsql['database'])

    # query the database
    activity_now = sd.selectFromSQL(con,time_now,this_lon,this_lat,tz)
    print 'Now: Selected %d entries' % (activity_now.shape[0])

    if activity_now.shape[0] > 0:
        offsetType = None
        foundHotspot = False
        for i,offset in enumerate(nowThenOffset_hours):
            # compare to the previous X hours
            startTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[0]) - pd.tseries.offsets.Hour(offset))
            endTime_then_UTC = pd.datetime.isoformat(pd.datetools.parse(time_now[1]) - pd.tseries.offsets.Hour(offset))
            time_then = [startTime_then_UTC, endTime_then_UTC]

            activity_then = sd.selectFromSQL(con,time_then,this_lon,this_lat,tz)
            print 'Then: Selected %d entries from %s' % (activity_then.shape[0],offsetTypes[i])

            if activity_then.shape[0] > 0:
                diffmore_lon, diffmore_lat = hap.findHotspots(\
                    activity_now=activity_now, activity_then=activity_then,\
                    nbins=[nbins_lon, nbins_lat], n_top_hotspots=n_top_hotspots,\
                    diffthresh=diffthresh, onlyUnique=onlyUnique)
                if len(diffmore_lon) > 0:
                    print 'found activity found compared to ' + offsetTypes[i]
                    foundHotspot = True
                    offsetType = offsetTypes[i]
                    break
            else:
                print 'no activity found compared to ' + offsetTypes[i]
                continue

        if foundHotspot:
            activity, n_clusters, keepClus, cluster_centers, message, success = hap.clusterActivity(\
                activity_now=activity_now, diffmore_lon=diffmore_lon, diffmore_lat=diffmore_lat,\
                nbins=[nbins_lon, nbins_lat],\
                min_nclusters=min_nclusters, max_nclusters=max_nclusters,\
                eps=eps, min_samples=min_samples)
        else:
            n_clusters = 0
            cluster_centers = []
            message = 'Sorry, no activity found during any baseline times!'
            success = False

        # activity, n_clusters, cluster_centers, message, success = hap.whatsHappening(\
        #     activity_now=activity_now, activity_then=activity_then,\
        #     nbins=[nbins_lon, nbins_lat],\
        #     min_nclusters=min_nclusters, max_nclusters=max_nclusters,\
        #     n_top_hotspots=n_top_hotspots,\
        #     diffthresh=diffthresh, eps=eps, min_samples=min_samples)
    # elif len(activity_now) > 0 and len(activity_then) == 0:
    #     n_clusters = 0
    #     cluster_centers = []
    #     message = 'Sorry, no activity found during the baseline time!'
    #     success = False
    # elif len(activity_now) == 0 and len(activity_then) > 0:
    #     n_clusters = 0
    #     cluster_centers = []
    #     message = 'Sorry, no activity found during this time!'
    #     success = False
    else:
        n_clusters = 0
        cluster_centers = []
        message = 'Sorry, no activity found in this region!'
        success = False

    print 'message: ' + message

    # close connection to database
    con.close()

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
    # for clusNum in range(n_clusters):
    for clusNum in range(len(keepClus)):
        print 'clusNum %d' % clusNum
        # print 'keepClus[clusNum]'
        print keepClus[clusNum]
        if keepClus[clusNum] == True:
            activity_thisclus = activity.loc[activity['clusterNum'] == clusNum]
            print 'keeping: len activity %d' % activity_thisclus.shape[0]
            if activity_thisclus.shape[0] > 0:
                tokens, freq_dist, clean_text = hap.cleanTextGetWordFrequency(activity_thisclus)
                word_freq.append(freq_dist)
            else:
                print 'keeping but len=0'
                pdb.set_trace()
                keepClus[clusNum] = False
        else:
            print 'not keeping'

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
    base_url = "https://twitter.com/search?f=realtime&q={}"
    dist_str = '5mi'
    twit_search_end = pd.to_datetime(request.args.get('endTime')).date().isoformat()
    twit_search_start = (pd.to_datetime(request.args.get('endTime')) - pd.tseries.offsets.Day(1)).date().isoformat()
    for i, clus in enumerate(cluster_centers):
        clus_centers.append(dict(lat=clus[1], long=clus[0], clusterid=int(clus[2])))
        this_array = []
        for word in word_freq[i].most_common(top_nWords):
            query = base_url.format( urllib.quote('%s near:"%s" within:%s since:%s until:%s' % (word[0],city,dist_str,twit_search_start,twit_search_end)) )
            this_array.append({'text': word[0], 'weight': word[1], 'link': query})
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
        offsetType = offsetType,\
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

@app.route('/slides')
def slides():
    # Renders slides_wide.html.
    return render_template('slides_wide.html')

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
timeWindow_hours = 3
# nowThenOffset_hours = 3
# nowThenOffset_hours = 24
nowThenOffset_hours = [timeWindow_hours, 24, 168]
offsetTypes = ['earlier today', 'yesterday', 'the same day last week']

# nowThenOffset_hours = [6]
# offsetTypes = ['earlier today']

# ["gray","orange","yellow","green","blue","purple"]
# clusterColor = ["D1D1E0","FF9933","FFFF66","00CC00","0066FF","CC0099"]
clusterColor = ["FF9933","FFFF66","00CC00","0066FF","CC0099"]
# FFFF66 # yellow
# CC0099 # purple
# E78AC3 # pink
# 8DA0CB # purplish

# ["gray","turquoise","orangy","purplish","pink","limey"]
# clusterColor = ["D1D1E0","66C2A5","FC8D62","8DA0CB","E78AC3","A6D854"]

#############
# set up some examples
#############

examples = [{"id": "1", "endTime": "2014-09-09T12:00:00", "name": "Tue Sep 9, 2014, 12 PM - Cupertino", "area_str": "apple_flint_center", "city": "Cupertino, CA"},
            {"id": "2", "endTime": "2014-09-09T13:00:00", "name": "Tue Sep 9, 2014, 1 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "3", "endTime": "2014-09-09T15:00:00", "name": "Tue Sep 9, 2014, 3 PM - Cupertino", "area_str": "apple_flint_center", "city": "Cupertino, CA"},
            {"id": "4", "endTime": "2014-09-09T21:00:00", "name": "Tue Sep 9, 2014, 9 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "5", "endTime": "2014-09-13T00:00:00", "name": "Sat Sep 13, 2014, 12 AM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "6", "endTime": "2014-09-13T12:00:00", "name": "Sat Sep 13, 2014, 12 PM - Embarcadero", "area_str": "embarc", "city": "The Embarcadero, San Francisco"},
            {"id": "7", "endTime": "2014-09-19T19:00:00", "name": "Fri Sep 19, 2014, 7 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "8", "endTime": "2014-09-19T21:00:00", "name": "Fri Sep 19, 2014, 9 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "9", "endTime": "2014-09-19T23:00:00", "name": "Fri Sep 19, 2014, 11 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "10", "endTime": "2014-09-20T12:00:00", "name": "Sat Sep 20, 2014, 12 PM - Mission", "area_str": "mission", "city": "Mission, San Francisco"},
            {"id": "11", "endTime": "2014-09-20T12:00:00", "name": "Sat Sep 20, 2014, 12 PM - Embarcadero", "area_str": "embarc", "city": "The Embarcadero, San Francisco"},
            {"id": "12", "endTime": "2014-09-27T14:00:00", "name": "Sat Sep 27, 2014, 2 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "13", "endTime": "2014-09-30T22:00:00", "name": "Tue Sep 30, 2014, 10 PM - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            {"id": "14", "endTime": "2014-10-04T15:30:00", "name": "Sat Oct 4, 2014, 3:30 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "15", "endTime": "2014-10-05T10:00:00", "name": "Sun Oct 5, 2014, 10 AM - Embarcadero", "area_str": "embarc", "city": "The Embarcadero, San Francisco"},
            {"id": "16", "endTime": "2014-10-19T12:00:00", "name": "Sun Oct 19, 2014, 12 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "17", "endTime": "2014-10-25T19:30:00", "name": "Sat Oct 25, 2014, 7:30 PM - SF", "area_str": "sf", "city": "San Francisco, CA"},
            {"id": "18", "endTime": "2014-10-26T18:30:00", "name": "Sun Oct 26, 2014, 6:30 PM - SF", "area_str": "sf", "city": "San Francisco, CA"}
            ]

            # {"id": "16", "endTime": "2014-09-10T00:15:00", "name": "Tue 9/9 Post Giants v Diamondbacks (7:15pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "17", "endTime": "2014-09-11T00:15:00", "name": "Wed 9/10 Post Giants v Diamondbacks (7:15pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "18", "endTime": "2014-09-11T17:45:00", "name": "Thr 9/11 Post Giants v Diamondbacks (12:25pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "19", "endTime": "2014-09-13T00:15:00", "name": "Fri 9/12 Post Giants v Dodgers (7:15pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "20", "endTime": "2014-09-13T23:05:00", "name": "Sat 9/13 Post Giants v Dodgers (6:05pm, L; no tweets?) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "21", "endTime": "2014-09-14T17:05:00", "name": "Sun 9/14 Post Giants v Dodgers (1:05pm, L) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "22", "endTime": "2014-09-26T00:15:00", "name": "Thr 9/25 Post Giants v Padres (7:15pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "23", "endTime": "2014-09-27T00:15:00", "name": "Fri 9/26 Post Giants v Padres (7:15pm, L) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "24", "endTime": "2014-09-27T17:05:00", "name": "Sat 9/27 Post Giants v Padres (1:05pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "25", "endTime": "2014-09-28T17:05:00", "name": "Sun 9/28 Post Giants v Padres (1:05pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "26", "endTime": "2014-10-06T19:07:00", "name": "Mon 10/6 Post Giants v Nationals (2:07pm, L) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"},
            # {"id": "27", "endTime": "2014-10-07T23:07:00", "name": "Tue 10/7 Post Giants v Nationals (6:07pm, W) - ATTPark", "area_str": "attpark", "city": "SoMa, San Francisco"}

            # {"id": "16", "endTime": "2014-09-10T00:15:00", "name": "Tue 9/9 Post Giants v Diamondbacks (7:15pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "17", "endTime": "2014-09-11T00:15:00", "name": "Wed 9/10 Post Giants v Diamondbacks (7:15pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "18", "endTime": "2014-09-11T17:45:00", "name": "Thr 9/11 Post Giants v Diamondbacks (12:25pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "19", "endTime": "2014-09-13T00:15:00", "name": "Fri 9/12 Post Giants v Dodgers (7:15pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "20", "endTime": "2014-09-13T23:05:00", "name": "Sat 9/13 Post Giants v Dodgers (6:05pm, L; no tweets?) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "21", "endTime": "2014-09-14T17:05:00", "name": "Sun 9/14 Post Giants v Dodgers (1:05pm, L) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "22", "endTime": "2014-09-26T00:15:00", "name": "Thr 9/25 Post Giants v Padres (7:15pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "23", "endTime": "2014-09-27T00:15:00", "name": "Fri 9/26 Post Giants v Padres (7:15pm, L) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "24", "endTime": "2014-09-27T17:05:00", "name": "Sat 9/27 Post Giants v Padres (1:05pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "25", "endTime": "2014-09-28T17:05:00", "name": "Sun 9/28 Post Giants v Padres (1:05pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "26", "endTime": "2014-10-06T19:07:00", "name": "Mon 10/6 Post Giants v Nationals (2:07pm, L) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"},
            # {"id": "27", "endTime": "2014-10-07T23:07:00", "name": "Tue 10/7 Post Giants v Nationals (6:07pm, W) - SoMa", "area_str": "soma", "city": "SoMa, San Francisco"}

            # {"id": "16", "endTime": "2014-09-10T00:15:00", "name": "Tue 9/9 Post Giants v Diamondbacks (7:15pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "17", "endTime": "2014-09-11T00:15:00", "name": "Wed 9/10 Post Giants v Diamondbacks (7:15pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "18", "endTime": "2014-09-11T17:45:00", "name": "Thr 9/11 Post Giants v Diamondbacks (12:25pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "19", "endTime": "2014-09-13T00:15:00", "name": "Fri 9/12 Post Giants v Dodgers (7:15pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "20", "endTime": "2014-09-13T23:05:00", "name": "Sat 9/13 Post Giants v Dodgers (6:05pm, L; no tweets?) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "21", "endTime": "2014-09-14T17:05:00", "name": "Sun 9/14 Post Giants v Dodgers (1:05pm, L) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "22", "endTime": "2014-09-26T00:15:00", "name": "Thr 9/25 Post Giants v Padres (7:15pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "23", "endTime": "2014-09-27T00:15:00", "name": "Fri 9/26 Post Giants v Padres (7:15pm, L) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "24", "endTime": "2014-09-27T17:05:00", "name": "Sat 9/27 Post Giants v Padres (1:05pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "25", "endTime": "2014-09-28T17:05:00", "name": "Sun 9/28 Post Giants v Padres (1:05pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "26", "endTime": "2014-10-06T19:07:00", "name": "Mon 10/6 Post Giants v Nationals (2:07pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"},
            # {"id": "27", "endTime": "2014-10-07T23:07:00", "name": "Tue 10/7 Post Giants v Nationals (6:07pm, W) - SF", "area_str": "sf", "city": "San Francisco, CA"}
