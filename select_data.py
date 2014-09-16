'''
select data: specified by latitude and longitude bounding box and datetime strings

also plots density
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pdb

def compute_distance_from_point(lon1, lat1, lon2, lat2, unit='meters'):
    if unit == 'meters':
        R_earth = 6378137.0 # meters
    elif unit == 'kilometers':
        R_earth = 6378.137 # kilometers
    elif unit == 'feet':
        R_earth = 20925524.9 # feet
    elif unit == 'miles':
        R_earth = 3963.1676 # miles
    else:
        print 'unit "%s" not support' % unit
        return
    print 'computing distance from X in ' + unit
    return R_earth * distance_on_unit_sphere(lon1, lat1, lon2, lat2)    

def distance_on_unit_sphere(lon1, lat1, lon2, lat2):
    'relative to a central point'

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
        
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
        
    # theta = longitude
    theta1 = lon1*degrees_to_radians
    theta2 = lon2*degrees_to_radians
        
    # Compute spherical distance from spherical coordinates.
        
    # For two locations in spherical coordinates 
    # (1, theta, phi) and (1, theta, phi)
    # cosine( arc length ) = 
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length
    
    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) + 
           np.cos(phi1)*np.cos(phi2))
    arc = np.arccos( cos )

    # Remember to multiply arc by the radius of the earth 
    # in your favorite set of units to get length.
    return arc

def spherical_dist_matrix(pos1, pos2, unit='meters'):
    '''
    http://stackoverflow.com/questions/19413259/efficient-way-to-calculate-distance-matrix-given-latitude-and-longitude-data-in
    http://stackoverflow.com/questions/20654918/python-how-to-speed-up-calculation-of-distances-between-cities
    '''
    if unit == 'meters':
        R_earth = 6378137.0 # meters
    elif unit == 'kilometers':
        R_earth = 6378.137 # kilometers
    elif unit == 'feet':
        R_earth = 20925524.9 # feet
    elif unit == 'miles':
        R_earth = 3963.1676 # miles
    else:
        print 'unit "%s" not support' % unit
        return
    # # convert to radians
    # pos1 = pos1 * np.pi / 180
    # pos2 = pos2 * np.pi / 180
    # cos_lat1 = np.cos(pos1[..., 1])
    # cos_lat2 = np.cos(pos2[..., 1])
    # cos_lat_d = np.cos(pos1[..., 1] - pos2[..., 1])
    # cos_lon_d = np.cos(pos1[..., 0] - pos2[..., 0])
    # return R_earth * np.arccos(cos_lat_d - cos_lat1 * cos_lat2 * (1 - cos_lon_d))

    # based on Spherical Law of Cosines
    pos1 = pos1/180*np.pi
    pos2 = pos2/180*np.pi
    lg1=pos1[0] #data format, (longitude, latitude)
    la1=pos1[1]
    lg2=pos2[0]
    la2=pos2[1]
    return R_earth * np.arccos(np.sin(la1)*np.sin(la2)+np.cos(la1)*np.cos(la2)*np.cos(lg1-lg2))

def selectSpaceBB(df,this_lon=[-180,180],this_lat=[-90,90]):
    '''
    select the data in space using a lon/lat bounding box
    '''
    withinBoundingBox = (df.longitude >= this_lon[0]) & (df.longitude <= this_lon[1]) & (df.latitude >= this_lat[0]) & (df.latitude <= this_lat[1])
    print 'Space: Selecting %d entries (out of %d)' % (sum(withinBoundingBox),len(df))
    df = df[withinBoundingBox]
    return df

def selectActivityFromPoint(df,this_lon,this_lat,unit='meters',radius=100,radius_increment=50,radius_max=200,min_activity=10):
    # select the data in space
    nFound = 0
    while nFound <= min_activity:
        distance_to_user = compute_distance_from_point(this_lon, this_lat, df.longitude, df.latitude, unit)
        within_range = distance_to_user <= radius
        nFound = sum(within_range)
        if nFound < min_activity:
            radius += radius_increment
            print 'Only found %d tweets, increasing radius by %d to %d %s and searching again' % (nFound,radius_increment,radius,unit)
        if radius > radius_max:
            print 'Radius larger than %d %s, stopping' % (radius_max,unit)
            radius -= radius_increment
            break
    df['distance'] = distance_to_user
    # df.loc[:,'distance'] = distance_to_user
    found_activity = df[(df.distance <= radius)]
    print 'returning %d tweets from a radius of %d %s.' % (nFound,radius,unit)
    return found_activity

def selectTime(df,tz='UTC',sinceDatetime=None,untilDatetime=None,rmnull=False):
    if rmnull:
        hasDatetime = ~df.datetime.isnull()
        print 'Time: Removing %d null entries (out of %d)' % (sum(~hasDatetime),len(df))
        df = df[hasDatetime]

    if sinceDatetime is None or sinceDatetime is '':
        # sinceDatetime = '2007-01-01 00:00:00'
        sinceDatetime = '1970-01-01 00:00:00'
    sinceDatetime = pd.to_datetime(sinceDatetime,utc=False)
    sinceDatetime = sinceDatetime.tz_localize(tz)
    if untilDatetime is None or untilDatetime is '':
        untilDatetime = str(pd.datetime.now())
    untilDatetime = pd.to_datetime(untilDatetime,utc=False)
    untilDatetime = untilDatetime.tz_localize(tz)
    if tz != 'UTC':
        df.datetime = df.datetime.apply(lambda x: x.tz_convert(tz), convert_dtype=False)

    # select the data in time
    withinDates = (df.datetime >= sinceDatetime) & (df.datetime <= untilDatetime)
    print 'Time: Selecting %d entries (out of %d) from %s to %s' % (sum(withinDates),len(df),str(sinceDatetime),str(untilDatetime))
    df = df[withinDates]
    return df

def selectUser(df,rmnull=True):
    if rmnull:
        hasUser = ~df.user_id.isnull()
        print 'Time: Removing %d null entries (out of %d)' % (sum(~hasUser),len(df))
        df = df[hasUser]
    return df

def make_hist(df,nbins=200,show_plot=False,savefig=False,figname='latlong_hist_plot'):
    H,xedges,yedges = np.histogram2d(np.array(df.longitude),np.array(df.latitude),bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)

    if show_plot:
        Hmasked = np.ma.masked_where(H==0,H) # mask pixels

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.pcolormesh(xedges,yedges,Hmasked)
        # plt.title('Density of tweets (%d bins)' % nbins)
        ax.set_title('Density of tweets (%d bins)' % nbins)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)
        cb = plt.colorbar()
        cb.set_label('Count')
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        if savefig:
            figname = 'data/' + figname + '.png'
            print 'saving figure to ' + figname
            plt.savefig(figname, bbox_inches='tight')
    return H, xedges, yedges

def choose_n_sorted(arr, n, srt='max', min_val=None, return_order='descend'):
    if srt == 'max':
        indices = arr.ravel().argsort()[-n:]
    elif srt == 'min':
        indices = arr.ravel().argsort()[::-1][-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    indices = list(indices)
    values = np.array([arr[i] for i in indices])
    idx = np.array([(i) for i in indices])
    if return_order == 'ascend':
        values = values[::-1]
        idx = idx[::-1]
    if min_val is not None:
        keepThese = values >= min_val
        values = values[keepThese]
        idx = idx[keepThese]
    return values, idx

def set_get_boundBox(area_str='sf'):
    '''
    Set and return coordinates for the bounding box of interest.
    Default is 'sf'.
    '''

    boundBox = {}

    boundBox['bayarea_lon'] = [-122.53,-121.8]
    boundBox['bayarea_lat'] = [36.94,38.0]

    boundBox['sf_lon'] = [-122.5686,-122.375]
    boundBox['sf_lat'] = [37.6681,37.8258]

    boundBox['fishwharf_lon'] = [-122.4231,-122.4076]
    boundBox['fishwharf_lat'] = [37.8040,37.8116]

    boundBox['embarc_lon'] = [-122.4089,-122.3871]
    boundBox['embarc_lat'] = [37.7874,37.7998]

    boundBox['att48_lon'] = [-122.3977,-122.3802]
    boundBox['att48_lat'] = [37.7706,37.7840]

    boundBox['pier48_lon'] = [-122.3977,-122.3838]
    boundBox['pier48_lat'] = [37.7706,37.7765]

    boundBox['attpark_lon'] = [-122.3977,-122.3802]
    boundBox['attpark_lat'] = [37.7765,37.7840]

    boundBox['levisstadium_lon'] = [-122.9914,-122.9465]
    boundBox['levisstadium_lat'] = [37.3777,37.4173]

    boundBox['mission_lon'] = [-122.4286,-122.3979]
    boundBox['mission_lat'] = [37.7481,37.7693]

    boundBox['sf_concerts_lon'] = [-122.4258,-122.4000]
    boundBox['sf_concerts_lat'] = [37.7693,37.7926]

    boundBox['nobhill_lon'] = [-122.4322,-122.3976]
    boundBox['nobhill_lat'] = [37.7845,37.8042]

    boundBox['mtview_caltrain_lon'] = [-122.0832,-122.0743]
    boundBox['mtview_caltrain_lat'] = [37.3897,37.3953]

    boundBox['apple_flint_center_lon'] = [-122.0550,-122.0226]
    boundBox['apple_flint_center_lat'] = [37.3121,37.3347]

    lon_str = '%s_lon' % area_str
    lat_str = '%s_lat' % area_str

    this_lon = boundBox[lon_str]
    this_lat = boundBox[lat_str]

    return this_lon, this_lat

def fullprint(*args, **kwargs):
    from pprint import pprint
    import numpy
    opt = numpy.get_printoptions()
    numpy.set_printoptions(threshold='nan')
    pprint(*args, **kwargs)
    numpy.set_printoptions(**opt)


def clusterThose(activity_now,nbins,diffmore_lon,diffmore_lat):
    X = np.vstack((activity_now.longitude, activity_now.latitude)).T
    scaler = StandardScaler(copy=True)
    X_centered = scaler.fit(X).transform(X)

    # eps = 0.00075
    eps = 0.75
    min_samples = 50

    n_clusters_ = 0
    n_tries = 0
    while n_clusters_ == 0:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_centered)
        # db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        n_tries += 1

        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == 0:
            eps *= 2.0
        if n_tries > 5:
            min_samples /= min_samples
        elif n_tries > 10:
            break

    print 'Estimated number of clusters: %d (eps=%f, min_samples=%d)' % (n_clusters_,eps,min_samples)

    if n_clusters_ > 0:
        binscale = 0.001
        core_samples = db.core_sample_indices_
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # X = scaler.inverse_transform(X_centered)

        unique_labels = np.unique(labels)

        # go through the found clusters
        keepClus = []
        cluster_centers = []
        clusterNums = np.repeat(-1,activity_now.shape[0])
        # for k, col in zip(unique_labels, colors):
        for k in unique_labels:
            if k != -1:
                # if in a cluster, set a mask for this cluster
                class_member_mask = (labels == k)

                # get the lat and long for this cluster
                cluster_lon = X[class_member_mask,0]
                cluster_lat = X[class_member_mask,1]

                # default setting for keeping the cluster
                keepThisClus = False

                # keep clusters that contain a hist2d hotspot
                for i in range(len(diffmore_lon)):
                    if diffmore_lon[i] > (min(cluster_lon) - nbins*binscale) and diffmore_lon[i] < (max(cluster_lon) + nbins*binscale) and diffmore_lat[i] > (min(cluster_lat) - nbins*binscale) and diffmore_lat[i] < (max(cluster_lat) + nbins*binscale):
                        keepThisClus = True
                        break

                keepClus.append(keepThisClus)
                if keepThisClus:
                    # fill in the cluster lable vector
                    clusterNums[class_member_mask] = k

                    # set the mean latitude and longitude
                    mean_lon = np.mean(X[core_samples_mask & class_member_mask,0])
                    mean_lat = np.mean(X[core_samples_mask & class_member_mask,1])
                    cluster_centers.append([mean_lon,mean_lat,k])

            # else:
            #     keepClus.append(False)
            #     # Black used for noise.
            #     # col = 'k'
        activity_now['clusterNum'] = clusterNums
        n_clusters_real = sum(keepClus)
        return activity_now, n_clusters_real, cluster_centers
    else:
        return activity_now, 0, []


# if __name__ == '__main__':
#     import plot_data
