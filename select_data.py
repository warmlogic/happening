'''
select data: specified by latitude and longitude bounding box and datetime strings

also plots density
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        # df['distance'] = distance_to_user
        df.loc[:,'distance'] = distance_to_user
        found_activity = df[(df.distance <= radius)]
        nFound += found_activity.shape[0]
        if nFound < min_activity:
            radius += radius_increment
            print 'Only found %d tweets, increasing radius by %d to %d %s and searching again' % (nFound,radius_increment,radius,unit)
        if radius > radius_max:
            print 'Radius larger than %d %s, stopping' % (radius_max,unit)
            radius -= radius_increment
            break
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

def make_hist(df,nbins=200,show_plot=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H,xedges,yedges = np.histogram2d(np.array(df.longitude),np.array(df.latitude),bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)

    if show_plot:
        Hmasked = np.ma.masked_where(H==0,H) # mask pixels

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
    return plt, H, xedges, yedges

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

# def n_min(arr, n):
#     indices = arr.ravel().argsort()[::-1][-n:]
#     indices = (np.unravel_index(i, arr.shape) for i in indices)
#     values = [arr[i] for i in indices]
#     idx = [(i) for i in indices]
#     if order == 'ascend':
#         values.reverse()
#         idx.reverse()
#     return values, idx

# if __name__ == '__main__':
#     import plot_data
