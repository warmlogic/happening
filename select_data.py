'''
select data: specified by latitude and longitude bounding box and datetime strings

also plots density
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_miles(lat1, long1, lat2, long2):
    print 'computing distance from San Francisco\n'
    R_earth = 3963.1676 # miles
    return R_earth * distance_on_unit_sphere(lat1, long1, lat2, long2)


def distance_on_unit_sphere(lat1, long1, lat2, long2):
    'relative to a central point'

    # Convert latitude and longitude to 
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0
        
    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
        
    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
        
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

def selectSpace(df,this_lat=[-180,180],this_lon=[-90,90]):
    # select the data in space
    withinBoundingBox = (df.latitude >= this_lat[0]) & (df.latitude <= this_lat[1]) & (df.longitude >= this_lon[0]) & (df.longitude <= this_lon[1])
    print 'Space: Selecting %d entries (out of %d)' % (sum(withinBoundingBox),len(df))
    df = df[withinBoundingBox]
    return df

def selectTime(df,sinceDatetime='2007-01-01 00:00:00',untilDatetime=pd.datetime.now(),rmnull=True):
    if rmnull:
        hasDatestr = ~df.datestr.isnull()
        print 'Time: Removing %d null entries (out of %d)' % (sum(~hasDatestr),len(df))
        df = df[hasDatestr]

    # convert to datetime format
    df.datestr[:] = pd.to_datetime(df.datestr,format='%a %b %d %H:%M:%S +0000 %Y')

    # select the data in time
    withinDates = (df.datestr >= sinceDatetime) & (df.datestr <= untilDatetime)
    print 'Time: Selecting %d entries (out of %d)' % (sum(withinDates),len(df))
    df = df[withinDates]
    return df

def plot_hist(df,nbins=200):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    H,xedges,yedges = np.histogram2d(np.array(df.latitude),np.array(df.longitude),bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H) # mask pixels

    plt.pcolormesh(xedges,yedges,Hmasked)
    # plt.title('Density of tweets (%d bins)' % nbins)
    ax.set_title('Density of tweets (%d bins)' % nbins)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    cb = plt.colorbar()
    cb.set_label('Count')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    return plt

if __name__ == '__main__':
    main()
