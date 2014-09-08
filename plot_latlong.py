"""
plot latitude and longitudes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.mlab as mlab
# import time
# import datetime
import pdb

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
    # ax = plt.axis() # not sure if this works
    H,xedges,yedges = np.histogram2d(np.array(df.latitude),np.array(df.longitude),bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    Hmasked = np.ma.masked_where(H==0,H) # mask pixels

    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.title('Counts')
    plt.colorbar()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    plt.show()

############
# Read the data
############

latlong = open("data/latlong_combined.csv")

print 'Reading locations...'
# df = pd.read_csv(latlong,header=None,names=['latitude', 'longitude'])
df = pd.read_csv(latlong,header=None,names=['id','datestr','latitude', 'longitude','text'])
print 'Done.'
latlong.close()

############
# Space
############

# choose only coordinates in our bounding box of interest

bayarea_lat = [-122.53,-121.8]
bayarea_lon = [36.94,38.0]

sf_lat = [-122.5686,-122.375]
sf_lon = [37.6681,37.8258]

nob_lat = [-122.4322,-122.3976]
nob_lon = [37.7845,37.8042]

# this_lat = bayarea_lat;
# this_lon = bayarea_long;

this_lat = sf_lat
this_lon = sf_lon

df = selectSpace(df,this_lat,this_lon)

############
# Time
############

# set the start and end datetimes
# sinceDatetime = '2014-09-05 09:00:00'
# untilDatetime = '2014-09-05 17:00:00'

# df = selectTime(df,sinceDatetime=sinceDatetime,untilDatetime=untilDatetime)

# # calculate distance
# sf_center = [-122.4167,37.7833]
# # castro_muni = [-122.43533,37.76263]
# distance_to_user = compute_miles(sf_center[0], sf_center[1], df.latitude, df.longitude)

############
# Plot
############

nbins = 50
plot_hist(df,nbins)

# plt.savefig('latlong_plot.png')


# if __name__ == '__main__':
#     main()


# n = 1e5
# x = y = np.linspace(-5, 5, 100)

# x = np.array(df.latitude)
# y = np.array(df.longitude)

# X, Y = np.meshgrid(x, y)
# Z1 = mlab.bivariate_normal(X, Y, 2, 2, 0, 0)
# Z2 = mlab.bivariate_normal(X, Y, 4, 1, 1, 1)
# ZD = Z2 - Z1
# x = X.ravel()
# y = Y.ravel()
# z = ZD.ravel()
# gridsize=30
# plt.subplot(111)

# # if 'bins=None', then color of each hexagon corresponds directly to its count
# # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# # the result is a pure 2D histogram 

# plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=1000)
# plt.axis([x.min(), x.max(), y.min(), y.max()])

# cb = plt.colorbar()
# cb.set_label('mean value')
# plt.show()
