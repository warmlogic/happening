"""
plot latitude and longitudes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
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


latlong = open("data/latlong_combined.csv")

print 'Reading locations...'
# df = pd.read_csv(latlong,header=None,names=['latitude', 'longitude'])
df = pd.read_csv(latlong,header=None,names=['id','datestr','latitude', 'longitude','text'])
print 'Done.'
latlong.close()

# # calculate distance
# sf_center = [-122.4167,37.7833]
# # castro_muni = [-122.43533,37.76263]
# distance_to_user = compute_miles(sf_center[0], sf_center[1], df.latitude, df.longitude)

# pdb.set_trace()

n = 1e5
x = y = np.linspace(-5, 5, 100)

x = np.array(df.latitude)
y = np.array(df.longitude)

X, Y = np.meshgrid(x, y)
Z1 = mlab.bivariate_normal(X, Y, 2, 2, 0, 0)
Z2 = mlab.bivariate_normal(X, Y, 4, 1, 1, 1)
ZD = Z2 - Z1
x = X.ravel()
y = Y.ravel()
z = ZD.ravel()
gridsize=30
plt.subplot(111)

# if 'bins=None', then color of each hexagon corresponds directly to its count
# 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then 
# the result is a pure 2D histogram 

plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=1000)
plt.axis([x.min(), x.max(), y.min(), y.max()])

cb = plt.colorbar()
cb.set_label('mean value')
plt.show()


# # plot
# fig = plt.figure()
# # ax = plt.axis() # not sure if this works
# nbins = 5000
# H,xedges,yedges = np.histogram2d(np.array(df.latitude),np.array(df.longitude),bins=nbins)
# H = np.rot90(H)
# H = np.flipud(H)
# Hmasked = np.ma.masked_where(H==0,H) # mask pixels

# plt.pcolormesh(xedges,yedges,Hmasked)
# plt.title('Counts')
# plt.colorbar()
# # ax.get_xaxis().set_visible(False)
# # ax.get_yaxis().set_visible(False)
# plt.show()

# plt.savefig('latlong_plot.png')



if __name__ == '__main__':
    main()
