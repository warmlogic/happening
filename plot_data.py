'''
check out the data
'''

import pandas as pd
import select_data

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

df = select_data.selectSpace(df,this_lat,this_lon)

############
# Time
############

# # # set the start and end datetimes
# sinceDatetime = '2014-09-05 09:00:00'
# untilDatetime = '2014-09-05 17:00:00'

# df = select_data.selectTime(df,sinceDatetime=sinceDatetime,untilDatetime=untilDatetime)

############
# Distance
############

# # calculate distance
# sf_center = [-122.4167,37.7833]
# # castro_muni = [-122.43533,37.76263]
# distance_to_user = select_data.compute_miles(sf_center[0], sf_center[1], df.latitude, df.longitude)

############
# Plot
############

# %matplotlib inline

nbins = 50
plt = select_data.plot_hist(df,nbins)

savefig = True
if savefig:
    figname = 'latlong_plot.png'
    print 'saving figure to ' + figname
    plt.savefig(figname)
plt.show()



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
