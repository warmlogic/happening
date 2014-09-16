import matplotlib.pyplot as plt
import mpld3
import numpy as np



def render_webfigure(var_dict):

  userlon = float(var_dict['userlon'])
  userlat = float(var_dict['userlat'])
  data = var_dict['data']

  fig, ax = plt.subplots(figsize=(15,10))

  #return antlons, antlats, scss, cats, separations, geodesics, contour_lons, contour_lats
  antlons = data[0]
  antlats = data[1]
  scss = data[2]
  cats = data[3]
  contour_lons = data[6]
  contour_lats = data[7]

  #print userlon
  plt.plot(userlon, userlat, 'g*', ms=40)
  plt.scatter(antlons, antlats, c='r')
  for i in xrange(len(antlons)):
    label = '{} ({})'.format(scss[i], cats[i])
    plt.text(antlons[i]+.05, antlats[i], label)
    plt.plot(contour_lons[i], contour_lats[i])
    #print contour_lons[i], contour_lats[i]

  import json
  execfile("/Users/mwoods/Work/OldJobs/JobSearch/Pre-Insight/plotUSA.py")
  jdata = json.load(open("/Users/mwoods/Work/OldJobs/JobSearch/Pre-Insight/states.json"))
  i=0
  for state in jdata['geometries']:
    i+=1
    j=0
    # State only has one border, do not enter into 'for each border' loop.
    if len(state['coordinates'][0]) != 1:
      x,y = np.array(state['coordinates']).T
      plt.plot(x,y, 'b', lw=3)
      continue
    # There is a list with multiple borders (islands)
    for shape in state['coordinates']:
      x,y = np.array(shape[0]).T
      plt.plot(x,y, 'b', lw=3)
  plt.xlim(userlon-1, userlon+1)
  plt.ylim(userlat-1, userlat+1)
  plt.xlabel("Longitude")
  plt.ylabel("Latitude")
  #plt.gca().set_aspect('equal')


  #plt.hist(np.random.random(100))
  #plt.scatter(np.random.random(100), np.random.random(100))
  fig_html = mpld3.fig_to_html(fig)

  return mpld3.fig_to_html(fig)
