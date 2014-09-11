import urllib2
import json
from app.helpers.secrets import get_API_key
from app.helpers.gmap_encoder import decode
from secrets import get_API_key
from gmap_encoder import decode
import itertools
import numpy as np

def geocode(search_term):
  API_KEY = get_API_key()
  base_url = "https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}"
  query = base_url.format( urllib2.quote(search_term), API_KEY)
  resp = urllib2.urlopen(query)
  data = json.load(resp)
  
  formatted_address = data['results'][0]['formatted_address'] 
  geom = data['results'][0]['geometry']
  lat, lon = geom['location']['lat'], geom['location']['lng']
  return lat, lon, formatted_address, data

def get_directions(origin, destination):
  API_KEY = get_API_key()

  base_url = "https://maps.googleapis.com/maps/api/directions/json?origin={}&destination={}&key={}"
  query = base_url.format(urllib2.quote(origin), urllib2.quote(destination), API_KEY)
  resp = urllib2.urlopen(query)
  data = json.load(resp)
  return data

def get_route_from_directions(directions):
  """ Take a JSON object and return a tuple of lon/lats """

  # Form a very dense array of points that describe the driving path.
  steps = directions['routes'][0]['legs'][0]['steps']
  step_polylines = [step['polyline']['points'] for step in steps]
  decoded_route_data = map(decode, step_polylines)
  route_data = itertools.chain.from_iterable(decoded_route_data)
  route_data = list(route_data)

  # Determine how many points to use.
  total_distance = directions['routes'][0]['legs'][0]['distance']['value'] # Meters
  total_duration = directions['routes'][0]['legs'][0]['duration']['value'] # Seconds
  if total_distance < 400000: # 400 km
    desired_num_nodes = 150
  elif total_distance < 4000000: # 4000 km
    desired_num_nodes = 200
  else:
    desired_num_nodes = 200
  #desired_num_nodes = 100

  distance1 = route_data[1:]
  distance2 = route_data[:-1]
  lon1, lat1 = zip(*distance1)
  lon2, lat2 = zip(*distance2)
  geodesics = haversine_dist(lon1, lat1, lon2, lat2) # Results in miles. Converted below.
  geodesics = np.r_[0, geodesics]
  meters_per_mile = 1609.34
  duration = np.cumsum(geodesics)*total_duration/(total_distance/meters_per_mile)

  def seconds_to_time(duration):
    if duration < 60:
      return '%.0f seconds' % duration
    elif duration < 60*60:
      return '%.0f minutes' % (duration/60.)
    elif duration < 60*60*24:
      return '%.1f hours' % (duration/60./60.)
    else:
      return '%.1f days' % (duration/60./60./24.)
  duration = map(seconds_to_time, duration)

  # Get the time between each of the "steps" in the driving path. Unfortunately,
  # len(route_data) = \Sum_N( n_i ) where n_i is the number of nodes decoded from
  # each of the N steps. I have the duration of each step, so let's extrapolate.
  # Duration times (in seconds)
  #step_durations = [step['duration']['value'] for step in steps]
  #step_distances = [step['distance']['value'] for step in steps]
  #step_lengths   = map(len, decoded_route_data)
  
  if desired_num_nodes > len(route_data):
    return route_data, duration

  resampled_route_data = route_data[::(len(route_data) // desired_num_nodes)]
  resampled_duration = duration[::(len(route_data) // desired_num_nodes)]
  print '--> Down sampled from', len(route_data), 'to', len(resampled_route_data), 'nodes'
  print '--> Down sampled from', len(duration), 'to', len(resampled_duration), 'durations'
  #print zip(route_data, duration)
  return resampled_route_data, resampled_duration


def resample_route_data(data):
  lats, lons = np.array(data).T
  lat1, lat2 = lats[:-1], lats[1:]
  lon1, lon2 = lons[:-1], lons[1:]
  separations = haversine_dist(lon1, lat1, lon2, lat2)
  
  return haversine_dist(lon1, lat1, lon2, lat2)

def haversine_dist(lon1, lat1, lon2, lat2):
  """
  Calculate the great circle distance between two points
  on the earth (specified in decimal degrees)
  """
  # convert decimal degrees to radians
  lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
  c = 2 * np.arcsin(np.sqrt(a))

  miles = 3956.27 * c
  return miles
  # 6367 km is the radius of the Earth
  #km = 6367 * c
  #return km 

def leg_to_js(route, settings = {}):
  """ I need to take input like:
      (21.291982, -157.821856),
      (-18.142599, 178.431),
      (23.982, -137.21856)
  and turn that into:
      var path1 = [new google.maps.LatLng(21.291982, -157.821856),
      new google.maps.LatLng(-18.142599, 178.431),
      new google.maps.LatLng(23.982, -137.21856)]

      var line1 = new google.maps.Polyline({
          path: path1,
          strokeColor: '#ff0000',
          strokeOpacity: 1.0,
          strokeWeight: 2
      });
  """

  make_LatLng_str = lambda x: 'new google.maps.LatLng'+str(x[::-1])
  path_strings = map(make_LatLng_str, route)
  path_var = ', '.join(path_strings)
  path_var = 'var path = [ {0} ]; \n'.format(path_var)
  color = settings.get('strokeColor', '#ff0000')
  opacity = settings.get('stokeOpacity', 1.0)
  weight = settings.get('stokeWeight', 10)
  if color == '#0C090A':
    weight = 2
    opacity = .7
  line_var = """var line = new google.maps.Polyline({{
  path: path,
  strokeColor: '{0}',
  strokeOpacity: {1},
  strokeWeight: {2}
}});
//// Add a new marker at the new plotted point on the polyline.
//var infoWindow = new google.maps.InfoWindow();
//for (i = 0; i < path.length; i++) {{ 
//    var marker = new google.maps.Marker({{
//        position: path[i],
//        map: map,
//        title: String(path[i])
//    }});
//    google.maps.event.addListener(marker, 'click', (function(marker) {{
//      return function() {{
//        infoWindow.setContent(marker.getTitle());
//        infoWindow.open(map, marker);
//      }}
//    }})(marker));
//}};
line.setMap(map);
  """.format(color, opacity, weight)

  return path_var + line_var


#def contour_to_js(contour, settings = {}):
#  """ I need to take input like:
#      (21.291982, -157.821856),
#      (-18.142599, 178.431),
#      (23.982, -137.21856)
#  and turn that into:
#      var path1 = [new google.maps.LatLng(21.291982, -157.821856),
#      new google.maps.LatLng(-18.142599, 178.431),
#      new google.maps.LatLng(23.982, -137.21856)]
#
#      var line1 = new google.maps.Polyline({
#          path: path1,
#          strokeColor: '#ff0000',
#          strokeOpacity: 1.0,
#          strokeWeight: 2
#      });
#  """
#
#  make_LatLng_str = lambda x: 'new google.maps.LatLng'+str(x[::-1])
#  path_strings = map(make_LatLng_str, contour)
#  path_var = ', '.join(path_strings)
#  path_var = 'var path = [ {0} ]; \n'.format(path_var)
#  color = settings.get('strokeColor', '#ff0000')
#  opacity = settings.get('stokeOpacity', 1.0)
#  weight = settings.get('stokeWeight', 10)
#  line_var = """var line = new google.maps.Polygon({{
#  paths: path,
#  strokeColor: '{0}',
#  strokeOpacity: {1},
#  strokeWeight: {2}
#}});
#line.setMap(map);
#  """.format(color, opacity, weight)
#
#  return path_var + line_var

def contour_to_js(leg):
  scs = leg['scs']
  color = leg['color']

  list_of_LatLngs = []
  if not leg['contour']:
    return ''
  for i in xrange(len(leg['contour'])):
    latlon = leg['contour'][i]
    #print scs, latlon
    #LatLng_str = 'new google.maps.LatLng(%, %f)'+(str(leg['contour'][i][::-1]))
    list_of_LatLngs.append('new google.maps.LatLng(%f, %f)'%(latlon[1], latlon[0]))
  LatLngs = ','.join(list_of_LatLngs)
  js = """
  var paths = [{0}];
  var shape = new google.maps.Polygon({{
    paths: paths,
    strokeColor: '{3}',
    strokeOpacity: 0.7,
    strokeWeight: 2,
    fillColor: '{3}',
    fillOpacity: 0.10,
    zIndex: -2
  }});
  shape.setMap(map);
  var marker = new google.maps.Marker({{
      position: paths[0],
      map: map,
      title: '{1}'
  }});
  var infoWindow = new google.maps.InfoWindow();
  google.maps.event.addListener(marker, 'click', (function(marker) {{
    return function() {{
      infoWindow.setContent('{2}');
      infoWindow.open(map, marker);
    }}
  }})(marker));
  google.maps.event.addListener(shape, 'click', (function(shape) {{
    return function() {{
      infoWindow.setContent('{2}');
      infoWindow.open(map, shape);
    }}
  }})(marker));
  
  """.format( LatLngs, scs, scs, color)
  return js
  #""".format( LatLngs, scs, get_wiki_table(scs))

def render_contours_and_legs(groups):
  """ The argument of this method is a dictionary that looks like:
  #'scs':    Short call sign e.g. KITS
  #'nodes':  A list of the LonLats
  #'empty':  A boolean to indicate no radio station available.
  """

  try:
    wiki_file = open("wiki.json")
  except:
    print "Problem opening wiki.json. Not surprised."
  wiki_dict = json.load( wiki_file )
  #print wiki_dict.keys()
  full_js_string = ''
  for group in groups:
    full_js_string += leg_and_contour_to_js(group, wiki_dict)
  return full_js_string

#def leg_and_contour_to_js(leg, wiki_dict):
#  scs = leg['scs']
#  color = leg['color']
#  freq = leg['freq']
#  duration = leg['dur']
#  wiki_data = wiki_dict.get(scs, scs)
#  list_of_LatLngs = []
#  if not leg['contour']:
#    contour_js = ""
#  else:
#    for i in xrange(len(leg['contour'])):
#      latlon = leg['contour'][i]
#      print scs, latlon
#      #LatLng_str = 'new google.maps.LatLng(%, %f)'+(str(leg['contour'][i][::-1]))
#      list_of_LatLngs.append('new google.maps.LatLng(%f, %f)'%(latlon[1], latlon[0]))
#    LatLngs = ','.join(list_of_LatLngs)
#    contour_js = """
#    var paths = [{0}];
#    var shape = new google.maps.Polygon({{
#      paths: paths,
#      strokeColor: '{1}',
#      strokeOpacity: 0.7,
#      strokeWeight: 2,
#      fillColor: '{1}',
#      fillOpacity: 0.10,
#      zIndex: -2
#    }});
#    shape.setMap(map);
#    //var marker = new google.maps.Marker({{
#    //    position: paths[0],
#    //    map: map,
#    //    title: '{2}'
#    //}});
#    var infoWindow = new google.maps.InfoWindow();
#    //google.maps.event.addListener(marker, 'click', (function(marker) {{
#    //  return function() {{
#    //    infoWindow.setContent('{3}');
#    //    infoWindow.open(map, marker);
#    //  }}
#    //}})(marker));
#    google.maps.event.addListener(shape, 'click', (function(shape) {{
#      return function() {{
#        infoWindow.setContent('{3}');
#        infoWindow.open(map, shape);
#      }}
#    //}})(marker));
#    }})(shape));
#    
#    """.format( LatLngs, color, scs, wiki_data)
#
#  # Ok, now here's the polyline.
#  opacity = 1
#  weight = 2
#  if color == '#0C090A':
#    weight = 2
#    opacity = .7
#  list_of_LatLngs = []
#  for i in xrange(len(leg['nodes'])):
#    latlon = leg['nodes'][i]
#    #print scs, latlon
#    #LatLng_str = 'new google.maps.LatLng(%, %f)'+(str(leg['contour'][i][::-1]))
#    list_of_LatLngs.append('new google.maps.LatLng(%f, %f)'%(latlon[1], latlon[0]))
#  LatLngs = ','.join(list_of_LatLngs)
#  leg_js = """
#    var polypath = [{0}];
#    var line = new google.maps.Polyline({{
#      path: polypath,
#      strokeColor: '{1}',
#      strokeOpacity: {2},
#      strokeWeight: {3},
#    }});
#    line.setMap(map);
#    var infoWindow = new google.maps.InfoWindow();
#    google.maps.event.addListener(polyline, 'click', (function(polyline) {{
#      return function() {{
#        infoWindow.setContent('{4}');
#        infoWindow.open(map, polyline);
#      }}
#    }})(polyline));
#  """.format( LatLngs, color, opacity, weight, wiki_data)
#
#  #make_LatLng_str = lambda x: 'new google.maps.LatLng'+str(x[::-1])
#  #path_strings = map(make_LatLng_str, route)
#  #path_var = ', '.join(path_strings)
#  #path_var = 'var path = [ {0} ]; \n'.format(path_var)
#  #color = settings.get('strokeColor', '#ff0000')
#  #opacity = settings.get('stokeOpacity', 1.0)
#  #weight = settings.get('stokeWeight', 10)
#  #if color == '#0C090A':
#    #weight = 2
#    #opacity = .7
#  #line_var = """var line = new google.maps.Polyline({{
#  #path: path,
#  #strokeColor: '{0}',
#  #strokeOpacity: {1},
#  #strokeWeight: {2}
##}});
#  #""".format( LatLngs, scs, get_wiki_table(scs))
#
#  return contour_js + leg_js

def leg_and_contour_to_js(leg, wiki_dict):
  scs = leg['scs']
  color = leg['color']
  freq = leg['freq']
  duration = leg['dur']
  wiki_data = wiki_dict.get(scs, scs)
  list_of_LatLngs = []
  if not leg['contour']:
    contour_js = ""
  else:
    for i in xrange(len(leg['contour'])):
      latlon = leg['contour'][i]
      #print scs, latlon
      #LatLng_str = 'new google.maps.LatLng(%, %f)'+(str(leg['contour'][i][::-1]))
      list_of_LatLngs.append('new google.maps.LatLng(%f, %f)'%(latlon[1], latlon[0]))
    LatLngs = ','.join(list_of_LatLngs)
    contour_js = """
    //var infoWindow = new google.maps.InfoWindow();

    var paths = [{0}];
    var shape = new google.maps.Polygon({{
      paths: paths,
      strokeColor: '{1}',
      strokeOpacity: 0.7,
      strokeWeight: 2,
      fillColor: '{1}',
      fillOpacity: 0.10,
      zIndex: -2
    }});
    shape.setMap(map);
  //  var marker = new google.maps.Marker({{
  //      position: paths[0],
  //      map: map,
  //      title: String(paths[0])
  //  }});
    //google.maps.event.addListener(marker, 'click', (function(marker) {{
    //  return function() {{
    //    infoWindow.setContent('{3}');
    //    infoWindow.setPosition(marker.latLng);
    //    infoWindow.open(map);
    //  }}
    //}})(marker));
    //google.maps.event.addListener(shape, 'click', (function(shape) {{
    //  return function() {{
    //    infoWindow.setContent('{3}');
    //    infoWindow.open(map, shape);
    //  }}
    //}})(polyline));
    
    """.format( LatLngs, color, scs, wiki_data)

  # Ok, now here's the polyline.
  opacity = 1
  weight = 6
  if color == '#0C090A':
    weight = 2
    opacity = .7
  list_of_LatLngs = []
  for i in xrange(len(leg['nodes'])):
    latlon = leg['nodes'][i]
    #print scs, latlon
    #LatLng_str = 'new google.maps.LatLng(%, %f)'+(str(leg['contour'][i][::-1]))
    list_of_LatLngs.append('new google.maps.LatLng(%f, %f)'%(latlon[1], latlon[0]))
  LatLngs = ','.join(list_of_LatLngs)
  leg_js = """
    var polypath = [{0}];
    var middle_point = polypath[Math.floor(polypath.length/2)];
    var polyline = new google.maps.Polyline({{
      path: polypath,
      strokeColor: '{1}',
      strokeOpacity: {2},
      strokeWeight: {3},
      pos: middle_point
    }});
    polyline.setMap(map);
    google.maps.event.addListener(polyline, 'click', (function(polyline) {{
      return function() {{
        infoWindow.setContent('{4}');
        //infoWindow.open(map, polyline);
        infoWindow.setPosition(this.pos);
        infoWindow.open(map);
      }}
    }})(polyline));
    // Click anywhere on the map to close the info window
    google.maps.event.addListener(map, "click", function () {{
      infoWindow.close();
    }});
  """.format( LatLngs, color, opacity, weight, wiki_data)

  #make_LatLng_str = lambda x: 'new google.maps.LatLng'+str(x[::-1])
  #path_strings = map(make_LatLng_str, route)
  #path_var = ', '.join(path_strings)
  #path_var = 'var path = [ {0} ]; \n'.format(path_var)
  #color = settings.get('strokeColor', '#ff0000')
  #opacity = settings.get('stokeOpacity', 1.0)
  #weight = settings.get('stokeWeight', 10)
  #if color == '#0C090A':
    #weight = 2
    #opacity = .7
  #line_var = """var line = new google.maps.Polyline({{
  #path: path,
  #strokeColor: '{0}',
  #strokeOpacity: {1},
  #strokeWeight: {2}
#}});
  #""".format( LatLngs, scs, get_wiki_table(scs))

  return contour_js + leg_js

def get_bounding_box(directions):
  bounding_NE = directions['routes'][0]['bounds']['northeast']
  bounding_SW = directions['routes'][0]['bounds']['southwest']
  NE_lat, NE_lon = bounding_NE['lat'], bounding_NE['lng']
  SW_lat, SW_lon = bounding_SW['lat'], bounding_SW['lng']
  bbox = 'var bbox = new google.maps.LatLngBounds( ' + \
      'new google.maps.LatLng( {0}, {1} ), ' + \
      'new google.maps.LatLng( {2}, {3} )); '
  bbox = bbox.format(SW_lat, SW_lon, NE_lat, NE_lon)
  return bbox


def consolidate_tunes(route, route_tunes, durations):
  num_nodes = len(route)
  legs = []
  current_leg = None
  first_node = route[0]
  first_tune = route_tunes[0]
  first_duration = durations[0]
  if first_tune == None:
    current_leg = {'scs':'NA', 'nodes':[], 'empty':True, 'contour':None, 'freq':None, 'dur':'to start'}
  else:
    contour = zip(*first_tune[6:8])
    freq = first_tune[8]
    current_leg = {'scs':first_tune[2], 'nodes':[], 'empty':False, 'contour':contour, 'freq':freq, 'dur':'to start'}

  # Loop over all each node. Add to group.
  for inode in xrange(0,num_nodes):
    #print durations[inode]
    #print '#>> ', inode, 'LatLng', route[inode],
    #if route_tunes[inode]:
      #print route_tunes[inode][:4]
    #else:
      #print route_tunes[inode]
    # Remember, some nodes don't have radio reception (set to None):
    if not route_tunes[inode]:
      # The current leg is a no-reception leg:
      if current_leg['empty']:
        current_leg['nodes'].append(route[inode])
      # The current leg is has reception. We moved to a no-reception region.
      else:
        legs.append(current_leg)
        current_leg = {'scs':'NA', 'nodes':[], 'empty':True, 'contour':None, 'freq':None}
        current_leg['nodes'].append(route[inode])
        current_leg['dur'] = durations[inode]
    # But if they do have radio reception:
    else:
      # This has the same station as the previous node.
      if current_leg['scs'] == route_tunes[inode][2]:
        current_leg['nodes'].append(route[inode])
      # This has a new station name.
      else:
        legs.append(current_leg)
        contour = zip(*route_tunes[inode][6:8])
        freq = route_tunes[inode][8]
        current_leg = {'scs':route_tunes[inode][2], 'nodes':[], 'empty':False, 'contour':contour, 'freq':freq}
        current_leg['nodes'].append(route[inode])
        current_leg['dur'] = durations[inode]
    # Let's close the gaps between legs. Average the position of this node
    # and the previous node.
    #if inode > 0:
      #this_node = current_leg['nodes'][-1]
      #prev_node = legs[-1]['nodes'][-1]
      #print '##', inode, ':', this_node, prev_node, this_node == prev_node
  legs.append(current_leg)

  # Add on points to each leg so that they touch.
  for ileg in xrange(len(legs)-1):
    leg = legs[ileg]
    next_leg = legs[ileg+1]
    leg_lastnode = leg['nodes'][-1]
    next_leg_firstnode = next_leg['nodes'][0]
    crossing_point = (leg_lastnode[0]+next_leg_firstnode[0])/2., (leg_lastnode[1]+next_leg_firstnode[1])/2.
    leg['nodes'].append( crossing_point )
    next_leg['nodes'].insert(0, crossing_point)

  # This is just to print stuff out!
  for i, leg in enumerate(legs):
    print "--> Leg %d: %s nodes and has call sign %s" % (i, len(leg['nodes']), leg['scs'])
  return legs

def assign_colors(grouped_nodes):
  """ The argument of this method is a dictionary that looks like:
  'scs':    Short call sign e.g. KITS
  'nodes':  A list of the LonLats
  'empty':  A boolean to indicate no radio station available.
  """

  colors = ['#00AB6F', '#0C5AA6', '#FF9700', '#FF5300']
  for group in grouped_nodes:
    group['color'] = '#0C090A'
    if group['scs'] != 'NA':
      group['color'] = colors[0]
      colors = colors[1:] + [colors[0]]
  return

def render_legs(grouped_nodes):
  """ The argument of this method is a dictionary that looks like:
  'scs':    Short call sign e.g. KITS
  'nodes':  A list of the LonLats
  'empty':  A boolean to indicate no radio station available.
  'color':  An html color
  """
  full_js_string = ''
  #colors = ['#ff0000', '#09CA32', '#00ff00', '#03C11D']
  #colors = ['#2E16B1',  '#640CAB',  '#FFF500',  '#FFCB00']
  # http://colorschemedesigner.com/#
  #colors = ['#00AB6F', '#0C5AA6', '#FF9700', '#FF5300']
  #colors = ['#444444']

  for group in grouped_nodes:
    leg = group['nodes']
    color = group['color']
    full_js_string += leg_to_js(leg, {'strokeColor':color})
  return full_js_string

def render_contours(grouped_nodes):
  """ The argument of this method is a dictionary that looks like:
  #'scs':    Short call sign e.g. KITS
  #'nodes':  A list of the LonLats
  #'empty':  A boolean to indicate no radio station available.
  """
  full_js_string = ''
  #colors = ['#ff0000', '#00ffff', '#00ff00', '#0f0f00']
  #colors = ['#00AB6F', '#0C5AA6', '#FF9700', '#FF5300']
  for group in grouped_nodes:
    full_js_string += contour_to_js(group)
  return full_js_string

