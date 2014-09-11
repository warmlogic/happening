import MySQLdb
import sys
import numpy as np
from matplotlib.path import Path


# Returns MySQL database connection
def con_db(host, port, user, passwd, db):
    try:
        con = MySQLdb.connect(host=host, port=port, user=user, passwd=passwd, db=db)

    except:
        print "Could not connect to database."
        #print "Error %d: %s" % (e.args[0], e.args[1])
        sys.exit(1)

    return con

# Make a query based on a given lon/lat

def query_db(cur, lon, lat, genre, subgenre):
  data_array = []

  # Query database
  #cur = con.cursor()
  query = get_haversine_query(lon, lat, genre=genre, subgenre=subgenre)
  #print query
  cur.execute(query)

  route_results = []

  raw_results = cur.fetchall()

  if len(raw_results) == 0:
    return None
  results = zip(*raw_results)
  antlons = np.array(results[2], dtype=float)
  antlats = np.array(results[3], dtype=float)
  scss = np.array(results[4])
  cats = np.array(results[5])
  separations = np.array(results[6], dtype=float)
  geodesics = np.array(results[7], dtype=float)
  contour_lons = [np.fromstring(i, sep=',') for i in results[8]]
  contour_lats = [np.fromstring(i, sep=',') for i in results[9]]
  frequencies = np.array(results[10], dtype=float)

  #cur.close()
  #con.close()
  return antlons, antlats, scss, cats, separations, geodesics, contour_lons, contour_lats, frequencies

def get_haversine_query(lon, lat, genre=None, subgenre=None):
  """
  r is in units of miles. Maybe not needed?
  """

  base_query = """
  SELECT {0} , {1}, b.antlon, b.antlat, b.scs, map.cat,
  2 * ASIN( 
  SQRT( 
      POW( SIN(   (b.antlat - {1})/360*2*PI()/2  )  , 2)
      + COS({1}/360*2*PI()) 
      * COS(b.antlat/360*2*PI()) 
      * POW(SIN( (b.antlon - {0})/360*2*PI()/2), 2)
    )
  ) * 3956.27 AS geod,
  SQRT(POW(({1} - b.antlat)*75.8, 2) + POW(({0} - b.antlon)*60,2)) AS separation,
  b.lons, b.lats
  FROM contours b
  JOIN contour_cat_map map
  ON b.id = map.contour_id
  WHERE 
      {1} > b.minlat
  AND {1} < b.maxlat
  AND {0} > b.minlon
  AND {0} < b.maxlon
  """
  # Try a variation without the great circle
  base_query = """
  SELECT {0} , {1}, b.antlon, b.antlat, b.scs, map.cat, 
  1 AS geod,
  2 AS separation,
  b.lons, b.lats, map.frequency
  FROM contours b
  JOIN contour_cat_map map
  ON b.id = map.contour_id
  AND   
      {1} > b.minlat
  AND {1} < b.maxlat
  AND {0} > b.minlon
  AND {0} < b.maxlon
  """
  query = base_query.format(lon, lat)
  # If there's a genre query, put that down as well.
  where_statement = None
  if genre and not subgenre:
    where_statement = ' AND cat = "'+genre + '" '
  elif genre and subgenre:
    where_statement = ' AND (cat = "{0}" OR cat = "{1}") '.format(genre, subgenre)
  if genre:
    query += where_statement

  # Order by the size of the contour to try to minimize short legs.
  if genre and not subgenre:
    query += 'ORDER BY b.size DESC'
  elif genre and subgenre:
    query += "ORDER BY CASE WHEN cat = '{0}' then 1 else 2 end, b.size DESC".format(genre)
  else:
    query += 'ORDER BY b.size DESC'

  #print query
  return query

def find_radio_stations(con, route, var_dict):
  """ This is the meatiest and most heavy lifting-est method in this project.
  This will loop over each point in the route (node) and determine the radio
  stations (if any) that it can receive.
  """
  cur = con.cursor()
  antennas_for_each_node = []
  i = -1
  for node in route:
    #print "Considering node:", i, 'of', len(route)
    i+= 1
    result = query_db(
        cur, node[0], node[1], var_dict['genre'], var_dict['subgenre'])
    # No radio towers exist near this node (based on the rectangular contour
    # approximation:
    if not result: 
      antennas_for_each_node.append(None)
      #print 'XX', i, result
      continue
    #print '>>', i, str(result[:3]).replace('\n', '')

    # There is at least one station whos rectangular coverage includes the node.
    found_in_contour=False
    antlons, antlats, scss, cats, separations, geodesics, contour_lons, contour_lats, frequencies = result
    antenna_dict = {}
    for antenna_num in xrange(len(contour_lons)):
      #print 'antenna_num', antenna_num
      lons = contour_lons[antenna_num]
      lats = contour_lats[antenna_num]
      path = Path(zip(lons, lats))
      #if i > 53 and i < 62:
        #print '\t', antenna_num, 'of', len(contour_lons), 'antennas.',
        #print '\tContour LonLat:', lons[0], lats[0], 'Node LonLat:', node[0], node[1]
      if path.contains_point(node) and scss[antenna_num] != 'NA':
        #print '\tQQQ'
        antennas_for_each_node.append(zip(*result)[antenna_num])
        found_in_contour = True
        #print '\tBreaking!'
        break
    if not found_in_contour:
      #print '\tDid not find found_in_contour'
      antennas_for_each_node.append(None)
    #print str(antennas_for_each_node[-1])[0]
  return antennas_for_each_node

#def in
