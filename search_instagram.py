from instagram.client import InstagramAPI

from authent import instaauth

# for when authentication is not needed
api = InstagramAPI(client_id=instaauth['id'], client_secret=instaauth['secret'])
popular_media = api.media_popular(count=20)
for media in popular_media:
    print media.images['standard_resolution'].url


# for when authentication is needed
access_token = instaauth['accesstoken']
api = InstagramAPI(access_token=access_token)
recent_media, next_ = api.user_recent_media(user_id="3318", count=10)
for media in recent_media:
   print media.caption.text


# http://instagram.com/developer/endpoints/media/

api.media_search(q, count, lat, lng, min_timestamp, max_timestamp)

# https://api.instagram.com/v1/media/search?lat=48.858844&lng=2.294351&access_token=access_token

