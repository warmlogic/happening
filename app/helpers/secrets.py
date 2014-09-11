from base64 import b64decode as _b
import socket
import os
from authent import googauth

def get_API_key():
  otherkey = googauth['otherkey']
  return _d(otherkey)

def generate_salted_API_key():
  """ This generates the key that is decoded in generate_API_key"""
  hostname = socket.gethostname()
  hostname = hostname*10
  # Insert real API key here, then run this function and save the result.
  API_KEY = googauth['apikey']
  key_len = len(API_KEY)
  salted_key = []
  for i in range(key_len):
    key = ord(API_KEY[i]) - ord(hostname[i])
    salted_key.append(str(key))
  return ",".join(salted_key)

def generate_API_key():
  # Salted Key
  salted_key = googauth['saltedkey']
  salted_key = eval(salted_key)
  # Subtract API 
  key_len = len(salted_key)
  hostname = socket.gethostname()
  hostname = hostname*10
  hostname = hostname[:key_len]
  API_KEY = []
  for i in range(key_len):
    key = ord(hostname[i]) + salted_key[i]
    API_KEY.append( chr(key % 256) )

  return "".join(API_KEY)

def _d(d):return _b('Q'+d+'n')
