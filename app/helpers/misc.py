import time

class Timer(object):
  def __init__(self):
    self.beginning = time.time()
    self.start = self.beginning
  def __call__(self, arg=None):
    # Provid tick/tock functionality
    now = time.time()
    if arg:
      print "%s took %.3f seconds. (%.1f runtime)" % (arg, now-self.start, now-self.beginning)
    else:
      print "At %.3f seconds. (%.1f runtime)" % (arg, now-self.start, now-self.beginning)
    self.start = now
