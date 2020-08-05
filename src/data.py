from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from PIL import Image
from io import open
import numpy as np
from util import *

#######################################################
### Data Object
#
# Extracts the data from the files, cleans the data,
# extracts metadata, and prepares batches
#######################################################

class Data(object):
  def __init__(self, files_list, data_file):
    self.files_list = files_list
    self.batch_size = batch_size
    self.data = np.load(data_file)
    self.metadata = self.prepare_data()

    self.batch_start_idx = 0 # Index to mark place in dataset

  def prepare_data(self):
    """
    Extracts the metadata for the video training data

    Params:
      None
    Returns: 
      train_data: List of metadata for training, each element of the form 
        (index, video_id, frame_id)
    """
    train_data = []
    idx = 0
    with open(self.files_list) as f:
      for line in f.readlines():
        path = line.strip()
        videoid, frameid = Data.get_videoid_frameid(path)
        train_data.append((idx, videoid, frameid)) # IMPORTANT FOR LATER CODE
        idx += 1
    np.random.shuffle(train_data) # Shuffling data for ease when batching data
    print u'Training on %d image files...' % len(train_data)
    return train_data

  def get_next_batch(self):
    """
    Provide next batch of data, starting at batch_start_idx

    Params:
      None
    Returns: 
      batch: List of image frames
      batch_annot: List of (videoid, frameid) metadata for batch
      epoch_completed: Boolean, True if complete batch or False exceed metadata length
    """
    curr_batch = self.metadata[self.batch_start_idx:
                  self.batch_start_idx+self.batch_size]
    self.batch_start_idx += self.batch_size
    epoch_completed = False
    if (self.batch_start_idx >= len(self.metadata)):
      self.batch_start_idx = 0
      epoch_completed = True
      np.random.shuffle(self.metadata) # shuffled again for fun? FIXME
    # load images and preprocess
    batch = []
    batch_annot = []
    data_indices = map(lambda x: x[0], curr_batch)
    batch = self.data[data_indices, :]
    batch_annot = map(lambda x: (x[1], x[2]), curr_batch)
    return batch, batch_annot, epoch_completed

  @staticmethod
  def get_videoid_frameid(path):
    """
    Use naming convention to extract metadata
      - gets image filename formatted as "path/to/dir/vid<vidid>_f<frameid>.jpg"

    Params:
      path: filename path given, should follow above naming convention 
    Returns: 
      videoid: Based on naming convention, return id for video
      frameid: Based on naming convention, return id for frame
    """
    try:
      path = path[:-4]                                      # remove extension
      filename = path[path.rfind(u'/')+1:]                  # remove path to directory
      videoid, frameid = filename.split(u'_')               # split videoid and frameid
      videoid = videoid[3:]                                 # extract videoid
      frameid = frameid[1:]                                 # extract frameid
      return videoid, frameid
    except:
      sys.exit(u'Invalid file name format!')

    
