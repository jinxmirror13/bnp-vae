from copy import deepcopy
from scipy.special import digamma
import sys
import numpy as np


#######################################################################
### Node object
# 
# Basis of the nCRP tree
#######################################################################

class Node(object):
  def __init__(self, node_id, num_vids, num_frames, parent, LATENT_CODE_SIZE, GAMMA):
    print 'Creating node ', node_id
    # Unique name to identify node
    #### TODO does this have anything to do with building the tree? Format of it?
    self.node_id = node_id

    # Boolean to see if this node is a leaf node
    self.isLeafNode = True

    # Attributes to recursively select nodes following a path
    self.parent = parent # I assume this is the parent's node_id
    self.children = []

    # Required node parameter sampled from a Normal dist
    self.alpha = np.random.normal(size=LATENT_CODE_SIZE)
    self.sigmasqr_inv = 1.0

    # Dataset specific fields for path features (see 4.2.1 in paper)
    #### parameter for multinomial dist to select path assignment given data x_{mn}
    self.phi = np.ones(shape=(num_frames, 1))

    #### defining the parameters for the Beta dist to sample v_{me} if at edge e (FIXME verify)
    self.gamma = np.ones(shape=(num_vids, 2))
    self.gamma[:, 1] *= GAMMA
    
