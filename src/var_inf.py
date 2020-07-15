from util import *
from copy import deepcopy
from scipy.special import digamma
from datetime import datetime
import sys
from node import Node
from util import *
import cPickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples

#######################################################
# Variational Inference Object
#
# 
#######################################################


class VarInf(object):
  def __init__(self):
    # initialize variational parameters
    self.root = None
    self.decay_coeff = 0.0
    self.vidid_frameid_to_idx = []
    self.vidid_to_idx = []
    self.kmeans_models = {}
    self.kmeans_labels = {}
    self.kmeans_ss = {}

  @staticmethod
  def weighted_kmeans(data, weights, K):
    """
    Calculate the centre points (K of them) for the data set

    Assumes points surround center [0]*D, relies on given weights

    Params:
      data: uhh... the data ?
      weights: given weights of the points for this procedure
      K: Number of centres
    Returns:
      centers: matrix [K x D] of selected data points that are "centres" ? mediods? FIXME
    """
    N = np.size(data, 0) # Number of data points
    D = np.size(data, 1) # Number of features
    # initialize centers
    centers = np.zeros(shape=(K, D)) # Empty array to hold multivariate centers
    init_weights = weights / np.sum(weights) # Normalise the initial weights
    closest_cluster_dist = np.ones(shape=N) / N # Random decimal initialisation? FIXME

    # For K rounds, select a data point based on the max weighted distance to be centre
    for k in range(K):
      idx = np.argmax(VarInf.normalize(init_weights * closest_cluster_dist))
      centers[k, :] = data[idx, :] # Use selected data point as centre
      for n in range(N):
        # Redefine distances with respect to next cluster center (always [0]*D)
        closest_cluster_dist[n] = np.min(np.linalg.norm(data[n, :] - centers[:k+1, :], axis=1))
    return centers

  def get_phi_leaves(self, idx):
    """
    Iterates through the tree to fetch leaf nodes and their phi values (??) FIXME

    Params:
      idx: An index
    Returns:
      result: [dictionary] Map of leaf nodes to respective phi[idx][0]
    """
    result = {}
    unvisited_nodes = [self.root]
    while len(unvisited_nodes) > 0:
      next_node = unvisited_nodes[0]
      unvisited_nodes = unvisited_nodes[1:]
      if next_node.isLeafNode:
        result[next_node] = next_node.phi[idx][0]
      else:
        unvisited_nodes = unvisited_nodes + next_node.children
    return result

  def get_true_path_mean(self, x_annot_batch):
    """
    Do standard VAE?? It returns a list [x_annot_batch] of list 
    [LATENT_CODE_SIZE] of zeros... FIXME

    Params:
      x_annot_batch: ?? FIXME
    Returns:
      true_path_mu_batch: List [x_annot_batch] of list 
          [LATENT_CODE_SIZE] of zeros
    """
    true_path_mu_batch = []
    indices = []
    for x_annot in x_annot_batch:
      # vidid: FIXME
      # frameid: FIXME
      (vidid, frameid) = x_annot
      # standard VAE
      true_path_mu_batch.append(np.zeros(shape=LATENT_CODE_SIZE))
      continue # So this skips the rest of the loop code??
      if len(self.vidid_frameid_to_idx) == 0:
        # no paths initiated
        true_path_mu_batch.append(np.zeros(shape=LATENT_CODE_SIZE))
      else:
        # Paths initiated
        idx = self.vidid_frameid_to_idx.index((vidid, frameid))
        phi = self.get_phi_leaves(idx)
        keys = phi.keys()
        values = VarInf.normalize(phi.values())
        sample_path_idx = np.random.choice(len(values), p=values)
        indices.append(sample_path_idx)
        true_path_mu_batch.append(keys[sample_path_idx].alpha)
    print indices
    return np.asarray(true_path_mu_batch)

  def get_matrix_from_dict(self, latent_codes):
    """
    TODO

    Params:
      latent_codes:
    Returns:
      result: 
    """
    if len(self.vidid_to_idx) == 0:
      for vidid in latent_codes:
        self.vidid_to_idx.append(vidid)
        for frameid in latent_codes[vidid]:
          self.vidid_frameid_to_idx.append((vidid, frameid))
    result = []
    for vidid in latent_codes:
      for frameid in latent_codes[vidid]:
        result.append(latent_codes[vidid][frameid])
    return np.asarray(result)

  def update_variational_parameters(self, latent_codes):
    """
    TODO - Self-editing of object variables

    Params:
      latent_codes:
    Returns:
      Nothing 
    """
    print 'Performing variational inference...'
    print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    latent_codes_matrix = self.get_matrix_from_dict(latent_codes)
    if self.root is None:
      # initialize tree
      self.root = Node('0', len(self.vidid_to_idx), len(self.vidid_frameid_to_idx), None, \
                       LATENT_CODE_SIZE, GAMMA)
      BF = 4
      l1_nodes = []
      for n in range(BF):
        node_tmp = Node('0-'+str(n), len(self.vidid_to_idx), len(self.vidid_frameid_to_idx), \
                        self.root, LATENT_CODE_SIZE, GAMMA)
        l1_nodes.append(node_tmp)
      l2_nodes = []
      for n in range(BF*BF):
        parent = l1_nodes[n/BF]
        node_tmp = Node(parent.node_id+'-'+str(n%BF), len(self.vidid_to_idx), \
                        len(self.vidid_frameid_to_idx), parent, LATENT_CODE_SIZE, GAMMA)
        l2_nodes.append(node_tmp)
      l3_nodes = []
      for n in range(BF*BF*BF):
        parent = l2_nodes[n/BF]
        node_tmp = Node(parent.node_id+'-'+str(n%BF), len(self.vidid_to_idx), \
                        len(self.vidid_frameid_to_idx), parent, LATENT_CODE_SIZE, GAMMA)
        l3_nodes.append(node_tmp)
      # mark all internal nodes
      self.root.isLeafNode = False
      self.root.children = l1_nodes
      for i, l in enumerate(l1_nodes):
        l.isLeafNode = False
        l.children = l2_nodes[BF*i: BF*(i+1)]
      for i, l in enumerate(l2_nodes):
        l.isLeafNode = False
        l.children = l3_nodes[BF*i: BF*(i+1)]
      # precompute gamma / phi mappings
      self.phi_vidid_list = []
      for i in range(len(self.vidid_to_idx)):
        vidid = self.vidid_to_idx[i]
        self.phi_vidid_list.append(filter(lambda idx: \
             self.vidid_frameid_to_idx[idx][0]==vidid, range(len(self.vidid_frameid_to_idx))))
      #initialize phi to one-hot
      alpha_leaves = self.get_alpha_leaves()
      alpha_keys = alpha_leaves.keys()
      alpha_values = alpha_leaves.values()
      phi_init = np.zeros(shape=(len(alpha_keys), len(self.vidid_frameid_to_idx)))
      for i, z in enumerate(latent_codes_matrix):
        phi_init[np.argmin(np.linalg.norm(z - alpha_values, axis=1))][i] = 1.0
      self.initialize_phi(self.root, dict(zip(alpha_keys, phi_init)))

    for iteration in range(1):
      self.compute_sigma_alpha(latent_codes_matrix)
      self.compute_gamma()
      self.compute_phi(latent_codes_matrix)
      self.print_phi(self.root)
    split = self.split_nodes(self.root, latent_codes_matrix, \
                             STDEV_THR + 100.0 * np.exp(-1.0*self.decay_coeff))
    if split:
      self.decay_coeff = 0.0
      self.compute_phi(latent_codes_matrix)
    else:
      self.decay_coeff += 1.0
    self.merge_nodes(self.root, latent_codes_matrix, \
                             STDEV_THR + 100.0 * np.exp(-1.0*self.decay_coeff))

  def print_stdev(self, node, latent_codes_matrix):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      stdev = np.sqrt(np.linalg.norm(np.sqrt(node.phi) * (latent_codes_matrix - node.alpha)) \
                      / np.sum(node.phi))
    else:
      for c in node.children:
        self.print_stdev(c, latent_codes_matrix)

  def print_all_param_values(self, node):
    print node.node_id
    print node.alpha
    print node.sigmasqr_inv
    print node.phi
    print node.gamma
    if not(node.isLeafNode):
      for c in node.children:
        self.print_all_param_values(c)

  def initialize_phi(self, node, init_values):
    """
    TODO

    Params:
      node:
      init_values:
    Returns:
      Nothing 
    """
    if node.isLeafNode:
      node.phi = np.reshape(init_values[node.node_id], (len(self.vidid_frameid_to_idx), 1))
    else:
      for c in node.children:
        self.initialize_phi(c, init_values)

  def print_phi(self, node):
    if node.isLeafNode:
      print 'phi', node.node_id, np.mean(node.phi)
    else:
      for c in node.children:
        self.print_phi(c)

  def print_gamma(self, node):
    if node.isLeafNode:
      if node.gamma is None:
        print 'gamma', node.node_id, node.gamma, node.gamma
      else:
        print 'gamma', node.node_id, np.mean(node.gamma, axis=0), node.gamma
    else:
      for c in node.children:
        self.print_gamma(c)

  def merge_nodes(self, node, latent_codes_matrix, split_thr):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
      split_thr:
    Returns:
      Nothing
    """
    print 'merge_nodes', node.node_id
    if node.isLeafNode:
      if np.mean(node.phi) < 10**-3:
        print 'Removing node ', node.node_id
        node.parent.children.remove(node)
    else:
      for c in node.children:
        self.merge_nodes(c, latent_codes_matrix, split_thr)
      if len(node.children) == 0:
        node.isLeafNode = True
      elif len(node.children) == 1:
        print 'Merge 1 child'
        node.isLeafNode = node.children[0].isLeafNode
        node.alpha = deepcopy(node.children[0].alpha)
        node.sigmasqr_inv = node.children[0].sigmasqr_inv
        node.phi = deepcopy(node.children[0].phi)
        node.gamma = deepcopy(node.children[0].gamma)
        node.children = deepcopy(node.children[0].children)
        for c in node.children:
          c.parent = node

  def split_nodes(self, node, latent_codes_matrix, split_thr):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
      split_thr:
    Returns:
      result:
    """
    print 'split_nodes', node.node_id
    if node.isLeafNode:
      # compute variance
      if np.mean(node.phi) < 10**-2:
        return False
      stdev = np.sqrt(np.linalg.norm(np.sqrt(node.phi) * (latent_codes_matrix - node.alpha)) \
                      / np.sum(node.phi))
      print node.node_id, stdev, split_thr
      if stdev > split_thr:
        best_K = args.num_splits
        node.isLeafNode = False
        new_centers = VarInf.weighted_kmeans(latent_codes_matrix, node.phi[:, 0], best_K)
        for k in range(best_K):
          new_node = Node(node.node_id + '-' + str(k), \
                          len(self.vidid_to_idx), len(self.vidid_frameid_to_idx), node, \
                          LATENT_CODE_SIZE, GAMMA)
          new_node.alpha = new_centers[k, :]
          new_node.sigmasqr_inv = node.sigmasqr_inv
          new_node.phi = node.phi / best_K
          node.children.append(new_node)
        return True
    else:
      result = False
      for c in node.children:
        result = self.split_nodes(c, latent_codes_matrix, split_thr) or result
      return result

  def compute_sigma_alpha_node(self, node, latent_codes_matrix):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      sum_phi = np.sum(node.phi)
      sum_phi_z = np.sum(node.phi * latent_codes_matrix, axis=0)
      node.sigmasqr_inv = SIGMA_B_sqrinv + sum_phi * SIGMA_Z_sqrinv
      try:
        node.alpha = deepcopy(node.parent.alpha)
      except AttributeError:
        node.alpha = deepcopy(ALPHA)
      node.alpha = (node.alpha * SIGMA_B_sqrinv + sum_phi_z * SIGMA_Z_sqrinv) \
                   / node.sigmasqr_inv
    else:
      # recursively find alpha's and sigma's of children nodes
      for c in node.children:
        self.compute_sigma_alpha_node(c, latent_codes_matrix)
      node.sigmasqr_inv = (1.0 + len(node.children)) * SIGMA_B_sqrinv
      try:
        node.alpha = deepcopy(node.parent.alpha)
      except AttributeError:
        node.alpha = deepcopy(ALPHA)
      for c in node.children:
        node.alpha += c.alpha
      node.alpha = node.alpha * SIGMA_B_sqrinv / node.sigmasqr_inv

  def compute_sigma_node(self, node, latent_codes_matrix):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      sum_phi = np.sum(node.phi)
      sum_phi_z = np.sum(node.phi * latent_codes_matrix, axis=0)
      node.sigmasqr_inv = SIGMA_B_sqrinv + sum_phi * SIGMA_Z_sqrinv
      print node.node_id, SIGMA_B_sqrinv, sum_phi, SIGMA_Z_sqrinv, node.sigmasqr_inv
    else:
      # recursively find alpha's and sigma's of children nodes
      for c in node.children:
        self.compute_sigma_node(c, latent_codes_matrix)
      node.sigmasqr_inv = (1.0 + len(node.children)) * SIGMA_B_sqrinv

  def compute_alpha_node(self, node, latent_codes_matrix):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      sum_phi = np.sum(node.phi)
      sum_phi_z = np.sum(node.phi * latent_codes_matrix, axis=0)
      try:
        node.alpha = deepcopy(node.parent.alpha)
      except AttributeError:
        node.alpha = deepcopy(ALPHA)
      node.alpha = (node.alpha * SIGMA_B_sqrinv + sum_phi_z * SIGMA_Z_sqrinv) \
                   / node.sigmasqr_inv
    else:
      # recursively find alpha's and sigma's of children nodes
      for c in node.children:
        self.compute_alpha_node(c, latent_codes_matrix)
      try:
        node.alpha = deepcopy(node.parent.alpha)
      except AttributeError:
        node.alpha = deepcopy(ALPHA)
      for c in node.children:
        node.alpha += c.alpha
      node.alpha = node.alpha * SIGMA_B_sqrinv / node.sigmasqr_inv

  def compute_sigma_alpha(self, latent_codes_matrix):
    """
    TODO

    Params:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    self.compute_sigma_node(self.root, latent_codes_matrix)
    self.compute_alpha_node(self.root, latent_codes_matrix)

  @staticmethod
  def digamma_add0(gamma_sum, new_gamma):
    """
    TODO

    Params:
      gamma_sum:
      new_gamma:
    Returns:
      calculation... TODO
    """
    return gamma_sum + digamma(new_gamma[:, 0]) - digamma(np.sum(new_gamma, axis=1))

  @staticmethod
  def digamma_add1(gamma_sum, new_gamma):
    """
    TODO

    Params:
      gamma_sum:
      new_gamma:
    Returns:
      calculation... TODO
    """
    if gamma_sum is None:
      return digamma(new_gamma[:, 1]) - digamma(np.sum(new_gamma, axis=1))
    return gamma_sum + digamma(new_gamma[:, 1]) - digamma(np.sum(new_gamma, axis=1))

  def compute_phi_node(self, node, latent_codes_matrix, gamma_sum_on, gamma_sum_before):
    """
    TODO

    Params:
      node:
      latent_codes_matrix:
      gamma_sum_on:
      gamma_sum_before:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      scaled_dist = 0.5 * SIGMA_Z_sqrinv * \
                    (np.linalg.norm(latent_codes_matrix - node.alpha, axis=1) + \
                     1.0 / node.sigmasqr_inv)
      for i in range(len(self.vidid_frameid_to_idx)):
        vidid, frameid = self.vidid_frameid_to_idx[i]
        node.phi[i] = gamma_sum_on[node][self.vidid_to_idx.index(vidid)] + \
                      gamma_sum_before[node][self.vidid_to_idx.index(vidid)] - \
                      scaled_dist[i]
    else:
      for c in node.children:
       self.compute_phi_node(c, latent_codes_matrix, gamma_sum_on, gamma_sum_before)

  def get_phi_min(self, node):
    """
    TODO

    Params:
      node:
    Returns:
      phi_min:
    """
    if node.isLeafNode:
      return node.phi
    else:
      phi_min = -np.inf * np.ones(shape=(len(self.vidid_frameid_to_idx), 1))
      for c in node.children:
        phi_min = np.maximum(phi_min, self.get_phi_min(c))
      return phi_min

  def get_phi_sum(self, node):
    """
    TODO

    Params:
      node:
    Returns:
      phi_sum:
    """
    if node.isLeafNode:
      return node.phi
    else:
      phi_sum = np.zeros(shape=(len(self.vidid_frameid_to_idx), 1))
      for c in node.children:
        phi_sum += self.get_phi_sum(c)
      return phi_sum

  def normalize_phi(self, node, phi_sum):
    """
    TODO

    Params:
      node:
      phi_sum:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      node.phi /= phi_sum
    else:
      for c in node.children:
        self.normalize_phi(c, phi_sum)

  def exponentiate_phi(self, node, offset):
    """
    TODO

    Params:
      node:
      offset:
    Returns:
      Nothing
    """
    if node.isLeafNode:
      node.phi = np.exp((node.phi - offset))
    else:
      for c in node.children:
        self.exponentiate_phi(c, offset)

  def compute_gamma_sum_on(self, node, curr_path_sum, result):
    """
    TODO

    Params:
      node:
      curr_path_sum:
      result:
    Returns:
      result:
    """
    if not(node.parent is None):
      curr_path_sum = VarInf.digamma_add0(curr_path_sum, node.gamma)
    if node.isLeafNode:
      result.update({node: curr_path_sum})
      return result
    else:
      for c in node.children:
        result = self.compute_gamma_sum_on(c, curr_path_sum, result)
      return result

  def compute_gamma_sum_before(self, node, curr_path_before, result):
    """
    TODO

    Params:
      node:
      curr_path_sum:
      result:
    Returns:
      digamma update? TODO
    """
    if node.isLeafNode:
      result.update({node: curr_path_before})
      return VarInf.digamma_add1(None, node.gamma), result
    else:
      sum_children = 0.0
      for c in node.children:
        sum_c, result = self.compute_gamma_sum_before(c, \
                             curr_path_before + sum_children, result)
        sum_children += sum_c
      #return curr_path_before + sum_children, result
      return VarInf.digamma_add1(sum_children, node.gamma), result

  def compute_phi(self, latent_codes_matrix):
    """
    TODO

    Params:
      latent_codes_matrix:
    Returns:
      Nothing
    """
    gamma_sum_on = self.compute_gamma_sum_on(self.root, \
                        np.zeros(shape=len(self.vidid_to_idx)), {})
    _, gamma_sum_before = self.compute_gamma_sum_before(self.root, \
                            np.zeros(shape=len(self.vidid_to_idx)), {})
    self.compute_phi_node(self.root, latent_codes_matrix, gamma_sum_on, gamma_sum_before)
    phi_min = self.get_phi_min(self.root)
    self.exponentiate_phi(self.root, phi_min)
    phi_sum = self.get_phi_sum(self.root)
    self.normalize_phi(self.root, phi_sum)
    phi_sum = self.get_phi_sum(self.root)

  def compute_gamma_node(self, node, sum_phi_before):
    """
    TODO

    Params:
      node:
      sum_phi_before:
    Returns:
      sum_phi_curr and sum_phi_children:
    """
    if node.isLeafNode:
      if node.parent is None:
        node.gamma = None
        return None
      sum_phi_curr = np.zeros(shape=len(self.vidid_to_idx))
      for i in range(len(self.vidid_to_idx)):
        phi_vidid = self.phi_vidid_list[i]
        sum_phi_curr[i] = np.sum(node.phi[phi_vidid])
      node.gamma[:, 0] = 1.0 + sum_phi_curr
      node.gamma[:, 1] = GAMMA + sum_phi_before
      return sum_phi_curr
    else:
      sum_phi_children = np.zeros(shape=len(self.vidid_to_idx))
      for c in node.children:
        sum_phi_children += self.compute_gamma_node(c, sum_phi_before + sum_phi_children)
      node.gamma[:, 0] = 1.0 + sum_phi_children
      node.gamma[:, 1] = GAMMA + sum_phi_before
      return sum_phi_children

  def compute_gamma(self):
    """
    TODO

    Params:
      None
    Returns:
      Compute Gamma node...
    """
    self.compute_gamma_node(self.root, np.zeros(shape=len(self.vidid_to_idx)))

  @staticmethod
  def normalize(p):
    """
    TODO

    Params:
      p:
    Returns:
      calculation... TODO
    """
    p = np.asarray(p)
    s = np.sum(p)
    if s > 0:
      return p/s
    else:
      return VarInf.normalize(np.ones(shape=np.shape(p)))

  def write_gamma(self, filename):
    """
    TODO

    Params:
      filename:
    Returns:
      Nothing
    """
    with open(filename, 'w') as f:
      for vidid in self.gamma:
        f.write(vidid + '\t' + unicode(self.gamma[vidid]) + '\n')

  def write_assignments(self, filename):
    """
    TODO

    Params:
      filename:
    Returns:
      Nothing
    """
    with open(filename, 'w') as f:
      for vidid in self.phi:
        for frameid in self.phi[vidid]:
          f.write(vidid + '\t' + frameid + '\t' + \
                  unicode(np.argmax(self.phi[vidid][frameid])) + '\n')

  def write_alpha_node(self, node, fileptr):
    """
    TODO

    Params:
      node:
      fileptr:
    Returns:
      Nothing
    """
    fileptr.write(node.node_id + ' ' + \
                  ' '.join(map(lambda x: str(np.round(x, 2)), node.alpha)) + \
                  '\n')
    if not(node.isLeafNode):
      for c in node.children:
        self.write_alpha_node(c, fileptr)

  def write_alpha(self, filename):
    """
    TODO

    Params:
      filename:
    Returns:
      Nothing
    """
    fileptr = open(filename, 'w')
    self.write_alpha_node(self.root, fileptr)
    fileptr.close()
      
  def write_sigma_node(self, node, fileptr):
    """
    TODO

    Params:
      node:
      fileptr:
    Returns:
      Nothing
    """
    fileptr.write(node.node_id + ' ' + \
                  ' '.join(map(lambda x: str(np.round(x, 2)), [node.sigmasqr_inv])) + \
                  '\n')
    if not(node.isLeafNode):
      for c in node.children:
        self.write_sigma_node(c, fileptr)

  def write_sigma(self, filename):
    """
    TODO

    Params:
      filename:
    Returns:
      Nothing
    """
    fileptr = open(filename, 'w')
    self.write_sigma_node(self.root, fileptr)
    fileptr.close()

  def get_nodes_list(self, node, result):
    """
    TODO

    Params:
      node:
      result:
    Returns:
      result
    """
    if node.isLeafNode:
      result.append(node)
    else:
      result.append(node)
      for c in node.children:
        result = self.get_nodes_list(c, result)
    return result

  def write_nodes(self, filename):
    """
    TODO

    Params:
      filename:
    Returns:
      Nothing
    """
    nodes_list = self.get_nodes_list(self.root, [])
    fileptr = open(filename, 'wb')
    cPickle.dump(nodes_list, fileptr)
    fileptr.close()

  def get_alpha_leaves_node(self, node, result):
    """
    TODO

    Params:
      node:
      result:
    Returns:
      result
    """
    print node.node_id, len(result)
    if node.isLeafNode:
      result.update({node.node_id: node.alpha})
      return result
    else:
      for c in node.children:
        result.update(self.get_alpha_leaves_node(c, result))
      return result

  def get_alpha_leaves(self):
    """
    TODO

    Params:
      None
    Returns:
      Leaf node
    """
    return self.get_alpha_leaves_node(self.root, {})
