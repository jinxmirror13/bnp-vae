import numpy as np
import scipy.stats
import sys
import copy
from node import Node
import cPickle
from PIL import Image

#######################################################
### Helper and Main Functions
#
# Setting up the hierarchy and perparing other objects
# FIXME
#######################################################

'''
python compute_acc_frame.py \
    /scratch/ans556/prasoon/all_models/models_test/alpha_140000.txt \
    /scratch/ans556/prasoon/all_models/models_test/z_150000.txt \
    /work/ans556/prediction_test/output_15k_z_train.txt 256 \
    /scratch/ans556/prasoon/data/MED/MED11/train_label.txt \
    /scratch/ans556/prasoon/data/MED/MED11/train_label.txt
'''

def get_alpha_leaves(node, result):
    """
    Alpha vector encodes the hierarchical relation 

    Params:
        node: current Node object (defined in node.py)
            - To fetch it's alpha, must find the leaf alpha
        result: dictionary of seen leaf nodes and their alpha
    Returns:
        result: dictionary of all leaf nodes and their alpha
    """
    if node.isLeafNode:
        result[node] = node.alpha
    else:
        for c in node.children:
            result = get_alpha_leaves(c, result)
    return result  

def get_internal_nodes(node, result):
    """
    Generate a list of every internal node (include root? TODO)

    Params:
        node: current Node object
        result: list of nodes that are not leaf nodes, seen so far
    Returns:
        result: list of nodes that are not leaf nodes
            - Does this combine lists smoothly while recursing? TODO
    """
    if node.isLeafNode:
        pass
    else:
        result.append(node)
        for c in node.children:
            result = get_internal_nodes(c, result)
    return result

def write_image(output_filename, input_filenames, K):
    """
    FIXME TODO
    Creates an image... to visualise output

    Params:
        output_filename: 
        input filenames: plural?
        K: scaling?
    Returns:
        None, saves an image under the specified output_filename
    """
    k1 = int(np.sqrt(K))
    new_im = Image.new('RGB', (k1 * 640, k1 * 360))
    for i in range(k1):
        for j in range(k1):
            curr_image = Image.open(input_filenames[i * k1 + j])
            if curr_image <> None:
                new_im.paste(curr_image.resize((636, 356)), (i * 640 + 2, j * 360 + 2))
    new_im.save(output_filename)

def get_node_to_imgs(node, result, z, K, filenames_list):
    """


    Params:
        node: current Node Object
        result:
        z: latent vector representations?
        K: scaling factor? FIXME
        filenames_list: names to store node images
    Returns:
        result: updated node information with frames_assigned (??) FIXME
    """
    if node.isLeafNode:
        frames_assigned = result[node]
    else:
        # Recurse until reach a leaf noode
        for c in node.children:
            result = get_node_to_imgs(c, result, z, K, filenames_list)
        # Frames Assigned? TODO
        frames_assigned = []
        for c in node.children:
            frames_assigned += list(result[c])

        # Result maps Node Objects to frrames_assigned?? TODO
        result.update({node: frames_assigned})
    # TODO
    dist_to_path = np.linalg.norm(node.alpha - z[frames_assigned, 2:], axis=1)
    top_k = sorted(range(len(dist_to_path)), key=lambda i: dist_to_path[i])[:K]
    filenames_curr_node = map(lambda j: filenames_list[j].strip(), \
                                                        [frames_assigned[i] for i in top_k])
    try:
        print node.node_id, '\t', node.parent.node_id, '\t', filenames_curr_node
    except:
        print node.node_id, '\t', 'Root node', '\t', filenames_curr_node
    write_image('./output_imgs4/' + node.node_id + '.jpg', filenames_curr_node, K)
    return result

#def main(nodes_file, z_train_file, z_test_file, bf, num_levels, num_paths, \
#         train_labels_file, test_labels_file, decay_factor):
def main(nodes_filename, train_labels_file, z_train_file, test_labels_file, z_test_file, \
                 decay_factor, filenames, K):
    # Fetch model and data details
    filenames_list = open(filenames).readlines()
    nodes_file = open(nodes_filename, 'rb')
    nodes_list = cPickle.load(nodes_file)
    nodes_file.close()

    # Construct Tree
    #### Find Root node
    root_node = None
    for node in nodes_list:
        if node.parent == None:
            root_node = node
            break
    #### Fetch Attrributes
    alpha_dict = get_alpha_leaves(root_node, {}) #dictionary of leaf node ids and their alphas
    num_paths = len(alpha_dict) # Number of leave nodes
    leaf_nodes_list = alpha_dict.keys() # List of leaf node ids
    alpha_values = alpha_dict.values() # List of leaf node alphas
    #### Load Training Data
    train_labels = np.loadtxt(train_labels_file) - 1
    num_classes = len(np.unique(train_labels))
    z_train = np.loadtxt(z_train_file)
    #### Minimize difference between leaf node alphas & z_train to find path assignment
    ######## Seems to be like mapping the data point to the closest cluster...
    path_assignments = [] # List of indices corresponding to leaf nodes for each point z
    for i, z in enumerate(z_train):
        closest_path_idx = np.argmin(np.linalg.norm((alpha_values - z[2:]), axis=1))
        path_assignments.append(closest_path_idx)
    #### Map leaf nodes to list of frames_assigned FIXME
    leaf2imgs = {}
    for path_id in range(num_paths):
        frames_assigned = np.where(np.asarray(path_assignments) == path_id)[0]
        leaf2imgs.update({leaf_nodes_list[path_id] : \
                                            list(frames_assigned)})
    nodes2imgs = get_node_to_imgs(root_node, leaf2imgs, z_train, K, filenames_list)

if __name__ == '__main__':
    nodes_file = sys.argv[1]
    train_labels_file = sys.argv[2]
    z_train_file = sys.argv[3]
    test_labels_file = sys.argv[2]
    z_test_file = sys.argv[3]
    decay_factor = 0.0
    filenames = sys.argv[4]
    K = eval(sys.argv[5])
    main(nodes_file, train_labels_file, z_train_file, test_labels_file, z_test_file, \
             decay_factor, filenames, K)
