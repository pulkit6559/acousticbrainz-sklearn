
from sklearn.svm import SVC
import numpy as np
import json
import yaml
import os

from train import transform


def load_and_filter_descriptors(filelist, filterdesc, preprocessing):
    """ Load a project filelist and perform transformations
        Returns:
            a dictionary of {filekey: filtereddata, ...}
    """

    ret = {}
    # TODO: If this doesn't exist?
    pre = preprocessing[filterdesc]
    for k, path in filelist.items():
        data = json.load(open(path))
        tr = transform.transform(data, pre)
        ret[k] = tr

    return ret


def load_filelist(projectroot, flfile):
    return yaml.load(open(os.path.join(projectroot, flfile)))


def load_groundtruth(projectroot, gtfile):
    return yaml.load(open(os.path.join(projectroot, gtfile)))


def enumerate_combinations(project):
    params = []
    for s in project["classifiers"]["svm"]:
        for pre in s["preprocessing"]:
            for k in s["kernel"]:
                for g in s["gamma"]:
                    for c in s["C"]:
                        params.append({"preprocessing": pre, "kernel": k, "gamma": g, "C": c})
    return params


def train_model_iteration(projectroot, project, iteration):
    param_iters = enumerate_combinations(project)
    if iteration >= len(param_iters) or iteration < 0:
        raise Exception("iteration parameter is out of bounds of number of available iterations (%s)" % (len(param_iters), ))

    params = param_iters[iteration]
    print(params)

    flfile = project["filelist"]
    gt = project["groundtruth"]
    groundtruth = load_groundtruth(projectroot, gt)
    classes = groundtruth.values()

    print("Loading...")
    data = load_and_filter_descriptors(load_filelist(projectroot, flfile), params["preprocessing"], project["preprocessing"])
    print("done")

    # loop through data items, get all keys
    keys = set()
    for k, v in data.items():
        keys.update(set(v.keys()))

    # map gt class names to numbers
    class_map = dict([(v, i) for i, v in enumerate(classes, 1)]) # class labels to numbers
    keys = sorted(list(keys))
    numkeys = len(keys)


    numitems = len(data)
    feature_data = np.empty(numitems)
    feature_classes = np.empty(numitems)

    for i, (k, d) in enumerate(data.items()):
        class_val = class_map[groundtruth[k]]
        feature_classes[i] = class_val
        # TODO: optimisation - make featuredata `numitems x numkeys`
        #       check how much memory this uses - will we replicate `data`?
        item_data = np.empty(numkeys)
        for j, featkey in enumerate(keys):
            # None or NaN?
            item_data[j] = d.get(featkey)
        feature_data[i] = item_data

    k = params["kernel"]
    g = params["gamma"]
    c = params["C"]
    clf = SVC(kernel=k, gamma=2**g, C=2**C)
    clf.train(feature_data, feature_classes)

    pickle.dump(clf, open("test.pkl"))



# Runner for cluster - read parameters and generate combinations, use index to select which combination
#                    - this must be reproducable each time

# Get transformations from projectfile / num permutation
# Perform descriptor filtering
# If number of descriptors is different in files in a class, perhaps data is missing.
#              - Ignore bad file? fill it in with NaN?
# Perform descriptor transformations (enumerate, normalize, gaussianize)

# Normal grid search - read parameters and put into scikitlearn method

# Test/train split - n folds in project file

# For each permutation, save 1 file with params
#                            1 file with ... results of test split?
#                                   Use groundtruth file to get accuracy

# Tool to look at results dir and see if any permutations failed to run, run them

# After all permutations are done, look in results and find params for run with best results
#                                  train model again using all data (no test split) and these params
# Save model

# Load model and perform classification

