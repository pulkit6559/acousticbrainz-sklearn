
from sklearn.svm import SVC

import acousticbrainz


def filter_descriptors(mbids, cache_dir, filterdesc):
    # TODO: filelist/groundtruth uses arbitrary file labels, we could use this instead of `mbid`
    # Load files from disk
    # filter, to remove params
    # turn into np array per class
    pass


def load_filelist(flfile):
    return yaml.load(open(flfile))


def load_groundtruth(gtfile):
    return yaml.load(open(gtfile))


def load_data(project):
    files = project["filelist"]
    gt = project["groundtruth"]

    if files is None:
        # If there are no files, it means the GT has mbids and we need to download them from AB
        # TODO: Is this the right place? Perhaps it should be in make_project?

        acousticbrainz.cache_mbids()

    keys = set()
    # cache data
    # for each class
    # - load and filter this classes files (transformation depends on our iteration)


def enumerate_combinations(project):
    params = []
    for s in project["classifiers"]["svm"]:
        for pre in s["preprocessing"]:
            for k in s["kernel"]:
                for g in s["gamma"]:
                    for c in s["C"]:
                        params.append({"preprocessing": pre, "kernel": k, "gamma": g, "C", c})
    return params

def train_model_iteration(project, iteration):
    param_iters = enumerate_combinations(project)
    if iteration >= len(param_iters) or iteration < 0:
        raise Exception("iteration parameter is out of bounds of number of available iterations (%s)" % (len(param_iters), ))

    params = param_iters[iteration]

    classes, data = load_data(project, params)

    class_map = dict([(v, i) for i, v in enumerate(classes, 1)]) # class labels to numbers


    cls = SVC(params)
    cls.train(data, classes)



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

