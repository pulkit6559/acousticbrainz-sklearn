import collections

from train import convert

def transform(data, transformlist):
    for t in transformlist:
        transform = t["transfo"]
        params = t.get("params")
        if transform == "remove":
            # TODO: Error if no params
            data = tr_remove(data, params["descriptorNames"])
    return data

def transform_all(data, transformlist):
    for t in transformlist:
        transform = t["transfo"]
        params = t.get("params")
        if transform == "enumerate":
            data = tr_all_enumerate(data, params["descriptorNames"])
        elif transform == "normalize":
            data = tr_all_normalize(data)
        elif transform == "gaussianize":
            data = tr_all_gaussianize(data)
    return data


def tr_remove(data, params):
    """ converts normal acousticbrainz structured data into {key.sub.sub: value}-style dict """
    return convert.convert(data, ignore=params)

def tr_all_enumerate(data, params):
    """ input: dict {key, datafile}
    Get all possible values for datafile[p in params]
    in all items in data
    create enumeration for each of these possible values
    replace datafile[p] with enumeration
    TODO: Could be more efficient by iterating once, adding to
          enumeration each time we find a new value
    """
    possible_values = collections.defaultdict(set)
    for k, v in data.items():
        for p in params:
            if p in v:
                possible_values[p].add(v[p])

    param_enum_map = {}
    for p in params:
        enum_map = dict([(v, i) for i, v in enumerate(sorted(list(possible_values[p])))])
        param_enum_map[p] = enum_map

    # Now apply enumerations
    for k, v in data.items():
        for p in params:
            if p in v:
                v[p] = param_enum_map[p][v[p]]

    return data

def tr_all_remove_varlength(data):
    """ Remove variable-lengths descriptors """
    # TODO: Iterate through data to find flattened lists of variable length
    # This transformation should be applied before computing the common layout
    # for the dataset, otherwise all variable-length lists would become
    # fixed-length with the minimum size encountered (which is maybe ok for some
    # tasks in future).
    pass

def tr_all_cleaner(data):
    """ Remove all descriptors that are either constant values,
    or contain NaN of Inf values. """
    # TODO: print a list of removed descriptors
    # Use numpy.sum() http://stackoverflow.com/a/6736970/603642
    # We can also do Nan / Inf check in convert()
    pass

def tr_all_normalize(data):
    # TODO: Normalize to [0, 1] (default Gaia behavior)
    #       - use sklearn.preprocessing.minmax_scale

    # To add in future:
    # - Standardize (center to the mean and scale to unit variance)
    #   (sklearn.preprocessing.scale)
    # - Robust standardize (sklearn.preprocessing.scale)
    # - Normalize to unit norm (sklearn.preprocessing.normalize)
    pass

def tr_all_gaussianize(data):
    pass


