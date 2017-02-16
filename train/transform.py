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

def tr_all_normalize(data):
    pass

def tr_all_gaussianize(data):
    pass
