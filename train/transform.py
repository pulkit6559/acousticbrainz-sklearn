from train import convert

def transform(data, transformlist):
    for t in transformlist:
        transform = t["transfo"]
        params = t.get("params")
        if transform == "remove":
            # TODO: Error if no params
            data = tr_remove(data, params["descriptorNames"])
        elif transform == "enumerate":
            data = tr_enumerate(data, params["descriptorNames"])
        elif transform == "normalize":
            data = tr_normalize(data)
        elif transform == "gaussianize":
            data = tr_gaussianize(data)
    return data


def tr_remove(data, params):
    """ converts normal acousticbrainz structured data into {key.sub.sub: value}-style dict """
    return convert.convert(data, ignore=params)


def tr_enumerate(data, params):
    return data


def tr_normalize(data):
    return data


def tr_gaussianize(data):
    return data
