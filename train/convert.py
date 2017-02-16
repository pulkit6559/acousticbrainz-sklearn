# Convert from MusicExtractor json to filtered dictionary

from fnmatch import fnmatch

def isMatch(name, patterns):
    if not patterns:
        return False
    for pattern in patterns:
        if fnmatch(name, pattern):
            return True
    return False


def convert(d, include=None, ignore=None):
    results = {}

    stack = [(k, k, v) for k, v in d.items()]
    while stack:
        name, k, v = stack.pop()
        if isinstance(v, dict):
            stack.extend([(name + '.' + k1, k1, v1) for k1, v1 in v.items()])
        elif isinstance(v, list):
            stack.extend([(name + '.' + str(i), i, v[i]) for i in range(len(v))])
        else:
            if include:
                # 'include' flag specified => apply both include and ignore
                if isMatch(name, include) and not isMatch(name, ignore):
                    results[name] = v
            else:
                # 'include' flag not specified => apply only ignore
                if not isMatch(name, ignore):
                    results[name] = v

    return results

