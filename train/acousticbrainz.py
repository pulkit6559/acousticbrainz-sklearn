# Get items from an AcousticBrainz website

import os
import json

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from train import util

ret = Retry(total=10, backoff_factor=0.2)
adaptor = HTTPAdapter(max_retries=ret)
session = requests.Session()
session.mount('http://', adaptor)
session.mount('https://', adaptor)

ACOUSTICBRAINZ_ROOT = "http://ac-acousticbrainz.s.upf.edu:8000"

def download_mbids(mbidlist):
    """ Do a bulk query of MBIDs and return them
    Returns a dict {mbid: data, mbid: data}
    If an mbid doesn't exist, it is not returned.
    """
    ret = {}
    url = os.path.join(ACOUSTICBRAINZ_ROOT, "api/v1/low-level")

    recids = ";".join(mbidlist)
    r = session.get(url, params={"recording_ids": recids})
    r.raise_for_status()
    for mbid, data in r.json().items():
        # TODO: Here we always assume we have offset 0
        ret[mbid] = data["0"]
    return ret

def dir_for(cache_root, mbid):
    return os.path.join(cache_root, mbid[:2])

def file_for(cache_root, mbid):
    return os.path.join(dir_for(cache_root, mbid), "%s.json" % mbid)

def cache_mbids(mbids, cache_root):
    """ Download a list of mbids to a cache directory if they don't already exist:
    Args:
        mbids: a list of mbids to download
        cache_root: a directory to save data to
    """
    toget = []
    for m in mbids:
        if not os.path.exists(file_for(cache_root, m)):
            toget.append(m)

    for chunk in util.chunks(toget, 20):
        print("dfownload")
        res = download_mbids(chunk)
        for mbid, data in res.items():
            util.mkdir_p(dir_for(cache_root, mbid))
            json.dump(data, open(file_for(cache_root, mbid), "w"))


