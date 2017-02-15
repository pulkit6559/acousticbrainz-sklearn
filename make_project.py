# Take parameters and use it to create a project file

import argparse
import yaml
import os
import csv

from train import util
import train.acousticbrainz

PROJECT_TEMPLATE = "project_template.yaml"

def acousticbrainz(args):
    # TODO: Check that the root is a directory where we have access to
    root = args.projectroot
    util.mkdir_p(root)

    # CSV data file is mbid,class
    # Turn into groundtruth file {"mbid": "class", ...}
    gt = {}
    with open(args.datasetfile) as fp:
        r = csv.reader(fp)
        for l in r:
            if len(l) == 2:
                gt[l[0]] = l[1]

    groundtruthname = "groundtruth.yaml"
    with open(os.path.join(root, groundtruthname), "w") as fp:
        yaml.dump(gt, fp)

    # Download acousticbrainz files:
    datadir = os.path.join(root, "data")
    util.mkdir_p(datadir)
    train.acousticbrainz.cache_mbids(list(gt.keys()), datadir)

    # and filelist file {"mbid": "path", ...}
    filelist = {}
    for k in gt.keys():
        filelist[k] = acousticbrainz.file_for(datadir, k)
    with open(os.path.join(root, filelistname), "w") as fp:
        yaml.dump(filelist, fp)

    template = open(PROJECT_TEMPLATE).read()

    resultsdir = os.path.join(root, "results")
    util.mkdir_p(resultsdir)

    projectfile = template % {"className": args.projectname,
            "datasetsDirectory": datadir,
            "resultsDirectory": results,
            "filelist": filelistname,
            "groundtruth": groundtruthname}

    with open(os.path.join(root, "project.yaml"), "w") as fp:
        fp.write(projectfile)


def data(projectname, projectroot, groundtruth, filelist):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("projectname", help="name of project")
    common.add_argument("projectroot", help="path to project structure")

    ab = sub.add_parser("ab", parents=[common], help="process an acousticbrainz dataset")
    ab.add_argument("datasetfile", help="acousticbrainz dataset file")
    ab.set_defaults(func=acousticbrainz)

    data = sub.add_parser("data", parents=[common], help="")
    data.add_argument("groundtruth", help="groundtruth project file")
    data.add_argument("filelist", help="filelist file")
    data.set_defaults(func=data)

    directory = sub.add_parser("dir", help="build a project from a directory of files")

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
