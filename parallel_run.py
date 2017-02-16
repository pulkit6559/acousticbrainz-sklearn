import argparse
import yaml
import os

from train import train

def main(projectfile, iteration):
    project = yaml.load(open(projectfile))
    projectroot = os.path.abspath(os.path.dirname(projectfile))

    train.train_model_iteration(projectroot, project, iteration)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="Yaml project file")
    parser.add_argument("iteration", type=int, help="iteration")
    args = parser.parse_args()
    main(args.project, args.iteration)
