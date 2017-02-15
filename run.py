import argparse
import yaml
from train import train

def main(projectfile):
    project = yaml.load(open(projectfile))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project", help="Yaml project file")
    args = parser.parse_args()
    main(args.project)
