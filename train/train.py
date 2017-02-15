
# Read project file
# Read list of mbids
# Download all data files to cache directory if they're not there

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

