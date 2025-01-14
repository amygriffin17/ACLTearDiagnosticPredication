# CS311 Final Project: ACL Tear Diagnostic Decision Tree vs. Random Forest

## Running the program

You can change the evaluation approach for the acl dataset by changing the optional arguments shown below.


Train and test decision tree learner

optional arguments:
  -h, --help            show this help message and exit
  -p PREFIX, --prefix PREFIX
                        Prefix for dataset files. Expects <prefix>.data.txt and <prefix>.label.txt files. Allowed values: acl.
  -m MODEL, --model MODEL
                        Which model to use (single decision tree or random forest). Allowed values: tree, forest
  -n TREES, --num_trees TREES
                        Number of trees for random forest (if applicable). Default = 10
  -f MAX_FEATURES, --max_features K_SPLITS
                        Number of features for random forest (if applicable). Default is NONE
  -r RAND_STATE, --random-state K_SPLITS
                        Random state for reproducibility. Default is 42.
```

For example, to train and test a tree, run the program as `python3 finalAI.py -p acl -m tree`.

To train and test a forest with default variables, run the program as `python3 finalAI.py -p acl -m forest`.