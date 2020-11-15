This is the solution of SoftServe team for the ['FORCE: Machine Predicted Lithology'](https://xeek.ai/challenges/force-well-logs) challenge. `train.csv` and `test.csv` are the training and public leaderboard datafiles from the challenge.

# Setup the environment
* Python3.8 was used during development
* `pip3 install --upgrade pip`
* `pip3 install -r requirements.txt`

# How to run
The solution assumes that there is a `train.csv` file in the same direcory with the python script `final_script.py`. It is supposed to be the very same training data used for the challenge.

There are two ways how to run the solution
1. Running the script
* Put `test.csv` file in the same directory with the script `final_script.py`
* `python3 final_script.py`
2. Use jupyter notebooks and follow `train_and_predict.ipynb`

The solution is expected to write some auxiliary files during training and inference.

The solution is expected to work about 12 hours on the Google Cloud Platform c2-standard-8 (8 vCPUs, 32 GB memory) machine.

The end result of the script is the `softserve_submission.csv` file with the corresponding predictions.

# How the solution works
There are a few major steps which the solution can be broken into
* **Data imputation.** It is done with LightGBM models only for those predictors that are good for predicting.
* **Determining holdout wells.** Wells that are similar to the test wells are found in this step and put in the holdout sample.
* **Feature preprocessing and feature generation.** Some features are generated according to the domain knowledge.
* **Training of the `OK model`**. At this step a LightGBM model that performs OK is built using hyperparameter tuning.
* **Ensemble training**. The most interesting part of the solution. At each step a LightGBM model with a random set of parameters is built using training wells and some of the holdout wells. Then using the rest of the holdout wells its performance is compared with the performance of the `OK model`. In case of good results, it is added to the list of good models.
* **Inference**. Get the predictions of all models and use the mode of the predicted labels as the final prediction.
