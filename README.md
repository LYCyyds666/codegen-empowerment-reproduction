# Empowerment for Code Assistance

# Setup
Create the conda environment from the environment.yml.\
Change directories to LiveCodeBench and run:
```
pip install -e .
```

Then, change directories back to the repository root and run:
```
pip install -e .
```
All of the necessary packages should now be installed!\
The next step is to set the `CODEGEN_ROOT` environment variable. This should point to the directory where you would like to save models and intermediate files. I recommend putting this on the largest disk volume available, because a single training run can create >100GB of files. You can set this variable so that it always loads when activating your conda environment, as follows:
```
conda env config vars set CODEGEN_ROOT=path_to_codegen_root
```
I also recommend setting your `HF_HOME` environment variable to the same large volume, such as `/raid/users/my_user/.cache`.

# Training
All of the configuration yamls are kept in `codegen/configs`.
The assistant model training config is stored at `finetune_agent.yaml`.

I recommend running a quick run to verify that everything is working properly. To do this, set `num_train_examples: 10` in `finetune_agent.yaml`.
The command to launch a training job is simple:
```
python -m codegen.training.finetune_agent
```

# Dataset Construction
We used a variety of datasets, all based on MatrixStudio/Codeforces-Python_submissions, an open dataset on HuggingFace.
This dataset comes with actual human submissions, which we do not need.
All we use are the problems themselves.
The script in scripts/format_dataset_like_livecodebench.py will take this dataset and format it correctly.
Then, we roll out LLM solutions to these problems by running evaluate/rollout_on_training_dataset.py.
Make sure to update the configs appropriately with the correct dataset at each step.
The script scripts/set_human_code_in_ds_from_evaluate.py will take those rollouts and add them back into the dataset.
The script scripts/construct_logit_threshold_ds.py will add in a column for the completions that are the most empowering, by computing the longest completion whose likelihood is above a threshold.
