# A Deep Metric Learning Approach to Account Linking

The code in this repository may be used to reproduce the author ID
results found in the article:

A Deep Metric Learning Approach to Account Linking. NAACL (2021)\
Aleem Khan, Elizabeth Fleming, Noah Schofield, Marcus Bishop, and Nicholas Andrews\
[[arxiv]](https://arxiv.org/abs/2105.07263) [[aclweb]](https://www.aclweb.org/anthology/2021.naacl-main.415/) [[bib]](https://www.aclweb.org/anthology/2021.naacl-main.415.bib)

Please cite the paper above if you find this code or ideas from the
paper useful in your own research.

## Dependencies

In order to run the software in this repository, the packages listed in 
`requirements.txt` must be installed. One way to do this is to make a 
conda environment (`conda create --name py --python=3.8`), activate it, 
and run `pip install -r requirements.txt`.

Including the location of this repository in your python path allows you 
to execute `import aid` in python:

```bash
export PYTHONPATH=$PYTHONPATH:${PATH_TO_THIS_REPOSITORY}
```

## Running an experiment

All the files related to experiments with particular datasets are 
intended to be contained in corresponding sub-directories of `expts`. 
For example, the files needed to reproduce the results with the Reddit 
dataset described in the paper listed above are contained in the 
`expts/reddit` sub-directory.

To run the experiment, first fetch and unpack the data files

```bash
https://storage.googleapis.com/naacl21_account_linking/1mil.tar.gz
https://storage.googleapis.com/naacl21_account_linking/test_queries.tar.gz
https://storage.googleapis.com/naacl21_account_linking/test_targets.tar.gz
```

and save them onto your filesystem. Next, edit the configuration files 
in `million_user_configs` by updating the values of `--training_queries`, 
`--train_tfrecord_path`,`--valid_tfrecord_path` to the locations where 
you saved the data files.

Next, update the `JOBS_DIR` variable in `run_training.sh` to point to 
the location on your filesystem where output files should be written.

Finally, to run the experiment corresponding with one of the 
configuration files, say, the model availing of all features, using 
triplet loss, and varying episode lengths between one and sixteen posts, 
run the command:

```bash
./run_training.sh full_model.cfg
```

Here, `run_training.sh` is a wrapper for the main trainer
`scripts/fit.py` and its argument is a configuration file, which
consists of a sequence of command-line arguments, one per line. These
will override any defaults specified in `scripts/fit.py`. You may also
provide additional command-line arguments at the end of the command
above. If you run the `run_training.sh` script on a machine with a
GPU, the script should automatically recognize and use a CUDA device.

## Pretrained checkpoints

The pre-trained checkpoints below may be used to reproduce
experimental results from the paper:

Checkpoints:
```bash
https://storage.googleapis.com/naacl21_account_linking/full_checkpoint.tar.gz
https://storage.googleapis.com/naacl21_account_linking/text_time_checkpoint.tar.gz
```

Pass the `rank` or `link` flags to `scripts/fit.py` to run experiments
with these models.
