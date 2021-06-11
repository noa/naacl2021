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
conda environment (`conda create --prefix ./env python=3.8`), activate it (`conda activate ./env`), 
and run `pip install -r requirements.txt`.

Then, include the location of this repository in your Python path:

```bash
export PYTHONPATH=$PYTHONPATH:${PATH_TO_THIS_REPOSITORY}
```

will enable you to import the `aid` (author ID) package.

## Running an experiment

To run an experiment, first fetch and unpack the data files

```bash
https://storage.googleapis.com/naacl21_account_linking/1mil.tar.gz
https://storage.googleapis.com/naacl21_account_linking/dev_queries.tar.gz
https://storage.googleapis.com/naacl21_account_linking/dev_targets.tar.gz
https://storage.googleapis.com/naacl21_account_linking/test_queries.tar.gz
https://storage.googleapis.com/naacl21_account_linking/test_targets.tar.gz
```

and save them onto your filesystem, specifically into separate
subdirectories `train`, `dev`, and `test`, to prevent name
conflicts. Unpack the files and then run the `json2tf.py` script to
produce sharded protocol buffer files for training. For example:

```bash
python json2tf.py --json /path/to/unpacked/json --tf /path/to/output/tfrecords --config /path/to/reddit/json/config`
```

> NOTE: This will take several hours for the `1mil.tar.gz` files.

Next, update the `JOBS_DIR` variable in `run_training.sh` to point to 
the location on your filesystem where output files should be written.

Finally, to run the experiment corresponding with one of the 
configuration files, say, the model availing of all features, using 
triplet loss, and varying episode lengths between one and sixteen posts, 
run the command:

```bash
./run_training.sh full_model.cfg --train_records=<TRAINING DATA> --train_tfrecord_path=<VALIDATION QUERIES> --valid_tfrecord_Path=<VALIDATION TARGETS>
```

> NOTE: When specifying the data location, use a wildcard to specify
  the set of all sharded files. For example, `dev/queries*`, which you
  might need to quote to prevent your shell from expanding it.

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

Pass the `rank` or `link` flags to `scripts/fit.py` to evaluate the
performance of these checkpoints. See `expts/reddit/test.sh`.
