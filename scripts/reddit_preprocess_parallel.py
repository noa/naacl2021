#! /usr/bin/env python

# Copyright 2021 Johns Hopkins University. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""The input consists of raw data from a Reddit dump and we output
preprocessed data (features) to a JSON file. The JSON file may
then be converted to TFRecords as a separate step.

Note that for simplicity all preprocessing happens in-memory. This
limits the size of datasets that may be preprocessed using this
script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
from aid.features import F
from aid.features import FeatureConfig
from datetime import datetime
from glob import glob
from tqdm import tqdm

import os
import pickle
import pandas as pd
import multiprocessing as mp
import sentencepiece as spm

try:
    import ujson as json
except ImportError:
    print("Install the `ujson` pip package to speed things up.")
    import json

args = flags.FLAGS

flags.DEFINE_string('df', None, 'Glob pointing to all pickled DataFrames')
flags.DEFINE_string('ids', None, 'Glob pointing to all id files')
flags.DEFINE_integer('num_proc', 8, 'Number of processers to user for parallel processing')
flags.DEFINE_string('subreddit_path', None, 'Path to subreddit pickle')
flags.DEFINE_string('vocab_model_path', None, 'Path to vocab')
flags.DEFINE_string('dev_ids', None, 'Path to dev ids to filter out (avoid train/dev contamination)')

flags.DEFINE_string('output_dir', '.', 'Output directory')
flags.DEFINE_string('model_dir', '.', 'Model directory')
flags.DEFINE_string('json_filename', 'examples.json', 'Output JSON file name.')
flags.DEFINE_string('config', 'reddit.json', 'Experiment configuration')
flags.DEFINE_string('unk_subreddit', '<unk>', 'Name of unknown subreddit')
flags.DEFINE_string('model_prefix', 'model', 'Prefix for subword model files')
flags.DEFINE_string('model_type', 'unigram', 'Model type')
flags.DEFINE_float('character_coverage', 1.0, 'Character coverage')
flags.DEFINE_integer('input_sentence_size', 1000000,
                     'Number of sentences used to fit subword model')
flags.DEFINE_integer('pad_id', 0, 'Padding ID')
flags.DEFINE_integer('bos_id', -1, 'BOS ID')
flags.DEFINE_integer('eos_id', 1, 'EOS ID')
flags.DEFINE_integer('unk_id', 2, 'Unk ID')
flags.DEFINE_float('min_ascii_fraction', 0.75,
                   'Filter comments with less than this fraction of ASCII')
flags.DEFINE_integer('min_chars', 1, 'Minimum comment length')
flags.DEFINE_integer('min_subwords', 10, 'Minimum number of subwords')
flags.DEFINE_string('text_key', 'body', 'Column name for text field')
flags.DEFINE_string('subreddit_key', 'subreddit', 'Column name for subreddit')
flags.DEFINE_integer('n_to_print', 1, 'Number of comments to print to console')
flags.DEFINE_string('sample_file_path', None, 'Path to JSON lines file with sample indices')


flags.mark_flags_as_required(['df', 'subreddit_path', 'vocab_model_path', 'dev_ids'])


def get_hour_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).hour


def get_day_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).weekday()


def get_minute_from_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).minute


def keep_comment(text, min_ascii_fraction=0.75, min_length=1):
    """ For purposes of vocabulary creation ignore non-ascii documents """
    len_total = len(text)
    if len_total < min_length:
        return False
    len_ascii = sum(c.isalpha() for c in text)
    frac_ascii = float(len_ascii) / float(len_total)
    if frac_ascii < min_ascii_fraction:
        return False
    return True


def fit_author_vocab(dfs):
    """ Keep track of author IDs """
    logging.info("Fitting full author vocab")
    author_map = {}
    for pickle_file in tqdm(dfs):
        df = pd.read_pickle(pickle_file)
        for i, a in enumerate(set(df['author'])):
            assert a not in author_map
            num_auths = len(author_map)
            author_map[a] = num_auths

    author_map_path = os.path.join(
        args.output_dir, 'authors.pickle')
    if os.path.exists(author_map_path):
        logging.info(f"Using existing author map: {author_map_path}")
        return

    logging.info(f"{len(author_map)} authors")
    with open(author_map_path, 'wb') as f:
        pickle.dump(author_map, f)


def print_examples(df, print_if_less_than=15):
    config = FeatureConfig.from_json(args.config)
    model_path = os.path.join(
        args.model_dir,
        f"{config.num_symbols}_{args.model_type}.model")
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    logging.info(f"Piece size: {sp.GetPieceSize()}")
    n_printed = 0
    for index, row in df.iterrows():
        raw_text = row[args.text_key]
        if len(raw_text.split()) < print_if_less_than:
            logging.info(raw_text)
            pieces = sp.EncodeAsPieces(raw_text)
            logging.info(" ".join(
                [f"{piece}:{sp.PieceToId(piece)}" for piece in pieces]))
            n_printed += 1
        if n_printed > args.n_to_print:
            break


def maybe_load_sample_file():
    if args.sample_file_path is None:
        return None
    samples = {}
    logging.info(f"Loading sample file: {args.sample_file_path}")
    with open(args.sample_file_path) as fh:
        for line in fh:
            sample = json.loads(line)
            samples[sample['author']] = sample
    assert samples
    return samples


def write_json(df, idx, ids_file, dev_ids):
    json_path = os.path.join(args.output_dir, f"queries{idx}.json")
    if os.path.exists(json_path):
        logging.info(f'{json_path} exists; delete to remake')
        return
    samples = maybe_load_sample_file()

    sp = spm.SentencePieceProcessor()

    sp.Load(args.vocab_model_path)

    with open(args.subreddit_path, 'rb') as fh:
        logging.info(f"Loading subreddit map: {args.subreddit_path}")
        subreddit_map = pickle.load(fh)
    author_map_path = os.path.join(
        args.output_dir, 'authors.pickle')
    with open(author_map_path, 'rb') as fh:
        logging.info(f"Loading author map: {author_map_path}")
        author_map = pickle.load(fh)
    logging.info(f"Writing preprocessed data to: {json_path}")
    N = len(open(ids_file).readlines())
    with open(json_path, 'w') as fout, \
            open(ids_file, 'r') as ids_file:
        for line in tqdm(ids_file, total=N):
            comment_ids = line.split()
            first_id = comment_ids[0]
            author = df.loc[first_id]['author']
            if samples:
                if author not in samples:
                    continue
                sample = samples[author]
                assert len(comment_ids) == sample['num_actions_total']
                start_index = sample['start_index']
                length = sample['episode_length']
                comment_ids = comment_ids[start_index:start_index + length]
                assert len(comment_ids) == length
            history = {
                F.SYMBOLS.value: [],
                F.HOUR.value: [],
                F.MINUTE.value: [],
                F.DAY.value: [],
                F.ACTION_TYPE.value: [],
                F.AUTHOR_ID.value: author_map[author]
            }
            for id_ in comment_ids:
                if id_ in dev_ids:
                    continue
                comment = df.loc[id_]
                history[F.SYMBOLS.value].append(sp.EncodeAsIds(comment['body']))
                history[F.HOUR.value].append(
                    get_hour_from_timestamp(comment['created_utc']))
                history[F.MINUTE.value].append(
                    get_minute_from_timestamp(comment['created_utc']))
                history[F.DAY.value].append(
                    get_day_from_timestamp(comment['created_utc']))
                subreddit_index = subreddit_map[args.unk_subreddit]
                if comment['subreddit'] in subreddit_map:
                    subreddit_index = subreddit_map[comment['subreddit']]
                history[F.ACTION_TYPE.value].append(subreddit_index)

            fout.write(json.dumps(history) + '\n')


def load_dev_ids():
    ids = open(args.dev_ids, 'r').readlines()
    id_set = set()
    for i in ids:
        for j in i.split():
            id_set.add(j)
    return id_set

def parallel_helper(input_tuple):
    idx = input_tuple[0]
    df_path = input_tuple[1]
    id_path = df_path.split('.')[0]
    df = pd.read_pickle(df_path)
    dev_ids = load_dev_ids()
    write_json(df, idx, id_path, dev_ids)

def main(argv):
    logging.info(f"Output directory: {args.output_dir}")

    all_pickles = glob(args.df)
    os.makedirs(args.output_dir, exist_ok=True)
    fit_author_vocab(all_pickles)
    with mp.Pool(args.num_proc) as pool:
        pool.map(parallel_helper, enumerate(all_pickles))



if __name__ == "__main__":
    app.run(main)
