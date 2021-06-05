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

import json
import os
import pickle
import sentencepiece

import argparse as ap
import numpy as np

from absl import logging
from aid.evaluation import author_linking
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def get_docs(args, sp):
    docs = []
    with open(args.tfidf_train_path, 'r') as inf:
        pbar = tqdm(total=args.tfidf_train_n)
        for line in inf:
            js = json.loads(line)
            for post in js['syms']:
                docs.append(sp.decode_ids(post))
                pbar.update(1)
                if len(docs) == args.tfidf_train_n:
                    return docs
    return docs


def get_tfidf_model(args, sp):
    model_path = 'tfidf_word' + f"_{args.tfidf_train_n}.pickle"
    if os.path.exists(model_path):
        logging.info(f"Loading existing model: {model_path}")
        with open(model_path, 'rb') as inf:
            return pickle.load(inf)

    vectorizer = TfidfVectorizer(norm='l2', ngram_range=(1, 1),
                                 analyzer='word', max_df=0.90,
                                 stop_words='english', lowercase=False,
                                 max_features=20000)

    logging.info("Fitting tfidf")
    docs = get_docs(args, sp)
    logging.info(f"Using {len(docs)} documents to fit TFIDF")
    vectorizer.fit(docs)

    logging.info(f"Serializing TFIDF model to {model_path}")
    with open(model_path, 'wb') as of:
        pickle.dump(vectorizer, of)

    return vectorizer


def main(args):
    logging.set_verbosity('info')
    q = open(args.queries, 'r').readlines()
    t = open(args.targets, 'r').readlines()

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(args.vocab_model_path)

    vectorizer = get_tfidf_model(args, sp)

    logging.info("Decoding queries..")
    queries = []
    query_labels = []
    for i in q:
        js = json.loads(i)
        query_str = ''
        for post in js['syms']:
            query_str += sp.decode_ids(post[:args.truncation])
            query_str += '\n'
        queries.append(query_str)
        query_labels.append(js['author_id'])

    assert len(queries) == len(query_labels)
    logging.info(f"{len(queries)} queries")

    logging.info("Decoding targets..")
    targets = []
    target_labels = []
    for i in t:
        js = json.loads(i)
        target_str = ''
        for post in js['syms'][:args.target_episode_len]:
            target_str += sp.decode_ids(post[:args.truncation])
            target_str += '\n'
        targets.append(target_str)
        target_labels.append(js['author_id'])

    assert len(targets) == len(target_labels)
    logging.info(f"{len(targets)} queries")

    logging.info("Transforming queries...")
    q_vec = vectorizer.transform(queries)
    logging.info("Transforming targets...")
    t_vec = vectorizer.transform(targets)

    q_labels = np.array(query_labels)
    t_labels = np.array(target_labels)
    logging.info(q_vec.shape)
    logging.info(t_vec.shape)

    logging.info("Performing evaluation:")

    metrics = author_linking(q_vec, q_labels, t_vec, t_labels, t_vec, t_labels,
                             n_jobs=args.n_jobs, metric='linear')
    logging.info(f"METRICS: {metrics}")


if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('--queries', required=True)
    p.add_argument('--targets', required=True)
    p.add_argument('--vocab-model-path', required=True)
    p.add_argument('--target-episode-len', default=4, type=int)
    p.add_argument('--n-jobs', default=16, type=int)
    p.add_argument('--tfidf-train-n', default=500000, type=int)
    p.add_argument('--tfidf-train-path', type=str)
    p.add_argument('--truncation', default=32, type=int,
                   help='Truncate text to this many subwords')
    args = p.parse_args()
    main(args)
