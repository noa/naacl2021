#! /usr/bin/env python

# Copyright 2019 Johns Hopkins University. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse as ap
import sentencepiece as spm
import pickle
from collections import Counter
from string import whitespace

from aid.generators import reddit_json_generator
from reddit_utils import keep_comment
from reddit_utils import parse_reddit_post

if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument('output_dir', help='Directory to output corpus, model and vocabulary files to.')
    p.add_argument('input_paths', nargs='+', help='Path(s) to compressed JSON file(s)')
    p.add_argument('--n_subreddit', type=int, default=1024)
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--model-type', type=str, default='unigram')
    p.add_argument('--character-coverage', type=float, default=1.0)
    p.add_argument('--input-sentence-size', type=int, default=1000000)
    p.add_argument('--pad-id', type=int, default=0)
    p.add_argument('--bos-id', type=int, default=-1)
    p.add_argument('--eos-id', type=int, default=1)
    p.add_argument('--unk-id', type=int, default=2)
    args = p.parse_args()
    n_written = 0
    n_total = 0

    if not os.path.isdir(args.output_dir):
        print("Making output directory: {}".format(args.output_dir))
        os.mkdir(args.output_dir)

    corpus_path = os.path.join(args.output_dir,
                               '{}_corpus.txt'.format(args.vocab_size))
    subreddit_path = os.path.join(args.output_dir,
                                  '{}_subreddits.txt'.format(args.n_subreddit))

    if not os.path.isfile(corpus_path) or not os.path.isfile(subreddit_path):
        with open(corpus_path, 'w') as f, open(subreddit_path, 'w') as g:
            for path in args.input_paths:
                print('[{} lines written of {}] Processing {}'.format(n_written, n_total, path))
                for comment in reddit_json_generator(path):
                    n_total += 1
                    post = parse_reddit_post(comment)
                    text = post.text
                    if keep_comment(text):
                        g.write(post.action + '\n')
                        text = text.replace('\n', ' ')
                        f.write(text + '\n')
                        n_written += 1
        print("Wrote {} lines to {}".format(n_written, corpus_path))

    print("Creating SubReddit map")
    subreddit_map_path = os.path.join(
        args.output_dir,
        '{}_subreddits.pickle'.format(args.n_subreddit))
    with open(subreddit_path) as f, open(subreddit_map_path, 'wb') as g:
        counts = Counter([x.rstrip() for x in f.readlines()])
        most_common = counts.most_common()
        print("Most common subreddits:")
        for sr, count in most_common[:5]:
            print("{} {}".format(sr, count))
        output_map = {}
        for i, sr in enumerate([x for x, _ in most_common[:args.n_subreddit - 1]]):
            output_map[sr] = i
        assert '<unk>' not in output_map
        output_map['<unk>'] = len(output_map)
        assert len(output_map) == args.n_subreddit
        pickle.dump(output_map, g)

    print("Creating Vocabulary with SentencePiece")
    model_prefix = os.path.join(args.output_dir,
                                "{}_{}".format(args.vocab_size, args.model_type))
    trainer_args = [
        '--input={}'.format(corpus_path),
        '--model_prefix={}'.format(model_prefix),
        '--vocab_size={}'.format(args.vocab_size),
        '--model_type={}'.format(args.model_type),
        '--character_coverage={}'.format(args.character_coverage),
        '--input_sentence_size={}'.format(args.input_sentence_size),
        '--shuffle_input_sentence=true',
        '--pad_id={}'.format(args.pad_id),
        '--eos_id={}'.format(args.eos_id),
        '--bos_id={}'.format(args.bos_id),
        '--unk_id={}'.format(args.unk_id)
    ]
    spm.SentencePieceTrainer.Train(' '.join(trainer_args))

    sp = spm.SentencePieceProcessor()
    sp.Load("{}.model".format(model_prefix))
    print("Piece size: {}".format(sp.GetPieceSize()))
    n_to_print = 5
    n_printed = 0
    for comment in reddit_json_generator(args.input_paths[0]):
        raw_text = comment['body']
        if len(raw_text.split()) < 15:
            print(raw_text)
            pieces = sp.EncodeAsPieces(raw_text)
            print(" ".join(["{}:{}".format(piece, sp.PieceToId(piece)) for piece in pieces]))
            n_printed += 1
        if n_printed > n_to_print:
            break
