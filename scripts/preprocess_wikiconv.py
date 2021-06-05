import argparse as ap
import json
from tqdm import tqdm
from glob import glob
import pickle
import sentencepiece as spm
from datetime import datetime
from absl import logging


def build_empty_json_obj():
  out_map = {'author_id' : -1,
             'syms' : [],
             'hour' : [],
             'minute' : [],
             'day' : [],
             'action_type' : []}
  return out_map


def build_user_map(fs):

  user_map = {}
  for i in tqdm(fs):
    with open(i, 'r') as handle:
      for i in handle:
        j = json.loads(i)
        if 'user_text' not in j or 'timestamp' not in j or 'cleaned_content' not in j:
          continue
        user = j['user_text']
        stamp = j['timestamp']
        text = j['cleaned_content']
        #simple heuristics...ip addresses will be allowed to pass
        if 'BOT' in user or 'bot' in user or 'Bot' in user:
          continue
        d = {'text':text, 'time':stamp}
        if user in user_map:
          user_map[user].append(d)
        else:
          user_map[user] = [d]
  return user_map


def main(args):

  if args.build_map:
    logging.info("BUILDING USER MAP FROM WIKI FILES..")
    user_map = build_user_map(glob(args.input_paths))
  else:
    logging.info(f"LOADING USER MAP FROM {args.map_path}")
    user_map = pickle.load(open(args.map_path))

  logging.info("DATA LOADED")

  # years to use for query/target
  train = set([str(i) for i in range(2000, 2017)])
  test = set(['2017', '2018'])

  train_map = {}
  test_map = {}

  logging.info("BUILDING QUERY/DEV MAPs")
  for k, v in tqdm(user_map.items()):
    train_hits = []
    test_hits = []
    for m in v:
      yr = m['time'][:4]
      if yr in train:
        train_hits.append(m)
      else:
        test_hits.append(m)
      if len(train_hits) > 0:
        train_map[k] = train_hits
      if len(test_hits) > 0:
        test_map[k] = test_hits

  sp = spm.SentencePieceProcessor()
  logging.info(f"LOADING VOCAB FROM {args.vocab_path}")
  sp.Load(args.vocab_path)
  logging.info("WRITING QUERY/TARGET JSON FILES ...")
  targeted_keys = set()

  with open(args.dev_query_output, 'w') as query_handle:
    with open(args.dev_target_output, 'w') as target_handle:
      with open(args.training_query_output, 'w') as training_handle:
        aid = -1
        for k, v in tqdm(train_map.items()):
          if len(v) >= args.spam_count_filter:
            continue

          if len(v) >= args.min_dev_query_len and k in test_map:
            if len(test_map[k]) >= args.min_dev_target_len:

              aid += 1
              targeted_keys.add(k)

              train_out = build_empty_json_obj()
              train_out['author_id'] = aid

              test_out = build_empty_json_obj()
              test_out['author_id'] = aid


              for m in v:
                dt = datetime.strptime(m['time'], '%Y-%m-%d %H:%M:%S UTC')
                sym = sp.EncodeAsIds(m['text'])
                train_out['syms'].append(sym)
                train_out['hour'].append(dt.hour)
                train_out['minute'].append(dt.minute)
                train_out['day'].append(dt.day)
                train_out['action_type'].append(0)

              for m in test_map[k]:
                dt = datetime.strptime(m['time'], '%Y-%m-%d %H:%M:%S UTC')
                sym = sp.EncodeAsIds(m['text'])
                test_out['syms'].append(sym)
                test_out['hour'].append(dt.hour)
                test_out['minute'].append(dt.minute)
                test_out['day'].append(dt.day)
                test_out['action_type'].append(0)

              query_handle.write(json.dumps(train_out) + '\n')
              target_handle.write(json.dumps(test_out) + '\n')

              # sprinkle in some more targets
        logging.info("ADDING MORE TARGETS FOR NOISE...")
        for k, v in tqdm(test_map.items()):
          if len(v) > 1000:
            continue
          if k not in targeted_keys and len(v) >= 16:
            aid += 1
            test_out = build_empty_json_obj()
            test_out['author_id'] = aid

            for m in test_map[k]:
              dt = datetime.strptime(m['time'], '%Y-%m-%d %H:%M:%S UTC')
              sym = sp.EncodeAsIds(m['text'])
              test_out['syms'].append(sym)
              test_out['hour'].append(dt.hour)
              test_out['minute'].append(dt.minute)
              test_out['day'].append(dt.day)
              test_out['action_type'].append(0)
            target_handle.write(json.dumps(test_out) + '\n')

if __name__ == "__main__":
  p = ap.ArgumentParser()
  p.add_argument('--input_paths', type=str, help='Glob of all wiki data')
  p.add_argument('--vocab_path', type=str, help='Path to sentencepiece vocab')
  p.add_argument('--build_map', dest='build_map', action='store_true',
                 help='Is the author map build? Or shall we build ourselves')
  p.add_argument('--map_path', type=str, default=None, help='Path to build map, if we are not building')
  p.add_argument('--min_training_query_len', type=int, default=100,
                 help='How many posts to constitute a training query?')
  p.add_argument('--min_dev_query_len', type=int, default=16, help='Min episode len for dev experiment')
  p.add_argument('--min_dev_target_len', type=int, default=16, help='Min episode len for dev experiment')
  p.add_argument('--spam_count_filter', type=int, default=1000, help='Max count to filter out spam')
  p.add_argument('--training_query_output', type=str, default='/exp/akhan/train.jsonl',
                 help='Path for training json')
  p.add_argument('--dev_query_output', type=str, default='/exp/akhan/queries.jsonl',
                 help='Path for dev query json')
  p.add_argument('--dev_target_output', type=str, default='/exp/akhan/targets.jsonl',
                 help='Path for dev query json')

  args = p.parse_args()
  if not args.build_map and args.map_path is None:
    raise ValueError("A path to a built map must be specified if we are not building one")

  logging.set_verbosity(logging.INFO)
  main(args)