"""
    Copyright 2022 Expedia, Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import pandas as pd
import numpy as np
from baselines.markov_baseline import MarkovBaseline
import tensorflow as tf
from metrics.hits_at_k import HitsAtK
import argparse


def load_dataset(path_to_data, clicks_col_name="clicks", is_eval=False):
    dataset = pd.read_csv(path_to_data, sep="\t")

    # join the last click
    if is_eval:
        dataset[clicks_col_name] = dataset[clicks_col_name] + "," + dataset["clicks_last"].apply(str)
    dataset[clicks_col_name] = dataset[clicks_col_name].apply(lambda x: [int(c) for c in x.split(",")])
    vocabulary = np.unique(np.concatenate(dataset[clicks_col_name].values).flatten())

    return dataset, vocabulary


def decode_clicks_file(line, lookup_table, is_eval=True):
    line_split = tf.strings.split(line, sep="\t")
    if is_eval:
        clicks = tf.strings.join([line_split[-2], line_split[-1]], separator=',')
    else:
        clicks = line_split[-1]
    clicks = tf.strings.split(clicks, sep=",")
    clicks = tf.strings.to_number(clicks, out_type=tf.int64)

    return {"features": lookup_table.lookup(clicks[-2]), "label": lookup_table.lookup(clicks[-1])}


def _input_fn(path_to_data, vocab_lookup_table, compression_type=None, batch_size=256, is_eval=True):
    dataset = tf.data.TextLineDataset(path_to_data, compression_type=compression_type) \
        .skip(1) \
        .map(lambda x: decode_clicks_file(x, vocab_lookup_table, is_eval)) \
        .repeat(1) \
        .batch(batch_size)

    return dataset


def evaluate(path_to_train, path_to_validation, k=5, batch_size=256):

    train_data, vocab_train = load_dataset(path_to_train)
    validation_data, vocab_val = load_dataset(path_to_validation, clicks_col_name="clicks_no_last", is_eval=True)

    full_vocab = np.unique(np.concatenate((vocab_train, vocab_val)))
    print("Vocabulary size {}".format(full_vocab.shape))

    mb = MarkovBaseline(window=1, vocabulary=full_vocab)
    mb.build_hotel2index()
    mb.fit(train_data["clicks"].values)

    init = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(mb.vocabulary, dtype=tf.int64),
                                               values=tf.constant([mb.hotel2index[h] for h in mb.vocabulary],
                                                                  dtype=tf.int64))
    table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)

    hits_at_k = HitsAtK(k=k)

    for batch in _input_fn(path_to_validation, table, compression_type='GZIP', batch_size=batch_size):
        logits = mb.predict_scores(batch["features"])
        hits_at_k.update_state(batch["label"], logits)

    print("Hits@5: {}".format(hits_at_k.result()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment setup")

    parser.add_argument("--train_path", action="store", help="Path to the training data")
    parser.add_argument("--eval_path", action="store", help="Path to the evaluation data")
    parser.add_argument('--hits_at_k', default=5, type=int)  # cutoff for the hits@k metric.
    parser.add_argument('--batch_size', default=256, type=int)
    params = parser.parse_args()

    evaluate(params.train_path, params.eval_path, params.hits_at_k, params.batch_size)
