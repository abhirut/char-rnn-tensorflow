#!/usr/bin/env python

from __future__ import print_function

import argparse
import unicodecsv as csv
import math
import os
from six.moves import cPickle


from six import text_type


parser = argparse.ArgumentParser(
                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--save_dir', type=str, default='save',
                    help='model directory to store checkpointed models')
parser.add_argument('--csv_path', type=str, default='../data/askubuntu_seg_cand_test_pooja_test_20190129_all_labels.csv',
                    help='csv containing segments to score')

args = parser.parse_args()

import tensorflow as tf
from model import Model

def get_prob(save_dir, text):
    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            log_prob = model.get_log_prob(sess, vocab, text)
            return math.exp(log_prob)

def score_segments(save_dir, csv_path):
    out_csv_path = csv_path.replace('.csv', '_probs.csv')

    with open(os.path.join(save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            with open(out_csv_path, 'w') as fw:
                csvw = csv.writer(fw)
                with open(csv_path) as fr:
                    csvr = csv.reader(fr)
                    row_num = 0
                    for row in csvr:
                        row_num += 1
                        if row_num == 1:
                            row.append('LM Prob')
                            csvw.writerow(row)
                            continue
                        print(row_num)
                        text = row[1]
                        log_prob = model.get_log_prob(sess, vocab, text)
                        row.append(log_prob)
                        csvw.writerow(row)



if __name__ == "__main__":
    #print(get_prob(args.save_dir, "What the hell"))
    score_segments(args.save_dir, args.csv_path)
