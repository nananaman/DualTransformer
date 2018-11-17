import argparse
import pandas as pd
import MeCab
import numpy as np
import torch

import transformer.Constants as Constants


def mecab_split(sentence, mecabTagger):
    node = mecabTagger.parseToNode(sentence)
    node = node.next
    words = []
    while node:
        word = node.surface
        node = node.next
        words.append(word)

    return words


def read_instances_from_file(inst_file, max_sent_len):
    # データ読み込み
    df = pd.read_excel(inst_file)

    word_insts = []
    trimmed_sent_count = 0

    # MeCabの準備
    mecabTagger = MeCab.Tagger("-Ochasen")
    mecabTagger.parse("")

    # リスト作成
    src_word_insts = []
    tgt_word_insts = []

    for index, row in df.iterrows():
        cmplx = row['#日本語(原文)']
        smply = row['#やさしい日本語']
        eng = row['#英語(原文)']

        # split
        src_words = mecab_split(cmplx, mecabTagger)
        tgt_words = mecab_split(smply, mecabTagger)
        
        if len(src_words) > max_sent_len:
            trimmed_sent_count += 1
        if len(tgt_words) > max_sent_len:
            trimmed_sent_count += 1

        # <eos>と<bos>を追加
        src_word_insts += [[Constants.BOS_SRC_WORD] +
                src_words + [Constants.EOS_WORD]]

        tgt_word_insts += [[Constants.BOS_TGT_WORD] +
                tgt_words + [Constants.EOS_WORD]]

    print('[Info] Get {} instances from {}'.format(len(src_word_insts), inst_file))
    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the ma sentence length {}.'
                .format(trimmed_sent_count, max_sent_len))

    return src_word_insts, tgt_word_insts

def build_vocab_idx(word_insts, min_word_count):
    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    # tagを登録
    word2idx = {
        Constants.BOS_SRC_WORD: Constants.BOS_SRC,
        Constants.BOS_TGT_WORD: Constants.BOS_TGT,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    # カウンターの初期化
    word_count = {w: 0 for w in full_vocab}

    # wordの出現回数をカウント
    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            # min_word_countより多く出現していれば登録
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            # 少なければ別にカウント
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {}'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print('[Info] Ignored word count = {}'.format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_data', default="./data/preprocessedData")
    parser.add_argument('-data', default= "./data/T15-2018.2.28.xlsx")
    parser.add_argument('-train_valid_ratio', default=0.8)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=1)
    parser.add_argument('-share_vocab', action='store_true')

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len

    # データ読み込み
    src_word_insts, tgt_word_insts = read_instances_from_file(opt.data, opt.max_word_seq_len)
    train_src_word_insts, valid_src_word_insts = np.split(src_word_insts, [int(len(src_word_insts) * opt.train_valid_ratio)])
    train_tgt_word_insts, valid_tgt_word_insts = np.split(tgt_word_insts, [int(len(tgt_word_insts) * opt.train_valid_ratio)])

    # build vocabulary
    if opt.share_vocab:
        print('[Info] Pre-defined vocabulary for source and target.')
        word2idx = build_vocab_idx(
            train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
        src_word2idx = tgt_word2idx = word2idx
    else:
        print('[Info] Build vocabulary for source.')
        src_word2idx = build_vocab_idx(
            train_src_word_insts, opt.min_word_count)
        print('[Info] Build vocabulary for target.')
        tgt_word2idx = build_vocab_idx(
                train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(
        train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(
        valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(
        train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(
        valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
