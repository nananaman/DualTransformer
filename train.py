import argparse
import time
import math
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate import bleu_score

import transformer.Constants as Constants
from transformer.Models import Transformer, Discriminator
from transformer.Optim import ScheduledOptim
from dataset import TranslationDataset, paired_collate_fn, collate_fn
from transformer.Translator import Translator


def prepare_dataloaders(data, opt):
    # Preparing DataLoader
    train_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['train']['src'],
            tgt_insts=data['train']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        TranslationDataset(
            src_word2idx=data['dict']['src'],
            tgt_word2idx=data['dict']['tgt'],
            src_insts=data['valid']['src'],
            tgt_insts=data['valid']['tgt']),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=paired_collate_fn)
    return train_loader, valid_loader


def cal_loss(pred, gold, smoothing):
    '''
    クロスエントロピー誤差の計算
    '''
    # tgtの整形
    gold = gold.contiguous().view(-1)

    if smoothing:
        # label smoothingを行う
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=Constants.PAD, reduction='sum')

    return loss


def cal_performance(pred, gold, smoothing=False):
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    # 正解数のカウント
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def noize(seqs):
    '''
    入力seqにドロップ、シャッフルを行う。
    :param seqs : 入力文
    :return : ノイズをかけた文
    '''
    p_wd = 0.1
    k = 3
    c_seqs = []
    for seq in seqs:
        # drop
        tmp = torch.rand(len(seq))
        mask = tmp < p_wd
        # 先頭と最後のタグ部分を除外
        mask[0] = 0
        index = np.where(seq == Constants.EOS)[0][0]
        mask[index:] = 0
        # マスク適用
        seq[mask] = Constants.UNK
        # shuffle
        # キーの列を生成
        q = np.arange(1, len(seq) - 1)
        # 0~self.kの乱数列を生成
        U = np.random.randint(k, size=len(seq) - 2)
        # 加算
        q = q + U
        # ソート
        for i in range(1, len(seq) - 2):
            tmp = np.array(q[i:i+k if i+k <
                             len(seq) - 2 else len(seq) - 2])
            # argmin
            t = np.argmin(tmp) + i
            seq[i], seq[t] = seq[t], seq[i]
        c_seqs.append(seq)
    c_seqs = torch.stack(c_seqs)
    return c_seqs


def auto_encoder(model, src_seq, src_pos, smoothing):
    # target
    gold = src_seq[:, 1:]

    # noise
    src_seq_noized = noize(src_seq)

    pred, z_src = model(src_seq_noized, src_pos, src_seq, src_pos)

    # calc loss
    loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
    non_pad_mask = gold.ne(Constants.PAD)
    n_word = non_pad_mask.sum().item()

    return loss, z_src, n_correct, n_word


def cross_domain(model, src_seq, src_pos, tgt_seq, tgt_pos, smoothing):
    # target
    gold = tgt_seq[:, 1:]

    # noise
    src_seq_noized = noize(src_seq)

    pred, z_src = model(src_seq_noized, src_pos, tgt_seq, tgt_pos)

    # calc loss
    loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
    non_pad_mask = gold.ne(Constants.PAD)
    n_word = non_pad_mask.sum().item()

    return loss, z_src, n_correct, n_word


def translate(translator, src_seq, src_pos, domain):
    src_word = Constants.BOS_SRC
    tgt_word = Constants.BOS_TGT
    if domain == Constants.BOS_TGT:
        src_word, tgt_word = tgt_word, src_word

    # s2t by previous model
    tgt_hyp, _ = translator.translate_batch(src_seq, src_pos, domain)
    tgt_hyp = [[tgt_word] + t_hyp[0] + [Constants.EOS] for t_hyp in tgt_hyp]
    tgt_seq_hyp, tgt_pos_hyp = collate_fn(tgt_hyp)

    return tgt_seq_hyp, tgt_pos_hyp


def discriminate(discriminator, z_list):
    loss = 0
    for domain, z in z_list:
        pred = discriminator(z)
        gold = torch.zeros_like(pred)
        if domain == Constants.BOS_SRC_WORD:
            gold += 1
        loss += F.binary_cross_entropy(pred, gold)

    return loss


def train_epoch(model, discriminator, training_data, optimizer, optimizer_d, opt, device, smoothing):
    model.train()
    discriminator.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)    ', leave=False):
        # prepare translator
        translator = Translator(opt, model, device)

        # prepare data
        src_seq, src_pos, tgt_seq, tgt_pos = map(lambda x: x.to(device), batch)
        tgt_seq_hyp, tgt_pos_hyp = translate(
            translator, src_seq, src_pos, Constants.BOS_SRC)
        src_seq_hyp, src_pos_hyp = translate(
            translator, tgt_seq, tgt_pos, Constants.BOS_TGT)
        tgt_seq_hyp = tgt_seq_hyp.to(device)
        tgt_pos_hyp = tgt_pos_hyp.to(device)
        src_seq_hyp = src_seq_hyp.to(device)
        src_pos_hyp = src_pos_hyp.to(device)

        # prepare counter
        n_correct, n_word = 0, 0

        # forward
        optimizer.zero_grad()
        optimizer_d.zero_grad()

        loss_auto_src, z_src_src, n_correct, n_word = auto_encoder(
            model, src_seq, src_pos, smoothing)
        # n_word_total += n_word
        # n_word_correct += n_correct

        loss_cd_src, z_tgt_src, n_correct, n_word = cross_domain(
            model, tgt_seq_hyp, tgt_pos_hyp, src_seq, src_pos, smoothing)
        n_word_total += n_word
        n_word_correct += n_correct

        loss_auto_tgt, z_tgt_tgt, n_correct, n_word = auto_encoder(
            model, tgt_seq, tgt_pos, smoothing)
        # n_word_total += n_word
        # n_word_correct += n_correct

        loss_cd_tgt, z_src_tgt, n_correct, n_word = cross_domain(
            model, src_seq_hyp, src_pos_hyp, tgt_seq, tgt_pos, smoothing)
        n_word_total += n_word
        n_word_correct += n_correct

        z_list = [
            [Constants.BOS_SRC_WORD, z_src_src],
            [Constants.BOS_TGT_WORD, z_tgt_src],
            [Constants.BOS_SRC_WORD, z_src_tgt],
            [Constants.BOS_TGT_WORD, z_tgt_tgt]]

        loss_adv = discriminate(discriminator, z_list)

        loss = loss_auto_src + loss_auto_tgt + loss_cd_src + loss_cd_tgt - loss_adv

        # backward
        loss.backward()

        # update params
        optimizer.step_and_update_lr()
        optimizer_d.step()

        # note keeping
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, discriminator, validation_data, opt, device):
    model.eval()
    discriminator.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    references = []
    hypotheses = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validation)  ', leave=False):
            # prepare translator
            translator = Translator(opt, model, device)

            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(
                lambda x: x.to(device), batch)
            tgt_seq_hyp, tgt_pos_hyp = translate(
                translator, src_seq, src_pos, Constants.BOS_SRC)
            src_seq_hyp, src_pos_hyp = translate(
                translator, tgt_seq, tgt_pos, Constants.BOS_TGT)

            for i, _ in enumerate(src_seq):
                # EOSでカット
                src = src_seq[i, 1:np.where(
                    src_seq[i] == Constants.EOS)[0][0]]
                tgt = tgt_seq[i, 1:np.where(
                    tgt_seq[i] == Constants.EOS)[0][0]]
                src_hyp = src_seq_hyp[i, 1:np.where(
                    src_seq_hyp[i] == Constants.EOS)[0][0]]
                tgt_hyp = tgt_seq_hyp[i, 1:np.where(
                    tgt_seq_hyp[i] == Constants.EOS)[0][0]]

                # idx2word
                src = [validation_data.dataset.src_idx2word[idx.item()]
                       for idx in src]
                tgt = [validation_data.dataset.tgt_idx2word[idx.item()]
                       for idx in tgt]
                src_hyp = [validation_data.dataset.src_idx2word[idx.item()]
                           for idx in src_hyp]
                tgt_hyp = [validation_data.dataset.tgt_idx2word[idx.item()]
                           for idx in tgt_hyp]
                references.extend([src])
                references.extend([tgt])
                hypotheses.extend([src_hyp])
                hypotheses.extend([tgt_hyp])

            tgt_seq_hyp = tgt_seq_hyp.to(device)
            tgt_pos_hyp = tgt_pos_hyp.to(device)
            src_seq_hyp = src_seq_hyp.to(device)
            src_pos_hyp = src_pos_hyp.to(device)

            # prepare counter
            n_correct, n_word = 0, 0

            # forward
            loss_auto_src, z_src_src, n_correct, n_word = auto_encoder(
                model, src_seq, src_pos, smoothing=False)
            n_word_total += n_word
            n_word_correct += n_correct

            loss_cd_src, z_tgt_src, n_correct, n_word = cross_domain(
                model, tgt_seq_hyp, tgt_pos_hyp, src_seq, src_pos, smoothing=False)
            n_word_total += n_word
            n_word_correct += n_correct

            loss_auto_tgt, z_tgt_tgt, n_correct, n_word = auto_encoder(
                model, tgt_seq, tgt_pos, smoothing=False)
            n_word_total += n_word
            n_word_correct += n_correct

            loss_cd_tgt, z_src_tgt, n_correct, n_word = cross_domain(
                model, src_seq_hyp, src_pos_hyp, tgt_seq, tgt_pos, smoothing=False)
            n_word_total += n_word
            n_word_correct += n_correct

            z_list = [
                [Constants.BOS_SRC_WORD, z_src_src],
                [Constants.BOS_TGT_WORD, z_tgt_src],
                [Constants.BOS_SRC_WORD, z_src_tgt],
                [Constants.BOS_TGT_WORD, z_tgt_tgt]]

            loss_adv = discriminate(discriminator, z_list)

            loss = loss_auto_src + loss_auto_tgt + loss_cd_src + loss_cd_tgt - loss_adv

            # note keeping
            total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    # calc bleu
    bleu = bleu_score.corpus_bleu(
        references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)
    return loss_per_word, accuracy, bleu


def train(model, discriminator, training_data, validation_data, optimizer, optimizer_d, device, opt):
    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch, loss, ppl, accuracy\n')
            log_vf.write('epoch, loss, ppl, accuracy\n')

    valid_accus = []
    valid_bleus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        # Train
        start = time.time()
        train_loss, train_accu = train_epoch(
            model, discriminator, training_data, optimizer, optimizer_d, opt, device, smoothing=opt.label_smoothing)
        print('  - (Training) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, '
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time() - start)/60))

        # Evaluate
        start = time.time()
        valid_loss, valid_accu, valid_bleu = eval_epoch(
            model, discriminator, validation_data, opt, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, bleu: {valid_bleu:3.3f}, '
              'elapse: {elapse:3.3f} min'.format(
                  ppl=math.exp(min(train_loss, 100)), accu=100*train_accu,
                  elapse=(time.time() - start)/60))

        valid_accus += [valid_accu]
        valid_bleus += [valid_bleu]

        model_state_dict = model.state_dict()
        discriminator_state_dict = discriminator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'discriminator': discriminator_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + \
                    '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                # if valid_accu >= max(valid_accus):
                if valid_bleu >= max(valid_bleus):
                    torch.save(checkpoint, model_name)
                    print('  - [INFO] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100*train_accu))
                log_vf.write('{epoch}, {loss: 8.5f}, {ppl: 8.5f}, {accu:3.3f}, {bleu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu, bleu=valid_bleu))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./data/preprocessedData')

    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=64)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=2048)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log', default='log')  # None
    parser.add_argument('-save_model', default='trained')  # None
    parser.add_argument('-save_mode', type=str,
                        choices=['all', 'best'], default='best')

    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")

    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-label_smoothing', action='store_true', default=True)

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    opt.d_word_vec = opt.d_model

    # Loading Dataset
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    # Preparing Model
    if opt.embs_share_weight:
        assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
            'The src/tgt word2idx table are different but asked to share word embedding.'

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    # device = torch.device('cpu')

    transformer = Transformer(
        opt.src_vocab_size,
        opt.tgt_vocab_size,
        opt.max_token_seq_len,
        tgt_emb_prj_weight_sharing=opt.proj_share_weight,
        emb_src_tgt_weight_sharing=opt.embs_share_weight,
        d_k=opt.d_k,
        d_v=opt.d_v,
        d_model=opt.d_model,
        d_word_vec=opt.d_word_vec,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        dropout=opt.dropout)

    discriminator = Discriminator(
        opt.d_model, 1024, opt.max_token_seq_len, device)

    #'''
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        transformer = nn.DataParallel(transformer)
    #    '''
    transformer.to(device)
    discriminator.to(device)

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, transformer.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    optimizer_d = optim.RMSprop(discriminator.parameters(), lr=5e-4)

    train(transformer, discriminator, training_data,
          validation_data, optimizer, optimizer_d, device, opt)


if __name__ == '__main__':
    main()
