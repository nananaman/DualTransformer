''' Translate input text with trained model. '''

import argparse
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from nltk.translate import bleu_score

from dataset import collate_fn, TranslationDataset
from transformer.Translator import Translator
from preprocess import read_instances_from_file, convert_instance_to_idx_seq
from transformer.Models import Transformer
import transformer.Constants as Constants
from train import translate, prepare_dataloaders


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model .pt file')
    parser.add_argument('-src', required=True,
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', default='data/preprocessedData',
                        help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5,
                        help='Beam size')
    parser.add_argument('-batch_size', type=int, default=30,
                        help='Batch size')
    parser.add_argument('-n_best', type=int, default=1,
                        help="""If verbose is set, will output the n_best
                        decoded sentences""")
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    output = opt.output

    # Loading Dataset
    data = torch.load(opt.vocab)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    training_data, validation_data = prepare_dataloaders(data, opt)
    test_loader = validation_data
    # test_loader = training_data

    checkpoint = torch.load(opt.model)
    opt = checkpoint['settings']
    model = Transformer(
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

    device = torch.device('cuda' if opt.cuda else 'cpu')
    for key in list(checkpoint['model'].keys()):
        checkpoint['model'][key[7:]] = checkpoint['model'].pop(key)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)

    translator = Translator(opt, model)

    references = []
    hypotheses = []

    with open(output, 'w') as f:
        for batch in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            # prepare data
            src_seq, src_pos, tgt_seq, tgt_pos = map(
                lambda x: x.to(device), batch)
            tgt_seq_hyp, _ = translate(
                translator, src_seq, src_pos, Constants.BOS_SRC)
            src_seq_hyp, _ = translate(
                translator, tgt_seq, tgt_pos, Constants.BOS_TGT)

            for i in range(len(src_seq)):
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
                src = [test_loader.dataset.src_idx2word[idx.item()]
                       for idx in src]
                tgt = [test_loader.dataset.tgt_idx2word[idx.item()]
                       for idx in tgt]
                src_hyp = [test_loader.dataset.src_idx2word[idx.item()]
                           for idx in src_hyp]
                tgt_hyp = [test_loader.dataset.tgt_idx2word[idx.item()]
                           for idx in tgt_hyp]
                src_line = 'src    : ' + ''.join(src)
                tgt_line = 'tgt    : ' + ''.join(tgt)
                src_hyp_line = 'src_hyp: ' + ''.join(src_hyp)
                tgt_hyp_line = 'tgt_hyp: ' + ''.join(tgt_hyp)
                pred_line = src_line + '\n' + tgt_line + '\n' + \
                    src_hyp_line + '\n' + tgt_hyp_line + '\n\n'
                f.write(pred_line)

                references.extend([src])
                references.extend([tgt])
                hypotheses.extend([src_hyp])
                hypotheses.extend([tgt_hyp])

                if i > 20:
                    break
            break

    # calc bleu
    bleu = bleu_score.corpus_bleu(
            references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)
    print('[Info] BLEU : {}'.format(bleu))
    print('[Info] Finished.')


if __name__ == "__main__":
    main()
