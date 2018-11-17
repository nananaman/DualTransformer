import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer
import transformer.Constants as Constants
import torch.nn.functional as F


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    '''
    Positional Encoding
    param n_position :
    param d_hid : ベクトル化したwordの次元数(隠れ層の次元？)
    param padding_idx :
    return
    '''
    def cal_angle(position, hid_idx):
        # PEのsin or cosの中身を計算
        # 2iを参照するので、切り捨て演算により偶奇問わず2iにする
        # d_hidはd_modelを指す
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        # 各次元?について計算
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    # 各posについて計算
    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    # 交互にsinとcosをとる
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    '''
    param seq_k : key seq
    param seq_q : 
    return padding_mask : マスク
    '''
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)  # b * lk
    # 次元を追加して拡張する
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b * lq * lk

    return padding_mask


def get_subsequent_mask(seq):
    '''
    Decode時に未知の単語をDecoderが受け取らないようにするためのマスク
    param seq : 入力seq
    '''
    sz_b, len_s = seq.size()
    # diagonal=1, 各要素が1, len_s * len_sの上三角行列を作成
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(
        0).expand(sz_b, -1, -1)  # b * ls * ls

    return subsequent_mask


class Encoder(nn.Module):
    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        '''
        Encoderの初期化
        param n_src_vocab : srcの語彙数
        param len_max_seq : 文の長さの最大値
        param d_word_vec : ベクトル化したwordの次元数
        param n_layers : layerの数N
        param n_head :
        param d_k :
        param d_v :
        param d_model :
        param d_inner :
        dropout : dropout rate
        '''

        n_position = len_max_seq + 1

        # src_word用のembedding
        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # pos用のembedding
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        # Encode Layerをスタック
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):
        '''
        forward
        param src_seq : srcの文
        param src_pos : srcのpos
        param return_attns : attentionを返すフラグ
        return : Encoderの出力
        '''
        enc_slf_attn_list = []

        # マスクの準備
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # forward
        # seqのembeddingしたものと、posをembeddingしたものを加算
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        # Encoderに通す
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            # attentionも返す
            if return_attns:
                return enc_output, enc_slf_attn_list
            return enc_output,


class Decoder(nn.Module):
    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # マスクの準備
        non_pad_mask = get_non_pad_mask(tgt_seq)

        # tgtをDecoderに入れる際、未知の部分を入れないようにするためのマスク
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        #
        slf_attn_mask_keypad = get_attn_key_pad_mask(
            seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                # attentionを返すためのリスト
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            # attentionも返す
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    def __init__(
            self,
            n_src_vocab, n_tgt_vocab, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        # Encoderの初期化
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # Decoderの初期化
        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        # output直前のLinear
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        # Xavier
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, "To facilitate the residual connections, thedimensions of all module outputs shall be the same."

        if tgt_emb_prj_weight_sharing:
            # output直前のLinearの重み共有
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            assert n_src_vocab == n_tgt_vocab, "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):
        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        # Encoderに通す
        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale
        return seq_logit.view(-1, seq_logit.size(2)), enc_output


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, max_word_seq_len, device):
        super().__init__()
        self.input_size = input_size
        self.max_word_seq_len = max_word_seq_len
        self.device = device

        self.i = nn.Linear(input_size * self.max_word_seq_len, hidden_size)
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, 1)

    def forward(self, x):
        max_len = x.size(1)
        batch_size = x.size(0)
        x = x.contiguous().view(batch_size, -1)
        if max_len < self.max_word_seq_len:
            z = torch.zeros(batch_size, (self.max_word_seq_len - max_len) * self.input_size, dtype=torch.float, device=self.device)
            x = torch.cat([x, z], dim=1)
        
        x = F.leaky_relu(self.i(x))
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        x = F.leaky_relu(self.l3(x))
        x = torch.sigmoid(self.o(x))

        return x
