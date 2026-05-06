# Built on stformer (https://github.com/csh3/stFormer).
# Some classes and functions are modified from scGPT (https://github.com/bowang-lab/scGPT) and scFoundation (https://github.com/biomap-research/scFoundation), 

from typing import Dict, Mapping, Optional, Any, Union, Callable
import itertools
import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones, _get_seq_len, _detect_is_causal_mask
from flash_attn.bert_padding import pad_input
from flash_attention import FlashMHA
from loss import grad_reverse

def SimSiamMLP(dim=768, projection_size=768, hidden_size=768*2):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
    )

class TransformerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        do_cls: bool = False,
        do_gcl: bool = False,
        do_lrc: bool = False,
        do_dab: bool = False,
        nlayers_cls: int = 3,
        n_cls: int = 2,
        nlayers_gcl: int = 3,
        n_gcl: int = 2,
        nlayers_lrc: int = 3,
        n_lrc: int = 2,
        dropout: float = 0.1,
        cell_emb_style: str = "max-pool",
        pre_norm: bool = False,
        scfoundation_token_emb1: Any = None,
        scfoundation_token_emb2: Any = None,
        scfoundation_pos_emb1: Any = None,
        scfoundation_pos_emb2: Any = None,
        num_batch_labels: Optional[int] = None, 
        use_batch_labels: bool = True, 
        do_mvc: bool = False, 
        mvc_decoder_style: str = "inner product", 
        explicit_zero_prob: bool = False, 
    ):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.do_dab = do_dab
        self.cell_emb_style = cell_emb_style
        self.pre_norm = pre_norm
        self.scfoundation_token_emb1 = scfoundation_token_emb1
        self.scfoundation_token_emb2 = scfoundation_token_emb2
        self.scfoundation_pos_emb1 = scfoundation_pos_emb1
        self.scfoundation_pos_emb2 = scfoundation_pos_emb2
        self.create_mlp_fn_pj = SimSiamMLP(dim=768, projection_size=768, hidden_size=768*4)
        self.create_mlp_fn_pd = SimSiamMLP(dim=768, projection_size=768, hidden_size=768*4)
        self.explicit_zero_prob = explicit_zero_prob
        self.use_batch_labels = use_batch_labels
        
        if cell_emb_style not in ["avg-pool", "max-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")

        decoder_layers = FlashTransformerDecoderLayer(
            d_model,
            nhead,
            d_hid,
            dropout,
            batch_first=True,
            norm_first=self.pre_norm,
        )

        # Batch Encoder
        if use_batch_labels and num_batch_labels != 0:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)
        else:
            self.batch_encoder = None

        self.transformer_decoder = BiasedTransformerDecoder(decoder_layers, nlayers)
        
        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        if do_cls:
            self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_gcl:
            self.gcl_decoder = GclDecoder(d_model, n_gcl, nlayers=nlayers_gcl)
        if do_lrc:
            self.lrc_decoder = LRCDecoder(d_model, n_lrc, nlayers=nlayers_lrc)
        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
    ) -> Tensor:
        src = self.scfoundation_pos_emb1(src)
        values = self.scfoundation_token_emb1(torch.unsqueeze(values, 2).float(), output_weight = 0)
        output = src + values
        return output  # (batch, seq_len, embsize)

    def _decode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor,
        cross_attn_bias: Tensor,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        batch_labels: Optional[Tensor] = None,  
    ) -> Tensor:
        src = self.scfoundation_pos_emb2(src)
        self.cur_gene_token_embs = src 
        values = self.scfoundation_token_emb2(torch.unsqueeze(values, 2).float(), output_weight = 0)
        tgt = src + values

        # if self.batch_encoder is not None and batch_labels is not None:
        #     batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        #     self.batch_emb = batch_emb #hongb
        #     tgt += batch_emb.unsqueeze(1).repeat(1, tgt.shape[1], 1)

        output = self.transformer_decoder(tgt, memory, cross_attn_bias, tgt_key_padding_mask=src_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask, tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal)
        return output

    def _pad_information_of_split_input(self, encoder_feature_lens: Tensor, ligand=False, max_seqlen=None):
        if max_seqlen is None:
            max_seqlen = encoder_feature_lens.max().item()
        if ligand:
            total_cell_num = encoder_feature_lens.size(0)
        else:
            total_cell_num = (encoder_feature_lens>0).sum().item()
        key_padding_mask = torch.zeros((total_cell_num, max_seqlen), dtype=torch.bool, device=encoder_feature_lens.device)
        if ligand:
            for i,val in enumerate(encoder_feature_lens):
                key_padding_mask[i, val:] = True
        else:
            for i,val in enumerate(encoder_feature_lens[encoder_feature_lens>0]):
                key_padding_mask[i, val:] = True
        indices = (~key_padding_mask.view(-1)).nonzero(as_tuple=True)[0]
        return indices, total_cell_num, max_seqlen, key_padding_mask

    def forward(
        self,
        encoder_src: Tensor,
        encoder_values: Tensor,
        encoder_src_key_padding_mask: Tensor,
        decoder_src: Tensor,
        decoder_values: Tensor,
        decoder_src_key_padding_mask: Tensor,
        cross_attn_bias: Tensor,
        batch_labels: Optional[Tensor] = None,
        niche_l: Optional[Tensor] = None,
        center_l: Optional[Tensor] = None,
        center_r: Optional[Tensor] = None,
        max_l_seqlen: Optional[int] = None,
        max_r_seqlen: Optional[int] = None,
        CLS: bool = False,
        GCL: bool = False,
        LRC: bool = False,
        MVC: bool = False, 
        output_gene_emb: bool = False,
    ) -> Mapping[str, Tensor]:
        
        memory = self._encode(encoder_src, encoder_values)
        decoder_output = self._decode(
            decoder_src, decoder_values, decoder_src_key_padding_mask, memory, encoder_src_key_padding_mask, cross_attn_bias, batch_labels=batch_labels)

        if self.batch_encoder is not None and batch_labels is not None:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
            self.batch_emb = batch_emb 

        output = {}

        mlm_output = self.decoder(
            decoder_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    decoder_output,
                    self.batch_emb.unsqueeze(1).repeat(1, decoder_output.shape[1], 1),
                ],
                dim=2,
            ),
        )

        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]


        if self.cell_emb_style == 'max-pool':
            cell_emb = torch.cat([torch.max(decoder_output[k][~decoder_src_key_padding_mask[k]], dim=0)[0].unsqueeze(0) for k in range(decoder_output.size(0))])
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.cat([torch.mean(decoder_output[k][~decoder_src_key_padding_mask[k]], dim=0).unsqueeze(0) for k in range(decoder_output.size(0))])
            
        output["cell_emb"] = cell_emb
        output["cell_emb_m"] = self.create_mlp_fn_pj(cell_emb)
        
        if output_gene_emb:
            output["gene_emb"] = decoder_output
        
        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if GCL:
            output["gcl_output"] = self.gcl_decoder(decoder_output)
        if LRC:
            split_src_indices, total_cell_num, max_seqlen, _ = self._pad_information_of_split_input(niche_l.sum(1), ligand=True)
            memory_l = pad_input(memory[niche_l], split_src_indices, total_cell_num, max_seqlen)
            cross_attn_bias_l = pad_input(torch.exp(cross_attn_bias[niche_l]).unsqueeze(-1), split_src_indices, total_cell_num, max_seqlen).squeeze(-1)

            niche_cell_num = (niche_l.sum(1)/center_l.sum(1)).cpu().numpy().astype(int)
            gene_num = center_l.sum(1).cpu().numpy()
            memory_l_pooling = []
            for center_cell in range(total_cell_num):
                for k in range(gene_num[center_cell]):
                    memory_l_pooling.append(np.sum([cross_attn_bias_l[center_cell][k+niche_cell*gene_num[center_cell]].item()*memory_l[center_cell][k+niche_cell*gene_num[center_cell]].cpu().numpy() for niche_cell in range(niche_cell_num[center_cell])], axis=0))
            
            split_src_indices, total_cell_num, max_seqlen, _ = self._pad_information_of_split_input(center_l.sum(1), ligand=True, max_seqlen = max_l_seqlen)
            memory_l_pooling = pad_input(torch.tensor(memory_l_pooling, device=decoder_output.device, dtype=decoder_output.dtype), split_src_indices, total_cell_num, max_seqlen)
            split_src_indices, total_cell_num, max_seqlen, _ = self._pad_information_of_split_input(center_r.sum(1), ligand=True, max_seqlen = max_r_seqlen)
            decoder_output_r = pad_input(decoder_output[center_r], split_src_indices, total_cell_num, max_seqlen)
            
            output["lrc_output"] = self.lrc_decoder(memory_l_pooling, decoder_output_r)

        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, self.batch_emb], dim=1),
                self.cur_gene_token_embs,
            )
            output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)
        return output

class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


#Flash attention decoder layer
class FlashTransformerDecoderLayer(nn.Module):
    r"""The class is modified from torch.nn.TransformerDecoderLayer to support the
    FlashAttention. It is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    
    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = FlashMHA(d_model, nhead, batch_first=batch_first, attention_dropout=dropout, 
                                attention_type='inner', bias=bias, **factory_kwargs)
        self.cross_attn = FlashMHA(d_model, nhead, batch_first=batch_first, attention_dropout=dropout,
                                attention_type='cross', bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = self._get_activation_fn(activation)
        else:
            self.activation = activation

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        cross_attn_bias: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        
        # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
        tgt_key_padding_mask_ = ~tgt_key_padding_mask
        memory_key_padding_mask_ = ~memory_key_padding_mask
        
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_key_padding_mask_, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, cross_attn_bias, tgt_key_padding_mask_, memory_key_padding_mask_, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_key_padding_mask_, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, cross_attn_bias, tgt_key_padding_mask_, memory_key_padding_mask_, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x, tgt_key_padding_mask, tgt_is_causal) -> Tensor:
        x = self.self_attn(x,tgt_key_padding_mask=tgt_key_padding_mask, tgt_is_causal=tgt_is_causal)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x, memory, attn_bias, tgt_key_padding_mask, memory_key_padding_mask, memory_is_causal) -> Tensor:
        x = self.cross_attn(x, mem=memory, attn_bias=attn_bias, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, memory_is_causal=memory_is_causal)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


#Biased attention decoder
class BiasedTransformerDecoder(nn.Module):
    r"""The class is modified from torch.nn.TransformerDecoder to support the
    biased cross-attention. BiasedTransformerDecoder is a stack of N decoder layers.

    Args:
        decoder_layer: an instance of the FlashTransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """

    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, cross_attn_bias: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, memory, cross_attn_bias=cross_attn_bias, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         tgt_is_causal=tgt_is_causal,
                         memory_is_causal=memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        return output

class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)


class ClsDecoder(nn.Module):
    """
    Decoder for cell classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)
    

class GclDecoder(nn.Module):
    """
    Decoder for gene classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_gcl: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()

        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_gcl)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class LRCDecoder(nn.Module):
    """
    Decoder for ligand-receptor pair classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_lr: int = 2,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        self.norm_l = nn.LayerNorm(d_model)

        d_model = 2*d_model
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_lr)

    def forward(self, ligand, receptor) -> Tensor:
        ligand = self.norm_l(ligand)
        x_lr = [list(itertools.product(ligand[i].cpu(), receptor[i].cpu())) for i in range(ligand.size(0))]
        x_lr = torch.tensor([[torch.concat(lr).tolist() for lr in cell] for cell in x_lr], device=ligand.device)
        for layer in self._decoder:
            x_lr = layer(x_lr)
        return self.out_layer(x_lr)

class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2)
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
