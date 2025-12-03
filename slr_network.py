import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTMLayer, TemporalConv
import modules.resnet as resnet

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, use_bn=False,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(resnet, c2d_type)()
        self.conv2d.fc = Identity()

        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None):

        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            framewise = self.conv2d(x.permute(0,2,1,3,4)).view(batch, temp, -1).permute(0,2,1) # btc -> bct
        else:
            framewise = x
        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len'].cpu()
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = None if self.training \
            else self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)
        return {
            "framewise_features": framewise,
            "visual_features": x,
            "temproal_features": tm_outputs['predictions'],
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    # def criterion_calculation(self, ret_dict, label, label_lgt):
    #     loss = 0
    #     for k, weight in self.loss_weights.items():
    #         if k == 'ConvCTC':
    #             loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
    #                                                   label.cpu().int(), ret_dict["feat_len"].cpu().int(),
    #                                                   label_lgt.cpu().int()).mean()
    #         elif k == 'SeqCTC':
    #             loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
    #                                                   label.cpu().int(), ret_dict["feat_len"].cpu().int(),
    #                                                   label_lgt.cpu().int()).mean()
    #         elif k == 'Dist':
    #             loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
    #                                                        ret_dict["sequence_logits"].detach(),
    #                                                        use_blank=False)
    #     return loss
    
    
    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0.0

        def _prep_ctc(logits, label, label_lgt, num_classes):
            # lengths (CPU, long) as required by the standard (non-cuDNN) implementation :contentReference[oaicite:1]{index=1}
            in_lens = ret_dict["feat_len"].detach().to("cpu", dtype=torch.long)
            tgt_lens = label_lgt.detach().to("cpu", dtype=torch.long)

            N = int(in_lens.numel())

            # logits must be (T, N, C)
            if logits.dim() != 3:
                raise ValueError(f"CTC logits must be 3D (T,N,C) or (N,T,C), got {tuple(logits.shape)}")

            # If logits look like (N,T,C), transpose to (T,N,C)
            if logits.size(1) != N and logits.size(0) == N:
                logits = logits.transpose(0, 1).contiguous()

            if logits.size(1) != N:
                raise ValueError(f"CTC expects logits shaped (T,N,C). Got {tuple(logits.shape)} with N={N}")

            T = int(logits.size(0))
            max_in = int(in_lens.max())
            if max_in > T:
                raise ValueError(
                    f"CTC invalid: input_lengths.max={max_in} > T={T}. "
                    f"Your logits are almost certainly still (N,T,C) somewhere."
                )

            # targets -> 1D concatenated (sum(target_lengths),)
            if label.dim() == 2:
                lab2 = label.detach().to("cpu")
                if lab2.size(0) != N and lab2.size(1) == N:
                    lab2 = lab2.transpose(0, 1).contiguous()

                if lab2.size(0) != N:
                    raise ValueError(f"Targets must have batch dim N={N}. Got targets {tuple(lab2.shape)}")

                S = int(lab2.size(1))
                max_tgt = int(tgt_lens.max())
                if max_tgt > S:
                    raise ValueError(f"CTC invalid: target_lengths.max={max_tgt} > S={S} (targets shape {tuple(lab2.shape)})")

                targets = torch.cat([lab2[i, :tgt_lens[i]] for i in range(N)], dim=0)
            else:
                targets = label.detach().to("cpu")

            targets = targets.to(torch.long).contiguous()

            # label id sanity (CTC targets must be class indices; and cannot be blank=0 inside the true target) :contentReference[oaicite:2]{index=2}
            if (targets < 0).any() or (targets >= num_classes).any():
                raise ValueError(
                    f"CTC target ids out of range: min={int(targets.min())}, max={int(targets.max())}, "
                    f"num_classes={num_classes}"
                )

            return logits, targets, in_lens, tgt_lens

        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                logits, targets, in_lens, tgt_lens = _prep_ctc(
                    ret_dict["conv_logits"], label, label_lgt, self.num_classes
                )
                loss = loss + weight * self.loss['CTCLoss'](
                    logits.log_softmax(-1), targets, in_lens, tgt_lens
                ).mean()

            elif k == 'SeqCTC':
                logits, targets, in_lens, tgt_lens = _prep_ctc(
                    ret_dict["sequence_logits"], label, label_lgt, self.num_classes
                )
                loss = loss + weight * self.loss['CTCLoss'](
                    logits.log_softmax(-1), targets, in_lens, tgt_lens
                ).mean()

            elif k == 'Dist':
                loss = loss + weight * self.loss['distillation'](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False
                )

        return loss
    
    
    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
