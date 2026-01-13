import time

from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices
# def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True, batch_size=100):
#     num_queries = similarity.size(0)
#     num_gallery = similarity.size(1)
#
#     all_indices = []
#     for start_idx in range(0, num_queries, batch_size):
#         end_idx = min(start_idx + batch_size, num_queries)
#         batch_similarity = similarity[start_idx:end_idx]
#
#         if get_mAP:
#             batch_indices = torch.argsort(batch_similarity, dim=1, descending=True)
#         else:
#             _, batch_indices = torch.topk(batch_similarity, k=max_rank, dim=1, largest=True, sorted=True)
#
#         all_indices.append(batch_indices)
#
#     indices = torch.cat(all_indices, dim=0)
#
#     pred_labels = g_pids[indices.cpu()]
#     matches = pred_labels.eq(q_pids.view(-1, 1))
#     pred_labels = g_pids[indices.cpu()]  # q * k
#     matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k
#
#     all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
#     all_cmc[all_cmc > 1] = 1
#     all_cmc = all_cmc.float().mean(0) * 100
#     # all_cmc = all_cmc[topk - 1]
#
#     if not get_mAP:
#         return all_cmc, indices
#
#     num_rel = matches.sum(1)  # q
#     tmp_cmc = matches.cumsum(1)  # q * k
#
#     inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
#     mINP = torch.cat(inp).mean() * 100
#
#     tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
#     tmp_cmc = torch.stack(tmp_cmc, 1) * matches
#     AP = tmp_cmc.sum(1) / num_rel  # q
#     mAP = AP.mean() * 100
#
#     return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.logger = logging.getLogger("PHP.eval")

    # def _compute_embedding(self, model):
    #     model = model.eval()
    #     device = next(model.parameters()).device
    #
    #     qids, gids, qfeats, gfeats = [], [], [], []
    #     t_time, i_time = [], []
    #     # text
    #     for pid, caption in self.txt_loader:
    #         caption = caption.to(device)
    #         with torch.no_grad():
    #             text_feat, ttime = model.encode_text(caption)
    #         qids.append(pid.view(-1)) # flatten
    #         qfeats.append(text_feat)
    #         t_time.append(ttime)
    #     qids = torch.cat(qids, 0)
    #     average_ttime = sum(t_time) / len(t_time)
    #     qfeats = torch.cat(qfeats, 0)
    #
    #     # image
    #     for pid, img in self.img_loader:
    #         img = img.to(device)
    #         with torch.no_grad():
    #             img_feat, itime = model.encode_image(img)
    #         gids.append(pid.view(-1)) # flatten
    #         gfeats.append(img_feat)
    #         i_time.append(itime)
    #     gids = torch.cat(gids, 0)
    #     average_itime = sum(i_time) / len(i_time)
    #     gfeats = torch.cat(gfeats, 0)
    #
    #     return qfeats, gfeats, qids, gids, average_ttime, average_itime
    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        # t_time, i_time = [], []
        # text
        # start_time = time.time()
        for pid, caption in self.txt_loader:
            caption = caption.to(device)
            with torch.no_grad():
                text_feat  = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten
            qfeats.append(text_feat)
        end_time = time.time()
        # t_time = end_time-start_time
        qids = torch.cat(qids, 0)
        # average_ttime = sum(t_time) / len(t_time)
        qfeats = torch.cat(qfeats, 0)

        # image
        # start_time = time.time()
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1))  # flatten
            gfeats.append(img_feat)
        end_time = time.time()
        # i_time = end_time - start_time
        gids = torch.cat(gids, 0)
        # average_itime = sum(i_time) / len(i_time)
        gfeats = torch.cat(gfeats, 0)

        return qfeats, gfeats, qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        # print(qfeats.shape,gfeats.shape)
        # print("itime and ttime", i_time, t_time)
        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
