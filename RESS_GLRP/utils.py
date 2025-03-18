import numpy as np
import copy
import torch


def Metrics(preds, trues):
    if preds.ndim == 3:
        return Metrics1(preds,trues)
    else:
        return Metrics2(preds,trues)


def Metrics1(preds, trues):

    pred_traj, target_traj = preds[:,:,:2], trues[:,:,:2],
    traj_ADE = np.linalg.norm(pred_traj - target_traj, axis=-1).mean(0)
    traj_FDE = np.linalg.norm(pred_traj - target_traj, axis=-1)[-1]
    ade = traj_ADE.mean()
    fde = traj_FDE.mean()

    Mes = [0,0,0,0]
    if preds.shape[-1] ==3:
        mae_sog = abs(preds[:, :, 2] - trues[:, :, 2]).mean()
        mse_sog = ((preds[:, :, 2] - trues[:, :, 2])**2).mean()
        Mes = np.array((mae_sog, mse_sog))
    if preds.shape[-1] ==4:
        mae_sog = abs(preds[:, :, 2] - trues[:, :, 2]).mean()
        mse_sog = ((preds[:, :, 2] - trues[:, :, 2]) ** 2).mean()
        mae_cog = abs(preds[:, :, 3] - trues[:, :, 3]).mean()
        mse_cog = ((preds[:, :, 3] - trues[:, :, 3])**2).mean()
        Mes = np.array((mae_sog, mse_sog, mae_cog, mse_cog))

    return ade, fde, Mes


def Metrics2(preds, trues):


    trues = np.tile(trues[:, :, None, :], (1, 1, 30, 1))

    pred_traj, target_traj = preds[:,:,:,:2], trues[:,:,:,:2],
    traj_ADE = np.linalg.norm(pred_traj - target_traj, axis=-1).mean(1)
    traj_FDE = np.linalg.norm(pred_traj - target_traj, axis=-1)[:, -1]
    ade = np.min(traj_ADE, axis=1).mean()
    fde = np.min(traj_FDE, axis=1).mean()

    Mes = [0,0,0,0]
    if preds.shape[-1] ==3:
        mae_sog = abs(preds[:, :, :,2] - trues[:, :, :,2]).mean()
        mse_sog = ((preds[:, :, :,2] - trues[:, :, :,2])**2).mean()
        Mes = np.array((mae_sog, mse_sog))
    if preds.shape[-1] ==4:
        mae_sog = abs(preds[:, :,:, 2] - trues[:, :, :,2]).mean()
        mse_sog = ((preds[:, :, :,2] - trues[:, :,:, 2]) ** 2).mean()
        mae_cog = abs(preds[:, :,:, 3] - trues[:, :, :,3]).mean()
        mse_cog = ((preds[:, :, :,3] - trues[:, :, :,3])**2).mean()
        Mes = np.array((mae_sog, mse_sog, mae_cog, mse_cog))

    return ade, fde, Mes



def Traj_loss(pred, target, goal=None):
    #see = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    if pred.dim() == 4:
        target = target.unsqueeze(2).repeat(1,1,30,1)

    traj_rmse = torch.sqrt(torch.sum((pred - target)**2, dim=-1)).sum(dim=0)
    loss_traj = traj_rmse.mean()

    return loss_traj




def get_node_index(seq_list):

    for idx, framenum in enumerate(seq_list):
        if idx == 0:
            node_indices = framenum > 0
        else:
            node_indices *= (framenum > 0)
    return node_indices

def update_batch_pednum(batch_pednum, ped_list):

    updated_batch_pednum_ = copy.deepcopy(batch_pednum).cpu().numpy()
    updated_batch_pednum = copy.deepcopy(batch_pednum)

    cumsum = np.cumsum(updated_batch_pednum_)
    new_ped = copy.deepcopy(ped_list).cpu().numpy()

    for idx, num in enumerate(cumsum):
        num = int(num)
        if idx == 0:
            updated_batch_pednum[idx] = len(np.where(new_ped[0:num] == 1)[0])
        else:
            updated_batch_pednum[idx] = len(np.where(new_ped[int(cumsum[idx - 1]):num] == 1)[0])

    return updated_batch_pednum

def mean_normalize_abs_input( node_abs, st_ed):

    node_abs = node_abs.permute(1, 0, 2)
    for st, ed in st_ed:
        mean_x = torch.mean(node_abs[st:ed, :, 0])
        mean_y = torch.mean(node_abs[st:ed, :, 1])

        node_abs[st:ed, :, 0] = ((node_abs[st:ed, :, 0] - mean_x))
        node_abs[st:ed, :, 1] = ((node_abs[st:ed, :, 1] - mean_y))

    return node_abs.permute(1, 0, 2)

def get_st_ed(batch_num):

    cumsum = torch.cumsum(batch_num, dim=0)
    st_ed = []
    for idx in range(1, cumsum.shape[0]):
        st_ed.append((int(cumsum[idx - 1]), int(cumsum[idx])))

    st_ed.insert(0, (0, int(cumsum[0])))

    return st_ed


def keep_full_tra(inputs):

    node_index = get_node_index(inputs[3])

    nodes_abs_ = inputs[0][:, node_index, :]
    nodes_norm = inputs[1][:, node_index, :]

    nei_lists = inputs[4][:, node_index, :]
    nei_list = nei_lists[:, :, node_index]
    batch_pednum = update_batch_pednum(inputs[6], node_index)
    st = get_st_ed(batch_pednum)
    nodes_abs = mean_normalize_abs_input(nodes_abs_, st)
    #nodes_abs = mean_normalize_abs_input_2(nodes_abs_)
    full_tra = (nodes_abs, nodes_norm, nei_list)

    return full_tra

def block_diagonal(self, matrices):

    shapes = [m.shape for m in matrices]

    total_rows = sum(shape[0] for shape in shapes)
    total_cols = sum(shape[1] for shape in shapes)

    result = torch.zeros(total_rows, total_cols)

    row_idx = 0
    col_idx = 0
    for matrix in matrices:
        h, w = matrix.shape
        result[row_idx:row_idx + h, col_idx:col_idx + w] = matrix
        row_idx += h
        col_idx += w

    return result

