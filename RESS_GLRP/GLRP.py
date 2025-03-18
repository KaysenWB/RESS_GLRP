import numpy as np
import torch.nn as nn
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import SimpleConv
from utils import Traj_loss



class GNN(torch.nn.Module):

    def __init__(self, feats_hidden):
        super(GNN, self).__init__()

        self.conv1 = SimpleConv(aggr='mean',combine_root='cat')
        self.conv2 = SimpleConv(aggr='mean',combine_root='cat')
        self.pro1 = nn.Linear(feats_hidden * 2, feats_hidden)
        self.pro2 = nn.Linear(feats_hidden * 2, feats_hidden)
        self.relu=nn.ReLU()

    def forward(self, data, edge, att):

        out = self.conv1(data,edge, att)
        out = self.relu(self.pro1(out))
        out = self.conv2(out, edge, att)
        out = self.relu(self.pro2(out))

        return out



class Decoder(nn.Module):
    def __init__(self, hidden_dim, feat_out, pred_len, K):
        super(Decoder,self).__init__()

        self.hidden_dim = hidden_dim
        self.feat_out = feat_out
        self.pred_len = pred_len
        self.K = K
        self.pro1 = nn.Sequential(nn.Linear(self.hidden_dim * 2,self.hidden_dim), nn.ReLU())
        self.pro2 = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim), nn.ReLU())
        self.grucell = nn.GRUCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim)
        self.pro3 = nn.Linear(self.hidden_dim, feat_out)

    def forward(self,dec_in, Z):

        if len(Z.shape) == 3:
            dec_in = dec_in.unsqueeze(1).repeat(1, self.K, 1)
        dec_h = torch.cat([dec_in, Z], dim=-1)

        forward_outputs = []
        forward_h = self.pro1(dec_h)
        forward_h = forward_h.view(-1, forward_h.shape[-1])
        forward_input = self.pro2(forward_h)

        for t in range(self.pred_len):  # the last step is the goal, no need to predict
            forward_h = self.grucell(forward_input, forward_h)
            forward_input = self.pro2(forward_h)
            forward_outputs.append(forward_h)

        forward_outputs = torch.stack(forward_outputs, dim=0)
        if len(Z.shape) == 3:
            forward_outputs = forward_outputs.view(self.pred_len,-1,self.K,self.hidden_dim)
        forward_outputs = self.pro3(forward_outputs)

        return forward_outputs



class VGAE(torch.nn.Module):

    def __init__(self, feats_hidden, K):
        super(VGAE, self).__init__()

        self.K = K
        self.Gnn_s = GNN(feats_hidden)
        self.pro3 = nn.Linear(feats_hidden, feats_hidden)
        self.pro4 = nn.Linear(feats_hidden, feats_hidden)
        self.norm = nn.LayerNorm(feats_hidden)
        self.G_loss = nn.BCEWithLogitsLoss()

    def forward(self, enc, edge_index, edge_attr, ADJ):

        enc = self.Gnn_s(enc, edge_index, edge_attr) + enc
        mu = self.pro3(enc).unsqueeze(1).repeat(1, self.K, 1)
        log_var = self.pro4(enc).unsqueeze(1).repeat(1, self.K, 1)
        Z = mu + torch.randn_like(log_var) * torch.exp(0.5 * log_var)

        kl_loss = -0.5 * torch.mean(
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
        Z_l = Z.mean(1)
        graph_loss = self.G_loss(Z_l @ Z_l.t(), ADJ)
        vae_loss = kl_loss + graph_loss

        return enc, Z, vae_loss

class GLRP(nn.Module):
    def __init__(self, args):
        super(GLRP, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length
        self.K = self.args.K

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.emb = nn.Parameter(torch.randn(self.obs_length, 1, self.feats_hidden))

        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=8)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.VGAE_En = VGAE(self.feats_hidden, self.K)

        self.Decoder = Decoder(self.feats_hidden,self.feats_out,self.pred_length,self.K)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        # data
        batch_in = inputs[0][:self.obs_length, :, :self.feats_out]
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        adj = inputs[2][:self.obs_length]
        adj = torch.mean(adj, dim=0)

        # ADJ
        mina,maxa = adj.min(),adj.max()
        ADJ = (adj - mina)/(maxa-mina)
        edge_index, edge_attr = dense_to_sparse(ADJ)

        # Trans encoder
        enc_ = self.pro2(self.relu(self.pro1(batch_in)))
        enc_ = enc_ * np.sqrt(self.feats_hidden) + self.emb.repeat(1, enc_.shape[1], 1)
        enc = self.encoder(enc_)[-1]

        # VGAE encoder
        enc, Z, Vae_loss = self.VGAE_En(enc, edge_index, edge_attr, ADJ)

        # Decode
        traj = self.Decoder(enc, Z)

        # Outputs
        traj = traj[-self.pred_length:, :, :]
        traj_loss = Traj_loss(traj, tar_y)
        loss = traj_loss + Vae_loss

        return traj, loss

