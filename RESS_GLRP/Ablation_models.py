from GLRP import *


class GLRP_ATT(nn.Module):
    def __init__(self, args):
        super(GLRP_ATT, self).__init__()
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
        self.pro5 = nn.Linear(self.feats_hidden, self.feats_out)

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
        traj = self.pro5(self.encoder(enc_))

        # VGAE encoder
        #enc, Z, Vae_loss = self.VGAE_En(enc, edge_index, edge_attr, ADJ)

        # Decode
        #traj = self.Decoder(enc, Z)

        # Outputs
        traj = traj[-self.pred_length:, :, :]
        traj_loss = Traj_loss(traj, tar_y)
        loss = traj_loss #+ Vae_loss

        return traj, loss


class GLRP_DE(nn.Module):
    def __init__(self, args):
        super(GLRP_DE, self).__init__()
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
        enc = self.pro2(self.relu(self.pro1(batch_in)))[-1]
        #enc_ = enc_ * np.sqrt(self.feats_hidden) + self.emb.repeat(1, enc_.shape[1], 1)
        #enc = self.encoder(enc_)[-1]

        # VGAE encoder
        #enc, Z, Vae_loss = self.VGAE_En(enc, edge_index, edge_attr, ADJ)

        # Decode
        traj = self.Decoder(enc, enc)

        # Outputs
        traj = traj[-self.pred_length:, :, :]
        traj_loss = Traj_loss(traj, tar_y)
        loss = traj_loss #+ Vae_loss

        return traj, loss


class GLRP_VG_DE(nn.Module):
    def __init__(self, args):
        super(GLRP_VG_DE, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.pred_length = self.args.pred_length
        self.K = self.args.K

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden ,self.feats_hidden)
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
        enc = self.pro2(self.relu(self.pro1(batch_in)))[-1]
        #enc_ = enc_ * np.sqrt(self.feats_hidden) + self.emb.repeat(1, enc_.shape[1], 1)
        #enc = self.encoder(enc_)[-1]

        # VGAE encoder
        enc, Z, Vae_loss = self.VGAE_En(enc, edge_index, edge_attr, ADJ)

        # Decode
        traj = self.Decoder(enc, Z)

        # Outputs
        traj = traj[-self.pred_length:, :, :]
        traj_loss = Traj_loss(traj, tar_y)
        loss = traj_loss + Vae_loss

        return traj, loss

class GLRP_ATT_DE(nn.Module):
    def __init__(self, args):
        super(GLRP_ATT_DE, self).__init__()
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
        #Z, Vae_loss = self.VGAE_En(enc, edge_index, edge_attr, ADJ)

        # Decode
        traj = self.Decoder(enc, enc)

        # Outputs
        traj = traj[-self.pred_length:, :, :]
        traj_loss = Traj_loss(traj, tar_y)
        loss = traj_loss #+ Vae_loss

        return traj, loss

