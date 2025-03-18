import numpy as np
import torch.nn as nn
from pytorch_tcn import TCN
import torch
from utils import Traj_loss

class Trans(nn.Module):
    def __init__(self, args):
        super(Trans, self).__init__()

        self.args = args
        self.device = self.args.device
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.heads = self.args.heads
        self.nlayer = self.args.layers_en

        self.pro_len1 = nn.Linear(self.obs_length, self.feats_hidden)
        self.pro_len2 = nn.Linear(self.feats_hidden, self.obs_length)

        self.layer = nn.TransformerEncoderLayer(d_model=self.feats_hidden, nhead=self.heads)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.layer2 = nn.TransformerDecoderLayer(d_model=self.feats_hidden,nhead=self.heads)
        self.decoder = nn.TransformerDecoder(self.layer2, num_layers=3)

        self.input_en1 = nn.Linear(self.feats_in, self.feats_hidden//2)
        self.input_en2 = nn.Linear(self.feats_hidden//2, self.feats_hidden)

        self.output_fc = nn.Linear(self.feats_hidden, self.feats_out)
        self.relu = nn.ReLU()

        self.emb = nn.Parameter(torch.randn(self.obs_length,1,self.feats_hidden))


    def forward(self, inputs, iftrain):

        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)
        #padding = torch.zeros_like(batch)
        #batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])
        batch_in = batch[:self.obs_length, :, :]

        his_enc = self.input_en2(self.relu(self.input_en1(batch_in)))
        #enc = self.pro_len1(his_enc.permute(1, 2, 0))
        his_enc = his_enc * np.sqrt(self.feats_hidden) + self.emb.repeat(1, his_enc.shape[1],1)

        traj = self.encoder(his_enc)
        traj = self.decoder(traj,his_enc)
        #traj = self.pro_len2(traj).permute(2, 0, 1)
        traj = self.output_fc(traj)
        traj = traj[-self.obs_length:, :, :]#.permute(1, 0, 2)#.unsqueeze(2).repeat(1, 1, 20, 1)


        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss



class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = self.args.obs_length

        self.lstm = nn.LSTM(self.feats_hidden, self.feats_hidden)
        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro4 = nn.Linear(self.feats_hidden//2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)
        padding = torch.zeros_like(batch)
        batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])

        enc = self.pro1(batch_in)
        enc = self.relu(enc)
        enc = self.pro2(enc)
        out, _ = self.lstm(enc)
        out = self.pro3(out)
        tra = self.pro4(out) # b2
        # b1
        traj = tra[-self.obs_length:, :, :]#.permute(1, 0, 2)#.unsqueeze(2).repeat(1, 1, 20, 1)

        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss



class GRU(nn.Module):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden
        self.obs_length = self.args.obs_length

        self.gru = nn.GRU(self.feats_hidden, self.feats_hidden)
        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_hidden//2)
        self.pro4 = nn.Linear(self.feats_hidden//2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in] # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]#.permute(1, 0, 2).to(self.device)
        padding = torch.zeros_like(batch)
        batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])

        enc = self.relu(self.pro1(batch_in))
        enc = self.relu(self.pro2(enc))
        out, _ = self.gru(enc)
        out = self.pro3(out)
        tra = self.pro4(out) # b2
        # b1
        traj = tra[self.obs_length:, :, :]#.permute(1, 0, 2)#.unsqueeze(2).repeat(1, 1, 20, 1)

        traj_loss = Traj_loss(traj, tar_y)
        return traj, traj_loss


class Seq2Seq(nn.Module):

    def __init__(self, args):
        super(Seq2Seq, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.pred_length = args.pred_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden

        self.encoder_se = nn.LSTM(self.feats_hidden,self.feats_hidden,num_layers=1)
        self.decoder_se = nn.LSTM(self.feats_hidden, self.feats_hidden, num_layers=1)

        self.pro1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.pro2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.pro3 = nn.Linear(self.feats_hidden, self.feats_out)
        self.pro4 = nn.Linear(self.feats_hidden // 2, self.feats_out)
        self.relu = nn.ReLU()

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in]  # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]  # .permute(1, 0, 2).to(self.device)
        #padding = torch.zeros_like(batch)
        #batch_in = torch.cat([batch[:self.obs_length, :, :], padding[self.obs_length:, :, :]])
        batch_in = batch[:self.obs_length, :, :]

        enc = self.pro2(self.relu(self.pro1(batch_in)))
        encout,state = self.encoder_se(enc)

        out,_ = self.decoder_se(enc,state)
        tra = self.pro3(out)
        traj_loss = Traj_loss(tra, tar_y)

        return tra, traj_loss



class T_CNN(nn.Module):

    def __init__(self, args):
        super(T_CNN, self).__init__()
        self.args = args
        self.device = self.args.device

        self.obs_length = args.obs_length
        self.pred_length = self.args.pred_length
        self.feats_in = self.args.feats_in
        self.feats_out = self.args.feats_out
        self.feats_hidden = self.args.feats_hidden

        self.project1 = nn.Linear(self.feats_in, self.feats_hidden)
        self.project2 = nn.Linear(self.feats_hidden, self.feats_hidden)
        self.project3 = nn.Linear(self.feats_hidden, self.feats_out)
        self.TCN = TCN(128, [128],
                       kernel_size=3)
        self.TCN2 = TCN(128, [128],
                       kernel_size=3)
        self.relu= nn.ReLU()
        self.emb = nn.Parameter(torch.randn(self.obs_length, 1, self.feats_hidden))

    def forward(self, inputs, iftrain):
        batch = inputs[0][:,:,:self.feats_in]  # B2
        tar_y = inputs[0][self.obs_length:, :, :self.feats_out]
        batch_in = batch[:self.obs_length, :, :]
        out = self.relu(self.project2(self.relu(self.project1(batch_in))))

        out = out.permute(1, 2, 0)
        out = self.TCN(out)[:,:,-1:]
        outputs = [out]
        for i in range(self.pred_length):
            outputs.append(self.TCN2(outputs[-1]))

        out = torch.cat(outputs,dim=-1)
        out = out.permute(2, 0, 1)
        tra = self.project3(out)[1:]

        traj_loss = Traj_loss(tra, tar_y)

        return tra, traj_loss



