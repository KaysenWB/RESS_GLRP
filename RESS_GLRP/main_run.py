import argparse
import os
import time

import numpy as np
import pandas as pd

from processor import processor
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser( description='GLRP')

parser.add_argument('--data_root', default='./AIS_process/AIS_processed_com.cpkl', type=str)
parser.add_argument('--save_dir', default='./output_f4_16/', help='Directory for saving caches and models.')

parser.add_argument('--train_model', default='GLRP', help='[GLRP, Trans, LSTM, GRU, Seq2Seq, TCNN]')
parser.add_argument('--data_rate', default=[8,0,2], type=list) # train, val, test
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--seq_length', default=16, type=int)
parser.add_argument('--obs_length', default=16, type=int)
parser.add_argument('--label_length', default=16, type=int)
parser.add_argument('--pred_length', default=16, type=int)
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--learning_rate', default=0.0005, type=float)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--dropout', default=0.1, type=float)

parser.add_argument('--layers_en', default=2, type=int)
parser.add_argument('--layers_de', default=1, type=int)
parser.add_argument('--heads', default=8, type=int, help='the number of heads in the multihead-attention models')

parser.add_argument('--device', default='cuda:2', type=str)

parser.add_argument('--feats_in', default=4, type=int)
parser.add_argument('--feats_hidden', default=128, type=int)
parser.add_argument('--feats_out', default=4, type=int)
parser.add_argument('--K', default=30, type=int)


args = parser.parse_args()


for pp in [16,32,48,64]:
    args.save_dir = './output_f4_{}/'.format(pp)
    args.seq_length = pp*2
    args.obs_length, args.label_length, args.pred_length = pp,pp,pp

    #model = ['GLRP', 'Trans', 'LSTM', 'GRU', 'Seq2Seq', 'TCNN']
    model = ['GLRP']
    #model = ['GLRP_ATT',  'GLRP_DE', 'GLRP_VG_DE', 'GLRP_ATT_DE', 'GLRP']

    name =  args.save_dir
    Metric = np.zeros((len(model), 8)) # six methods and seven indicator

    for m_id, m in enumerate(model):
        args.train_model = m
        args.save_dir = name + args.train_model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        trainer = processor(args)

        ts = time.time()
        trainer.train()
        train_time = time.time()-ts

        ADE, FDE, ME, infer_time = trainer.test()
        print('train_time: {}-----------------'.format(train_time))
        print('infer_time: {}-----------------'.format(infer_time))

        trainer.pred()
        Metric[m_id,:] = [ADE, FDE, infer_time,train_time] + list(ME)

    save_m = pd.DataFrame(Metric, columns=[['ADE', 'FDE', 'Infer_time','Train_time', 'MAE_SOG', 'MSE_SOG', 'MAE_COG', 'MSE_COG']])
    save_m['Model'] = model
    save_m = save_m[['Model', 'ADE', 'FDE', 'MAE_SOG', 'MSE_SOG', 'MAE_COG', 'MSE_COG', 'Train_time','Infer_time']]
    table = open(name + 'Table','w')
    save_m.to_csv(table)
    table.close()
