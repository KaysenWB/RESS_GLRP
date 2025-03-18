import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def getDistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda x: x/180*np.pi, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(a**0.5)
    r = 6371000  # meter
    return (c * r)/1852


def getDegree(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(lambda x: x / 180 * np.pi, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = np.degrees(np.arctan2(y, x))
    brng = (brng + 360) % 360
    return brng


def cal(node1, node2):

    lon1, lat1, cog1, sog1 = node1[0], node1[1], node1[3], node1[2]
    lon2, lat2, cog2, sog2 = node2[:,0], node2[:,1], node2[:,3], node2[:,2]

    v1_x, v1_y = sog1 * np.sin(cog1 / 180 * np.pi), sog1 * np.cos(cog1 / 180 * np.pi)
    v2_x, v2_y = sog2 * np.sin(cog2 / 180 * np.pi), sog2 * np.cos(cog2 / 180 * np.pi)

    vr_x, vr_y = v2_x - v1_x, v2_y - v1_y
    vr = np.sqrt(vr_x ** 2 + vr_y ** 2)

    wr = np.degrees(np.arctan2(vr_x, vr_y))
    wr[wr < 0] += 360

    Dr = getDistance(lon1, lat1, lon2, lat2)
    aT = getDegree(lon1, lat1, lon2, lat2)

    oT = (aT - cog1) % 360
    Cr = (cog1 - cog2) % 360
    DCPA = Dr * np.sin((wr - aT - 180) / 180 * np.pi)
    TCPA = Dr * np.cos((wr - aT - 180) / 180 * np.pi).round(decimals=2) / vr.round(decimals=2)

    return DCPA, TCPA

def D_Tcpa (batch):
    B, L, F = batch.shape

    dcpa = np.zeros((L, B, B))
    tcpa = np.zeros((L, B, B))
    for s_l in range(L):
        pic = batch[:, s_l, :]
        for s_i, ss in enumerate(pic):
            dcpa[s_l, s_i, :], tcpa[s_l, s_i, :] = cal(ss, pic)
    dcpa = dcpa.round(decimals=4)
    tcpa = (tcpa * 60).round(decimals=3)
    tcpa = np.nan_to_num(tcpa, nan=0.0, posinf=0.0, neginf=0.0)

    return dcpa, tcpa

def infer_sc(batch2):
    L, B, F = batch2.shape
    sog = np.zeros((L, B))
    cog = np.zeros((L, B))

    for i in range(B):
        tra = batch2[:, i, :]

        #plt.scatter(tra[:, 0], tra[:, 1], s=3, c='red')
        #imp = plt.imread('/Users/yangkaisen/MyProject/Data/map/map_apple.jpg')
        #plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
        #plt.show()

        diff_  =  tra[1:, :] - tra[:-1, :] #np.concatenate((tra[3:, :], tra[-3:,:])) - tra #
        #see = diff_[-1:,:]
        diff = np.concatenate((diff_, diff_[-1:,:]))
        diff_new = np.zeros_like(diff)
        mean_step = 1
        for n in range(len(diff)):
            diff_new[n,:] = diff[n - mean_step: n + 1 + mean_step, :].mean(0)
        diff_new[:mean_step, :] = diff_new[mean_step, :]

        sog[:, i] = np.sqrt(np.sum(diff_new ** 2 ,axis=1)) * 360 * 60
        cog_tr = np.arctan2(diff_new[:, 1], diff_new[:, 0]) * 180 / np.pi
        #cog_tr = np.degrees(np.arctan2(diff_new[:, 0], diff_new[:, 1]))
        cog_tr[cog_tr < 0] += 360
        cog[:, i] = cog_tr
    sc = np.stack((sog,cog),axis=-1)
    return sc


def inverse(Reals, Preds, ff):
    mean_inver = Reals.mean(axis=(0, 1), keepdims = True)[:, :, :ff]
    std_inver = Reals.std(axis=(0, 1), keepdims = True)[:, :, :ff]
    Preds = Preds * std_inver + mean_inver
    return Reals, Preds

def filter_data(reals, preds, keep):

    nodes = preds.shape[1]
    keep_list = [False] * nodes
    for k in keep:
        keep_list[k] = True
    Reals =  reals[:, keep_list, :]
    Preds =  preds[:, keep_list, :]

    return Reals, Preds

def show_tra(Reals, Preds):
    plt.scatter(Reals[:, : ,0], Reals[:,: ,1],s=3,c='black',alpha=1)
    plt.scatter(Preds[:,: ,0], Preds[:,: ,1],s=3,c='red')
    imp = plt.imread('/Data/map/map_apple.jpg')
    plt.imshow(imp, extent=[114.099003, 114.187537, 22.265695, 22.322062])
    plt.show()
    print(';')


def show_risk(Reals, Preds, step_list):

    for step in step_list:

        plt.scatter(Reals[:, :, 0], Reals[:, :, 1], s=5, c='blue', alpha=0.35)
        nodes_r = Reals[step, :, :]
        nodes_p = Preds[step, :, :]
        nodes_show_r = np.zeros((Reals.shape[1] * 2, Reals.shape[-1]))
        nodes_show_p = np.zeros((Reals.shape[1] * 2, Reals.shape[-1]))


        for no in nodes_r:
            nodes_show_p[1::2] = nodes_r
            nodes_show_p[::2] = no
            plt.plot(nodes_show_p[::, 0], nodes_show_p[::, 1], color='blue', markersize=1, alpha=1, linewidth = 2, )



        for no in nodes_p:
            nodes_show_r[1::2] = nodes_p
            nodes_show_r[::2] = no
            plt.plot(nodes_show_r[::, 0], nodes_show_r[::, 1], color='red', markersize=1, alpha=0.5, linewidth = 2)

        plt.scatter(Reals[step, :, 0], Reals[step, :, 1], s=70, c='blue',)
        plt.scatter(Preds[step, :, 0], Preds[step, :, 1], s=70, c='red', )
        imp = plt.imread('/Data/map/map_apple_local.jpg')
        plt.imshow(imp, extent=[114.126503, 114.161003, 22.285695, 22.315695], alpha=0.6)
        plt.savefig(fname='/Users/yangkaisen/MyProject/GTGF_ship/A_scene/{}_steps.jpg'.format(step), dpi=500,
                    format='jpg', bbox_inches='tight')

        plt.show()
        print(';')
        # border_x = [114.126503, 114.161003, 114.161003,  114.126503, 114.126503]
        # border_y = [22.315695,  22.315695,   22.285695,  22.285695, 22.315695]
        # plt.plot(border_x, border_y, color = 'red')


def save_risk_of_a_scene(dcpa_real, dcpa_gt, tcpa_real, tcpa_gt):
    DR,DG,TR,TG = [],[],[],[]
    for sh in range(dcpa_real.shape[-1]):
        DR.append(dcpa_real[:,sh,:])
        DG.append(dcpa_gt[:,sh,:])
        TR.append(tcpa_real[:,sh,:])
        TG.append(tcpa_gt[:,sh,:])
    DR,DG,TR,TG = np.concatenate(DR), np.concatenate(DG), np.concatenate(TR), np.concatenate(TG),
    risk_five_out = pd.DataFrame(np.concatenate((DR,DG,TR,TG),axis=1))
    save_table = open('/GTGF_ship/A_scene/A_scen.csv', 'w')
    risk_five_out.to_csv(save_table)
    save_table.close()
    print(';')

def metrics(dcpa_real, dcpa_gt, tcpa_real,tcpa_gt):
    mae_d = abs(abs(dcpa_real) - abs(dcpa_gt)).mean()

    #mse_d = ((dcpa_real - dcpa_gt)**2).mean()
    mae_t = abs((tcpa_real - tcpa_gt)).mean()
    #mse_t = (((tcpa_real - tcpa_gt)/60)**2).mean()
    return (mae_d, mae_t)


def get_metrics_risk():

    model_list = ['LSTM', 'GRU', 'Seq2Seq', 'TCNN', 'Trans', 'GT']
    Me_all = []

    for step in [16, 32, 48, 64]:
        Reals = np.load('/Users/yangkaisen/MyProject/GTGF_ship/output_f4_{}/GT/Reals.npy'.format(step))

        ME_setp = np.zeros((len(model_list),2))

        for m_id, model in enumerate(model_list):
            Preds = np.load('/Users/yangkaisen/MyProject/GTGF_ship/output_f4_{}/{}/Preds.npy'.format(step, model))

            pred_len, _, ff = Preds.shape
            reals, preds = inverse(Reals, Preds, ff)
            reals = reals[pred_len:, :, :ff]
            preds = preds[:, :, :ff]

            keep = [1, 2, 9, 10]
            reals, preds = filter_data(reals, preds, keep)

            #reals = np.concatenate((reals, infer_sc(reals[:,:,:2])), axis=-1)
            #preds = np.concatenate((preds, infer_sc(preds[:,:,:2])), axis=-1)

            dcpa_P, tcpa_P = D_Tcpa(np.transpose(preds, (1, 0, 2)))
            dcpa_R, tcpa_R = D_Tcpa(np.transpose(reals, (1, 0, 2)))

            # cal mae and mse (error) of D/T, keep
            ME_setp[m_id, :] = metrics(dcpa_R, dcpa_P, tcpa_R, tcpa_P)

        Me_all.append(ME_setp)
    Me_table = pd.DataFrame(np.concatenate(Me_all,axis=0))
    save_table = open('/Users/yangkaisen/MyProject/GTGF_ship/Table_risk', 'w')
    Me_table.to_csv(save_table)
    save_table.close()


def get_aship_risk_and_error(dcpa_real, dcpa_gt, tcpa_real, tcpa_gt, ship_id):
    dcpa_p = dcpa_gt[:, ship_id, :]
    tcpa_p = tcpa_gt[:, ship_id, :]

    error_d = np.abs(dcpa_real[:, ship_id, :]- dcpa_p)
    error_t = np.abs(tcpa_real[:, ship_id, :] - tcpa_p)

    #save_data = pd.DataFrame(np.concatenate((dcpa_p, tcpa_p), axis=1)) # save DT
    save_data = pd.DataFrame(np.concatenate((error_d[:,0],error_d[:,1],error_d[:,3], error_t[:,0],error_t[:,1],error_t[:,3]), axis=0))  # save error
    save_table = open('/GTGF_ship/Draw/A_scene/A_scen_single.csv', 'w')
    save_data.to_csv(save_table)
    save_table.close()
    print(';')


def get_error_for_box():

    model_list = ['LSTM', 'GRU', 'Seq2Seq', 'TCNN', 'Trans', 'GT']
    Error_D = []
    Error_T = []

    for step in [16, 32, 48, 64]:
        Reals = np.load('/Users/yangkaisen/MyProject/GTGF_ship/output_f4_{}/GT/Reals.npy'.format(step))

        ME_setp = np.zeros((len(model_list),2))

        for m_id, model in enumerate(model_list):
            Preds = np.load('/Users/yangkaisen/MyProject/GTGF_ship/output_f4_{}/{}/Preds.npy'.format(step, model))

            pred_len, _, ff = Preds.shape
            reals, preds = inverse(Reals, Preds, ff)
            reals = reals[pred_len:, :, :ff]
            preds = preds[:, :, :ff]

            keep = [1, 2, 9, 10]
            reals, preds = filter_data(reals, preds, keep)

            #reals = np.concatenate((reals, infer_sc(reals[:,:,:2])), axis=-1)
            #preds = np.concatenate((preds, infer_sc(preds[:,:,:2])), axis=-1)

            dcpa_P, tcpa_P = D_Tcpa(np.transpose(preds, (1, 0, 2)))
            dcpa_R, tcpa_R = D_Tcpa(np.transpose(reals, (1, 0, 2)))

            #see = abs(abs(dcpa_R) - abs(dcpa_P)).mean(axis=(2))
            #see1 = abs(abs(dcpa_R) - abs(dcpa_P)).mean(axis=(2)).flatten()

            Error_D.append(abs(abs(dcpa_R) - abs(dcpa_P)).mean(axis=(2)).flatten())
            Error_T.append(abs(tcpa_R - tcpa_P).mean(axis=(2)).flatten())

    return Error_D, Error_T





if __name__ == "__main__":

    # cal mae and mse (error) of D/T
    #get_metrics_risk()


    # read
    Reals = np.load('/Users/yangkaisen/MyProject/GLRP_ship/output_f4_32/GT/Reals.npy')
    Preds = np.load('/Users/yangkaisen/MyProject/GLRP_ship/output_f4_32/GT/Preds.npy')
    pred_len, _, ff = Preds.shape

    # inverse
    Reals, Preds = inverse(Reals, Preds, ff)
    Reals =  Reals[pred_len:, :, :ff]
    Preds =  Preds[:, :, :ff]

    # filter_data if single sence
    keep = [1, 2, 9, 10] 
    Reals, Preds = filter_data(Reals, Preds, keep)

    # infer sog and cog by tra
    if ff ==2:
        Reals = np.concatenate((Reals, infer_sc(Reals)),axis=-1)
        Preds = np.concatenate((Preds, infer_sc(Preds)),axis=-1)


    # cal DCPA and TCPA
    dcpa_gt, tcpa_gt = D_Tcpa(np.transpose(Preds,(1,0,2)))
    dcpa_real, tcpa_real = D_Tcpa(np.transpose(Reals,(1,0,2)))

    # show_tra
    #show_tra(Reals, Preds)

    # show_risk
    #step_list = [4,16,28]
    #show_risk(Reals, Preds, step_list)
    get_aship_risk_and_error(dcpa_real, dcpa_gt, tcpa_real, tcpa_gt, 2)

    # save data of a scene
    save_risk_of_a_scene(dcpa_real, dcpa_gt, tcpa_real, tcpa_gt)


