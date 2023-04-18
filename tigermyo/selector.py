from tigermyo.utils import T1mapCal
import numpy as np

# Calculate R2 and FQI
def calculateFQI(im, Errmap):
    SS_im_mean = np.mean(im, axis=0)
    SS_total = np.sum(np.square(im - SS_im_mean), axis=0)
    SS_res = np.square(Errmap)
    with np.errstate(divide='ignore', invalid='ignore'):
        R2_map = 1 - SS_res/SS_total
    return R2_map, np.nanmean(R2_map)

def selector(reged_ims, invtime):
    reged_ims_table = []
    FQIs = []
    for i in range(len(reged_ims)):
        if reged_ims[i] is not None:
            reged_ims_table.append(T1mapCal(reged_ims[i], invtime))
            _, FQI = calculateFQI(reged_ims[i], reged_ims_table[i]['Errmap'])
            FQIs.append(FQI)
        else:
            reged_ims_table.append(None)
            FQIs.append(float('-inf'))
    
    return reged_ims_table, FQIs