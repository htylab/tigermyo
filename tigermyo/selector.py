from tigermyo.utils import T1mapCal
import numpy as np

def selector(reged_ims, invtime):
    reged_ims_table = []
    fit_err = []
    for i in range(len(reged_ims)):
        reged_ims_table.append(T1mapCal(reged_ims[i], invtime))
        fit_err.append(np.sum(reged_ims_table[i]['Errmap'])/reged_ims_table[i]['Errmap'].size)
    
    return reged_ims_table, np.argmin(np.array(fit_err))
    