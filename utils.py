import numpy as np
import glob
import pydicom as dicom

def read_molli_dir(dirname):
    '''
    example call: im, Invtime = read_molli_dir(dirname)
    '''
    im=[]
    Invtime=[]
    count=0
    files = glob.glob(dirname+'\*')
    for file in files:
        try:
            temp = dicom.read_file(file)
            im = np.append(im,temp.pixel_array)
            mat_m, mat_n = temp.pixel_array.shape
            Invtime = np.append(Invtime, temp.InversionTime)
            count += 1
        except:
            print("invalid dicom file, ignore this %s." % file)
            pass

    #print("Total dicom file:%d" % count)
    im = np.reshape(im, (count, mat_m, mat_n))
    temp = np.argsort(Invtime)
    Invtime = Invtime[temp]
    im = im[temp]
    return im, Invtime

def T1mapCal(im, Invtime, threshold = 40):
    '''
    example call: T1map, Amap_pre, Bmap,T1starmap,Errmap = T1LLmap(im, Invtime)
    '''
    import ctypes
    import os
    import platform
    from numpy.ctypeslib import ndpointer
    import numpy as np

    TI_num = im.shape[0]
    mat_m = im.shape[1]
    mat_n = im.shape[2]
    mat_size = mat_m * mat_n



    if platform.system() == 'Windows':
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.dll')
        lb = ctypes.CDLL(dllpath)
        lib = ctypes.WinDLL("",handle=lb._handle)
    else: #linux
        dllpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'T1map.so')
        lib = ctypes.CDLL(dllpath)

    syn = lib.syn    

    # Define the types of the output and arguments of this function.
    syn.restype = None
    syn.argtypes = [ndpointer(ctypes.c_double), 
                    ctypes.c_int,
                    ndpointer(ctypes.c_double),
                    ctypes.c_long,ctypes.c_double,
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double),
                    ndpointer(ctypes.c_double)]

    T1map = np.empty((1, mat_size), dtype=np.double)
    Amap = np.empty((1, mat_size), dtype=np.double)
    Bmap = np.empty((1, mat_size), dtype=np.double)
    T1starmap = np.empty((1, mat_size), dtype=np.double)
    Errmap = np.empty((1, mat_size), dtype=np.double)
    # We execute the C function, which will update the array.
    # 40 is threshold
    syn(Invtime, TI_num, im.flatten('f'), mat_size,
        threshold, T1map, Amap, Bmap, T1starmap, Errmap)


    if platform.system() == 'Windows':

        from ctypes.wintypes import HMODULE
        ctypes.windll.kernel32.FreeLibrary.argtypes = [HMODULE]
        ctypes.windll.kernel32.FreeLibrary(lb._handle)
    else:
        pass
        #lib.dlclose(lib._handle)

    dT1LL = dict()

    for key in ('T1map', 'Amap', 'Bmap', 'T1starmap', 'Errmap'):
        dT1LL[key] = np.reshape(locals()[key], (mat_m, mat_n), 'f')
    
    return dT1LL