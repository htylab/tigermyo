import numpy as np
import glob
import pydicom as dicom
import os
import sys

# determine if application is a script file or frozen exe
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(os.path.abspath(__file__))

files_path = os.path.join(application_path, 'files')
os.makedirs(files_path, exist_ok=True)

def read_molli_dir(dirname):
    '''
    example call: im, Invtime = read_molli_dir(dirname)
    '''
    im=[]
    Invtime=[]
    count=0
    files = glob.glob(dirname+'/*')
    
    if len(files) == 0:
        raise ValueError('No files. Please check the path.')
    
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
        dll = 'T1map.dll'
        dll = checkFileExit(dll, "https://github.com/htylab/tigermyo/releases/download/v0.0.1/T1map.dll")
        lb = ctypes.CDLL(dll)
        lib = ctypes.WinDLL("",handle=lb._handle)
    else: #linux
        dll = 'T1map.so'
        dll = checkFileExit(dll, "https://github.com/htylab/tigermyo/releases/download/v0.0.1/T1map.so")
        lib = ctypes.CDLL(dll)

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

def download(url, file_name):
    import urllib.request
    import certifi
    import shutil
    import ssl
    context = ssl.create_default_context(cafile=certifi.where())
    #urllib.request.urlopen(url, cafile=certifi.where())
    with urllib.request.urlopen(url,
                                context=context) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        
def checkFileExit(file, url):
    file_path = os.path.join(files_path, file)
    if not os.path.exists(file_path):
        try:
            print(f"Missing {file}")
            print('Downloading file....')
            print(url, file_path)
            download(url, file_path)
            download_ok = True
            print('Download finished...')
        except:
            download_ok = False

        if not download_ok:
            raise ValueError('Server error. Please check the model name or internet connection.')
        
    return file_path