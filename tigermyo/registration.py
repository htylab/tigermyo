import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import onnxruntime
import os
from scipy.signal import medfilt2d
import ctypes
import platform
from numpy.ctypeslib import ndpointer

def regMOLLI(im, target_im):
    reged_im = np.zeros_like(im)
    try:
        for i in tqdm(range(im.shape[0]), leave = False):
            fixed_np = target_im[i, ...]#讀目標影像
            moving_np = im[i,:,:]
            
            fixed_mask_np=None
            meshsize=10

            fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np), sitk.sitkFloat32)
            moving = sitk.Cast(sitk.GetImageFromArray(moving_np), sitk.sitkFloat32)

            sitk.sitkUInt8
            transfromDomainMeshSize=[meshsize]*moving.GetDimension()
            tx = sitk.BSplineTransformInitializer(fixed,
                                                    transfromDomainMeshSize )


            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMattesMutualInformation(128)
            R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                                        numberOfIterations=200,
                                                        convergenceMinimumValue=1e-5,
                                                        convergenceWindowSize=9)
            R.SetOptimizerScalesFromPhysicalShift( )
            R.SetInitialTransform(tx)
            R.SetInterpolator(sitk.sitkLinear)
            R.SetShrinkFactorsPerLevel([6,2,1])
            R.SetSmoothingSigmasPerLevel([6,2,1])
            if not (fixed_mask_np is None):
                fixed_mask  = sitk.Cast(sitk.GetImageFromArray(fixed_mask_np),
                                            sitk.sitkUInt8)
                R.SetMetricFixedMask(fixed_mask)

            outTx = R.Execute(fixed, moving)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(outTx)

            moving_reg = resampler.Execute(moving)
            moving_reg_np = sitk.GetArrayFromImage(moving_reg)
            
            reged_im[i, ...] = moving_reg_np
            
        return reged_im
    except:
        return None
    
def regMOLLI_MI(im, target_im):
    reged_im = np.zeros_like(im)
    try:
        for i in tqdm(range(im.shape[0]), leave = False):
            moving_np = im[i,:,:]
            k = i
            mi_max = float('-inf')
            for j in range(target_im.shape[0]):
                mi = mutual_information(target_im[j, ...], moving_np)
                if mi > mi_max:
                    mi_max = mi
                    k = j
            if mutual_information(im[-1,:,:], moving_np) > mi_max:
                k = 7
                fixed_np = im[-1,:,:]
            else:
                fixed_np = target_im[k, ...]
        
            fixed_mask_np=None
            meshsize=10

            fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np), sitk.sitkFloat32)
            moving = sitk.Cast(sitk.GetImageFromArray(moving_np), sitk.sitkFloat32)

            sitk.sitkUInt8
            transfromDomainMeshSize=[meshsize]*moving.GetDimension()
            tx = sitk.BSplineTransformInitializer(fixed,
                                                    transfromDomainMeshSize )


            R = sitk.ImageRegistrationMethod()
            R.SetMetricAsMattesMutualInformation(128)
            R.SetOptimizerAsGradientDescentLineSearch(learningRate=1.0,
                                                        numberOfIterations=200,
                                                        convergenceMinimumValue=1e-5,
                                                        convergenceWindowSize=9)
            R.SetOptimizerScalesFromPhysicalShift( )
            R.SetInitialTransform(tx)
            R.SetInterpolator(sitk.sitkLinear)
            R.SetShrinkFactorsPerLevel([6,2,1])
            R.SetSmoothingSigmasPerLevel([6,2,1])
            if not (fixed_mask_np is None):
                fixed_mask  = sitk.Cast(sitk.GetImageFromArray(fixed_mask_np),
                                            sitk.sitkUInt8)
                R.SetMetricFixedMask(fixed_mask)

            outTx = R.Execute(fixed, moving)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetTransform(outTx)

            moving_reg = resampler.Execute(moving)
            moving_reg_np = sitk.GetArrayFromImage(moving_reg)
            
            reged_im[i, ...] = moving_reg_np
        
        reged_im[-1, ...] = im[-1,:,:]
    
        return reged_im
    except:
        return None
    
def regToLast(im):
    target_im = np.zeros_like(im)
    for i in range(im.shape[0]):
        target_im[i, ...] = im[-1, ...]
    reged_im = regMOLLI(im, target_im)
    return reged_im

# Calculate the similarity between images
def mutual_information(image1, image2):
    hgram, x_edges, y_edges = np.histogram2d(image1.ravel(),image2.ravel(),bins=20)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# Perform Xue's paper
def fitting_and_reg(img, Invtime, iteration=4):
    Img = img.copy()
    dT1LL = T1LLmap(Img, Invtime)
    T1map = dT1LL['T1map']
    Amap = dT1LL['Amap']
    Bmap = dT1LL['Bmap']
    T1starmap = dT1LL['T1starmap']
    for jj in tqdm(range(iteration), leave = False):
        synImg = synImg_cal2(Invtime, Amap, Bmap, T1starmap)
        for ii in range(Img.shape[0]):
            syn_temp = synImg[ii]
            syn_temp[syn_temp<0] = 0
            threshold = np.percentile(syn_temp,99)
            syn_temp[syn_temp > threshold] = threshold
            syn_temp = medfilt2d(syn_temp)
            _, resampler = reg_spline_MI(syn_temp, medfilt2d(Img[ii]), meshsize=7)
            Img[ii] = reg_move_it(resampler,Img[ii])
        dT1LL = T1LLmap(Img, Invtime)
        T1map = dT1LL['T1map']
        Amap = dT1LL['Amap']
        Bmap = dT1LL['Bmap']
        T1starmap = dT1LL['T1starmap']
        
    return Img, dT1LL

# Calculate synthetic images
def synImg_cal2(synTI, Amap, Bmap, T1starmap):
    '''
    synTI = np.arange(0,2500,10)
    synImg = synImg_cal(synTI, Amap_pre, Bmap_pre, T1starmap_pre)
    '''
    j,k = Amap.shape
    Amap[np.nonzero(T1starmap==0)]=0
    Bmap[np.nonzero(T1starmap==0)]=0
    T1starmap[np.nonzero(T1starmap==0)]=1e-10

    if isinstance(synTI, np.ndarray):
        synImg = np.empty((synTI.size,j,k), dtype=np.double)
        for ii in range(synTI.size):
            synImg[ii] = abs(Amap-Bmap*np.exp(-synTI[ii]/T1starmap))
    else:
        synImg = Amap * 0.0
        synImg = abs(Amap-Bmap*np.exp(-synTI/T1starmap))

    return synImg

# Calculate function parameters and T1map
def T1LLmap(im, Invtime, threshold = 40):
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

# Regstration
def reg_spline_MI(fixed_np, moving_np, fixed_mask_np=None, meshsize=10):
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed_np), sitk.sitkFloat32)
    moving  = sitk.Cast(sitk.GetImageFromArray(moving_np), sitk.sitkFloat32)

    sitk.sitkUInt8
    transfromDomainMeshSize=[meshsize]*moving.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed,
                                          transfromDomainMeshSize )


    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(128)
    R.SetOptimizerAsGradientDescentLineSearch(learningRate=0.1,
                                              numberOfIterations=500,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    R.SetOptimizerScalesFromPhysicalShift( )
    R.SetInitialTransform(tx)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetShrinkFactorsPerLevel([6,2,1])
    R.SetSmoothingSigmasPerLevel([6,2,1])
    if not (fixed_mask_np is None):
        fixed_mask  = sitk.Cast(sitk.GetImageFromArray(fixed_mask_np),
                                sitk.sitkUInt8)
        R.SetMetricFixedMask(fixed_mask)

    outTx = R.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(outTx)

    moving_reg = resampler.Execute(moving)
    moving_reg_np = sitk.GetArrayFromImage(moving_reg)
    
    return moving_reg_np, resampler

def reg_move_it(resampler, img_np):
    if len(img_np.shape) == 2:
        img  = sitk.Cast(sitk.GetImageFromArray(img_np), sitk.sitkFloat32)
        img_reg = resampler.Execute(img)
        img_reg_np = sitk.GetArrayFromImage(img_reg)
    else: #suppose to be 3
        img_reg_np=img_np.copy()
        for ii in range(img_np.shape[0]):
            img  = sitk.Cast(sitk.GetImageFromArray(img_np[ii]), sitk.sitkFloat32)
            img_reg = resampler.Execute(img)
            img_reg_np[ii] = sitk.GetArrayFromImage(img_reg)

    return img_reg_np

def regToVMTX(im, invtime, iteration = 4):
    model_onnx_path = 'tigermyo/VMT.onnx'
    session = onnxruntime.InferenceSession(model_onnx_path)
    session.get_modelmeta()

    last_im = im[-1, ...]
    x = (last_im/last_im.max()).astype(np.float32)
    x = np.stack([x])
    x = np.stack([x])

    output = session.run(None, {"input": x})
    target_im = np.squeeze(np.array(output[0]))
    for i in range(im.shape[0]-1):
        target_im[i, ...] = target_im[i, ...] * im[i, ...].max()

    print("Reg to VMT")
    reged_im = regMOLLI_MI(im, target_im)

    if iteration > 0:
        print("Use Xue's method to improve motion correction")
    reged_im, dT1LL = fitting_and_reg(reged_im, invtime, iteration=iteration)

    return reged_im, dT1LL