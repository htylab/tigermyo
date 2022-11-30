import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import onnxruntime

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
    
def regToLast(im):
    target_im = np.zeros_like(im)
    for i in range(im.shape[0]):
        target_im[i, ...] = im[-1, ...]
    reged_im = regMOLLI(im, target_im)
    return reged_im

def regToLast2All(im):
    model_onnx_path = 'tigermyo/Last2AllNonpretrain.onnx'
    session = onnxruntime.InferenceSession(model_onnx_path)
    session.get_modelmeta()
    
    last_im = im[-1, ...]
    x = (last_im/last_im.max()).astype(np.float32)
    x = np.stack([x])
    x = np.stack([x])

    output = session.run(None, {"input": x})
    target_im = np.squeeze(np.array(output[0]))
    for i in range(im.shape[0]):
        target_im[i, ...] = target_im[i, ...] * im[i, ...].max()
    
    reged_im = regMOLLI(im, target_im)
    return reged_im

def regToLast2T1starAB(im, invtime, amap, bmap):
    model_onnx_path = 'tigermyo/Last2T1starABPretrain.onnx'
    session = onnxruntime.InferenceSession(model_onnx_path)
    session.get_modelmeta()
    
    last_im = im[-1, ...]
    x = (last_im/last_im.max()).astype(np.float32)
    x = np.stack([x])
    x = np.stack([x])

    output = session.run(None, {"input": x})
    fake_3ch = np.squeeze(np.array(output[0]))
    
    fake_t1star = fake_3ch[0, ...] * 2500
    fake_t1star[fake_t1star<300] = 300
    amap_th = np.percentile(amap, 95)
    fake_amap = fake_3ch[1, ...] * amap_th
    fake_amap[fake_amap<0] = 0
    bmap_th = np.percentile(bmap, 95)
    fake_bmap = fake_3ch[2, ...] * bmap_th
    fake_bmap[fake_bmap<0] = 0
    
    target_im = np.zeros_like(im)
    for j, each_invtime in enumerate(invtime):
        target_im[j, ...] = np.abs(fake_amap-fake_bmap*np.exp(-each_invtime/fake_t1star))
    
    reged_im = regMOLLI(im, target_im)
    return reged_im