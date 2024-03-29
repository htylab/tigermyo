import numpy as np
import onnxruntime
import matplotlib.pyplot as plt
from tigermyo.utils import checkFileExit

def predict_mask(t1map):
    model_onnx = 'SegT1map.onnx'
    
    model_onnx = checkFileExit(model_onnx, "https://github.com/htylab/tigermyo/releases/download/v0.0.1/SegT1map.onnx")
    
    session = onnxruntime.InferenceSession(model_onnx)
    session.get_modelmeta()
    
    t1map[t1map<300] = 0
    t1map[t1map>2500] = 2500

    x = (t1map/t1map.max()).astype(np.float32)
    x = np.stack([x])
    x = np.stack([x])

    output = session.run(None, {"input": x})
    mask = np.array(output[0].argmax(1).squeeze(), dtype='uint8')
    return mask

if __name__ == '__main__':
    
    ###############   ENTER FILE PATH   ###############
    mat = np.load("FILE PATH")
    ###################################################
    
    mask = predict_mask(mat["T1map"])
    plt.figure()
    plt.imshow(mask, vmax=3, vmin=0)
    plt.show()