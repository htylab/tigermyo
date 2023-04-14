from tigermyo import utils
from tigermyo import predict
from tigermyo import layering
from tigermyo import aha
from tigermyo import registration
from tigermyo import selector
import argparse
import numpy as np
import platform
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_B', '--input_B',type=str, default=None, help="Path to the input basal section of MOLLI RAW data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-input_M', '--input_M', type=str, default=None, help="Path to the input mid section of MOLLI RAW data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-input_A', '--input_A', type=str, default=None, help="Path to the input apical section of MOLLI RAW data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-input_B_MOCO', '--input_B_MOCO', type=str, default=None, help="Path to the input basal section of MOLLI MOCO data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-input_M_MOCO', '--input_M_MOCO', type=str, default=None, help="Path to the input mid section of MOLLI MOCO data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-input_A_MOCO', '--input_A_MOCO', type=str, default=None, help="Path to the input apical section of MOLLI MOCO data,it's a folder for the specific format(nii.gz)")
    parser.add_argument('-iteration_VMT', "--iteration_VMT", type=int, default=4, help="Iterative VMTX registration times")
    parser.add_argument('-layering_mode', "--layering_mode", type=bool, default=False, help="Take the middle layer of LVM as output")
    parser.add_argument('-output', "--output", type=str, default=None, help="Path to output AHA17 result,if there is no value will only display the figure")
    args = parser.parse_args()
    run_args(args)
    
def run(input_B = None, input_M = None, input_A = None, input_B_MOCO = None, input_M_MOCO = None, input_A_MOCO = None, output = None, iteration_VMT = 4, layering_mode = False):
    args = argparse.Namespace()

    args.input_B = input_B
    args.input_M = input_M
    args.input_A = input_A
    args.input_B_MOCO = input_B_MOCO
    args.input_M_MOCO = input_M_MOCO
    args.input_A_MOCO = input_A_MOCO
    args.iteration_VMT = iteration_VMT
    args.layering_mode = layering_mode
    args.output = output
    
    return run_args(args)       

def run_args(args):
    path_bundles = [[args.input_B, args.input_B_MOCO], [args.input_M, args.input_M_MOCO], [args.input_A, args.input_A_MOCO]]
    
    T1maps = []
    masks = []
    
    for i, path_bundle in enumerate(path_bundles):
        if path_bundle[0] is not None:
            if i == 0: print("Start processing basal section")
            elif i == 1: print("Start processing mid section")
            elif i == 2: print("Start processing apical section")
            
            print(f"Load Data: {path_bundle[0]}")
            im, invtime = utils.read_molli_dir(path_bundle[0])
            if path_bundle[1] is not None:
                print(f"Load Data: {path_bundle[1]}")
                moco_im, invtime = utils.read_molli_dir(path_bundle[1])
            else:
                moco_im = None
            
            print("Starting LMT Reg\r", end='')
            regedToLast_im = registration.regToLast(im)
            print("Finish LMT Reg")
            print(f"VMTX Reg will Iterate {args.iteration_VMT} times")
            print("Starting VMTX Reg\r", end='')
            regedToVMTX_im, _ = registration.regToVMTX(im, invtime, iteration=args.iteration_VMT)
            print("Finish VMTX Reg")
            
            reged_ims = [im, regedToLast_im, regedToVMTX_im, moco_im]
            print("\nSelecting best registration result")
            reged_ims_table, FQIs = selector.selector(reged_ims, invtime)
            best_idx = np.argmax(np.array(FQIs))
            print("FQI")
            print(f"RAW: {FQIs[0]:03f}")
            print(f"LMT: {FQIs[1]:03f}")
            print(f"VMTX: {FQIs[2]:03f}")
            if FQIs[3] != float('-inf'):
                print(f"MOCO: {FQIs[3]:03f}\n")
            
            if best_idx == 0: print("Select RAW result")
            elif best_idx == 1: print("Select LMT result")
            elif best_idx == 2: print("Select VMTX result")
            else: print("Select MOCO result")
            print()
                
            T1map = reged_ims_table[best_idx]['T1map']
            
            mask = predict.predict_mask(T1map)
            
            if args.layering_mode:
                mask_layer = layering.layering_percent(mask, 2/3, 1/3)
                
                import matplotlib.pyplot as plt
                
                plt.imshow(mask)
                plt.show()
                plt.imshow(mask_layer[0])
                plt.show()
                plt.imshow(mask_layer[1])
                plt.show()
                plt.imshow(mask_layer[2])
                plt.show()
                
                LV = mask == 1
                LVM = mask_layer[1]
                RV = mask == 3
                
                mask = LV * 1 + LVM * 2 + RV * 3
                
            
            T1maps.append(T1map)
            masks.append(mask)
        else:
            if i == 0: print("No basal section data")
            elif i == 1: print("No mid section data")
            elif i == 2: print("No apical section data")
            
            T1maps.append(None)
            masks.append(None)
            
            print()
    
    data = aha.get_aha17(masks[0], masks[1], masks[2], T1maps[0], T1maps[1], T1maps[2])
    
    aha.draw_aha17(data, args.output)   
    
if __name__ == "__main__":
    main()
    
    if platform.system() == 'Windows':
        os.system('pause')