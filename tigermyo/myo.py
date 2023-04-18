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
    parser.add_argument('-only_use', "--only_use", type=int, default=0, help="Only use a certain method to calculate (1:input 2:LMT 3:VMTX)")
    parser.add_argument('-iteration_VMTX', "--iteration_VMTX", type=int, default=4, help="Iterative VMTX registration times")
    parser.add_argument('-layering_mode', "--layering_mode", action='store_true', help="Take the middle layer of LVM as output")
    parser.add_argument('-output', "--output", type=str, default=None, help="Path to output AHA17 result,if there is no value will only display the figure")
    args = parser.parse_args()
    run_args(args)
    
def run(config):
    args = argparse.Namespace()

    args.input_B = config['input_B'] if 'input_B' in config.keys() else None
    args.input_M = config['input_M'] if 'input_M' in config.keys() else None
    args.input_A = config['input_A'] if 'input_A' in config.keys() else None
    args.input_B_MOCO = config['input_B_MOCO'] if 'input_B_MOCO' in config.keys() else None
    args.input_M_MOCO = config['input_M_MOCO'] if 'input_M_MOCO' in config.keys() else None
    args.input_A_MOCO = config['input_A_MOCO'] if 'input_A_MOCO' in config.keys() else None
    args.only_use = config['only_use'] if 'only_use' in config.keys() else 0
    args.iteration_VMTX = config['iteration_VMTX'] if 'iteration_VMTX' in config.keys() else 4
    args.layering_mode = config['layering_mode'] if 'layering_mode' in config.keys() else False
    args.output = config['output'] if 'output' in config.keys() else None

    return run_args(args)       

def run_args(args):
    path_bundles = [[args.input_B, args.input_B_MOCO], [args.input_M, args.input_M_MOCO], [args.input_A, args.input_A_MOCO]]

    T1maps = []
    masks = []

    for i, path_bundle in enumerate(path_bundles):
        if path_bundle[0] is not None:
            if i == 0: print(f"Start processing basal section {'with layering mode' if args.layering_mode else ''}")
            elif i == 1: print(f"Start processing mid section {'with layering mode' if args.layering_mode else ''}")
            elif i == 2: print(f"Start processing apical section {'with layering mode' if args.layering_mode else ''}")

            if args.only_use == 0: print('Process all methods!')
            elif args.only_use == 1: print('Only process input!')
            elif args.only_use == 2: print('Only process LMT!')
            elif args.only_use == 3: print('Only process VMTX!')

            print(f"Load Data: {path_bundle[0]}")
            im, invtime = utils.read_molli_dir(path_bundle[0])
            if path_bundle[1] is not None:
                print(f"Load Data: {path_bundle[1]}")
                moco_im, invtime = utils.read_molli_dir(path_bundle[1])
            else:
                moco_im = None
            print("Finish Data Loading!\n")

            reged_ims = []
            if args.only_use == 0 or args.only_use == 1:
                reged_ims.append(im)

            if args.only_use == 0 or args.only_use == 2:
                print("Starting LMT Reg...")
                regedToLast_im = registration.regToLast(im)
                reged_ims.append(regedToLast_im)
                print("Finish LMT Reg!")

            if args.only_use == 0 or args.only_use == 3:
                print(f"VMTX Reg will iterate {args.iteration_VMTX} times")
                print("Starting VMTX Reg...")
                regedToVMTX_im, _ = registration.regToVMTX(im, invtime, iteration=args.iteration_VMTX)
                reged_ims.append(regedToVMTX_im)
                print("Finish VMTX Reg!\n")

            if args.only_use == 0 and path_bundle[1] is not None:
                reged_ims.append(moco_im)

            print("Selecting best registration result...")
            reged_ims_table, FQIs = selector.selector(reged_ims, invtime)
            best_idx = np.argmax(np.array(FQIs))
            print("FQI")
            if args.only_use == 0:
                print(f"RAW: {FQIs[0]:03f}")
                print(f"LMT: {FQIs[1]:03f}")
                print(f"VMTX: {FQIs[2]:03f}")
                if FQIs[3] != float('-inf'):
                    print(f"MOCO: {FQIs[3]:03f}")
            elif args.only_use == 1:
                print(f"RAW: {FQIs[0]:03f}")
            elif args.only_use == 2:
                print(f"LMT: {FQIs[0]:03f}")
            elif args.only_use == 3:
                print(f"VMTX: {FQIs[0]:03f}")

            if args.only_use == 0:
                if best_idx == 0: print("Select RAW result!\n")
                elif best_idx == 1: print("Select LMT result!\n")
                elif best_idx == 2: print("Select VMTX result!\n")
                else: print("Select MOCO result!\n")
            elif args.only_use == 1:
                print("Select RAW result!\n")
            elif args.only_use == 2:
                print("Select LMT result!\n")
            elif args.only_use == 3:
                print("Select VMTX result!\n")
                
            T1map = reged_ims_table[best_idx]['T1map']
            
            mask = predict.predict_mask(T1map)
            
            if args.layering_mode:
                mask_layer = layering.layering_percent(mask, 2/3, 1/3)
                
                LV = mask == 1
                LVM = mask_layer[1]
                RV = mask == 3
                
                mask = LV * 1 + LVM * 2 + RV * 3
                
            T1maps.append(T1map)
            masks.append(mask)
        else:
            if i == 0: print("No basal section data!\n")
            elif i == 1: print("No mid section data!\n")
            elif i == 2: print("No apical section data!\n")
            
            T1maps.append(None)
            masks.append(None)
            
            print()
    
    data = aha.get_aha17(masks[0], masks[1], masks[2], T1maps[0], T1maps[1], T1maps[2])
    
    aha.draw_aha17(data, args.output)
    
if __name__ == "__main__":
    main()
    
    if platform.system() == 'Windows':
        os.system('pause')