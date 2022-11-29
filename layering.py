from skimage.morphology import dilation, disk
import math

def layering(mask, num_area = 3):
    from skimage.morphology import dilation, disk
    
    temp = mask*0
    i = 1
    mask_con = []
    while(temp.sum()!=(mask==2).sum()):
        temp = (dilation(mask==1, disk(i))&(mask==2))*1
        if i == 1:
            mask_con.append(temp)
        else:
            mask_con.append(temp-temp_con)
        temp_con = temp
        i = i + 1

    if num_area<len(mask_con):
        rem = len(mask_con)%(num_area)
        div = len(mask_con)//(num_area)
        idx = len(mask_con)-1
        mask_out = []
        while(idx>=0):
            temp = mask*0
            for i in range(div):
                temp = temp + mask_con[idx-i]
            if rem>0:
                temp = temp + mask_con[idx-div]
                rem = rem - 1
                idx = idx - div - 1
            else:
                idx = idx - div
            mask_out.append(temp)
    else:
        print(f'Max num is {len(mask_con)-1}')
        mask_out = mask_con
        
    return mask_out

def layering_percent(mask, max_percent = 0.8, min_percent = 0.2):
    temp = mask*0
    i = 1
    mask_con = []
    while(temp.sum()!=(mask==2).sum()):
        temp = (dilation(mask==1, disk(i))&(mask==2))*1
        if i == 1:
            mask_con.append(temp)
        else:
            mask_con.append(temp-temp_con)
        temp_con = temp
        i = i + 1

    if (max_percent>=min_percent) & (max_percent<=1) & (min_percent>=0):
        idx_max = math.ceil((len(mask_con)-1)*max_percent)
        idx_min = math.floor((len(mask_con)-1)*min_percent)
        mask_out = []
        
        temp = mask*0
        for i in range(idx_min):
            temp = temp + mask_con[i]
        mask_out.append(temp)
        
        temp = mask*0
        for i in range(idx_min,idx_max+1):
            temp = temp + mask_con[i]
        mask_out.append(temp)
        
        temp = mask*0
        for i in range(idx_max+1,len(mask_con)):
            temp = temp + mask_con[i]
        mask_out.append(temp)
    else:
        print('wrong range!')
        mask_out = mask_con
        
    return mask_out