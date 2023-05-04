import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_ahaseg(mask, nseg=6):
    from scipy import ndimage

    def mid_to_angles(mid, seg_num):
        anglelist = np.zeros(seg_num)
        if seg_num == 4:
            anglelist[0] = mid - 45 - 90
            anglelist[1] = mid - 45
            anglelist[2] = mid + 45
            anglelist[3] = mid + 45 + 90

        if seg_num == 6:
            anglelist[0] = mid - 120
            anglelist[1] = mid - 60
            anglelist[2] = mid
            anglelist[3] = mid + 60
            anglelist[4] = mid + 120
            anglelist[5] = mid + 180
        anglelist = (anglelist + 360) % 360
        
        angles = np.append(anglelist, anglelist[0])
        angles = np.rad2deg(np.unwrap(np.deg2rad(angles)))
    

        return angles.astype(int)

    def circular_sector(theta_range, lvb):
        cx, cy = ndimage.center_of_mass(lvb)
        max_range = np.min(np.abs(lvb.shape-np.array([cx, cy]))).astype(np.int64)
        r_range = np.arange(0, max_range, 0.1)
        theta = theta_range/180*np.pi
        z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
        xall = -np.imag(z) + cx
        yall = np.real(z) + cy
        
        
        smask = lvb * 0
        xall = np.round(xall.flatten())
        yall = np.round(yall.flatten())
        mask = (xall >= 0) & (yall >= 0) & \
               (xall < lvb.shape[0]) & (yall < lvb.shape[1])
        xall = xall[np.nonzero(mask)].astype(int)
        yall = yall[np.nonzero(mask)].astype(int)
        smask[xall, yall] = 1
        
        return smask


    lvb = (mask == 1)
    lvw = (mask == 2)
    rvb = (mask == 3)
    lx, ly = ndimage.center_of_mass(lvb)
    rx, ry = ndimage.center_of_mass(rvb)
    j = (-1)**0.5
    lvc = lx + j*ly
    rvc = rx + j*ry
    mid_angle = np.angle(rvc - lvc)/np.pi*180 - 90

    angles = mid_to_angles(mid_angle, nseg)
    AHA_sector = lvw * 0    
    for ii in range(angles.size-1):
        angle_range = np.arange(angles[ii], angles[ii + 1], 0.1)
        smask = circular_sector(angle_range, lvb)
        AHA_sector[smask > 0] = (ii + 1)

    label_mask = AHA_sector * lvw
    return label_mask

def bullseye_plot(ax, data, seg_bold=None, cmap=None, norm=None):
    """
    Bullseye representation for the left ventricle.

    Parameters
    ----------
    ax : axes
    data : list of int and float
        The intensity values for each of the 17 segments
    seg_bold : list of int, optional
        A list with the segments to highlight
    cmap : ColorMap or None, optional
        Optional argument to set the desired colormap
    norm : Normalize or None, optional
        Optional argument to normalize data into the [0.0, 1.0] range

    Notes
    -----
    This function creates the 17 segment model for the left ventricle according
    to the American Heart Association (AHA) [1]_

    References
    ----------
    .. [1] M. D. Cerqueira, N. J. Weissman, V. Dilsizian, A. K. Jacobs,
        S. Kaul, W. K. Laskey, D. J. Pennell, J. A. Rumberger, T. Ryan,
        and M. S. Verani, "Standardized myocardial segmentation and
        nomenclature for tomographic imaging of the heart",
        Circulation, vol. 105, no. 4, pp. 539-542, 2002.
    """
    if seg_bold is None:
        seg_bold = []

    linewidth = 2
    data = np.ravel(data)

    if cmap is None:
        cmap = plt.cm.viridis

    if norm is None:
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())

    theta = np.linspace(0, 2 * np.pi, 768)
    r = np.linspace(0.2, 1, 4)

    # Create the bound for the segment 17
    for i in range(r.shape[0]):
        ax.plot(theta, np.repeat(r[i], theta.shape), '-k', lw=linewidth)

    # Create the bounds for the segments 1-12
    for i in range(6):
        theta_i = np.deg2rad(i * 60)
        ax.plot([theta_i, theta_i], [r[1], 1], '-k', lw=linewidth)

    # Create the bounds for the segments 13-16
    for i in range(4):
        theta_i = np.deg2rad(i * 90 - 45)
        ax.plot([theta_i, theta_i], [r[0], r[1]], '-k', lw=linewidth)

    # Fill the segments 1-6
    r0 = r[2:4]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2)) * data[i]
        ax.pcolormesh(theta0, r0-0.139, z, cmap=cmap, norm=norm, shading='auto')
        if i + 1 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[2], r[3]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[2], r[3]], '-k', lw=linewidth + 1)

    # Fill the segments 7-12
    r0 = r[1:3]
    r0 = np.repeat(r0[:, np.newaxis], 128, axis=1).T
    for i in range(6):
        # First segment start at 60 degrees
        theta0 = theta[i * 128:i * 128 + 128] + np.deg2rad(60)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((128, 2)) * data[i + 6]
        ax.pcolormesh(theta0, r0-0.139, z, cmap=cmap, norm=norm, shading='auto')
        if i + 7 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[1], r[2]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[1], r[2]], '-k', lw=linewidth + 1)

    # Fill the segments 13-16
    r0 = r[0:2]
    r0 = np.repeat(r0[:, np.newaxis], 192, axis=1).T
    for i in range(4):
        # First segment start at 45 degrees
        theta0 = theta[i * 192:i * 192 + 192] + np.deg2rad(45)
        theta0 = np.repeat(theta0[:, np.newaxis], 2, axis=1)
        z = np.ones((192, 2)) * data[i + 12]
        ax.pcolormesh(theta0, r0-0.139, z, cmap=cmap, norm=norm, shading='auto')
        if i + 13 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)
            ax.plot(theta0[0], [r[0], r[1]], '-k', lw=linewidth + 1)
            ax.plot(theta0[-1], [r[0], r[1]], '-k', lw=linewidth + 1)

    # Fill the segments 17
    if data.size == 17:
        r0 = np.array([0, r[0]])
        r0 = np.repeat(r0[:, np.newaxis], theta.size, axis=1).T
        theta0 = np.repeat(theta[:, np.newaxis], 2, axis=1)
        z = np.ones((theta.size, 2)) * data[16]
        ax.pcolormesh(theta0, r0, z, cmap=cmap, norm=norm, shading='auto')
        if 17 in seg_bold:
            ax.plot(theta0, r0, '-k', lw=linewidth + 2)

    ax.set_ylim([0, 1])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
def get_aha17(mask_B, mask_M, mask_A, T1map_B, T1map_M, T1map_A):
    T1 = []
    
    #########  Slice B  ############
    aha_seg_B = np.empty((1,1))
    aha_seg_B[:] = np.nan
    if mask_B.size != 1:
        aha_seg_B = get_ahaseg(mask_B, nseg=6)
        T1map_i = np.ndarray(shape=(6,T1map_B.shape[0],T1map_B.shape[1]))
        for i in range(6):
            T1map_i[i] = T1map_B*(aha_seg_B==i+1)
            T1map_i[i][T1map_i[i] == 0] = np.nan
            if np.nanmedian(T1map_i[i]) == np.nan:
                T1.append(0)
            else:
                T1.append(np.nanmedian(T1map_i[i]))     
    else:
        for i in range(6):
            T1.append(0)

    #########  Slice M  ############
    aha_seg_M = np.empty((1,1))
    aha_seg_M[:] = np.nan
    if mask_M .size != 1:
        aha_seg_M = get_ahaseg(mask_M, nseg=6)
        T1map_i = np.ndarray(shape=(6,T1map_M.shape[0],T1map_M.shape[1]))
        for i in range(6):
            T1map_i[i] = T1map_M*(aha_seg_M==i+1)
            T1map_i[i][T1map_i[i] == 0] = np.nan
            if np.nanmedian(T1map_i[i]) == np.nan:
                T1.append(0)
            else:
                T1.append(np.nanmedian(T1map_i[i]))     
    else:
        for i in range(6):
            T1.append(0)

    #########  Slice A  ############
    aha_seg_A = np.empty((1,1))
    aha_seg_A[:] = np.nan
    if mask_A.size != 1:
        aha_seg_A = get_ahaseg(mask_A, nseg=4)
        T1map_i = np.ndarray(shape=(4,T1map_A.shape[0],T1map_A.shape[1]))
        for i in range(4):
            T1map_i[i] = T1map_A*(aha_seg_A==i+1)
            T1map_i[i][T1map_i[i] == 0] = np.nan
            if np.nanmedian(T1map_i[i]) == np.nan:
                T1.append(0)
            else:
                T1.append(np.nanmedian(T1map_i[i]))     
    else:
        for i in range(4):
            T1.append(0)
        
    return T1[0:6], T1[6:12], T1[12:16], aha_seg_B, aha_seg_M, aha_seg_A

def draw_aha17(data, path=None):

    # Make a figure and axes with dimensions as desired.
    fig, ax = plt.subplots(figsize=(8, 8), nrows=1, ncols=1,
                        subplot_kw=dict(projection='polar'))
    fig.canvas.manager.set_window_title('Left Ventricle Bulls Eyes (AHA)')

    # Create the axis for the colorbars
    axl = fig.add_axes([0.06, 0.05, 0.9, 0.05])


    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=1100, vmax=1400)
    # Create an empty ScalarMappable to set the colorbar's colormap and norm.
    # The following gives a basic continuous colorbar with ticks and labels.
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
                cax=axl, orientation='horizontal') # cax=axl, orientation='horizontal', label='T1(ms)')
    cbar.ax.tick_params(labelsize=16)


    # Create the 17 segment model
    bullseye_plot(ax, data,cmap=cmap, norm=norm)
    # ax.set_title('Bulls Eye (AHA)')
    # temp_T = np.around(T1, decimals=2, out=None)
    temp_T = [int(x) for x in data]

    for text,xytext, color in zip(*[temp_T,[(0.465, 0.825),(0.2, 0.675),(0.165, 0.35),(0.465, 0.15),(0.775, 0.35),(0.735, 0.675),(0.465, 0.715),(0.275, 0.615),(0.275, 0.365),(0.465, 0.255),(0.66, 0.365),(0.66, 0.615),(0.465, 0.615),(0.345, 0.475),(0.465, 0.365),(0.595, 0.475)],['k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k']]):
        ax.annotate(text,
                    xy=(0,0),  # theta, radius
                    xytext=xytext,    # fraction, fraction
                    textcoords='figure fraction',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    color=color,
                    size=17
                    )

    if path is None:
        plt.show()
    else:
        plt.savefig(f'{path}.png')