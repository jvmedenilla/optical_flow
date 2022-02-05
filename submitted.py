'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def smooth_video(x, sigma, L):
    '''
    y = smooth_video(x, sigma, L)
    Smooth the video using a sampled-Gaussian smoothing kernel.

    x (TxRxC) - a video with T frames, R rows, C columns
    sigma (scalar) - standard deviation of the Gaussian smoothing kernel
    L (scalar) - length of the Gaussian smoothing kernel
    y (TxRxC) - the same video, smoothed in the row and column directions.
    '''
    num_frames, num_rows, num_columns = x.shape
    
    y = np.zeros((num_frames, num_rows, num_columns))
    
    tmp = np.zeros((num_frames, num_rows, num_columns))
    h_row = np.zeros((L))
    for n in range(0, L):
        h_row[n] = (1/np.sqrt(2*np.pi*(sigma**2)) * np.exp(((-0.5)*((n-(L-1)/2)/sigma)**2)))    

    for k in range(0, num_frames):
        for m in range(0, num_rows):
            #z = np.convolve(h_row, x[1][m])
            tmp[k,m,:] = np.convolve(h_row, x[k,m,:], mode='same')
        for n in range(0, num_columns):
            y[k,:,n] = np.convolve(h_row, tmp[k,:,n], mode='same')
            
    return y

def gradients(x):
    '''
    gt, gr, gc = gradients(x)
    Compute gradients using a first-order central finite difference.

    x (TxRxC) - a video with T frames, R rows, C columns
    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    '''
    h  = [0.5, 0, -0.5]
    T = x.shape[0]
    R = x.shape[1]
    C = x.shape[2]
    
    gt = np.zeros((T, R, C))
    gr = np.zeros((T, R, C))
    gc = np.zeros((T, R, C))

    for row in range(R):
        for column in range(C):
            gt[:,row,column] = np.convolve(x[:,row,column], h, mode='same')
            
    for t in range(T):
        for row in range(R):
            gc[t,row,:] = np.convolve(x[t,row,:], h, mode='same')
         
        for column in range(C):
            gr[t,:,column] = np.convolve(x[t,:,column], h, mode='same')
                    
    gt[0,:,:]=0
    gt[-1,:,:]=0 
    gr[:,0,:]=0 
    gr[:,-1,:]=0 
    gc[:,:,0]=0
    gc[:,:,-1]=0
    
    return gt,gr,gc


def lucas_kanade(gt, gr, gc, H, W):
    '''
    vr, vc = lucas_kanade(gt, gr, blocksize)

    gt (TxRxC) - gradient in the time direction
    gr (TxRxC) - gradient in the vertical direction
    gc (TxRxC) - gradient in the horizontal direction
    H (scalar) - height (in rows) of each optical flow block
    W (scalar) - width (in columns) of each optical flow block

    vr (Txint(R/H)xint(C/W)) - pixel velocity in vertical direction
    vc (Txint(R/H)xint(C/W)) - pixel velocity in horizontal direction
    '''
    T = gt.shape[0]
    R = gt.shape[1]
    C = gt.shape[2]
    vr = np.empty((T, R//H, C//W))
    vc = np.empty((T, R//H, C//W))

    for t in range(T):
        for row in range(R//H):
            for column in range(C//W):
                A2_matrix = gr[t,row*H:(row+1)*H,column*W:(column+1)*W].reshape(-1,1)
                A1_matrix = gc[t,row*H:(row+1)*H,column*W:(column+1)*W].reshape(-1,1)
                
                A_combined = np.hstack((A1_matrix, A2_matrix))
                b_vec = -1*gt[t,row*H:(row+1)*H,column*W:(column+1)*W].reshape(-1,1)
                r_vec = np.matmul(np.linalg.pinv(A_combined),b_vec)
                #print(r_vec)
                vr[t,row,column] = r_vec[1,0]
                vc[t,row,column] = r_vec[0,0]
        
    return vr, vc

def medianfilt(x, H, W):
    '''
    y = medianfilt(x, H, W)
    Median-filter the video, x, in HxW blocks.

    x (TxRxC) - a video with T frames, R rows, C columns
    H (scalar) - the height of median-filtering blocks
    C (scalar) - the width of median-filtering blocks
    y (TxRxC) - y[t,r,c] is the median of the pixels x[t,rmin:rmax,cmin:cmax], where
      rmin = max(0,r-int((H-1)/2))
      rmax = min(R,r+int((H-1)/2)+1)
      cmin = max(0,c-int((W-1)/2))
      cmax = min(C,c+int((W-1)/2)+1)
    '''
    T, R, C = x.shape
    y = x.copy()
    
    for t in range(T):
        for r in range(R):
            for c in range(C):
                rmin = max(0,r-int((H-1)/2))
                rmax = min(R,r+int((H-1)/2)+1)
                cmin = max(0,c-int((W-1)/2))
                cmax = min(C,c+int((W-1)/2)+1)
                y[t,r,c] = np.median(x[t, rmin:rmax, cmin:cmax])
                
                   
    return y

            
def interpolate(x, U):
    '''
    y = interpolate(x, U)
    Upsample and interpolate an image using bilinear interpolation.

    x (TxRxC) - a video with T frames, R rows, C columns
    U (scalar) - upsampling factor
    y (Tx(U*R)x(U*C)) - interpolated image
    '''
    T, R, C = x.shape
    y = np.zeros((T,R*U,C*U))
    tmp = y.copy()
    
    h_pwl = np.zeros(((2*U) +1))

    for n in range(0, U+1):
        h_pwl[U+n] = (1/U)*(U-n)
        h_pwl[U-n] = (1/U)*(U-n)
    
    for t in range(T):
        for r in range(R):
            for c in range(C):
                tmp[t,r*U,c*U] = x[t,r,c]
                
    tmp2 = tmp.copy()

    for t in range(T):
        for r in range(R*U):
            tmp2[t, r,:] = np.convolve(tmp[t,r,:],h_pwl,mode='same')
        for c in range(C*U):
            y[t, :,c] = np.convolve(tmp2[t,:,c],h_pwl,mode='same')

    return y

def scale_velocities(v, U):
    '''
    delta = scale_velocities(v, U)
    Scale the velocities in v by a factor of U,
    then quantize them to the nearest integer.
    
    v (TxRxC) - T frames, each is an RxC velocity image
    U (scalar) - an upsampling factor
    delta (TxRxC) - integers closest to v*U
    '''
    delta = v.copy()
    T,R,C = delta.shape
    for t in range(T):
        for r in range(R):
            delta[t, r,:] = np.round(np.multiply((v[t,r,:]),U),0)
         #   print(int.delta[t, r,:])
        for c in range(C):
            delta[t, :,c] = np.round(np.multiply((v[t,:,c]),U),0)
            
                
    return delta

def velocity_fill(x, vr, vc, keep):
    '''
    y = velocity_fill(x, vr, vc, keep)
    Fill in missing frames by copying samples with a shift given by the velocity vector.

    x (T,R,C) - a video signal in which most frames are zero
    vr (T,Ra,Cb) - the vertical velocity field, integer-valued
    vc (T,Ra,Cb) - the horizontal velocity field, integer-valued
        Notice that Ra and Cb might be less than R and C.  If they are, the remaining samples 
        of y should just be copied from y[t-1,r,c].
    keep (array) -  a list of frames that should be kept.  Every frame not in this list is
     replaced by samples copied from the preceding frame.

    y (T,R,C) - a copy of x, with the missing frames filled in.
    '''
    T, R, C = x.shape
    Ra, Cb = vr.shape[1:3]
    y = x.copy()

    
    for t in range(T):
        if t in keep:
            continue
        else:
            for r in range(R):
                for c in range(C):
                    #if t == 1:
                    #    continue
                    if r>=Ra or c>=Cb:
                        y[t,r,c] = y[t-1,r,c]
                    else:
                        y[t,r,c] = y[t-1, int(max(0,min(R-1,r - (vr[t-1,r,c])))), int(max(0, min(C-1,c - (vc[t-1,r,c]))))]
                        
    return y