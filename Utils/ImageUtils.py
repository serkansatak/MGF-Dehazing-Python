import numpy as np
import cv2
from numpy import matlib


def getImagePyramid(frame):
    images = []
    images.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
    for i in range(3):
        images.append(cv2.pyrDown(images[-1]))
    return images


def boxfilter(imSrc: np.ndarray, r: float) -> np.ndarray:
    """
    BOXFILTER   O(1) time box filtering using cumulative sum

    - Definition imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)));
    - Running time independent of r; 
    - Equivalent to the function: colfilt(imSrc, [2*r+1, 2*r+1], 'sliding', @sum);
    - But much faster.
    """
    height, width = imSrc.shape[:2]
    imDst = np.zeros((height, width), dtype=np.float32)
    
    # Cumulative sum over Y axis
    imCum = np.cumsum(imSrc, 0)
    
    # Difference over Y axis
    imDst[0: r+1, :] = imCum[r: 2*r+1, :]
    imDst[r+1: height-r, :] = imCum[2*r+1: height, :] - imCum[0: height-2*r-1, :]
    imDst[height-r: height, :] = matlib.repmat(imCum[np.newaxis, height-1, :], r, 1) - imCum[height-2*r-1: height-r-1, :]
    
    # Cumulative sum over X axis
    imCum = np.cumsum(imDst, 1)
    
    # Difference over Y axis
    imDst[:, 0: r+1] = imCum[:, r: 2*r+1]
    imDst[:, r+1: width-r] = imCum[:, 2*r+1: width] - imCum[:, 0: width-2*r-1]
    
    imDst[:, width-r: width] = matlib.repmat(imCum[:, width-1, np.newaxis], 1, r) - imCum[:, width-2*r-1: width-r-1]
    return imDst


def maxk(mat: np.ndarray, dim: int, k: int) -> np.ndarray:
    """
    Sorts ndarrays along 'dim' dimension and 
    returns 'k' number of maximums for each.
    """
    mat = mat.flatten('F')[:,np.newaxis]
    
    # Sort the matrix along the specified dimension and get the indices
    indices = np.argsort(mat, axis=dim)
    # Get the indices of the maximum k values along the specified dimension
    max_indices = np.take(indices, np.arange(-k, 0), axis=dim)
    # Get the corresponding values in the matrix
    max_values = np.take_along_axis(mat, max_indices, axis=dim)
    return max_values


def split_xy(mat):
    """
    Divides matrix into 4 parts
    """
    row, col = mat.shape[:2]
    upL = mat[:round(row/2), :round(col/2)]
    upR = mat[:round(row/2), round(col/2):]
    lowL = mat[round(row/2):, :round(col/2)]
    lowR = mat[round(row/2):, round(col/2):]
    return upL, upR, lowL, lowR


def ham_chia_tren(im, args):

    # Split image into 4 parts    
    splitParts = split_xy(im)

    # Find mean value of each part
    means = [np.mean(mat) for mat in splitParts]    

    # Find max of mean value
    x = means.copy()
    x.sort(reverse=True)
    x = x[:2]

    im = splitParts[means.index(x[0])]
    args.ST = x[0]-x[1]
    return im


def get_size_image_i(Gray_image, args):
    
    # Divide image to 4 parts.
    upL, upR, _ , _ =split_xy(Gray_image)
    mUpL, mUpR = [np.mean(mat) for mat in [upL, upR]]
    
    # find max of mean value
    x = max(mUpL, mUpR)
    if x == mUpL:
        im = upL
        while args.ST > args.threshold:
            im = ham_chia_tren(im, args)
    elif x == mUpR:
        im = upR
        while args.ST > args.threshold:
            im = ham_chia_tren(im, args)        
    return im


def fast_gradient(im:np.ndarray, p: np.ndarray, N: np.ndarray, N1: np.ndarray, args) -> np.ndarray:
    """
    GUIDEDFILTER   O(1) time implementation of guided filter.

    - guidance image: I (should be a gray-scale/single channel image)
    - filtering input image: p (should be a gray-scale/single channel image)
    - regularization parameter: eps
    """
    
    eps = args.eps ** 2
    
    s_end = tuple(reversed(im.shape[:2]))
    s_start = (args.width, args.height)

    im_sub = cv2.resize(im, s_start, interpolation=cv2.INTER_NEAREST) # NN is often enough
    p_sub = cv2.resize(p, s_start, interpolation=cv2.INTER_NEAREST)
    
    mean_im = np.divide(boxfilter(im_sub, args.r), N)
    mean_p = np.divide(boxfilter(p_sub, args.r), N)
    mean_imp = np.divide(boxfilter(np.multiply(im_sub, p_sub), args.r), N)
    # Covariance matrix of (im, p) in each local patch
    cov_imp = mean_imp - np.multiply(mean_im, mean_p) 
    mean_imim = np.divide(boxfilter(np.multiply(im_sub, im_sub), args.r), N)
    var_im = mean_imim - np.multiply(mean_im, mean_im)
    
    # Weight
    epsilon = (0.01 * (np.max(p_sub) - np.min(p_sub)))**2
    
    # N1 = boxfilter(np.ones((args.heigt, args.width), dtype=np.float32), args.r1);  
    # the size of each local patch; N=(2r+1)^2 except for boundary pixels.
    
    mean_im1 = np.divide(boxfilter(im_sub, args.r1), N1)
    mean_imim1 = np.divide(boxfilter(np.multiply(im_sub,im_sub), args.r1), N1)
    var_im1 = mean_imim1 - np.multiply(mean_im1, mean_im1)
    
    chi_im = np.sqrt(np.abs(var_im1, var_im))
    weight = (chi_im + epsilon) / (np.mean(chi_im) + epsilon)
    
    gamma = (4/np.mean(chi_im) - np.min(chi_im)) * (chi_im - np.mean(chi_im))
    gamma = 1 - np.divide(1, (1 + np.exp(gamma)))
    
    # Result
    a = (cov_imp + np.divide(np.multiply(np.divide(eps, weight), gamma), (var_im + np.divide(eps, weight))))
    b = mean_p - np.multiply(a, mean_im)
    
    mean_a = np.divide(boxfilter(a, args.r), N)
    mean_b = np.divide(boxfilter(b, args.r), N)
    mean_a = cv2.resize(mean_a, s_end, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_b, s_end, interpolation=cv2.INTER_LINEAR)
    
    q = np.multiply(mean_a, im) + mean_b
    return q


def pyramid_dehazing(rgb, im0, im1, im2, args, A=None):
    
    N = boxfilter(np.ones((args.height, args.width), dtype=np.float32), args.r)
    NN = boxfilter(np.ones((args.height, args.width), dtype=np.float32), args.rr)
    
    if A==None:
        # Estimation of A
        im_dv = get_size_image_i(im2, args)
        u, v = im_dv.shape[:2]
        A0 = np.mean(maxk(im_dv, dim=0, k=round(u*v*0.05)))
        A = min(A0, 0.99)
    
    # Estimation of T
    max_im = np.max(im2)
    min_im = np.min(im2)
    
    q = np.multiply(args.gamma * (im2-min_im) / (max_im - min_im) , im2)
    
    bounds = [3, 2, 1, 0]
    
    for bound in bounds:
        if min_im > bound/10:
            j2 = q + ( im2 / (2**bound) )
    
    # Level 2
    j2 = np.minimum(j2, max_im)
    t2 = np.multiply((A-im2), (A-j2))
    t2 = np.abs(t2)
    
    t2_refine = t2
    t2_refine[im2>=A] = 1 / np.max(t2) * t2[im2>=A] 
    t2 = t2_refine
    
    m, n = t2.shape[:2]
    r1 = (m*2, n*2)
    
    # Level 1
    t1_raw = cv2.resize(t2, r1, interpolation = cv2.INTER_LINEAR)
    print(t1_raw)
    t1_final = fast_gradient(im1,t1_raw, N, NN, args)
    t1_final = np.maximum(t1_final, 0.05)
    
    m, n = t1_final.shape[:2]
    r0 = (m*2, n*2)
    
    # Level 0
    t0_raw = cv2.resize(t1_final, r0, interpolation = cv2.INTER_LINEAR)
    t0_rough = t0_raw.copy()
    
    t0 = fast_gradient(im0, t0_raw, N, NN, args)
    t0 = np.maximum(t0, 0.05)
    
    # Haze-free image
    result = np.multiply(rgb-A, np.repeat(t0[:,:,np.newaxis], 3, axis=2)) - A
    return result, t0_rough, A
