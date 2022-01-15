import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance_matrix
import pandas as pd
from sklearn.cluster import KMeans
import tqdm

def load_dataset(dir_sc, images_per_class=None):
    inames = []
    ilabels = []
    cnames = sorted(os.listdir(dir_sc))
    for ilabel, cl in enumerate(cnames):
        dir_cl = os.path.join(dir_sc, cl)
        for iname in os.listdir(dir_cl)[:images_per_class]:
            inames.append(os.path.join(cl, iname))
            ilabels.append(ilabel)
    ilabels = np.array(ilabels)
    return inames, ilabels, cnames

def ComputeHoG(im, show = False):
    fd, hog_image = hog(im, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)
    if show == True:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(im, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return fd, hog_image

def resize_image(img, shape = (1024,1024)):
    return cv.resize(img, shape, interpolation =cv.INTER_LINEAR)

def ComputeHoGs(inames):

    Hogs = []
    features = []
    for i, x in tqdm.tqdm(enumerate(inames)):
        p = os.path.join(path, x)
        img = cv.imread(p)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = resize_image(gray)
        feature, hog = ComputeHoG(gray)
        Hogs.append(hog)
        features.append(feature)
    
    return features, Hogs


def ComputeSift_CV(I):
    gray= cv.cvtColor(I,cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    img = cv.drawKeypoints(gray,kp,I,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des

def ComputeSiftDataset_CV(inames):
    keypoints = []
    descriptors = []
    for x in tqdm.tqdm(inames):
        p = os.path.join(path, x)
        img = cv.imread(p)
        kp, des = ComputeSift_CV(img)
        keypoints.append(kp)
        descriptors.append(des)
    
    return keypoints, descriptors


def compute_split(length, seed=1337, pc=0.80):
    train_ids = np.random.RandomState(seed=seed).choice(
        length,
        size=int(length * pc),
        replace=False)
    test_ids = np.array(list(set(np.arange(length)) - set(train_ids)))
    return train_ids, test_ids


def compute_visual_dict(sift, n_clusters=1000, n_init=1, verbose=1):
    # reorder data
    dim_sift = sift[0].shape[-1]
    sift = [s.reshape(-1, dim_sift) for s in sift]
    sift = np.concatenate(sift, axis=0)
    # remove zero vectors
    keep = ~np.all(sift==0, axis=1)
    sift = sift[keep]
    # randomly pick sift
    ids, _ = compute_split(sift.shape[0], pc=0.05)
    sift = sift[ids]

    zeros_vect = np.zeros((128))
    kmeans = KMeans(n_clusters=n_clusters).fit(sift)
    centers = kmeans.cluster_centers_
    np.append(centers, zeros_vect)
    vdict = centers

    return vdict



def display_images(images):
    n_images,w,h = images.shape
    n = int(np.ceil(np.sqrt(n_images)))
    im = np.zeros((n*w, n*h))
    for k in range(n_images):
        i = k % n
        j = k // n
        im[i*w:i*w+w, j*h:j*h+h] = images[k]

    plt.figure(figsize=(0.7*n,0.7*n))
    plt.gray()
    plt.imshow(im)
    plt.axis('off')
    plt.show()
    
    
def dense_sampling(im, s=8):
    w, h = im.shape
    x = np.arange(0, w, s)
    y = np.arange(0, h, s)
    return x, y

def auto_padding(im, k=16, s=8):
    w, h = im.shape
    x = np.arange(0, w, s)
    y = np.arange(0, h, s)
    # last region could be smaller
    last_r = im[x[-1]:x[-1]+k, y[-1]:y[-1]+k]
    if last_r.shape == (k, k):
        return im
    dif_w = k - last_r.shape[0]
    dif_h = k - last_r.shape[1]
    n_im = np.zeros((w+dif_w, h+dif_h))
    id_w = dif_w // 2
    id_h = dif_h // 2
    n_im[id_w:id_w+w, id_h:id_h+h] = im
    return n_im


def conv_separable(im, h_x, h_y, pad=1):
    h_x = h_x.reshape(1,3)
    h_y = h_y.reshape(3,1)

    im_w, im_h = im.shape
    hx_w, hx_h = h_x.shape
    hy_w, hy_h = h_y.shape

    h_x_t = h_x.transpose()
    h_y_t = h_y.transpose()

    if hx_w != 1:
        raise ValueError()
    if hx_h % 2 != 1:
        raise ValueError()
    if hy_h != 1:
        raise ValueError()
    if hy_w % 2 != 1:
        raise ValueError()
    if hx_h != hy_w:
        raise ValueError()
        
        
def gaussian_mask(size=16, sigma=0.5):
    sigma *= size
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)


def compute_grad(I):

    gX = cv.Sobel(I, ddepth=cv.CV_32F, dx=1, dy=0, ksize=3)
    gY = cv.Sobel(I, ddepth=cv.CV_32F, dx=0, dy=1, ksize=3)
    #print(gX)
    #print(gY)
    return gX, gY

def compute_grad_ori(g_x, g_y, g_m, b=8):
    ori = np.zeros((b, 2))
    for i in range(b):
        ori[i,0] = np.cos(2 * np.pi * i / b)
        ori[i,1] = np.sin(2 * np.pi * i / b)
    w, h = g_m.shape
    # TODO: algebraic form
    g_o = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            if g_m[i,j] > 0:
                v = np.array([g_y[i,j], -g_x[i,j]])
                v = v / np.linalg.norm(v, ord=2)
                prod = np.dot(ori,v)
                g_o[i,j] = np.argmax(prod)
            else:
                g_o[i,j] = -1
    g_o = g_o.astype(int)
    return g_o


def compute_grad_mod_ori(I):


    Ix, Iy = compute_grad(I)

    Gn = np.sqrt(Ix**2 + Iy**2)
    Go = compute_grad_ori(Ix, Iy, Gn, 8)#np.arctan(Iy/Ix)
    return Gn, Go

def compute_histogram(g_n, g_o):
    """
    g_n and g_o are 4x4 matrices that contain the norm, and the discretized orientation.
    """
    hist = np.zeros((8))
    for i in range(8):
        hist[i] = g_n[g_o == i].sum()
    return hist



def compute_sift_region(Gn, Go, mask=None):
    t_min=.5
    t_max=.2
    with_l2 = True

    patch_size = 16
    sift = np.zeros((128)) 

    if mask is not None:
        Gn = Gn * mask
    
    idx = 0
    for k in range(0, patch_size, 4):
        for l in range(0, patch_size, 4):
            hist = compute_histogram(Gn[l:l+4,k:k+4], Go[l:l+4,k:k+4])            
            sift[idx:idx+8] = hist
            idx += 8

    norm = np.linalg.norm(sift, ord=2)
    # min thresholding on norm
    if norm <= t_min:
        return np.zeros((128))
    # l2-normalization
    if with_l2:
        sift = sift / norm
    # max thresholding on values
    sift[sift >= t_max] = t_max
    # l2-normalization
    if with_l2:
        norm = np.linalg.norm(sift, ord=2)
        sift = sift / norm
    return sift

def compute_sift_image(I):
    x, y = dense_sampling(I)
    im = auto_padding(I)
    m = gaussian_mask()
    
    # Here, compute on the global image (norm, gradients)
    Gn, Go = compute_grad_mod_ori(I)
    
    sifts = np.zeros((len(x), len(y), 128))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            if xi + 16 <= Gn.shape[0] and yj + 16 <= Gn.shape[1]:  # it was usefull afterall
                sifts[i, j, :] = compute_sift_region(Gn[xi:xi+16, yj:yj+16], Go[xi:xi+16, yj:yj+16], m) # TODO SIFT du patch de coordonnee (xi, yj)
    return sifts

def ComputeSiftDataset(inames):

    descriptors = []
    for x in tqdm.tqdm(inames):
        p = os.path.join(path, x)
        img = cv.imread(p)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        gray = resize_image(gray)
        #print(gray.shape)
        des = compute_sift_image(gray)
        des = (des * 255).astype('uint8')
        descriptors.append(des)
    
    return descriptors

def compute_regions(im, k=16, s=8):
    x, y = dense_sampling(im) # before padding
    im = auto_padding(im)
    images = np.zeros((x.shape[0], y.shape[0], k, k))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            images[i,j] = im[x[i]:x[i]+k, y[j]:y[j]+k]
    return images


def get_regions(inames):
    vdpaths = [os.path.join(path, iname) for iname in inames]

    regions = []
    for p in vdpaths:
        im = cv.imread(p)
        gray= cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        regions.append(compute_regions(gray))

    k = regions[0].shape[-1]
    n_reg = np.array([r.shape[0]*r.shape[1] for r in regions])
    cs_reg = np.cumsum(n_reg)

    regions = [r.reshape(-1, k, k) for r in regions]
    regions = np.concatenate(regions, axis=0)

    return regions


def compute_feats(vdict, descriptors):
    """
    vdict: (num_clusters, 128): visual dictionnary containing all clusters.
    image_sifts: (H, W, 128) all sift features from the given image
    """
    # flatten sifts
    sifts = np.array(descriptors).reshape(-1, 128)  # (N, 128)
    #print(sifts.shape)
    feats = np.zeros(vdict.shape[0])
    
    distances = distance_matrix(sifts, vdict)
    best_feature = np.argmin(distances, axis=1)
    for i in best_feature:
      feats[i]+=1
    
    norm = np.linalg.norm(feats, ord=2)
    feats = feats/norm
    return feats


def ComputeLaplacian(img):
    
    src = cv.GaussianBlur(img, (3, 3), 0)
    gray= cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    dst = cv.Laplacian(gray, ddepth, ksize=kernel_size)
    gray = resize_image(dst, shape = (512,512))
    abs_dst = cv.convertScaleAbs(gray)

    return abs_dst


def ComputeLaplacians(inames):
    laplacians = []
    for x in tqdm.tqdm(inames):
        p = os.path.join(path, x)
        img = cv.imread(p)
        laplacian = ComputeLaplacian(img)
        laplacian = np.ravel(laplacian)
        laplacians.append(laplacian)
    return laplacians