from distutils.log import error
import numpy as np
import numba as nb
import sklearn as sk
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.cluster import MeanShift, estimate_bandwidth

import random
import colorsys
import logging
import scipy.sparse.linalg

import scipy
from scipy.ndimage import convolve
import skimage
from skimage.morphology import flood, flood_fill
from skimage.segmentation import slic, quickshift, mark_boundaries
from skimage.util import img_as_float
from skimage import io

import pymeanshift as pms

import argparse

import math



def gkern(l=16, sig=4.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)



@nb.njit(parallel=True, fastmath=True)
def compute_similarity_scores(x,y):

    gabor_x = x[:,3:43]
    gabor_y = y[:,3:43]

    sift_x = x[:,43:]
    sift_y = y[:,43:]
    
    #w = np.array([0.3,0.1,0.2])
    pixels = np.empty((x.shape[0]))
    scores = np.empty((x.shape[0]))

    for i in nb.prange(x.shape[0]):
        tmp = np.empty(y.shape[0])
        for j in range(y.shape[0]):
            tmp[j] = 0.4*np.abs(x[i,0]-y[j,0])
            tmp[j]+= 0.1*np.abs(x[i,1]-y[j,1])
            tmp[j]+= 0.0*np.abs(x[i,2]-y[j,2]) # not used
            tmp[j]+= 0.2*np.sqrt(np.sum((gabor_x[i] - gabor_y[j])**2))
            tmp[j]+= 0.3*np.sqrt(np.sum((sift_x[i] - sift_y[j])**2))
        pixels[i] = np.argmin(tmp)
        scores[i] = np.min(tmp)
            
    return pixels, scores

def compute_similarity_scores_test(x,y):

    gabor_x = x[:,3:43]
    gabor_y = y[:,3:43]

    sift_x = x[:,43:]
    sift_y = y[:,43:]
    
    #w = np.array([0.3,0.1,0.2])
    z = np.empty((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i,j] = 0.3*np.abs(x[i,0]-y[j,0])
            z[i,j]+= 0.1*np.abs(x[i,1]-y[j,1])
            z[i,j]+= 0.1*np.abs(x[i,2]-y[j,2])
            z[i,j]+= 0.2*np.sqrt(np.sum((gabor_x[i] - gabor_y[j])**2))
            z[i,j]+= 0.2*np.sqrt(np.sum((sift_x[i] - sift_y[j])**2))
    return z


@nb.njit(parallel=True, fastmath=True)
def get_sift_features(g_m, g_d, gkern):
    h, w = g_m.shape
    sift_features = np.empty((h-16, w-16, 128))
    hist_val = np.array([0.0,  np.pi/4, np.pi/2,  3*np.pi/4, np.pi, 5* np.pi/4, 6* np.pi/4, 7* np.pi/4, 2*np.pi])
    for x in nb.prange(8, h-8):
        for y in range(8, w-8):
            magnitude_block = g_m[x-8:x+8, y-8:y+8]*gkern
            direction_block = g_d[x-8:x+8, y-8:y+8]
            bins = np.zeros((4,4,8))
            for i in range(4):
                for j in range(4):
                    #construct histogram
                    dir_sub_block = direction_block[i*4:(i+1)*4,j*4:(j+1)*4]
                    mag_sub_block = magnitude_block[i*4:(i+1)*4,j*4:(j+1)*4]
                    for angle_id in range(8):
                        below = (hist_val[angle_id] < dir_sub_block)
                        above = (dir_sub_block < hist_val[angle_id+1])
                        selected = below * above
                        tmp =  mag_sub_block * selected
                        bins[i,j,angle_id] += tmp.sum()
            bins = bins.reshape((-1))
            bins += 1e-7 # avoid div by 0
            bins /= bins.sum() # normalize
        
            bins[bins > 0.2] = 0.2 # apparently makes it illumination independent
            bins /= bins.sum() # renormalize
            sift_features[x-8, y-8] = bins
    return sift_features


def gradient(img):
    gx = cv2.filter2D(img,-1, kernel=np.array([[-1,0],[0,1]]))
    gy = cv2.filter2D(img,-1, kernel=np.array([[0,1],[-1,0]]))
    g_magnitude = np.sqrt(np.power(gx,2)+np.power(gy,2))
    g_dir = np.pi + np.arctan2(gy, gx)
    return g_magnitude, g_dir

wd_width = 1

# the window class, find the neighbor pixels around the center.
class WindowNeighbor:
    def __init__(self, width, center, pic):
        # center is a list of [row, col, Y_intensity]
        self.center = [center[0], center[1], pic[center][0]]
        self.width = width
        self.neighbors = None
        self.find_neighbors(pic)
        self.mean = None
        self.var = None

    def find_neighbors(self, pic):
        self.neighbors = []
        ix_r_min = max(0, self.center[0] - self.width)
        ix_r_max = min(pic.shape[0], self.center[0] + self.width + 1)
        ix_c_min = max(0, self.center[1] - self.width)
        ix_c_max = min(pic.shape[1], self.center[1] + self.width + 1)
        for r in range(ix_r_min, ix_r_max):
            for c in range(ix_c_min, ix_c_max):
                if r == self.center[0] and c == self.center[1]:
                    continue
                self.neighbors.append([r,c,pic[r,c,0]])

    def __str__(self):
        return 'windows c=(%d, %d, %f) size: %d' % (self.center[0], self.center[1], self.center[2], len(self.neighbors))


# affinity functions, calculate weights of pixels in a window by their intensity.
def affinity_a(w):
    nbs = np.array(w.neighbors)
    sY = nbs[:,2] # neighbors intensity
    cY = w.center[2] # center intensity
    diff = sY - cY
    sig = np.var(np.append(sY, cY))
    if sig < 1e-6: # to not have weight explosion
        sig = 1e-6  
    wrs = np.exp(- np.power(diff,2) / (sig * 2.0)) # weight is proportional to the difference in intensity
    wrs = - wrs / np.sum(wrs) # normalize
    nbs[:,2] = wrs # the weights given to each neighbors
    return nbs

# translate (row,col) to/from sequential number
def to_seq(r, c, rows):
    return c * rows + r

def fr_seq(seq, rows):
    r = seq % rows
    c = int((seq - r) / rows)
    return (r, c)

# combine 3 channels of YUV to a RGB photo: n x n x 3 array
def yuv_channels_to_rgb(cY,cU,cV,pic_yuv):
    pic_cols = pic_yuv.shape[1]
    pic_rows = pic_yuv.shape[0]
    ansRGB = [colorsys.yiq_to_rgb(cY[i],cU[i],cV[i]) for i in range(len(cY))]
    ansRGB = np.array(ansRGB)
    pic_ansRGB = np.zeros(pic_yuv.shape)
    pic_ansRGB[:,:,0] = ansRGB[:,0].reshape(pic_rows, pic_cols, order='F')
    pic_ansRGB[:,:,1] = ansRGB[:,1].reshape(pic_rows, pic_cols, order='F')
    pic_ansRGB[:,:,2] = ansRGB[:,2].reshape(pic_rows, pic_cols, order='F')
    return pic_ansRGB

class Colorizer:
    def __init__(self, segmentation = "meanshift", verbose=False):
        self.segmentation = segmentation
        self.verbose = verbose
        self.name = None
    def colorize(self, target, reference):
        print("Transfer color from the reference to the target image")
        color_transfer_mask, target_img_YUV = self.transfer_color(target, reference)

        pic_yuv = target_img_YUV
        weightData = []
        pic_cols = pic_yuv.shape[1]
        pic_rows = pic_yuv.shape[0]
        pic_size = pic_cols*pic_rows # there is exactly one linear constraint by pixel

        for c in range(pic_cols):
            for r in range(pic_rows):
                w = WindowNeighbor(wd_width, (r,c), pic_yuv)
                if not color_transfer_mask[r,c]: # if not in the color pseudo scribble, we add up the constraints
                    weights = affinity_a(w)
                    for e in weights:
                        weightData.append([w.center,(e[0],e[1]), e[2]]) # here e[2] is the weight of one neighbor of the pixel
                # if it is in the color transfer mask, this set up the constraint U(r) = u, where u is the pixel color transfered by our algorithm
                weightData.append([w.center, (w.center[0],w.center[1]), 1.]) 

        sp_idx_rc_data = [[to_seq(e[0][0], e[0][1], pic_rows), to_seq(e[1][0], e[1][1], pic_rows), e[2]] for e in weightData]
        sp_idx_rc = np.array(sp_idx_rc_data, dtype=np.int64)[:,0:2]
        sp_data = np.array(sp_idx_rc_data, dtype=np.float64)[:,2]

        matA = scipy.sparse.csr_matrix((sp_data, (sp_idx_rc[:,0], sp_idx_rc[:,1])), shape=(pic_size, pic_size))

        b_u = np.zeros(pic_size)
        b_v = np.zeros(pic_size)
        idx_colored = np.nonzero(color_transfer_mask.reshape(pic_size, order='F'))
        pic_u_flat = pic_yuv[:,:,1].reshape(pic_size, order='F')
        b_u[idx_colored] = pic_u_flat[idx_colored] # b_u is zero everywhere but where the colored where transfered, where it is u

        pic_v_flat = pic_yuv[:,:,2].reshape(pic_size, order='F')
        b_v[idx_colored] = pic_v_flat[idx_colored] # same

        print("Propagate the color through the image")
        ansY = pic_yuv[:,:,0].reshape(pic_size, order='F')
        ansU = scipy.sparse.linalg.spsolve(matA, b_u)
        ansV = scipy.sparse.linalg.spsolve(matA, b_v)

        pic_ans = yuv_channels_to_rgb(ansY,ansU,ansV,pic_yuv)
        return pic_ans

    def get_features(self, image):
        ## compute luminance ##
        luminance = cv2.cvtColor(image,cv2.COLOR_BGR2Luv)[:,:,0].astype(float)/255.0

        if self.verbose:
           print("Computing Saliency")
        ## compute saliency ##
        #saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        #saliency = cv2.saliency.ObjectnessBING_create()
        (success, saliencyMap) = saliency.computeSaliency(image)
        
        if self.verbose:
            plt.imshow(saliencyMap)
            plt.show()
            print("Computing standard derivation")


        ## compute standard derivation via integral image ##
        pad_img = np.pad(luminance, (2,2), 'edge')
        sum_1, square_sum = cv2.integral2(pad_img)
        n = 25
        
        s1 = sum_1[5:,5:] + sum_1[:-5,:-5] - sum_1[:-5,5:] - sum_1[5:,:-5]
        s2 = square_sum[5:,5:] + square_sum[:-5,:-5] - square_sum[:-5,5:] - square_sum[5:,:-5]
        std = np.sqrt((s2 - (s1**2)/n)/n)
        std[np.isnan(std)] = 0.0
        std /= std.max() #normalize
        
        if self.verbose:
            plt.imshow(std)
            plt.show()
            print("Computing Gabor")

        ## gabor filters ##
        gabor_features = np.empty((image.shape[0],image.shape[1], 8*5))
        #pad_img = np.pad(luminance, (4,4), 'edge')
        filters = []
        for i in range(8):
            for j in range (5):
                filters.append(cv2.getGaborKernel((21,21),9,i*np.pi/8,np.exp(j), 0.5, 0, ktype=cv2.CV_32F))

        for filter_id in range(len(filters)):
            gabor_features[:,:,filter_id] = cv2.filter2D(luminance,-1,filters[filter_id], borderType=cv2.BORDER_REPLICATE)
        gabor_features =  gabor_features / gabor_features.sum(2)[:,:,np.newaxis]
        #print(gabor_features.sum(2)) 

        if self.verbose:
            print("Computing SIFT features")

        ## SIFT ##
        g_magnitude, g_dir = gradient(luminance)
        pad_g_magnitude = np.pad(g_magnitude, (8,8), 'edge')
        pad_g_dir = np.pad(g_dir, (8,8), 'edge')
        sift_features = get_sift_features(pad_g_magnitude, pad_g_dir, gkern())
        #print(sift_features.shape, g_magnitude.shape)



        return luminance, std, saliencyMap, gabor_features, sift_features

    def get_segmentation(self, image):

        if self.verbose:
            print("Computing Segmentation features")

        h,w = image.shape[0], image.shape[1]
        image_float = image.astype(float)/255.0
        
        if self.segmentation == "slic":
            segments = slic(image_float, n_segments = 1000,compactness=5., sigma = 1.0)
        elif self.segmentation == "bad_meanshift": #do not use
            est_bandwidth = estimate_bandwidth(image_float.reshape((h*w,3)), quantile=0.05, n_samples=500)
            mean_shift = MeanShift(bandwidth=est_bandwidth, n_jobs=-1, bin_seeding=True)
            segments = mean_shift.fit(image_float.reshape((h*w,3)))
            segments = segments.labels_.reshape((h,w))
        elif self.segmentation == "meanshift":
            print(image.shape)
            (_, segments, _) = pms.segment(image, spatial_radius=8, 
                                                              range_radius=8, min_density=100)
            
        else:
            raise error("segmentation method not defined")
        if self.verbose:
            plt.imshow(mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments))
            plt.imsave(f"boundaries_{self.name}.jpg", mark_boundaries(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), segments))
            plt.show()
        numSections = segments.max()
        pixels_sets = [None]*numSections
        for i in range(numSections):
            pixels_sets[i] = []
        for i in range(h):
            for j in range(w):
                pixels_sets[segments[i,j]-1].append((i,j))
        return segments, pixels_sets

    def compute_segments_features(self, segments, pixels_sets, luminance, std, saliencyMap, gabor_features, sift_features):
        numSections = segments.max()
        superpixels = np.zeros((numSections, 3 + 40 + 128))
        superpixels_pixels = [None]*numSections
        for i in range(1,1+numSections):
            segment = segments == i
            number_of_pixels = np.count_nonzero(segment)

            superpixels[i-1,0] = (luminance[segment].sum()/number_of_pixels)
            superpixels[i-1,1] = (std[segment].sum()/number_of_pixels)
            superpixels[i-1,2] = (saliencyMap[segment].sum()/number_of_pixels)
            for filter_id in range(40):
                superpixels[i-1, 3+filter_id] = (gabor_features[segment, filter_id].sum()/number_of_pixels)
            for sift_feature_id in range(128):
                superpixels[i-1, 43+sift_feature_id] = (sift_features[segment, sift_feature_id].sum()/number_of_pixels)

            target_pixels_set = pixels_sets[i-1]
            superpixels_pixels[i-1] = np.empty((len(target_pixels_set), 3+40+128)) 
            for pixel_id in range(len(target_pixels_set)):
                x,y = target_pixels_set[pixel_id]
                superpixels_pixels[i-1][pixel_id, 0] = luminance[x,y]
                superpixels_pixels[i-1][pixel_id, 1] = std[x,y]
                superpixels_pixels[i-1][pixel_id, 2] = saliencyMap[x,y]
                for filter_id in range(40):
                    superpixels_pixels[i-1][pixel_id,3+filter_id] = gabor_features[x,y,filter_id]
                
                for sift_feature_id in range(128):
                    superpixels_pixels[i-1][pixel_id,43+sift_feature_id] = sift_features[x,y,sift_feature_id]

        return superpixels, superpixels_pixels

    def compute_best_pixels_pair(self, target, reference):
        if self.verbose:
            print("working on the target image")
            self.name= "target"

        segments, target_pixels_sets = self.get_segmentation(target[:,:,0])
        luminance, std, saliencyMap, gabor_features, sift_features = self.get_features(target)
        if self.verbose:
            print("Computing segments feature")
        target_superpixels, target_superpixels_pixels = self.compute_segments_features(segments,
                                                                        target_pixels_sets,
                                                                        luminance,
                                                                        std,
                                                                        saliencyMap,
                                                                        gabor_features,
                                                                        sift_features)
        
        
        if self.verbose:
            print("working on the reference image")
            self.name= "reference"
        segments, reference_pixels_sets = self.get_segmentation(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
        luminance, std, saliencyMap, gabor_features, sift_features = self.get_features(reference)
        if self.verbose:
            print("Computing segments feature")
        reference_superpixels, reference_superpixels_pixels = self.compute_segments_features(segments,
                                                                                            reference_pixels_sets,
                                                                                            luminance,
                                                                                            std,
                                                                                            saliencyMap,
                                                                                            gabor_features,
                                                                                            sift_features)


        if self.verbose:
            print("computing best superpixel")

        #best_superpixels = np.argmin(compute_similarity_scores(target_superpixels, reference_superpixels), axis=1)
        best_superpixels, _ = compute_similarity_scores(target_superpixels, reference_superpixels)
        best_superpixels = best_superpixels.astype(int)
        pixels_pairs = [None]*len(best_superpixels)
        pixels_pairs_scores = [None]*len(best_superpixels)
        for i in range(len(best_superpixels)):
            #print(i)
            #if i == 514:
            #    print("grabbing")
            #    tmp = compute_similarity_scores_test(target_superpixels_pixels[i], reference_superpixels_pixels[best_superpixels[i]])
            #    print (tmp.max(), tmp.min())
            #tmp = compute_similarity_scores(target_superpixels_pixels[i], reference_superpixels_pixels[best_superpixels[i]])
            #pixels_pairs[i] = np.argmin(tmp , axis=1)
            #pixels_pairs_scores[i] = np.min(tmp , axis=1)
            pixels_pairs[i], pixels_pairs_scores[i] = compute_similarity_scores(target_superpixels_pixels[i], reference_superpixels_pixels[best_superpixels[i]])
            pixels_pairs[i] = pixels_pairs[i].astype(int)


        return pixels_pairs, pixels_pairs_scores, best_superpixels, target_pixels_sets, reference_pixels_sets

    def transfer_color(self, target, reference):
        pixels_pairs, pixels_pairs_scores, best_superpixels, target_pixels_sets, reference_pixels_sets = self.compute_best_pixels_pair(target, reference)

        if self.verbose:
            print("Transfering color from reference to target")
        color_rgb = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB).astype(float)/255.0
        reference_img_Y, reference_img_U, reference_img_V = colorsys.rgb_to_yiq(color_rgb[:,:,0], color_rgb[:,:,1], color_rgb[:,:,2])
        reference_img_YUV = np.dstack((reference_img_Y, reference_img_U, reference_img_V))
        target_img_YUV = target.copy().astype(float)/255.0
        target_img_YUV[:,:,1] = 0.0
        target_img_YUV[:,:,2] = 0.0

        color_transfer_mask = np.zeros(target_img_YUV.shape[:2], dtype=bool)
        for i in range(len(best_superpixels)):
            target_pixels_set = target_pixels_sets[i]
            reference_pixels_set = reference_pixels_sets[best_superpixels[i]]
            scores = pixels_pairs_scores[i]
            pixel_map = pixels_pairs[i]
            #print((target_pixels_set[0], reference_pixels_set[pixel_map[0]], scores[0]) )
            target_reference_score_triplets = [(target_pixels_set[j], reference_pixels_set[pixel_map[j]], scores[j]) for j in range(len(target_pixels_sets[i]))]
            top_15percent = sorted(target_reference_score_triplets, key=lambda x: x[2])[:math.ceil(0.15*len(target_reference_score_triplets))]
            for target, selected, _ in top_15percent:
                color_transfer_mask[target[0], target[1]] = True
                target_img_YUV[target[0], target[1], 1] = reference_img_YUV[selected[0],selected[1],1]
                target_img_YUV[target[0], target[1], 2] = reference_img_YUV[selected[0],selected[1],2]

        return color_transfer_mask, target_img_YUV


# unused
def get_128vector(g_m, g_d, gkern, x, y):
    
    h, w = g_m.shape
    magnitude_block = g_m[x-8:x+8, y-8:y+8]*gkern
    direction_block = g_d[x-8:x+8, y-8:y+8]
    bins = np.zeros((4,4,8))
    hist_val = np.array([0.0,  np.pi/4, np.pi/2,  3*np.pi/4, np.pi, 5* np.pi/4, 6* np.pi/4, 7* np.pi/4, 2*np.pi])
    for i in range(4):
        for j in range(4):
            #construct histogram
            dir_sub_block = direction_block[i*4:(i+1)*4,j*4:(j+1)*4]
            mag_sub_block = magnitude_block[i*4:(i+1)*4,j*4:(j+1)*4]
            for angle_id in range(8):
                bins[i,j,angle_id] += mag_sub_block[(hist_val[angle_id] < dir_sub_block) * (dir_sub_block < hist_val[angle_id+1])].sum()
    bins = bins.reshape((-1))
    bins += 1e-7 # avoid div by 0
    bins /= bins.sum() # normalize
   
    #bins[bins > 0.2] = 0.2 # apparently makes it illumination independent
    #bins /= bins.sum() # renormalize
    return bins


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=str, help="the path to the color reference image", required=True)
    parser.add_argument("--target", type=str, help="the path to the gray target image", required=True)
    parser.add_argument("--segmentation", type=str, help="either meanshift or slic. default is meanshift")
    parser.add_argument("-v","--verbose", action='store_true', help="set if you wnat to show plot the different steps")
    args = parser.parse_args()


    #path_to_color = "ColorfulOriginal"
    #path_to_gray = "Gray"
    #fruit_types = ["Apple", "Banana", "Brinjal", "Broccoli",
    #            "CapsicumGreen","Carrot","Cherry", "ChilliGreen",
    #            "Corn", "Cucumber", "LadyFinger", "Lemon", "Orange",
    #            "Peach", "Pear", "Plum", "Pomegranate", "Potato",
    #            "Strawberry", "Tomato"]

    #color_img_path = os.path.join("CapsicumGreen", "Capsicum" + "1.jpg")
    #gray_img_path = os.path.join("CapsicumGreen", "Capsicum" + "1.jpg")
    #color_path = os.path.join(path_to_color, color_img_path)
    #gray_path = os.path.join(path_to_gray, gray_img_path)
    color_path = args.reference
    gray_path = args.target
    

    reference_image = cv2.imread(color_path)
    target_image = cv2.imread(gray_path)
    segmentation = args.segmentation if args.segmentation else "meanshift"

    
    verbose = args.verbose if args.verbose else False
    
    colorizer = Colorizer(segmentation=segmentation, verbose=verbose)
    colorized_img = colorizer.colorize(target_image, reference_image)

    fig = plt.figure()
    fig.add_subplot(1,2,1).set_title('Reference')
    imgplot = plt.imshow(cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB))
    fig.add_subplot(1,2,2).set_title('Colorized')
    imgplot = plt.imshow(colorized_img)
    plt.show()
    plt.imsave("result.png", colorized_img)
