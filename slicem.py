import time
import mrcfile
import argparse
import numpy as np
import multiprocessing
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from skimage import transform, measure

SHIFT = ['Euclidean', 'L1', 'cosine'] # Metrics requiring real space translation

def main():
    """calculates similarity between line projections from 2D class averages"""
    
    parser = argparse.ArgumentParser(description='compare similarity of 2D class averages based on common lines')
    
    parser.add_argument('-i', '--input', action='store', dest='mrc_input', required=True,
                        help='path to mrcs file of 2D class averages')
    
    parser.add_argument('-o', '--outpath', action='store', dest='outpath', required=True,
                        help='path for output files')
    
    parser.add_argument('-m', '--metric', action='store', dest='metric', required=False, 
                        default='Euclidean', choices=['Euclidean', 'L1', 'cosine', 'EMD', 'correlate'],
                        help='choose scoring method, default Euclidean')
    
    parser.add_argument('-s', '--scale_factor', action='store', dest='scale_factor', required=False, type=float, default=1,
                        help='scale factor for downsampling. (e.g. -s 2 converts 200pix box --> 100pix box)')
    
    parser.add_argument('-c', '--num_workers', action='store', dest='num_workers', required=False, type=int, default=1,
                        help='number of CPUs to use, default 1')
    
    parser.add_argument('-d', '--domain', action='store', dest='domain', required=False, 
                        default='Fourier', choices=['Fourier', 'Real'], help='Fourier or Real space, default Fourier')
        
    parser.add_argument('-t', '--translate', action='store', dest='translate', required=False, 
                        default='full', choices=['full', 'valid'],
                        help='indicate size of score vector, numpy convention, default full')
    
    parser.add_argument('-a', '--angular_sampling', action='store', dest='angular_sampling', required=False, 
                        type=int, default=5, help='angle sampling for 1D projections in degrees, default 5')

    args = parser.parse_args()

    if args.domain == 'Fourier':
        rotation_degrees = np.arange(0, 180, args.angular_sampling)
    else:
        rotation_degrees = np.arange(0, 360, args.angular_sampling)
    
    shape, projection_2D = get_projection_2D(mrcs=args.mrc_input, factor=args.scale_factor)
    
    num_class_avg = len(projection_2D)
    num_1D = num_class_avg*len(rotation_degrees)
    
    print("number of 2D class averages: {}".format(num_class_avg))
    print("number of 1D projection vectors: {}".format(num_1D))
    print("total number of pairwise scores: {}".format(int(num_1D*(num_1D-1)/2)))

    if args.metric == 'Euclidean':
        pairwise_score = pairwise_l2
    elif args.metric == 'L1':
        pairwise_score = pairwise_l1
    elif args.metric == 'cosine':
        pairwise_score = pairwise_cosine
    elif args.metric == 'EMD':
        pairwise_score = pairwise_wasserstein
    elif args.metric == 'correlate':
        pairwise_score = pairwise_correlate
                
    if args.metric in SHIFT:
        wrapper_function = wrapper_slide_function
    else:
        wrapper_function = wrapper_single_function
        
    final_scores = {}
    
    with multiprocessing.Pool(args.num_workers) as pool:
        for i in range(num_class_avg-1):
            line_projections_1 = vectorize(i, projection_2D[i], rotation_degrees, shape, args.domain)
            for j in range(i+1, num_class_avg):
                line_projections_2 = vectorize(j, projection_2D[j], rotation_degrees, shape, args.domain)
                           
                projection_pairs = []
                for line_1 in line_projections_1.values():
                    for line_2 in line_projections_2.values():
                        projection_pairs.append((line_1, line_2))
            
                pair_scores = pool.starmap(
                    wrapper_function, 
                    [(pair, pairwise_score, args.translate, args.domain) for pair in projection_pairs]
                )

                optimum = min(pair_scores, key = lambda x: x[4])

                avg_1, deg_1, avg_2, deg_2, score = [value for value in optimum]

                final_scores[(avg_1, avg_2)] = (deg_1, deg_2, score)
                final_scores[(avg_2, avg_1)] = (deg_2, deg_1, score)
    
    write_scores(final_scores, outpath=args.outpath)

    
class Projection:
    """for 1D projection vectors"""
    
    def __init__(self, 
                 class_avg,
                 angle,
                 vector): 

        self.class_avg = class_avg
        self.angle = angle
        self.vector = vector
    
    def size(self):
        return len(self.vector)

    
def get_projection_2D(mrcs, factor):
    """read, scale and extract class averages"""
    
    projection_2D = {}

    with mrcfile.open(mrcs) as mrc:
        for i, data in enumerate(mrc.data):
            projection_2D[i] = data
        mrc.close()

    shape = transform.rotate(projection_2D[0].copy(), 45, resize=True).shape[0]    
    
    for k, avg in projection_2D.items():
        if factor == 1:
            projection_2D[k] = extract_class_avg(avg.copy())
        else:
            scaled_img = transform.rescale(
                avg, 
                scale=(1/factor), 
                anti_aliasing=True, 
                multichannel=False,  # Add to supress warning
                mode='constant'      # Add to supress warning
            )     
            projection_2D[k] = extract_class_avg(scaled_img)
            
    return shape, projection_2D


def extract_class_avg(avg):
    """fit in minimal bounding box"""
    
    image = avg.copy()
    image[image < 0] = 0

    struct = np.ones((2, 2), dtype=bool)
    dilate = ndi.binary_dilation(image, struct)

    labeled = measure.label(dilate, connectivity=2)
    rprops = measure.regionprops(labeled, image, cache=False)

    if len(rprops) == 1:
        select_region = 0
        
    else:
        img_y, img_x = image.shape

        if labeled[int(img_y/2), int(img_x/2)] != 0: # Check for central region
            select_region = labeled[int(img_y/2), int(img_x/2)] - 1 # For index

        else:
            distances = [
                (i, np.linalg.norm(np.array((img_y/2, img_x/2)) - np.array(r.weighted_centroid))) 
                for i, r in enumerate(rprops)
            ]

            select_region = min(distances, key=lambda x: x[1])[0] # Pick first closest region    

    y_min, x_min, y_max, x_max = [p for p in rprops[select_region].bbox]

    return image[y_min:y_max, x_min:x_max]


def vectorize(key, image, rotation_degrees, shape, domain):
    """
    takes image and creates 1D projections
    similar to Radon transform
    """
    projection_1D = {}
    projection_1D_FT = {}
    
    for degree in rotation_degrees:
        proj_1D = transform.rotate(image, degree, resize=True).sum(axis=0).astype('float32')
        trim_1D = np.trim_zeros(proj_1D, trim='fb')
        
        pad_1D = np.pad(proj_1D, (0, shape-len(proj_1D))) # Pad to largest possible shape from 2D 
        F = abs(np.fft.rfft(pad_1D))
        
        projection_1D[(key, degree)] = Projection(class_avg=key, angle=degree, vector=trim_1D)
        projection_1D_FT[(key, degree)] = Projection(class_avg=key, angle=degree, vector=F)
    
    if domain == 'Fourier':
        return projection_1D_FT
    else:
        return projection_1D
    
    
def pairwise_l2(a, b):
    return np.linalg.norm(a - b)


def pairwise_l1(a, b):
    return np.linalg.norm(a - b, 1)


def pairwise_cosine(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pairwise_correlate(a, b, translate):
    s = np.correlate(a, b, mode=translate)
    return 1 / (1 + np.amax(s)) # Convert to distance


def pairwise_wasserstein(a, b, translate):
    return wasserstein_distance(a, b)


def slide_score(a, b, pairwise_score, translate, domain):
    """
    finds minimum pairwise score for translations of 1D projections
    a, b are instances of the Projection class
    'valid' is elements without zero padding
    'full' is scores at all translations
    """
    scores = []
    
    if domain == 'Fourier':
        scores.append(pairwise_score(a.vector[1:], b.vector[1:])) #Drop 0th seems to help
        
    else:
        if a.size() > b.size():    
            l, s = a.vector, b.vector
        else:
            l, s = b.vector, a.vector

        l_size, s_size = len(l), len(s)

        if translate == 'valid':
            diff_of_len = abs(l_size - s_size)

            if diff_of_len == 0:
                scores.append(pairwise_score(l, s))      
            else:
                pad_s = np.pad(s, pad_width=(diff_of_len, diff_of_len))
                for i in range(0, diff_of_len+1):
                    shift_s = pad_s[i:i+l_size]
                    scores.append(pairwise_score(l, shift_s))

        elif translate == 'full':
            pad_l = np.pad(l, pad_width=(s_size-1, s_size-1))
            pad_s = np.pad(s, pad_width=(l_size+s_size-2, l_size+s_size-2))

            for i in range(0, l_size+s_size-1):
                shift_s = pad_s[i:i+len(pad_l)]
                scores.append(pairwise_score(pad_l, shift_s))
            
    return min(scores)


def wrapper_slide_function(pair, pairwise, translate, domain):
    """
    pair is tuple from Projection class to be scored
    pairwise is function to score vectores (e.g. Euclidean)
    """
    score = slide_score(pair[0], pair[1], pairwise, translate, domain)
    return [pair[0].class_avg, pair[0].angle, pair[1].class_avg, pair[1].angle, score]


def wrapper_single_function(pair, pairwise, translate, domain):
    """same as above but for correlate and EMD"""
    score = pairwise(pair[0].vector[1:], pair[1].vector[1:], translate) # Skip 0th component 
    return [pair[0].class_avg, pair[0].angle, pair[1].class_avg, pair[1].angle, score]

                
def write_scores(final_scores, outpath):
    """
    tab separted file of final scores
    load scores into the slicem gui
    """
    stamp = time.strftime('%Y%m%d_%H%M%S')
    
    header = ['projection_1', 'degree_1', 'projection_2', 'degree_2', 'score']
    
    with open(outpath+'/slicem_scores_{0}.txt'.format(stamp), 'w') as f:
        for h in header:
            f.write(h+'\t')
        f.write('\n')
        for p, v in final_scores.items():
            f.write(str(p[0])+'\t'+str(v[0])+'\t'+str(p[1])+'\t'+str(v[1])+'\t'+str(v[2])+'\n')          

            
if __name__ == "__main__":
    starttime = time.time()
    main()
    print('Runtime: {} minutes'.format((time.time() - starttime)/60))