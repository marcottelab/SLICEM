import time
import mrcfile
import argparse
import numpy as np
import multiprocessing
from skimage import transform, measure
from scipy import ndimage as ndi
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

ROTATION_DEGREES = np.arange(0, 360, 5)
SLIDE = ['Euclidean', 'L1', 'cosine']  # Metrics requiring translation

def main():
    """
    calculates similarity between line projections from 2D class averages
    """
    parser = argparse.ArgumentParser(description='compare similarity of 2D class averages based on common lines')
    
    parser.add_argument('-i', '--input', action='store', dest='mrc_input', required=True,
                        help='path to mrcs file of 2D class averages')
    
    parser.add_argument('-o', '--outpath', action='store', dest='outpath', required=True,
                        help='path for output files')
    
    parser.add_argument('-m', '--metric', action='store', dest='metric', required=False, 
                        default='Euclidean', choices=['Euclidean', 'L1', 'cosine', 'EMD'],
                        help='choose scoring method, Euclidean default')
    
    parser.add_argument('-s', '--scale_factor', action='store', dest='scale_factor', required=False, type=float, default=1,
                        help='scale factor for downsampling. (e.g. -s 2 converts 200pix box --> 100pix box)')
    
    parser.add_argument('-c', '--num_workers', action='store', dest='num_workers', required=False, type=int, default=1,
                        help='number of CPUs to use')

    args = parser.parse_args()

    final_scores = {}
    
    projection_2D = get_projection_2D(mrcs=args.mrc_input, factor=args.scale_factor)
      
    num_class_avg = len(projection_2D)
    num_1D = num_class_avg*len(ROTATION_DEGREES)
    
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
    
    if args.metric in SLIDE:
        wrapper_function = wrapper_slide_function
    else:
        wrapper_function = wrapper_wasserstein
    
    with multiprocessing.Pool(args.num_workers) as pool:
        for i in range(num_class_avg-1):
            line_projections_1 = vectorize(i, projection_2D[i])
            for j in range(i+1, num_class_avg):
                line_projections_2 = vectorize(j, projection_2D[j])

                projection_pairs = []
                for line_1 in line_projections_1.values():
                    for line_2 in line_projections_2.values():
                        projection_pairs.append((line_1, line_2))
            
                pair_scores = pool.starmap(
                    wrapper_function, 
                    [(pair, pairwise_score) for pair in projection_pairs]
                )

                optimum =  min(pair_scores, key = lambda x: x[4])

                avg_1, deg_1, avg_2, deg_2, score = [val for val in optimum]

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
        l = len(self.vector)
        return(l)

    
def get_projection_2D(mrcs, factor):
    """read, scale and extract class averages"""
    
    projection_2D = {}

    with mrcfile.open(mrcs) as mrc:
        for i, data in enumerate(mrc.data):
            projection_2D[i] = data
        mrc.close()

    for k, avg in projection_2D.items():
        if factor == 1:
            projection_2D[k] = extract_class_avg(avg)
        else:
            scaled_img = transform.rescale(
                avg, 
                scale=(1/factor), 
                anti_aliasing=True, 
                multichannel=False,  # Add to supress warning
                mode='constant'      # Add to supress warning
            )     
            projection_2D[k] = extract_class_avg(scaled_img)
            
    return projection_2D


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
                (i, euclidean(np.array((img_y/2, img_x/2)), np.array(r.weighted_centroid))) 
                for i, r in enumerate(rprops)
            ]

            select_region = min(distances, key=lambda x: x[1])[0] # Pick first closest region    

    y_min, x_min, y_max, x_max = [p for p in rprops[select_region].bbox]

    extract = image[y_min:y_max, x_min:x_max]
    
    return extract


def vectorize(key, image):
    """
    takes image and creates 1D projections
    similar to Radon transform
    """
    projection_1D = {}
    for degree in ROTATION_DEGREES:
        proj_1D = transform.rotate(image, degree, resize=True).sum(axis=0)
        trim_1D = np.trim_zeros(proj_1D, trim='fb').astype('float32')
        projection_1D[(key, degree)] = Projection(class_avg=key, angle=degree, vector=trim_1D)
    return projection_1D


def pairwise_l2(a, b):
    score = euclidean(a.vector, b.vector)
    return score


def pairwise_l1(a, b):
    score = sum(abs(a.vector - b.vector))
    return score


def pairwise_cosine(a, b):
    score = cosine(a.vector, b.vector)
    return score 


def pairwise_wasserstein(a, b):
    score = wasserstein_distance(a.vector, b.vector)
    return score


def slide_score(a, b, pairwise_score):
    """
    a, b are instances of the Projection class
    finds minimum pairwise score for all translations of 1D projections
    """
    lp1 = a.size()
    lp2 = b.size()
    
    dol = abs(lp1 - lp2)
    
    if dol == 0:
        score = pairwise_score(a, b)

    else:
        projection_shifts = []
        
        if lp1 > lp2:    
            static = a
            shift = b
        else:
            static = b
            shift = a
            
        for i in range(0, dol+1):
            v = np.pad(shift.vector, pad_width=(i, dol-i), mode='constant')
            
            projection_shifts.append(
                Projection(
                    class_avg=str(shift.class_avg)+'_'+str(i),
                    angle = shift.angle,
                    vector = v
                )
            )
        
        scores = []
        
        for shifted in projection_shifts:
            scores.append(pairwise_score(static, shifted))
        score = min(scores)    
    return score


def wrapper_slide_function(pair, pairwise):
    """
    - pair is tuple from Projection class to be scored
    - pairwise is function to score vectores (e.g. Euclidean)
    """
    score = slide_score(pair[0], pair[1], pairwise)

    return [pair[0].class_avg, pair[0].angle, pair[1].class_avg, pair[1].angle, score]


def wrapper_wasserstein(pair, pairwise):
    """same as above but without slide_score"""
    
    score = pairwise(pair[0], pair[1])
    
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
