import time
import mrcfile
import argparse
import numpy as np
import skimage as ski
import itertools as it
import multiprocessing 
from scipy import ndimage as ndi
from scipy import spatial, signal, stats

def main():
    """
    calculates similarity between line projections from 2D class averages
    """
    parser = argparse.ArgumentParser(description='compare and cluster 2D class averages based on common lines')
    
    parser.add_argument('-i', '--input', action='store', dest='mrc_input', required=True,
                        help='path to mrcs file of 2D class averages')
    parser.add_argument('-o', '--outpath', action='store', dest='outpath', required=True,
                        help='path for output files')
    parser.add_argument('-d', '--description', action='store', dest='description', required=True,
                        help='name for output file')
    parser.add_argument('-m', '--metric', action='store', dest='metric', required=True,
                        help='choose scoring method (Euclidean, L1, cross-correlation, cosine)')
    parser.add_argument('-n', '--normalize', action='store_true', required=False,
                        help='zscore normalize 1D projections before scoring (not recommended)')
    parser.add_argument('-p', '--pixel_size', action='store', dest='pixel_size', type=int, required=True,
                        help='pixel size of 2D class averages in A/pixel')
    parser.add_argument('-c', '--num_workers', action='store', dest='num_workers', type=int, required=False, default=1,
                        help='number of CPUs to use')

    args = parser.parse_args()

    starttime = time.time()
    
    projection_2D, projection_1D = get_1D_projections(mrcs=args.mrc_input,
                                                      pixel_size=args.pixel_size,
                                                      norm=args.normalize)
                                                      
    num_class_avg = len(projection_2D)

    n = len(projection_1D)
    
    print("number of 2D class averages: {}".format(num_class_avg))
    print("number of 1D projection vectors: {}".format(n))
    print("total number of pairwise scores: {}".format(int(n*(n-1)/2)))

    #remove pairs from same 2D class average
    all_pairs = list(it.combinations(projection_1D.values(), 2))
    projection_pairs = [pair for pair in all_pairs if pair[0].class_avg != pair[1].class_avg]
    
    scores = multiproc_pairwise(projection_pairs,
                                num_workers=args.num_workers,
                                metric=args.metric)

    pairs = list(it.combinations(list(range(num_class_avg)), 2))

    score_matrix = {pair: [] for pair in pairs}

    for s in scores:
        #format is [projection_1, projection_2] = [(angle_1, angle_2, score)]
        score_matrix[s[0], s[2]].append((s[1], s[3], s[4]))

    complete_scores = optimum_scores(score_matrix, 
                                     metric=args.metric)

    write_scores(complete_scores,
                 outpath=args.outpath, 
                 description=args.description)

    print('That took: {} minutes'.format((time.time() - starttime)/60))

    
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

    
def get_1D_projections(mrcs, pixel_size, norm):
    """
    creates dictionary of 1D projections, key is [class avg number, projection angle] 
    """
    projection_2D = {}
    
    projection_1D = {}
    
    angles = np.arange(0, 360, 5)
    
    with mrcfile.open(mrcs) as mrc:
        for i, data in enumerate(mrc.data):
            projection_2D[i] = data.astype('float64')
            
    for k, avg in projection_2D.items():
        projection_2D[k] = extract_class_avg(avg, pixel_size)
    
    for k, avg in projection_2D.items():
        for a in angles:
            #to recreate original version, don't resize or trim avgs
            #make 1D projections as in Radon transform
            proj_1D = ski.transform.rotate(avg, a, resize=True).sum(axis=0)
            trim_1D = np.trim_zeros(proj_1D, trim='fb')
            projection_1D[(k, a)] = Projection(class_avg = k,
                                               angle = a,
                                               vector = trim_1D)

    if norm:
        for p, v in projection_1D.items():
            projection_1D[p].vector = stats.zscore(projection_1D[p].vector)
    
    return projection_2D, projection_1D  


def extract_class_avg(avg, pixel_size):
    """
    keep positive values from normalized class average
    and remove extra densities in class average
    fit in minimal bounding box
    """
    #EV: could try different region extraction or masking
    pos_img = avg
    pos_img[pos_img < 0] = 0
    
    #dilate by pixel size to connect neighbor regions
    extra = 3 #Angrstroms to dilate (set minimum of 3A)
    extend = int(np.ceil((pixel_size/extra)**-1))

    struct = np.ones((extend, extend), dtype=bool)
    dilate = ndi.binary_dilation(input=pos_img, structure=struct)

    labeled = ski.measure.label(dilate, connectivity=2, background=False)

    #select a single region from each 2D class average
    rprops = ski.measure.regionprops(labeled, cache=False)
    bbox = [r.bbox for r in rprops]

    if len(bbox) == 1:
        #use only region in the image
        selected = 1

    elif len(bbox) > 1:
        img_x_center = len(avg)/2
        img_y_center = len(avg[:,0])/2
        img_center = (img_x_center, img_y_center)
        #for use in distance calculation
        x1 = np.array(img_center)

        box_range = {}
        for i, box in enumerate(bbox):
            width_coord = list(range(box[1], box[3]+1))
            length_coord = list(range(box[0], box[2]+1))
            box_range[i+1] = [width_coord, length_coord]

        box_centers = {}
        for i, box in enumerate(bbox):
            y_max, y_min = box[0], box[2]
            x_max, x_min = box[1], box[3]
            center_xy = (((x_max+x_min)/2, (y_max+y_min)/2))
            #i+1 because 0 is the background region
            box_centers[i+1] = center_xy

        selected = 'none'

        for region, bound in box_range.items():
            #first check if there is a region in the center
            if img_x_center in bound[0] and img_y_center in bound[1]:
                #use center region
                selected = region

        if selected == 'none':
            #find box closest to the center    
            distance = {}
            for region, center in box_centers.items():
                x2 = np.array(center)
                distance[region] = spatial.distance.euclidean(x1, x2)  
            region = min(distance, key=distance.get) 
            #use region closest to center
            selected = region

    selected_region = (labeled == selected)

    properties = ski.measure.regionprops(selected_region.astype('int'))
    bbox = properties[0].bbox

    y_min, y_max = bbox[0], bbox[2]
    x_min, x_max = bbox[1], bbox[3]

    #keep only true pixels in bounding box
    true_region = np.empty(avg.shape)

    for i, row in enumerate(avg):
        for j, pixel in enumerate(row):
            if selected_region[(i,j)] == True:
                true_region[(i,j)] = pos_img[(i, j)]
            else:
                true_region[(i,j)] = 0

    new_region = true_region[y_min:y_max, x_min:x_max]
    
    return new_region


def pairwise_l2(a, b):
    score = spatial.distance.euclidean(a.vector, b.vector)
    return score

def pairwise_l1(a, b):
    score = sum(abs(a.vector - b.vector))
    return score

def pairwise_cosie(a, b):
    score = spatial.distance.cosine(a.vector - b.vector)
    return score 

def pairwise_xcorr(a, b):
    score = signal.correlate(a.vector, b.vector, mode='valid')
    score_max = np.amax(score)
    return score_max


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
            projection_shifts.append(Projection(class_avg = str(shift.class_avg)+'_'+str(i),
                                                angle = shift.angle,
                                                vector = v))

        scores = []
        
        for shifted in projection_shifts:
            scores.append(pairwise_score(static, shifted))
        score = min(scores)    
    return score


def wrapper_xcorr(pair, pairwise):
    """
    - pair is tuple of instances from Projection class
    - func is how to pairwise score vectors defined from user
    """
    score = pairwise(pair[0], pair[1])
    
    return [pair[0].class_avg, 
            pair[0].angle, 
            pair[1].class_avg, 
            pair[1].angle,
            score]


def wrapper_function(pair, slide, pairwise):
    """
    - same as above but passes pairwise_score to slide_score
    """
    score = slide(pair[0], pair[1], pairwise)

    return [pair[0].class_avg, 
            pair[0].angle, 
            pair[1].class_avg, 
            pair[1].angle,
            score]


def multiproc_pairwise(projection_pairs, num_workers, metric):
    """
    """
    with multiprocessing.Pool(num_workers) as pool:
        if metric == 'Euclidean':
            pair_scores = pool.starmap(wrapper_function, 
                                       [(pair, slide_score, pairwise_l2) for pair in projection_pairs])
        elif metric == 'L1':
            pair_scores = pool.starmap(wrapper_function, 
                                       [(pair, slide_score, pairwise_l1) for pair in projection_pairs])
        elif metric == 'cosine':    
            pair_scores = pool.starmap(wrapper_function, 
                                       [(pair, slide_score, pairwise_cosine) for pair in projection_pairs])
        elif metric == 'cross-correlation':
            pair_scores = pool.starmap(wrapper_xcorr, 
                                       it.product(projection_pairs, [pairwise_xcorr]))
    return pair_scores


def optimum_scores(score_matrix, metric):
    """
    best pairwise score for all 1D projections
    """
    optimum_pairwise_scores = {}
    
    complete_score_matrix = {}

    #cross-correlation optimum is max, other metrics are min
    if metric != 'cross-correlation': 
        for pair, scores in score_matrix.items():
            optimum_pairwise_scores[pair] = min(score_matrix[pair], key = lambda x: x[2])
    else:
        for pair, scores in score_matrix.items():
            optimum_pairwise_scores[pair] = max(score_matrix[pair], key = lambda x: x[2])

    #complete the array
    for pair, value in optimum_pairwise_scores.items():
        if pair[0] == pair[1]:
            next
        else:
            complete_score_matrix[pair] = value
            #now flip pair
            score = value[2]
            angle_1 = value[0]
            angle_2 = value[1]
            complete_score_matrix[(pair[1], pair[0])] = (angle_2, angle_1, score)
        
    return complete_score_matrix
        
        
def write_scores(complete_scores, outpath, description):
    """
    tab separted text file of complete score matrix
    to load into the slicem gui
    """
    with open(outpath+'/slicem_scores_{0}.txt'.format(description), 'w') as f:
        f.write('projection_1' + '\t' + 'angle_1' + '\t' +'projection_2' + '\t' + 'angle_2' + '\t' + 'score' + '\n')
        for p, v in complete_scores.items():
            f.write(str(p[0])+'\t'+str(v[0])+'\t'+str(p[1])+'\t'+str(v[1])+'\t'+str(v[2])+'\n')
            
            
if __name__ == "__main__":
    main()