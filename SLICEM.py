import time
import mrcfile
import argparse
import numpy as np
import skimage as ski
import multiprocessing
from scipy import spatial
from scipy import ndimage as ndi

ROTATION_DEGREES = np.arange(0, 360, 5)

def main():
    """
    calculates similarity between line projections from 2D class averages
    """
    parser = argparse.ArgumentParser(description='compare similarity of 2D class averages based on common lines')
    
    parser.add_argument('-i', '--input', action='store', dest='mrc_input', required=True,
                        help='path to mrcs file of 2D class averages')
    
    parser.add_argument('-o', '--outpath', action='store', dest='outpath', required=True,
                        help='path for output files')
    
    parser.add_argument('-m', '--metric', action='store', dest='metric', required=False, default='Euclidean',
                        choices=['Euclidean', 'L1', 'cosine'],
                        help='choose scoring method, Euclidean default')
    
    parser.add_argument('-p', '--pixel_size', action='store', dest='pixel_size', type=float, required=True,
                        help='pixel size of 2D class averages in A/pixel')
    
    parser.add_argument('-s', '--scale_factor', action='store', dest='scale_factor', type=float, required=False, default=1,
                        help='scale factor for downsampling. (e.g. -s 2 converts 100pix box --> 50pix box)')
    
    parser.add_argument('-c', '--num_workers', action='store', dest='num_workers', type=int, required=False, default=1,
                        help='number of CPUs to use')

    args = parser.parse_args()

    final_scores = {}
    
    projection_2D = get_projection_2D(mrcs=args.mrc_input,
                                      pixel_size=args.pixel_size,
                                      factor=args.scale_factor)
      
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
    
    for i in range(num_class_avg-1):
        line_projections_1 = vectorize(i, projection_2D[i])
        for j in range(i+1, num_class_avg):
            line_projections_2 = vectorize(j, projection_2D[j])

            projection_pairs = []
            for line_1 in line_projections_1.values():
                for line_2 in line_projections_2.values():
                    projection_pairs.append((line_1, line_2))

            with multiprocessing.Pool(args.num_workers) as pool:
                pair_scores = pool.starmap(
                    wrapper_function, 
                    [(pair, slide_score, pairwise_score) for pair in projection_pairs]
                )

            optimum =  min(pair_scores, key = lambda x: x[4])
            avg_1 = optimum[0]
            deg_1 = optimum[1]
            avg_2 = optimum[2]
            deg_2 = optimum[3]
            score = optimum[4]
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

    
def get_projection_2D(mrcs, pixel_size, factor):
    """
    read, scale and extract class averages 
    """
    projection_2D = {}

    with mrcfile.open(mrcs) as mrc:
        for i, data in enumerate(mrc.data):
            projection_2D[i] = data
        mrc.close()

    for k, avg in projection_2D.items():
        if factor == 1:
            projection_2D[k] = extract_class_avg(avg, pixel_size)
        else:
            scaled_img = ski.transform.rescale(avg, 
                                               scale=(1/factor), 
                                               anti_aliasing=True, 
                                               multichannel=False, #add to supress warning
                                               mode='constant') #add to supress warning
            projection_2D[k] = extract_class_avg(scaled_img, pixel_size*factor)
            
    return projection_2D


def extract_class_avg(avg, pixel_size):
    """
    keep positive values from normalized class average
    remove extra densities in class average
    fit in minimal bounding box
    """
    pos_img = avg.copy()
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


def vectorize(key, image):
    """
    takes image and creates 1D projections
    similar to Radon transform
    """
    projection_1D = {}
    for degree in ROTATION_DEGREES:
        proj_1D = ski.transform.rotate(image, degree, resize=True).sum(axis=0)
        trim_1D = np.trim_zeros(proj_1D, trim='fb').astype('float32')
        projection_1D[(key, degree)] = Projection(class_avg=key, angle=degree, vector=trim_1D)
    return projection_1D


def pairwise_l2(a, b):
    score = spatial.distance.euclidean(a.vector, b.vector)
    return score


def pairwise_l1(a, b):
    score = sum(abs(a.vector - b.vector))
    return score


def pairwise_cosie(a, b):
    score = spatial.distance.cosine(a.vector - b.vector)
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
            projection_shifts.append(Projection(class_avg = str(shift.class_avg)+'_'+str(i),
                                                angle = shift.angle,
                                                vector = v))

        scores = []
        
        for shifted in projection_shifts:
            scores.append(pairwise_score(static, shifted))
        score = min(scores)    
    return score


def wrapper_function(pair, slide, pairwise):
    """
    - pair is tuple from Projection class to be scored
    - pairwise is function to score vectores (e.g. Euclidean)
    """
    score = slide(pair[0], pair[1], pairwise)

    return [pair[0].class_avg, 
            pair[0].angle, 
            pair[1].class_avg, 
            pair[1].angle,
            score]

                
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
