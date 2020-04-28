import mrcfile
import numpy as np
import skimage as ski
import networkx as nx
from igraph import Graph

from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from scipy import ndimage as ndi
from scipy import spatial, signal, stats


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
    
    @property
    def dimension(self):
        return self.vector.ndim


def nearest_neighbors(complete_scores, num_class_avg, neighbors, metric):
    """
    group k best scores for each class average to construct graph 
    """
    order_scores = {avg: [] for avg in range(num_class_avg)}
    
    projection_knn = {}

    #reorganize score matrix to sort for nearest neighbors
    #projection_knn[projection_1] = [projection_2, angle_1, angle_2, score]
    for pair, values in complete_scores.items():
        p1 = pair[0]
        p2 = pair[1]
        a1 = values[0]
        a2 = values[1]
        s = values[2]
        
        #option: convert distance to similarity
        #if metric != 'cross-correlation':
        #    s = 1/(1 + values[2])
        #else:
        #    s = values[2]
            
        c = [p2, a1, a2, s]
        order_scores[p1].append(c)
        
    #zscore scores by each projection, use as edgeweight in graph
    for projection, scores in order_scores.items():
        all_scores = []
        for v in scores:
            all_scores.append(v[3])
        
        u = np.mean(all_scores)
        s = np.std(all_scores)
    
        for v in scores:
            zscore = (v[3] - u)/s
            v[3] = zscore

    for avg, scores in order_scores.items():
        if metric != 'cross-correlation':
            sort = sorted(scores, reverse=False, key=lambda x: x[3])[:neighbors]
        else:
            sort = sorted(scores, reverse=True, key=lambda x: x[3])[:neighbors]
        projection_knn[avg] = sort
        
    return projection_knn


def remove_outliers(clusters, projection_2D):
    """
    use median absolute deviation of summed 2D projections to remove outliers
    inspect outliers for further processing
    """
    pixel_sums = {}
    
    outliers = []

    for cluster, nodes in clusters.items():
        pixel_sums[cluster] = []
        for node in nodes:
            pixel_sums[cluster].append(sum(sum(projection_2D[node])))

    for cluster, psums in pixel_sums.items():
        med = np.median(psums)
        m_psums = [abs(x - med) for x in psums]
        mad = np.median(m_psums)

        for i, proj in enumerate(psums):
            #Boris Iglewicz and David Hoaglin (1993)
            z = 0.6745*(proj - med)/mad
            if abs(z) > 3.5:
                outliers.append((cluster, clusters[cluster][i]))

    clusters["outliers"] = [o[1] for o in outliers]
    
    for outlier in outliers:
        cluster = outlier[0]
        node = outlier[1]
        clusters[cluster].remove(node)
        print('class_avg node {0} was removed from cluster {1} as an outlier'.format(node, cluster))
        
        
def create_network(metric,
                   community_detection,
                   projection_2D, 
                   num_class_avg, 
                   complete_scores, 
                   network_from, 
                   neighbors, 
                   top):
    """
    get new clusters depending on plotting input options
    """

    if network_from == 'top_n':
        sort_by_scores = []

        for pair, score in complete_scores.items():
            sort_by_scores.append([pair[0], pair[1], score[2]])

        if metric != 'cross-correlation':
            top_n = sorted(sort_by_scores, reverse=False, key=lambda x: x[2])[:top]

            #convert from distance to similarity (not zscore like with knn)
            for score in top_n:
                c = 1/(1 + score[2])
                score[2] = c
        else:
            top_n = sorted(sort_by_scores, reverse=True, key=lambda x: x[2])[:top]

        #change to tuple for iGraph and networkx
        flat = [tuple(pair) for pair in top_n]

    else:
        flat = []

        projection_knn = nearest_neighbors(complete_scores=complete_scores, 
                                           num_class_avg=num_class_avg, 
                                           neighbors=neighbors, 
                                           metric=metric)

        for projection, knn in projection_knn.items():
            for n in knn:
                #(proj_1, proj_2, score)
                #abs score to correct for distance vs similarity
                flat.append((projection, n[0], abs(n[3])))

    #recluster the new knn or top_n scores
    clusters = {}

    g = Graph.TupleList(flat, weights=True)

    if community_detection == 'walktrap':
        wt = Graph.community_walktrap(g, weights='weight', steps=4)
        cluster_dendrogram = wt.as_clustering()
    elif community_detection == 'betweenness':
        ebs = Graph.community_edge_betweenness(g, weights='weight', directed=True)
        cluster_dendrogram = ebs.as_clustering()

    for community, projection in enumerate(cluster_dendrogram.subgraphs()):
        clusters[community] = projection.vs['name']

    #convert node IDs back to ints
    for cluster, nodes in clusters.items():
        clusters[cluster] = [int(node) for node in nodes]
        clusters[cluster] = sorted(nodes)

    #and remove outliers
    remove_outliers(clusters=clusters, projection_2D=projection_2D)

    #adds singles to clusters if they are not in top n scores
    clustered = []

    for cluster, nodes in clusters.items():
        for n in nodes:
            clustered.append(n)

    clusters['singles'] = []

    for node_key in projection_2D:
        if node_key not in clustered:
            clusters['singles'].append(node_key)
      
    G = nx.Graph()

    for pair in flat:
        G.add_edge(int(pair[0]), int(pair[1]), weight=pair[2])

    #if you want to see directionality in the networkx plot
    #G = nx.MultiDiGraph(G)

    #adds singles if not in top n scores
    for node_key in projection_2D:
        if node_key not in G.nodes:
            G.add_node(node_key)
    
    return clusters, G


def random_color():
    return tuple(np.random.rand(1,3)[0])

def get_plot_colors(clusters, graph):

    color_list = []

    preset_colors = [color for colors in [cm.Set3.colors] for color in colors]

    for i in range(len(clusters)):
        if i < len(preset_colors):
            color_list.append(preset_colors[i])
        else:
            color_list.append(random_color())   
    
    colors = []

    for i, node in enumerate(graph.nodes):
        for cluster, projections in clusters.items():
            if cluster == 'singles':
                if node in projections:
                    colors.append((0.85, 0.85, 0.85))
            elif cluster == 'outliers':
                if node in projections:
                    colors.append((0.35, 0.35, 0.35))    
            elif node in projections:
                colors.append((color_list[cluster]))
                
    return colors


def network_positions(graph, network_from):
    if network_from == 'knn':
        pos = nx.spring_layout(graph, weight='weight', k=0.2, scale=3.5,)
    else:
        pos = nx.spring_layout(graph, weight='weight')
        
    return pos


def plot_network(graph, positions, colors, save, outpath):
    
    fig, ax = plt.subplots(figsize=(16, 8))


    nx.draw_networkx(graph, pos=positions, with_labels=True, node_size=300, width=1, 
                     node_color=colors, edge_color='grey', font_size=10, linewidths=1.5,
                     node_shape=('o'), font_weight='bold', alpha=0.85)


    plt.tick_params(axis='both',   
                    which='both',     
                    bottom=False,      
                    top=False,
                    left=False,
                    labelbottom=False,
                    labelleft=False)

    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.collections[0].set_edgecolor('k')

    if save:
        plt.savefig(outpath+'slicem_network.pdf')
    else:
        plt.show()     

        
def tile_2D_clusters(clusters, graph, mrc_input, colors, save, outpath):
    """
    plot 2D class avgs sorted and colored by cluster
    """
    
    projection_2D = {}
    
    with mrcfile.open(mrc_input) as mrc:
        for i, data in enumerate(mrc.data):
            projection_2D[i] = data.astype('float64') 
    
    ordered_projections = []
    colors_2D = []
    flat_clusters = []

    for cluster, nodes in clusters.items():
        for n in nodes:
            ordered_projections.append(projection_2D[n])

        for n in nodes:
            flat_clusters.append(n)

        for i, n in enumerate(graph.nodes):
            if n in nodes:
                colors_2D.append(colors[i])
    
    gc = int(np.ceil(np.sqrt(len(ordered_projections))))

    if len(ordered_projections) <= (gc**2 - gc):
        gr = gc - 1
    else:
        gr = gc

    #assuming images are same size, get shape
    l, w = ordered_projections[0].shape

    #add blank images to pack in grid
    while len(ordered_projections) < gr*gc:
        ordered_projections.append(np.zeros((l, w)))

    fig = plt.figure(figsize=(16., 16.))

    grid = ImageGrid(fig, 111, #similar to subplot(111)
                     nrows_ncols=(gr, gc), #creates grid of axes
                     axes_pad=0.05) #pad between axes in inch

    props = dict(boxstyle='round', facecolor='white')

    for i, (ax, im) in enumerate(zip(grid, ordered_projections)):
        #iterate over the grid returns the axes
        ax.imshow(im, cmap='gray')

        ax.spines['bottom'].set_color(colors_2D[i])
        ax.spines['bottom'].set_linewidth(3)

        ax.spines['top'].set_color(colors_2D[i])
        ax.spines['top'].set_linewidth(3)

        ax.spines['left'].set_color(colors_2D[i])
        ax.spines['left'].set_linewidth(3)

        ax.spines['right'].set_color(colors_2D[i])
        ax.spines['right'].set_linewidth(3)

        ax.set_xticklabels([])
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.set_yticklabels([])
        ax.yaxis.set_tick_params(which='both', length=0)

        text = str(flat_clusters[i])
        ax.text(10, 10, text, va='top', ha='left', bbox=props)

    if save:
        plt.savefig(outpath+'2D_clustered.pdf')
    else:
        plt.show()        
        
        
def plot_projections(projection1, angle1, projection2, angle2, save, outpath, proj_1, proj_2):
    """
    plots pixel extracted class averages and line projection for class average pair
    at the angle giving the most similar line projection
    """    
    ref = ski.transform.rotate(projection1, angle1, resize=True)
    comp = ski.transform.rotate(projection2, angle2, resize=True)
    
    f, axarr = plt.subplots(2, 2, figsize=(8, 8))

    rx, ry = np.shape(ref)
    cx, cy = np.shape(comp)

    m = max(rx, ry, cx, cy)

    if rx < m:    
        if (m-rx) % 2  == 0:
            rx_pad = int((m-rx)/2)
            ref = np.pad(ref, pad_width=((rx_pad,rx_pad), (0,0)), mode='constant')
        else:
            rx_pad = int((m-rx)/2)
            ref = np.pad(ref, pad_width=((rx_pad+1,rx_pad), (0,0)), mode='constant')  


    if ry < m:
        if (m-ry) % 2 == 0:
            ry_pad = int((m-ry)/2)
            ref = np.pad(ref, pad_width=((0,0), (ry_pad,ry_pad)), mode='constant')
        else:
            ry_pad = int((m-ry)/2)
            ref = np.pad(ref, pad_width=((0,0), (ry_pad+1,ry_pad)), mode='constant')


    if cx < m:    
        if (m-cx) % 2  == 0:
            cx_pad = int((m-cx)/2)
            comp = np.pad(comp, pad_width=((cx_pad,cx_pad), (0,0)), mode='constant')
        else:
            cx_pad = int((m-cx)/2)
            comp = np.pad(comp, pad_width=((cx_pad+1,cx_pad), (0,0)), mode='constant')


    if cy < m:
        if (m-cy) % 2 == 0:
            cy_pad = int((m-cy)/2)
            comp = np.pad(comp, pad_width=((0,0), (cy_pad,cy_pad)), mode='constant')
        else:
            cy_pad = int((m-cy)/2)
            comp = np.pad(comp, pad_width=((0,0), (cy_pad+1,cy_pad)), mode='constant')   
        
        
    ref_intensity = ref.sum(axis=0)
    ref_list = range(1, len(ref_intensity)+1)
    
    comp_intensity = comp.sum(axis=0)
    comp_list = range(1, len(comp_intensity)+1)
    
    ref_max = np.amax(ref_intensity)
    comp_max = np.amax(comp_intensity)
    axis_max = max(ref_max, comp_max) + 0.1*max(ref_max, comp_max)
    
    ref_min = np.amin(ref_intensity)
    comp_min = np.amin(comp_intensity)
    axis_min = min(ref_min, comp_min)

    #  PROJECTION_1
    #2D projection image
    axarr[0, 0].imshow(ref, cmap=plt.get_cmap('gray'), aspect='equal') 
    axarr[0, 0].axis('off')
    
    #1D line projection
    axarr[1, 0].plot(ref_list, ref_intensity, color='black')
    axarr[1, 0].xaxis.set_visible(True)
    axarr[1, 0].set_ylabel('Intensity')
    #axarr[1, 0].yaxis.set_visible(False)
    axarr[1, 0].set_ylim([axis_min, axis_max+0.5])
    axarr[1, 0].fill_between(ref_list, 0, ref_intensity, alpha=0.5, color='deepskyblue')
    
    #  PROJECTION_2
    #2D projection image
    axarr[0, 1].imshow(comp, cmap=plt.get_cmap('gray'), aspect='equal')
    axarr[0, 1].axis('off')
    
    #lD line projection
    axarr[1, 1].plot(comp_list, comp_intensity, color='black')
    axarr[1, 1].xaxis.set_visible(True)
    #axarr[1, 1].set_ylabel('Intensity')
    axarr[1, 1].yaxis.set_visible(False)
    axarr[1, 1].set_ylim([axis_min, axis_max+0.5])
    axarr[1, 1].fill_between(comp_list, 0, comp_intensity, alpha=0.5, color='red')
    
    plt.tight_layout()
    
    if save:
        plt.savefig(outpath+'{0}_{1}.pdf'.format(proj_1, proj_2))
    else:
        plt.show()


#add pairwise scoring like in main but keep track of optimum location
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
    
    lp1 = a.size()
    lp2 = b.size()
    if lp1 < lp2 or lp1 == lp2:
        shift = a
        static = b
    else:
        shift = b
        static = a
    
    loc = np.argwhere(score == np.amax(score))
    #if multiple maximum occur pick the first
    loc = loc[0]
    
    info = (static, shift, loc)
    
    return score_max, info

def slide_score(a, b, pairwise_score):
    """
    a, b are instances of the Projection class
    finds minimum pairwise score for all translations
    """
    lp1 = a.size()
    lp2 = b.size()
    
    dol = abs(lp1 - lp2)
    
    if dol == 0:
        score = pairwise_score(a, b)
        info = (a, b, np.array([0]))

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
        loc = np.argwhere(scores == np.amin(scores))
        #if multiple maximum occur pick the first
        loc = loc[0]
        info = (static, shift, loc)
        
    return score, info


def overlay_lines(projection1, projection2, metric, save, outpath):
    """
    overlays line projections at optimum angle between two class averages
    """
    
    #first get location where line projections have optimum overlap
    if metric == 'Euclidean':
        score, info = slide_score(projection1, projection2, pairwise_l2)
    elif metric == 'L1':
        score, info = slide_score(projection1, projection2, pairwise_l1)
    elif metric == 'cosine':    
        score, info = slide_score(projection1, projection2, pairwise_cosie)
    elif metric == 'cross-correlation':
        score, info = pairwise_xcorr(projection1, projection2)
    
    loc = info[2]
    dol = abs(projection1.size() - projection2.size())
    
    if projection1.size() >= projection2.size():
        ref = info[0]
        comp = info[1]
        ref_intensity = ref.vector
        comp_intensity = np.pad(comp.vector, pad_width=(loc[0], dol-loc), mode='constant')
    else:
        ref = info[1]
        comp = info[0]        
        ref_intensity = np.pad(ref.vector, pad_width=(loc[0], dol-loc), mode='constant')    
        comp_intensity = comp.vector

    ref_list = range(1, len(ref_intensity)+1)
    comp_list = range(1, len(comp_intensity)+1)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    x_axis_max = max(len(ref_intensity), len(comp_intensity))+1
    y_axis_max = max(np.amax(ref_intensity), np.amax(comp_intensity))
    y_axis_min = min(np.amin(ref_intensity), np.amin(comp_intensity))

    ax.plot(ref_list, ref_intensity, color='black')
    ax.plot(comp_list, comp_intensity, color='black')
    
    ax.fill_between(ref_list, 0, ref_intensity, alpha=0.35, color='deepskyblue')
    ax.fill_between(comp_list, 0, comp_intensity, alpha=0.35, color='red')

    ax.set_ylabel('Intensity')
    ax.set_ylim([y_axis_min, (y_axis_max + 0.025*y_axis_max)])
    
    ax.set_xticks(np.arange(0, x_axis_max, step=10))
    ax.set_xlim(0, x_axis_max)
    ax.set_xlabel('Pixel')
    
    if save:
        plt.savefig(outpath+'overlap_{0}_{1}.pdf'.format(projection1.class_avg, projection2.class_avg))
    else:
        plt.show()