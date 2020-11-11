import os
import mrcfile
import numpy as np
import pandas as pd
import skimage as ski
import networkx as nx
from igraph import Graph
from scipy import ndimage as ndi
from scipy import spatial, signal, stats

import tkinter as tk
from tkinter import ttk
import tkinter.filedialog

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use('TkAgg')


class SLICEM_GUI(tk.Tk):
    
    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "SLICEM_GUI")

        tabControl = ttk.Notebook(self) 
        input_tab = ttk.Frame(tabControl) 
        network_tab = ttk.Frame(tabControl) 
        projection_tab = ttk.Frame(tabControl)
        output_tab = ttk.Frame(tabControl)
        
        tabControl.add(input_tab, text='Inputs') 
        tabControl.add(network_tab, text='Network Plot')
        tabControl.add(projection_tab, text='Projection Plot')
        tabControl.add(output_tab, text='Outputs')
        tabControl.pack(expand=1, fill="both") 
        
        self.cwd = os.getcwd()
        
        #########################     INPUT TAB     ############################## 
        mrc_label = ttk.Label(input_tab, text="path to 2D class averages (mrcs): ")
        mrc_label.grid(row=0, column=0, sticky=tk.E, pady=10) 
        self.mrc_entry = ttk.Entry(input_tab, width=20)
        self.mrc_entry.grid(row=0, column=1, sticky=tk.W, pady=10) 
        self.mrc_button = ttk.Button(input_tab, 
                                     text="Browse", 
                                     command=lambda: self.set_text(text=self.askfile(),
                                                                   entry=self.mrc_entry))
        self.mrc_button.grid(row=0, column=2, sticky=tk.W, pady=2)

        
        scores_label = ttk.Label(input_tab, text="path to SLICEM scores: ")
        scores_label.grid(row=1, column=0, sticky=tk.E, pady=10) 
        self.score_entry = ttk.Entry(input_tab, width=20)
        self.score_entry.grid(row=1, column=1, sticky=tk.W, pady=10) 
        self.score_button = ttk.Button(input_tab, 
                                       text="Browse", 
                                       command=lambda: self.set_text(text=self.askfile(),
                                                                     entry=self.score_entry))
        self.score_button.grid(row=1, column=2, sticky=tk.W, pady=2)        
        
        
        pixel_label = ttk.Label(input_tab, text="pixel size of 2D classes ("u"\u212B""/pix): ")
        pixel_label.grid(row=2, column=0, sticky=tk.E, pady=10) 
        self.pixel_entry = ttk.Entry(input_tab, width=20)
        self.pixel_entry.grid(row=2, column=1, sticky=tk.W, pady=10) 
        
        
        metric_label = ttk.Label(input_tab, text="metric used for scoring: ")
        metric_label.grid(row=3, column=0, sticky=tk.E, pady=2)
        options = ['Euclidean', 'L1', 'cosine', 'cross-correlation',
                   'norm-xcorr', 'norm-Euclidean']
        self.metric = tk.StringVar(input_tab)
        self.metric.set(options[0])
        self.opt_menu = ttk.OptionMenu(input_tab, self.metric, options[0], *options)
        self.opt_menu.config(width=16)
        self.opt_menu.grid(row=3, column=1, sticky=tk.E, pady=10)
        
        
        self.load_button = ttk.Button(input_tab,
                                      text='Load Inputs',
                                      command=lambda: self.load_inputs(self.mrc_entry,
                                                                       self.score_entry,
                                                                       self.pixel_entry,
                                                                       self.metric))
        self.load_button.grid(row=4, column=1, pady=20)
        ############################################################################           
    
    
    
    
        #########################     NETWORK TAB     ##############################
        network_tab.grid_rowconfigure(0, weight=1)
        network_tab.grid_columnconfigure(0, weight=1)
        
        #TOP FRAME
        nettopFrame = tk.Frame(network_tab, bg='lightgrey', width=600, height=400)
        nettopFrame.grid(row=0, column=0, sticky='nsew')

        self.netcanvas = None
        self.nettoolbar = None
        

        #BOTTOM FRAME
        netbottomFrame = ttk.Frame(network_tab, width=600, height=100)
        netbottomFrame.grid(row=1, column=0, sticky='nsew')
        netbottomFrame.grid_propagate(0)

        
        self.detection = tk.StringVar(network_tab)
        self.detection.set('walktrap')

        comm_label = ttk.Label(netbottomFrame, text='community detection algorithm: ')
        comm_label.grid(row=0, column=0, sticky=tk.E)        

        self.comm1 = ttk.Radiobutton(netbottomFrame, 
                                     text='walktrap', 
                                     variable=self.detection, 
                                     value='walktrap')
        self.comm1.grid(row=0, column=1, padx=5, sticky=tk.W)
        
        self.comm2 = ttk.Radiobutton(netbottomFrame, 
                                     text='betweenness', 
                                     variable=self.detection,
                                     value='betweenness')
        self.comm2.grid(row=0, column=2, padx=3, sticky=tk.W)
        
        
        self.network = tk.StringVar(network_tab)
        self.network.set('knn')
        
        net_label = ttk.Label(netbottomFrame, text='construct network from: ')
        net_label.grid(row=1, column=0, sticky=tk.E)
        
        self.net1 = ttk.Radiobutton(netbottomFrame, 
                                    text='nearest neighbors', 
                                    variable=self.network, 
                                    value='knn')
        self.net1.grid(row=1, column=1, padx=5, sticky=tk.W)
        
        self.net2 = ttk.Radiobutton(netbottomFrame, 
                                    text='top n scores', 
                                    variable=self.network,
                                    value='top_n')
        self.net2.grid(row=2, column=1, padx=5, sticky=tk.W)

        knn_label = ttk.Label(netbottomFrame, text='(# of k): ')
        knn_label.grid(row=1, column=2, sticky=tk.W)
        self.knn_entry = ttk.Entry(netbottomFrame, width=6)
        self.knn_entry.insert(0, 0)
        self.knn_entry.grid(row=1, column=2, sticky=tk.E)
        
        topn_label = ttk.Label(netbottomFrame, text='(# of n): ')
        topn_label.grid(row=2, column=2, sticky=tk.W)
        self.topn_entry = ttk.Entry(netbottomFrame, width=6)
        self.topn_entry.insert(0, 0)
        self.topn_entry.grid(row=2, column=2, sticky=tk.E)
        
        
        self.cluster = ttk.Button(netbottomFrame,
                                  width=12,
                                  text='cluster', 
                                  command=lambda: self.slicem_cluster(metric,
                                                                      self.detection.get(),
                                                                      extract_2D,
                                                                      complete_scores,
                                                                      self.network.get(),
                                                                      int(self.knn_entry.get()),
                                                                      int(self.topn_entry.get())))
        self.cluster.grid(row=0, column=3, sticky=tk.W, padx=10, pady=2)
        
        
        self.net_plot = ttk.Button(netbottomFrame,
                                   width=12,
                                   text='plot network', 
                                   command=lambda: self.plot_slicem_network(self.network.get(),
                                                                            nettopFrame))
        self.net_plot.grid(row=1, column=3, sticky=tk.W, padx=10, pady=2)
        
        
        self.tiles = ttk.Button(netbottomFrame, 
                                width=12,
                                text='plot 2D classes', 
                                command=lambda: self.plot_tiles(nettopFrame))
        self.tiles.grid(row=2, column=3, sticky=tk.W, padx=10, pady=2)
        ############################################################################

        
        
        
        #########################     PROJECTION TAB      ########################## 
        projection_tab.grid_rowconfigure(0, weight=1)
        projection_tab.grid_columnconfigure(0, weight=1)
        
        #TOP FRAME
        projtopFrame = tk.Frame(projection_tab, bg='lightgrey', width=600, height=400)
        projtopFrame.grid(row=0, column=0, sticky='nsew')
        projtopFrame.grid_rowconfigure(0, weight=1)
        projtopFrame.grid_columnconfigure(0, weight=1)
        
        self.projcanvas = None
        self.projtoolbar = None


        #BOTTOM FRAME
        projbottomFrame = ttk.Frame(projection_tab, width=600, height=50)
        projbottomFrame.grid(row=1, column=0, sticky='nsew')
        projbottomFrame.grid_propagate(0)
        
        
        avg1_label = ttk.Label(projbottomFrame, text='class average 1: ')
        avg1_label.grid(row=0, column=0, sticky=tk.E, padx=2)
        self.avg1 = ttk.Entry(projbottomFrame, width=5)
        #self.avg1.insert(0, 0)
        self.avg1.grid(row=0, column=1, padx=2)
        
        avg2_label = ttk.Label(projbottomFrame, text='class avereage 2: ')
        avg2_label.grid(row=0, column=2, sticky=tk.E, padx=2)
        self.avg2 = ttk.Entry(projbottomFrame, width=5)
        #self.avg2.insert(0, 1)
        self.avg2.grid(row=0, column=3, padx=2)
        
        
        self.proj_button = ttk.Button(projbottomFrame, 
                                      text='plot projections',
                                      command=lambda: self.plot_projections(int(self.avg1.get()),
                                                                            int(self.avg2.get()),
                                                                            projtopFrame))
        self.proj_button.grid(row=0, column=4, padx=20)
        
        self.overlay_button = ttk.Button(projbottomFrame,
                                         text='plot overlap',
                                         command=lambda: self.overlay_lines(int(self.avg1.get()),
                                                                            int(self.avg2.get()),
                                                                            projtopFrame))
        self.overlay_button.grid(row=0, column=5, padx=12)
    ################################################################################     

    
    
    
    ###########################     OUTPUT TAB     ################################# 
        star_label = ttk.Label(output_tab, text='path to corresponding star file (star): ')
        star_label.grid(row=0, column=0, sticky=tk.E, pady=10)
        self.star_entry = ttk.Entry(output_tab, width=20)
        self.star_entry.grid(row=0, column=1, stick=tk.W, pady=10)
        self.star_button = ttk.Button(output_tab, 
                                      text="Browse", 
                                      command=lambda: self.set_text(text=self.askfile(),
                                                                    entry=self.star_entry))
        self.star_button.grid(row=0, column=2, sticky=tk.W, pady=2)
        
        outdir_label = ttk.Label(output_tab, text='directory to save files in: ')
        outdir_label.grid(row=1, column=0, sticky=tk.E, pady=10)
        self.out_entry = ttk.Entry(output_tab, width=20)
        self.out_entry.grid(row=1, column=1, sticky=tk.W, pady=10)
        self.out_button = ttk.Button(output_tab, 
                                     text="Browse", 
                                     command=lambda: self.set_text(text=self.askpath(),
                                                                   entry=self.out_entry))
        self.out_button.grid(row=1, column=2, sticky=tk.W, pady=2)
        
        
        self.write_button = ttk.Button(output_tab,
                                       text='Write Star Files',
                                       command=lambda: self.write_star_files(self.star_entry.get(),
                                                                             self.out_entry.get()))
        self.write_button.grid(row=2, column=1, pady=20)
        
        
        self.write_edges = ttk.Button(output_tab,
                                      text='Write Edge List',
                                      command=lambda: self.write_edge_list(self.network.get(),
                                                                           self.out_entry.get()))
        self.write_edges.grid(row=3, column=1, pady=10)
    ################################################################################      
    

    
    
    ###############################   GUI METHODS   ################################
    def load_scores(self, score_file):
        complete_scores = {}
        with open(score_file, 'r') as f:
            next(f)
            for line in f:
                l = line.rstrip('\n').split('\t')
                complete_scores[(int(l[0]), int(l[2]))] = (int(l[1]), int(l[3]), float(l[4]))
        return complete_scores
    
    
    def load_class_avg(self, mrc_input):
        projection_2D = {}
        with mrcfile.open(mrc_input) as mrc:
            for i, data in enumerate(mrc.data):
                projection_2D[i] = data.astype('float64')
        return projection_2D
    
    
    def load_inputs(self, mrc_entry, score_entry, pixel_entry, metric_entry):
        global projection_2D, extract_2D, complete_scores, pixel_size, metric
        
        projection_2D = self.load_class_avg(mrc_entry.get())
        complete_scores = self.load_scores(score_entry.get())
        pixel_size = float(pixel_entry.get())
        metric = metric_entry.get()
        
        extract_2D = {}
        for k, avg in projection_2D.items():
            extract_2D[k] = extract_class_avg(avg)
            
        print('Inputs Loaded!')

        
    def askfile(self):
        file = tk.filedialog.askopenfilename(initialdir=self.cwd)
        return file
    
    
    def askpath(self):
        path = tk.filedialog.askdirectory(initialdir=self.cwd)
        return path
    
        
    def set_text(self, text, entry):
        entry.delete(0, tk.END)
        entry.insert(0, text)
        
        
    def show_dif_class_msg(self):
        tk.messagebox.showwarning(None, 'Select different class averages')
        
       
    def show_cluster_fail(self):
        tk.messagebox.showwarning(None, 'Clustering failed.\nTry adjusting # of edges or switching methods')
      
    
    def slicem_cluster(self, 
                       metric, 
                       community_detection, 
                       projection_2D, 
                       complete_scores, 
                       network_from, 
                       neighbors, 
                       top):
        
        global flat, clusters, G, colors
        
        num_class_avg = len(projection_2D)
        
        flat, clusters, G = self.create_network(metric=metric,
                                               community_detection=community_detection,
                                               projection_2D=projection_2D, 
                                               num_class_avg=num_class_avg, 
                                               complete_scores=complete_scores, 
                                               network_from=network_from, 
                                               neighbors=neighbors, 
                                               top=top)
        
        colors = get_plot_colors(clusters, G)
        
        print('clusters computed!')   
        
        
    def create_network(self,
                       metric,
                       community_detection,
                       projection_2D, 
                       num_class_avg, 
                       complete_scores, 
                       network_from, 
                       neighbors, 
                       top):
        """
        get new clusters depending on input options
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
            try:
                wt = Graph.community_walktrap(g, weights='weight', steps=4)
                cluster_dendrogram = wt.as_clustering()
            except:
                self.show_cluster_fail()
        elif community_detection == 'betweenness':
            try:
                ebs = Graph.community_edge_betweenness(g, weights='weight', directed=True)
                cluster_dendrogram = ebs.as_clustering()
            except:
                self.show_cluster_fail()

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

        return flat, clusters, G

        
    def plot_slicem_network(self, network_from, frame):
        
        if network_from == 'knn':
            positions = nx.spring_layout(G, weight='weight', k=0.3, scale=3.5)
        else:
            positions = nx.spring_layout(G, weight='weight', k=0.18, scale=1.5)
            
        f = Figure(figsize=(8,5))
        a = f.add_subplot(111)
        
        a.axis('off')

        nx.draw_networkx_nodes(G, positions, ax=a, edgecolors='black', linewidths=2, 
                               node_size=300, alpha=0.65, node_color=colors)
        nx.draw_networkx_edges(G, positions, ax=a, width=1, edge_color='grey')
        nx.draw_networkx_labels(G, positions, ax=a, font_weight='bold', font_size=10)
        
        if self.netcanvas:
            self.netcanvas.get_tk_widget().destroy()
            self.nettoolbar.destroy()

        self.netcanvas = FigureCanvasTkAgg(f, frame)
        self.netcanvas.draw()
        self.netcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.netcanvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.nettoolbar = NavigationToolbar2Tk(self.netcanvas, frame)
        self.nettoolbar.update()


    def plot_tiles(self, frame):
        """
        plot 2D class avgs sorted and colored by cluster
        """
        ordered_projections = []
        colors_2D = []
        flat_clusters = []

        for cluster, nodes in clusters.items():
            for n in nodes:
                ordered_projections.append(projection_2D[n])

            for n in nodes:
                flat_clusters.append(n)

            for i, n in enumerate(G.nodes):
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

        f = Figure(figsize=(4,4), dpi=200)

        grid = ImageGrid(f, 111, #similar to subplot(111)
                         nrows_ncols=(gr, gc), #creates grid of axes
                         axes_pad=0.05) #pad between axes in inch

        props = dict(boxstyle='round', facecolor='white')

        for i, (ax, im) in enumerate(zip(grid, ordered_projections)):
            #iterate over the grid returns the axes
            ax.imshow(im, cmap='gray')

            ax.spines['bottom'].set_color(colors_2D[i])
            ax.spines['bottom'].set_linewidth(2)

            ax.spines['top'].set_color(colors_2D[i])
            ax.spines['top'].set_linewidth(2)

            ax.spines['left'].set_color(colors_2D[i])
            ax.spines['left'].set_linewidth(2)

            ax.spines['right'].set_color(colors_2D[i])
            ax.spines['right'].set_linewidth(2)

            ax.set_xticklabels([])
            ax.xaxis.set_tick_params(which='both', length=0)
            ax.set_yticklabels([])
            ax.yaxis.set_tick_params(which='both', length=0)

            text = str(flat_clusters[i])
            ax.text(6, 6, text, va='top', ha='left', bbox=props, size=3)
            
        newWindow = tk.Toplevel(frame)

        canvas = FigureCanvasTkAgg(f, newWindow)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas.figure.tight_layout()
        
        toolbar = NavigationToolbar2Tk(canvas, newWindow)
        toolbar.update()      
        

    def plot_projections(self, p1, p2, frame):
        
        if p1 == p2:
            self.show_dif_class_msg()
        else:
            projection1 = extract_2D[p1]
            projection2 = extract_2D[p2]

            angle1 = complete_scores[p1, p2][0]
            angle2 = complete_scores[p1, p2][1]

            ref = ski.transform.rotate(projection1, angle1, resize=True)
            comp = ski.transform.rotate(projection2, angle2, resize=True)

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

            f = Figure(figsize=(4,4))
            spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f)
            tl = f.add_subplot(spec[0, 0])
            tr = f.add_subplot(spec[0, 1])
            bl = f.add_subplot(spec[1, 0])
            br = f.add_subplot(spec[1, 1])

            #  PROJECTION_1
            #2D projection image
            tl.imshow(ref, cmap=plt.get_cmap('gray'), aspect='equal') 
            tl.axis('off')

            #1D line projection
            bl.plot(ref_list, ref_intensity, color='black')
            bl.xaxis.set_visible(False)
            #bl.set_ylabel('Intensity')
            bl.yaxis.set_visible(False)
            bl.set_ylim([axis_min, axis_max+0.5])
            bl.fill_between(ref_list, 0, ref_intensity, alpha=0.5, color='deepskyblue')

            #  PROJECTION_2
            #2D projection image
            tr.imshow(comp, cmap=plt.get_cmap('gray'), aspect='equal')
            tr.axis('off')

            #lD line projection
            br.plot(comp_list, comp_intensity, color='black')
            br.xaxis.set_visible(False)
            br.yaxis.set_visible(False)
            br.set_ylim([axis_min, axis_max+0.5])
            br.fill_between(comp_list, 0, comp_intensity, alpha=0.5, color='red')

            asp = np.diff(bl.get_xlim())[0] / np.diff(bl.get_ylim())[0]
            bl.set_aspect(asp)
            asp1 = np.diff(br.get_xlim())[0] / np.diff(br.get_ylim())[0]
            br.set_aspect(asp)

            f.tight_layout()

            if self.projcanvas:
                self.projcanvas.get_tk_widget().destroy()
                self.projtoolbar.destroy()

            self.projcanvas = FigureCanvasTkAgg(f, frame)
            self.projcanvas.draw()
            self.projcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.projcanvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.projtoolbar = NavigationToolbar2Tk(self.projcanvas, frame)
            self.projtoolbar.update()
        
        
    def overlay_lines(self, p1, p2, frame):
        """
        overlays line projections at optimum angle between two class averages
        """
        if p1 == p2:
            self.show_dif_class_msg()
        
        else:
            a1 = complete_scores[p1, p2][0]
            a2 = complete_scores[p1, p2][1]
            
            projection1 = make_1D(extract_2D[p1], a1)
            projection2 = make_1D(extract_2D[p2], a2)

            if metric == 'norm-xcorr' or metric == 'norm-Euclidean':
                projection1.vector = stats.zscore(projection1.vector)
                projection2.vector = stats.zscore(projection2.vector)
            
            #first get location where line projections have optimum overlap
            if metric == 'Euclidean' or metric == 'norm-Euclidean':
                score, info = slide_score(projection1, projection2, pairwise_l2)
            elif metric == 'L1':
                score, info = slide_score(projection1, projection2, pairwise_l1)
            elif metric == 'cosine':    
                score, info = slide_score(projection1, projection2, pairwise_cosie)
            elif metric == 'cross-correlation' or metric == 'norm-xcorr':
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

            f = Figure(figsize=(4,4))
            ax = f.add_subplot(111)

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
            
            f.tight_layout()
            
            if self.projcanvas:
                self.projcanvas.get_tk_widget().destroy()
                self.projtoolbar.destroy()

            self.projcanvas = FigureCanvasTkAgg(f, frame)
            self.projcanvas.draw()
            self.projcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.projcanvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.projtoolbar = NavigationToolbar2Tk(self.projcanvas, frame)
            self.projtoolbar.update()
            
            
    def write_star_files(self, star_input, outpath):
        """
        split star file into new star files based on clusters
        """
        with open(star_input, 'r') as f:
            table = parse_star(f)

        cluster_star = {}

        for cluster, nodes in clusters.items():
            if nodes:
                #convert to str to match df
                #add 1 to match RELION indexing
                avgs = [str(node+1) for node in nodes]
                subset = table[table['ClassNumber'].isin(avgs)]
                cluster_star[cluster] = subset

        for cluster, table in cluster_star.items():
            with open(outpath+'/slicem_cluster_{0}.star'.format(cluster), 'w') as f:
                #write the star file
                print('data_', file=f)
                print('loop_', file=f)
                for i, name in enumerate(table.columns):
                    print('_rln' + name + ' #' + str(i+1), file=f)
                table.to_csv(f, sep='\t', index=False, header=False)

        with open(outpath+'/slicem_clusters.txt', 'w') as f:
            for cluster, averages in clusters.items():
                f.write(str(cluster) + '\t' + str(averages) + '\n')
                
        print('star files written!')      

        
    def write_edge_list(self, network, outpath):
        with open(outpath+'/slicem_edge_list.txt', 'w') as f:
            f.write('projection_1'+'\t'+'projection_2'+'\t'+'score'+'\n')
            for t in flat:
                f.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\n')
            if network == 'top_n':
                if clusters['singles']:
                    for single in clusters['singles']:
                        f.write(str(single)+'\n')
                        
        print('edge list written!')   
        
        
#utility functions from main script to make GUI standalone        
def extract_class_avg(avg):
    """
    keep positive values from normalized class average
    remove extra densities in class average
    fit in minimal bounding box
    """
    pos_img = avg.copy()
    pos_img[pos_img < 0] = 0
    
    #dilate by pixel size to connect neighbor regions
    #extra = 3 #Angrstroms to dilate (set minimum of 3A)
    #extend = int(np.ceil((pixel_size/extra)**-1))
    #struct = np.ones((extend, extend), dtype=bool)
    
    struct = np.ones((2, 2), dtype=bool)
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

def make_1D(projection, angle):
    proj_1D = ski.transform.rotate(projection, angle, resize=True).sum(axis=0)
    trim_1D = np.trim_zeros(proj_1D, trim='fb')
    p = Projection(class_avg=projection, angle=angle, vector=trim_1D)
    return p
    
def pairwise_l2(a, b):
    score = spatial.distance.euclidean(a.vector, b.vector)
    return score

def pairwise_l1(a, b):
    score = sum(abs(a.vector - b.vector))
    return score

def pairwise_cosine(a, b):
    score = spatial.distance.cosine(a.vector, b.vector)
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


def parse_star(f):
    """
    functions to parse star file adapted from Tristan Bepler
    https://github.com/tbepler/topaz
    https://www.nature.com/articles/s41592-019-0575-8
    """
    return parse(f)

def parse(f):
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        if line.startswith('data_'): 
            return parse_star_body(lines[i+1:])
        
def parse_star_body(lines):
    #data_images line has been read, next is loop
    for i in range(len(lines)):
        if lines[i].startswith('loop_'):
            lines = lines[i+1:]
            break
    header,lines = parse_star_loop(lines)
    #parse the body
    content = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('data'): # done with image block
            break
        if line.startswith('#') or line.startswith(';'): # comment lines
            continue
        if line != '':
            tokens = line.split()
            content.append(tokens)
            
    table = pd.DataFrame(content, columns=header)
    
    return table      
    
def parse_star_loop(lines):
    columns = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if not line.startswith('_'):
            break
        name = line[1:]
        #strip trailing comments from name
        loc = name.find('#')
        if loc >= 0:
            name = name[:loc]
        #strip 'rln' prefix
        if name.startswith('rln'):
            name = name[3:]
        name = name.strip()
        
        columns.append(name)
        
    return columns, lines[i:]


app = SLICEM_GUI()
app.geometry('900x700')
app.mainloop()