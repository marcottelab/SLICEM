# SLICEM

Compare and cluster 2D class averages of multiple distinct macromolecular assemblies from cryo-EM projections based on common lines

input is normalized 2D class averages as .mrcs and the accompanying particle.star (using RELION conventions)

output is new particle.star files for each cluster of particles to be used for 3D reconstruction

Example mrcs and star file provided, corresponding cryo-EM data at EMPIAR-10268




usage: SLICEM.py [-h] -i MRC_INPUT -s STAR_INPUT -o OUTPATH -d DESCRIPTION -m
                 METRIC [-n] -p PIXEL_SIZE [-k NEIGHBORS] -g
                 COMMUNITY_DETECTION [-c NUM_WORKERS]

compare and cluster 2D class averages based on common lines

optional arguments:
  -h, --help            show this help message and exit
  -i MRC_INPUT, --input MRC_INPUT
                        path to mrcs file of 2D class averages
  -s STAR_INPUT, --star_input STAR_INPUT
                        path to star file for particles in 2D class averages
                        (uses RELION conventions)
  -o OUTPATH, --outpath OUTPATH
                        path for output files
  -d DESCRIPTION, --description DESCRIPTION
                        name for output file
  -m METRIC, --metric METRIC
                        choose scoring method (Euclidean, L1, cross-
                        correlation, cosine)
  -n, --normalize       zscore normalize 1D projections before scoring (not
                        recommended)
  -p PIXEL_SIZE, --pixel_size PIXEL_SIZE
                        pixel size of 2D class averages in A/pixel
  -k NEIGHBORS, --neighbors NEIGHBORS
                        number of edges for each node in the graph
  -g COMMUNITY_DETECTION, --community_detection COMMUNITY_DETECTION
                        community detection algorithm to use (betweenness,
                        walktrap)
  -c NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPUs to use




python3.7 SLICEM.py -i /path/to/mixture_2D.mrcs -s /path/to/particles.star -o /path/to/output/ -d mixture -m L1 -p 4 -k 5 -g betweenness -c 30




Link to JSB paper:
Separating distinct structures of multiple macromolecular assemblies from cryo-EM projections
https://www.sciencedirect.com/science/article/pii/S1047847719302370

Link to bioRxiv: https://www.biorxiv.org/content/10.1101/611566v1
