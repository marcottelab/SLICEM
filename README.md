# SLICEM

Compare and cluster 2D class averages of multiple distinct macromolecular assemblies from cryo-EM projections based on common lines

input is normalized 2D class averages as .mrcs and the accompanying particle.star (using RELION conventions)

output is new particle.star files for each cluster of particles to be used for 3D reconstruction

Example mrcs and star file provided, corresponding cryo-EM data at EMPIAR-10268



python3.7 SLICEM.py -i /path/to/mixture_2D.mrcs -s /path/to/particles.star -o /path/to/output/ -d mixture -m L1 -p 4 -k 5 -g betweenness -c 30




Link to JSB paper:
Separating distinct structures of multiple macromolecular assemblies from cryo-EM projections
https://www.sciencedirect.com/science/article/pii/S1047847719302370

Link to bioRxiv: https://www.biorxiv.org/content/10.1101/611566v1
