# SLICEM

Algorithm that takes 2D projection images from cryo-EM data of multiple macromolecular assemblies and separates them into homogenous groups

code is currently in notebook form SLICEM.ipynb

input is zero-mean normalized 2D class averages as .mrcs and the accompanying particle.star 

output is new particle.star file for each cluster of particles to be used for 3D reconstruction

Example mrcs and star file provided, corresponding cryo-EM data at EMPIAR-10268
