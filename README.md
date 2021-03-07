# SLICEM
Compare and cluster 2D class averages of multiple distinct structures from cryo-EM data based on common lines.  
Example mrcs and star file provided in /data, corresponding cryo-EM data at EMPIAR-10268.  
See **manual.pdf** for a brief tutorial.

# New in 2021_Mar_7 update
- Default scoring now uses FT of projections, use `-d Real` for real space projections
- GUI now includes option to remove nodes from the graph
***upcoming updates***
- Improved scoring and clustering
- GPU support

# New in 2020_Nov_6 update
- Fixed memory error with multiprocessing  
- Added option to downscale class averages `-s` (--> faster processing)  
- Added Wasserstein (Earth mover) distance
- Removed Jupyter Notebook (out of date)  

# Installation
Download the software and create a conda environment -
```
git clone https://github.com/marcottelab/SLICEM.git
conda env create -f environment.yml
source activate SLICEM 
source deactivate #to return to base env
```

# Usage
First generate a score file using slicem.py
```
usage: SLICEM_DEV.py [-h] -i MRC_INPUT -o OUTPATH
                     [-m {Euclidean,L1,cosine,EMD,correlate}]
                     [-s SCALE_FACTOR] [-c NUM_WORKERS] [-d {Fourier,Real}]
                     [-t {full,valid}] [-a ANGULAR_SAMPLING]

compare similarity of 2D class averages based on common lines

optional arguments:
  -h, --help            show this help message and exit
  -i MRC_INPUT, --input MRC_INPUT
                        path to mrcs file of 2D class averages
  -o OUTPATH, --outpath OUTPATH
                        path for output files
  -m {Euclidean,L1,cosine,EMD,correlate}, --metric {Euclidean,L1,cosine,EMD,correlate}
                        choose scoring method, default Euclidean
  -s SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        scale factor for downsampling. (e.g. -s 2 converts 200pix box --> 100pix box)
  -c NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPUs to use, default 1
  -d {Fourier,Real}, --domain {Fourier,Real}
                        Fourier or Real space, default Fourier
  -t {full,valid}, --translate {full,valid}
                        indicate size of score vector, numpy convention, default full
  -a ANGULAR_SAMPLING, --angular_sampling ANGULAR_SAMPLING
                        angle sampling for 1D projections in degrees, default 5
```
**command line example**
```
(SLICEM): python slicem.py -i path/to/input.mrc -o path/to/output/ -c 8
```
alternative example if running remotely
```
(SLICEM): nohup python slicem.py -i path/to/input.mrc -o path/to/output/ -m L1 -s 2 -c 10 > log.txt &
```

<br/>
Next load the scores and class averages into the GUI for clustering. See manual.pdf for more details

```
python slicem_gui.py
```

---

<br/>

## Introduction

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Intro.gif?raw=true" />

<br clear="left"/><br/>


## Line projections

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Proteasome.gif?raw=true"  />

<br clear="left"/><br/>


## Comparison of line projections

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/ProtRibo.gif?raw=true" />

<br clear="left"/><br/>


## Clustering

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Graph.gif?raw=true"  />

<br clear="left"/><br/>



## Reference 
**Separating distinct structures of multiple macromolecular assemblies from cryo-EM projections**
<br/>
https://doi.org/10.1016/j.jsb.2019.107416
<br/>
### Bugs and suggestions
report as a GitHub issue or email eric.verbeke@utexas.edu
