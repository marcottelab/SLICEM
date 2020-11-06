# SLICEM
Compare and cluster 2D class averages of multiple distinct structures from cryo-EM data based on common lines.  
Example mrcs and star file provided in /data, corresponding cryo-EM data at EMPIAR-10268.  
See **slicem_manual.pdf** for a brief tutorial.

# New in 2020_Nov_6 update
- Fixed memory error with multiprocessing  
- Added option to downscale class averages `-s` (--> faster processing)  
- Removed support for normalization and cross-correlation  
- Removed Jupyter Notebook (out of date)  
***upcoming updates***
- GPU support
- Faster, more accurate comparison of line projections
- Improved GUI

# Installation
Download the software and create a conda environment -
```
git clone https://github.com/marcottelab/SLICEM.git
conda env create -f environment.yml
source activate SLICEM 
source deactivate #to return to base env
```

# Usage
First generate a score file using SLICEM.py
```
usage: SLICEM.py [-h] -i MRC_INPUT -o OUTPATH [-m {Euclidean,L1,cosine}] -p
                    PIXEL_SIZE [-s SCALE_FACTOR] [-c NUM_WORKERS]

compare similarity of 2D class averages based on common lines

optional arguments:
  -h, --help            show this help message and exit
  -i MRC_INPUT, --input MRC_INPUT
                        path to mrcs file of 2D class averages
  -o OUTPATH, --outpath OUTPATH
                        path for output files
  -m {Euclidean,L1,cosine}, --metric {Euclidean,L1,cosine}
                        choose scoring method, Euclidean default
  -p PIXEL_SIZE, --pixel_size PIXEL_SIZE
                        pixel size of 2D class averages in A/pixel
  -s SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        scale factor for downsampling. (e.g. -s 2 converts 100pix box --> 50pix box)
  -c NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPUs to use
```
**command line example**
```
(SLICEM): python SLICEM.py -i path/to/input.mrc -o path/to/output/ -p 1 -s 4 -c 8
```
alternative example if running remotely
```
(SLICEM): nohup python SLICEM.py -i path/to/input.mrc -o path/to/output/ -p 1 -s 4 -c 8 > log.txt &
```

<br/>
Next load the scores and class averages into the GUI for clustering. See slicem_manual.pdf for more details

```
python SLICEM_GUI.py
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
