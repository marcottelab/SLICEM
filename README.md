# SLICEM
---
Compare and cluster 2D class averages of multiple distinct structures from cryo-EM data based on common lines.  
Example mrcs and star file provided in /data, corresponding cryo-EM data at EMPIAR-10268.  
See **slicem_manual.pdf** for a brief tutorial.


# Installation
---
Download the software and create a conda environment -
```
git clone https://github.com/marcottelab/SLICEM.git
conda env create -f environment.yml
source activate slicem 
source deactivate #to return to base env
```

# Usage
---
First generate a score file using SLICEM.py
```
usage: SLICEM.py [-h] -i MRC_INPUT -o OUTPATH -d DESCRIPTION -m METRIC [-n] -p
                 PIXEL_SIZE [-c NUM_WORKERS]

compare and cluster 2D class averages based on common lines

optional arguments:
  -h, --help            show this help message and exit
  -i MRC_INPUT, --input MRC_INPUT
                        path to mrcs file of 2D class averages
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
  -c NUM_WORKERS, --num_workers NUM_WORKERS
                        number of CPUs to use
```
```
example: python SLICEM.py -i path/to/input.mrc -o path/to/output/ -d mixture -m Euclidean -p 1 -c 8
```

<br/>
Next load the scores and class averages into the GUI for clustering. See **slicem_manual.pdf** for more details
```
python SLICEM_GUI.py
```

---

<br/><br/>
## Introduction
---

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Intro.gif?raw=true" />

<br clear="left"/><br/>

---
## Line projections

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Proteasome.gif?raw=true"  />

<br clear="left"/><br/>

---
## Comparison of line projections

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/ProtRibo.gif?raw=true" />

<br clear="left"/><br/>

---
## Clustering

<img align="left" width="800" src="https://github.com/EricVerbeke/EricVerbeke.github.io/blob/master/media/Graph.gif?raw=true"  />

<br clear="left"/><br/>

---

## Reference 
**Separating distinct structures of multiple macromolecular assemblies from cryo-EM projections**
<br/>
https://doi.org/10.1016/j.jsb.2019.107416
<br/>
### Bugs and suggestions
report as a GitHub issue or email eric.verbeke@utexas.edu
