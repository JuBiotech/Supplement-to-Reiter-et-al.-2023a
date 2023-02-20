[![DOI](https://zenodo.org/badge/604218880.svg)](https://zenodo.org/badge/latestdoi/604218880)

# Supplement code

__Microbial Production of L-Histidine with Corynebacterium glutamicum in minimal media__

Engineering in Life Sciences

Alexander Reiter<sup>1,2</sup> (ORCID: 0000-0003-3499-2901),<br>
Lars Wesseling<sup>1</sup><br>
Wolfgang Wiechert<sup>1,3</sup> (ORCID: 0000-0001-8501-0694),<br>
Marco Oldiges<sup>1,2</sup> (ORCID: 0000-0003-0704-5597)<br> 

<sup>1</sup> Forschungszentrum Jülich GmbH, Institute of Bio- and Geosciences, IBG-1: Biotechnology, Jülich 52425, Germany<br>
<sup>2</sup> RWTH Aachen University, Institute of Biotechnology, Aachen 52062, Germany<br>
<sup>3</sup> RWTH Aachen University, Computational Systems Biotechnology, Aachen 52062, Germany<br>

Corresponding author: Prof. Dr. Marco Oldiges, mail: m.oldiges@fz-juelich.de, phone: +49 2461 61-3951. fax: +49 2461 61-3870

# Description
Supplement code for database creation (database), method creation (development) and evaluation (scan).<br>
Example files are provided in the database folder.

## General procedure:
* Create the necessary database files with database code
* Create the analysis method (project) for the organism with development code
* Transfer the methods created to your instrument
* Create a plate layout for your samples and run batch creation
* Integrate your data with the corresponding vendor software
* Use development and experiment files for data evaluation scan code

### Database

Automated creation of a database based on KEGG, Pubchem and ChEMBL:
* Organisms
* Pathways
* Metabolites
* CFM-ID MS/MS Predictions

### Development

Automated method development:
* Organism specific analysis methods 
* Analysis batch creation

### Scan

Automated data evaluation:
* Preprocessing
* Univariate analysis
* Multivariate analysis
* Cluster analysis
* Pathway analysis

## Requirements

### Software:
- Anaconda, Python = 3.9
- Git

### Channels:
- defaults
- conda-forge

### Dependencies:
- adjusttext=0.7.3.1
- notebook=6.4.5
- numpy=1.21.3
- scipy=1.7.1
- pandas=1.3.4
- xlrd=2.0.1
- openpyxl=3.0.9
- joblib=1.1.0
- matplotlib=3.4.3
- seaborn=0.11.2
- ipython=7.28.0
- python=3.9.7
- pathlib=1.0.1
- statsmodels=0.13.0
- biopython=1.79
- pubchempy=1.0.4
- requests=2.26.0
- selenium=3.141.0
- scikit-learn=1.0.1
- networkx=2.6.3

### Anaconda

1. Install Python Environment manager with Python 3.9 (Anaconda with Anaconda Navigator).
2. Create new environment.
3. Open environment with terminal.
4. Add python packages based on dependencies above with conda install.

## Using DSFIApy from the repository

1. In a shell, navigate to the folder where you want the repository to be located. 
2. Open a terminal / shell and clone the repository via `git clone {repo adress}`.
3. Select the new environment in Anaconda Navigator.
4. Open terminal.
5. With `cd {absolute path}`, navigate to the location where the DSFIApy example folder is now found.
6. Open jupyter lab window with `jupyter notebook`.
7. The methods in this package are made to be used with the jupyter notebooks provided in the example folder.

## Additional licences

### pyChemometrics
BSD 3-Clause License

Copyright (c) 2017, Gonçalo dos Santos Correia
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

