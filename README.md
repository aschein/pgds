# Poisson—Gamma Dynamical Systems
Source code for the paper: [Poisson—Gamma Dynamical Systems] (http://people.cs.umass.edu/~aschein/ScheinZhouWallach2016_paper.pdf) by Aaron Schein, Mingyuan Zhou, and Hanna Wallach, presented at NIPS 2016.

The MIT License (MIT)

Copyright (c) 2016 Aaron Schein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What's included:
* [pgds.pyx](https://github.com/aschein/pgds/blob/master/src/pgds.pyx): The main code file.  Implements Gibbs sampling inference for PGDS.
* [mcmc_model.pyx](https://github.com/aschein/pgds/blob/master/src/mcmc_model.pyx): Implements Cython interface for MCMC models.  Inherited by pgds.pyx.
* [sample.pyx](https://github.com/aschein/pgds/blob/master/src/sample.pyx): Implements fast Cython method for sampling various distributions.
* [lambertw.pyx](https://github.com/aschein/pgds/blob/master/src/lambertw.pyx): Code for computing the Lambert-W function.
* [Makefile](https://github.com/aschein/pgds/blob/master/src/Makefile): Makefile (cd into this directoy and type 'make' to compile).
* [icews_example.ipynb](https://github.com/aschein/pgds/blob/master/src/icews_example.ipynb): Jupyter notebook with an examples of how to use the code to run PGDS on ICEWS data for exploratory and predictive analyses.

## Dependencies:
* numpy
* scipy
* matplotlib
* seaborn
* pandas
* argparse
* path
* scikit-learn
* cython
* GSL
