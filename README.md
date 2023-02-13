When in the repo directory, in order to produce the test results, one can use target "test". In order to produce results with the downloaded data, target name "all" can be used. To make use of these targets, simply cd into the repo's director and input "python3 run <target>"

The project directory contains the following:

data folder contains test data, the notebook for generating the test data, as well as the complete allegations data

Dockerfile from which necessary packages will be installed to run project

report folder contains the overleaf report pdf

run.py is the code in which results are produced (main coding file)

src folder contains python code for data cleaning and generation of missing data (methods called in run.py). It also contains the models for ensuring that the model is fair and producing predictions so that fairness notion measurements can be produced. 