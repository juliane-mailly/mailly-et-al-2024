# Mailly et al., 2024
Model source code, model data analysis script, literature review analysis script.

## [/model](/model) folder
This folder contains all the necessary scripts to generate and analyse the data from the article.
 - The [parameters.xlsx](/model/parameters.xlsx) file describes the values that each parameter takes in each simulation set.
 - The [/src](/model/src) folder contains the source code for the model

### [/src](/src) folder
This folder contains everything necessary to reproduce the data from the article.
Use the [parameters.py](/model/src/parameters.py) file to specify the parameters of the simulations.
To run the simulations, execute the [main.py](/model/src/main.py) file.
The environments used in the simulations are stored in the [/Arrays](/model/src/Arrays) folder. This folder also contains two files that are used for the data analyses, that give a correspondance grid between the number of flowers and density of a patch and the average nearest-enighbour distance between the flowers of the patch. This equivalence sheet was generated using [convert_density_NN.ipynb](/model/src/convert_density_NN.ipynb).
The simulations outputs will be stored in a /Output folder. After generating the data, use the [data_analysis.ipynb](/model/src/data_analysis.ipynb) file to generate the corresponding plots.

## [/literature_review](/literature_review) folder
This folder contains the [data](/literature_review/litrev_data.csv) and the [analysis script](/literature_review/litrev_analysis.ipynb) to reproduce the plots from the article.
