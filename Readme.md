# Multimodal emotion recognition method
The goal of this project is to recognise emotions in most situations. For this, four kinds of data are taken: face, posture, body and context/environment features. Each of these are processed independently and then combined with a merging method called EmbraceNet+, wich is an extention of the [EmbraceNet](https://github.com/idearibosome/embracenet).

## Dataset
![alt text](https://github.com/juan1t0/multimodalDLforER/blob/master/figures/pre-pross.png)
The used data are shared [here](https://drive.google.com/file/d/1JAGejLFaymrIsq44icV42IdaAdydSdk9/view?usp=sharing), this zip contains all the data for each modality in numpy array format.

This dataset is acquire form the original [EMOTIC dataset](http://sunai.uoc.edu/emotic/download.html).

## Execution
Currently, only the execution with a [notebook](https://github.com/juan1t0/multimodalDLforER/blob/master/EmbraeNet_Plus.ipynb) is available.

## Acknowledgments
This research was supported by the FONDO NACIONAL DEDESARROLLO CIENTÍFICO, TECNOLÓGICO Y DE INNOVACIÓN TECNOLÓGICA - FONDECYT as executing entity of CONCYTEC under grant agreement no.01-2019-FONDECYT-BM-INC.INV in the project RUTAS: Robots for Urban Tourism,Autonomous and Semantic web based.
