Surrogate model for TORIC, toric_surrogate_PRD_v2.0-beta

This repo contains information relating to the surrogate model for TORIC.

Input features:

freq = ICRF wave frequency in MHz, training range = 120±1

T_avol = volume average temperature in keV, training range = 8.2±3.5

n_avol = volume average density in 10^20 m^-3, training range = 3.1±2.0

min_conc = He3 minority concentration relative to the electron density in %, training range = 3±2

gap_inner = distance from last closed flux surface (LCFS) to the inner wall in m, training range = 0.015±0.015

gap_outer = distance from LCFS to the outer midplane limiter in m, training range = 0.025±0.025

gap_inner and gap_outer can be converted to R0 and a with the following equations: R_inner = 1.24+gap_inner; R_outer = 2.43-gap_outer; a = (R_outer-R_inner)/2; R0 = R_outer-a; Note that this assumes that the inner and outer gaps do not vary as a function of z0, which is a good assumption for the inner gap (nearly vertical inner wall) but perhaps less good of an assumption for the outer gap where there is significant curvature to the outer limiter.

z0 = vertical position of magnetic axis in m, training range = 0±0.05

nu_T = temperature profile peaking factor, training range = 2.7±0.5

nu_n = density profile peaking factor, training range = 1.3±0.25

DT_ratio = ratio of D/(D+T), training range = 0.5±0.5

Realistic functional forms for T and n for TORIC-NN (v.0.3) by P. Rodriguez-Fernandez 2020.

This release extends the TORIC surrogate model for SPARC to the Primary Reference Discharge (PRD) H-mode scenario. The model is very similar to the QGT1LMode model but with density and temperature profile parameters consistent with the PRD. The other significant change is the addition of the D/(D+T) ratio as an input parameter.

Description of the files in this release:
TORIC_MLP_PCA_PW*.lzma and TORIC_MLP_PCA_Pw*.pkl contain the principal component analysis (PCA) for the species indicated by the '*'. Here the species are 2T (2nd harmonic tritium), He3 (Helium-3), and E (electrons). The PCA inverse transform is used to convert the principal components predicted by the neural network into radial profiles.

TORIC_MLP_PCA_scaler.lzma and TORIC_MLP_PCA_scaler.pkl contain the scaling factors for normalizing the input variables.

TORIC_MLP_Pw*.lzma and TORIC_MLP_Pw*.pkl contain the multi-layer perceptron (MLP) neural network for predicting the principal components.

TORIC_PRD_DATA_SPLIT.csv contains the database of simulations used for training the surrogate model.

train_MLP_PCA_TORIC_He3_T_e_hyperparam.ipynb is a Jupyter notebook used for training the surrogate models.

Hyperparameters used for each of the three models:

Species = He3
Maxiter = 50
Alfa = 0.01
Learningrate_init = 0.01
Num_components = 10
Activ = relu

Species = 2T
Maxiter = 400
Alfa = 0.1
Learningrate_init = 0.01
Num_components = 5
Activ = relu

Species = E
Maxiter = 200
Alfa = 0.1
Learningrate_init = 0.01
Num_components = 10
Activ = relu

Usage of the surrogate model is the same as for the previous version.