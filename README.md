# toric_surrogate
Surrogate model for TORIC

This repo contains information relating to the surrogate model for TORIC.  The initial MLP + PCA surrogate model for TORIC on SPARC is contained within TORIC_MLP_v1.  This assumes a 50:50 DT ratio for discharges anticipated in the early Q > 1 L-mode campaign.  Input features:
freq = ICRF wave frequency in MHz, training range = 119 to 121
T_avol = volume average temperature in keV, training range = 1.5 to 8.5
n_avol = volume average density in 10^20 m^-3, training range = 1.1 to 5.1
min_conc = He3 minority concentration relative to the electron density in %, training range = 1 to 5
gap_inner = distance from last closed flux surface (LCFS) to the inner wall in m, training range = 0 to 0.03
gap_outer = distance from LCFS to the outer midplane limiter in m, training range = 0 to 0.05
z0 = vertical position of magnetic axis in m, training range = -0.05 to 0.05
nu_T = temperature profile peaking factor, training range = 2 to 3
nu_n = density profile peaking factor, training range = 1.15 to 1.65
B0 = toroidal magnetic field on axis in T, training range = 11.5 to 12.5

Output power deposition profiles are for He3 minority species, 2nd harmonic T, and electron heating.  For integrated modeling purposes the most straightforward way to implement is to use a Stix distribution to estimate the "temperature" of the He3 tail.  When the tail temperature is below the critical temperature, He3 power goes to the ions, and when the tail temperature is above the critical temperature it goes to electrons.  2nd harmonic T and electron power deposition profiles can be treated as going directly to bulk T and electrons, respectively.

More details to be added to this README file as additional models get uploaded...  -20240924 GMW
