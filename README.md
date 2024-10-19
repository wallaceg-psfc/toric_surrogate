# toric_surrogate
Surrogate model for TORIC

This repo contains information relating to the surrogate model for TORIC.  The initial MLP + PCA surrogate model for TORIC on SPARC is contained within TORIC_MLP_v1.  This assumes a 50:50 DT ratio for discharges anticipated in the early Q > 1 L-mode campaign.  Input features:

freq = ICRF wave frequency in MHz, training range = 119 to 121

T_avol = volume average temperature in keV, training range = 1.5 to 8.5

n_avol = volume average density in 10^20 m^-3, training range = 1.1 to 5.1

min_conc = He3 minority concentration relative to the electron density in %, training range = 1 to 5

gap_inner = distance from last closed flux surface (LCFS) to the inner wall in m, training range = 0 to 0.03

gap_outer = distance from LCFS to the outer midplane limiter in m, training range = 0 to 0.05

gap_inner and gap_outer can be converted to R0 and a with the following equations:
R_inner = 1.24+gap_inner;
R_outer = 2.43-gap_outer;
a = (R_outer-R_inner)/2;
R0 = R_outer-a;
Note that this assumes that the inner and outer gaps do not vary as a function of z0, which is a good assumption for the inner gap (nearly vertical inner wall) but perhaps less good of an assumption for the outer gap where there is significant curvature to the outer limiter.

z0 = vertical position of magnetic axis in m, training range = -0.05 to 0.05

nu_T = temperature profile peaking factor, training range = 2 to 3

nu_n = density profile peaking factor, training range = 1.15 to 1.65

Realistic functional forms for T and n for TORIC-NN	(v.0.3) by P. Rodriguez-Fernandez 2020

    Description:
        This functional form imposes:
            1) tanh pedestal for T and n.
            2) Linear aLT profile from 0 at rho=0 to X at rho=x_a,
                where X is the specified core aLT value (default 2.0)
                and x_a is calculated by matching specified nu_T (peaking)
            3) Flat aLT profile from rho=x_a to rho=1-width_ped, where
                width_ped is the pedestal width (default 0.05).
            4) T and n share the same x_a, and aLn is calculated by matching
                specified nu_n (peaking). Density profile also has a linear
                aLn from 0 to rho=x_a and a flat one until rho=1-width_ped
            5) Pedestal is rescaled to match specified volume averages for
                T and n.

    Notes:
        - Not all combinations of aLT and nu_T are valid. If aLT is too low,
            nu_T cannot be excessively high and viceversa. The code will not
            crash, but will give profiles that do not match the specified
            temperature	peaking.
            e.g. aLT = 2.0 requires nu_T to be within [1.5,3.0]
        - The following values are hardcoded:  aLT=2.0,width_ped=0.05


            

B0 = toroidal magnetic field on axis in T, training range = 11.8 to 12.5

Other equilibrium features are fixed at delta = 0.54, kappa = 1.97, double null configuration, and nu = 0.1.  Analytic equilibrium documentation in Guazzotto, L., and J. P. Freidberg. "Simple, general, realistic, robust, analytic tokamak equilibria. Part 1. Limiter and divertor tokamaks." Journal of Plasma Physics 87.3 (2021): 905870303.

Toroidal mode number = 44 corresponding to launched k|| = 18.1 m-1.  Upper and lower antennas located at z_upper = 0.20805, R_upper = 2.4205, z_lower = -0.2137, R_lower = 2.4197 m.  Output profiles are on a uniform normalized sqrt toroidal flux grid from 0 to 1.

Output power deposition profiles are for He3 minority species, 2nd harmonic T, and electron heating.  For integrated modeling purposes the most straightforward way to implement is to use a Stix distribution to estimate the "temperature" of the He3 tail.  When the tail temperature is below the critical temperature, He3 power goes to the ions, and when the tail temperature is above the critical temperature it goes to electrons.  2nd harmonic T and electron power deposition profiles can be treated as going directly to bulk T and electrons, respectively.

More details to be added to this README file as additional models get uploaded...  -20240930 GMW
