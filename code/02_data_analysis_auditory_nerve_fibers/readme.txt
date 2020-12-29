The matlab files of this folder compute the results shown in Figure 7B and Supplementary Figures S5, S7

(I) fit_single_tuning_curve.m

Fit the neuronal response of auditory neural fiber (ANF) of every unit in every trial. 
The result can be used to detect and remove outliers in the data.

With that the following figures can be generated:
	- The histogram of normalized spontaneous activity r/r_m in every trial (Fig S5A)

(II) fit_tuning_curve.m

Fit the neuronal response of auditory neural fiber (ANF) of every unit, combining all valid trials.

With that the following figures can be generated:
	- Example of tuning curve fitting (Fig 7B)
	- The histogram of normalized spontaneous activity r/r_m, and the dynamic range sigma  (Fig S5B)

(III) get_stat_2types.m

Calculate statistics of normalized spontaneous activity r/r_m, dynamics range sigma, and threshold theta of ANFs.
Classify all the data points into 2 types.

With that the following figures can be generated:
	- The classified data (Fig S5C)
	- Regression between r/r_m and sigma (Fig S5D)
	- Regression between sigma and theta (Fig S5E)

(IV) get_stat_2types.m

Calculate statistics of normalized spontaneous activity r/r_m, dynamics range sigma, and threshold theta of ANFs.
Classify all the data points into 3 types.

With that the following figures can be generated:
	- The classified data (Fig S7A)
	- Regression between r/r_m and sigma (Fig S7B)
	- Regression between sigma and theta (Fig S7C)

