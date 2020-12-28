The python files of this folder compute the results shown in Figures 3-6 and Supplementary Figures S1, S2, S3, S9

(I) independent.py

Optimize thresholds which maximize mutual information for the independent-coding channel.

With that the following figures can be generated:
    - Max. mutual information for each input and output noise combination (Fig 3A)
    - Optimal thresholds and optimal threshold diversity (Fig. 4A-E)
    - How the bifurcations of optimal thresholds are related to derivates of maximal information (Fig. 5 A+B; Sup. Fig. S3 A+C)
    - How the bifurcations of optimal thresholds are related to the eigenvalues of the Hesse matrix of the information landscape (Fig. 6 A+D)




(II) lumped.py

As above but for the lumped-coding channel.

With that the following figures can be generated:
    - Max. mutual information for each input and output noise combination (Fig 3B)
    - Optimal thresholds and optimal threshold diversity (Fig. 4F-J)
    - How the bifurcations of optimal thresholds are related to derivates of maximal information (Fig. 5 C+D; Sup. Fig. S3 B+D)
    - How the bifurcations of optimal thresholds are related to the eigenvalues of the Hesse matrix of the information landscape (Fig. 6 E)


(III) The Figures 3 C-F are obtained by substraction or division of (I) and (II), respectively.

(IV) Sup. Figures S1B and S2 are obtained by using (I) with increasing the neuron number, see independent_N=4.py Be aware of local maxima, especially for N=6.

(V) Supp. Fig. S6 is obtained by using (I) with a modified stimulus distribution (A+B) or a modified input noise distribution (C+D). Just uncomment the respective distributions in independent.py . 


