#!/usr/bin/python

'''
Optimize thresholds which maximize mutual information for the independent-coding channel.

This code goes through a grid of input and output noise values. For each combination of noise values:
    - find optimal thresholds which maximize information
    - get eigenvalues of Hessian of the information landscape
    - get derivatives of information with respect to both noise sources

With that the following figures can be generated:
    - Max. mutual information for each input and output noise combination (Fig 3A)
    - Optimal thresholds and optimal threshold diversity (Fig. 4A-E)
    - How the bifurcations of optimal thresholds are related to derivates of maximal information (Fig. 5 A+B)
    - How the bifurcations of optimal thresholds are related to the eigenvalues of the Hesse matrix of the information landscape (Fig. 6 A+D)

Modifications:
    - Changing the input noise distribution: modify  H[:,n] = .... in the function Info_3N_ind() (Supp. Fig. S6 C+D)
    - Similarly for chaning the stimulus distribution: modify  Ps = ...  (Supp. Fig. S6 A+B)

Tested and run with Python 2.7.
The Python version and all the required packages for running this code can be loaded with Anaconda by using the provided yml-file:
$ conda env create -f py2.yml

(For more info regarding installation these packages with Anaconda, see: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file )

    
'''

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
plt.ioff()  #figure windows are only opened with plt.show
#plt.ion()  #figure windows pop up
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size #for heatmap with fixed (quadratic size)
import scipy.misc
from scipy.optimize import fmin 
from scipy.special import erfc 
import numdifftools as nd   #for calculating Hesse matrix numerically
from joblib import Parallel, delayed #for parallelizing loops, https://pythonhosted.org/joblib/parallel.html
import os 
eps= np.finfo(float).eps #machine precision
import time
print time.strftime("%d %b %Y %H:%M:%S")
starttime = time.time()
 


'''setups'''

load_data_from_file = False #if calculations have been performed once, optimal thresholds, max Information and eigenvectors of Hessian can be loaded from file

cpus = 105 #number of cpus used for parallel calculation
print "number of cpus:", cpus



'''parameters'''

N = 3     #number of neurons. Attention!: Also number of loops and factors have to be adjusted in the function Info_3N_ind  (in specific: P_comb = Pk[:,0,i1]*Pk[:,1,i2]*Pk[:,2,i3]) )

Rsteps= 200 #number of data points for output noise
maxR= 2.0 #maximum R
minR = 0.001  #standard 0.001, usually not touched

sigsteps = 200 #numer of data points for input noise
maxsig = 0.7 #maximum sigma
minsig = 0.01 #small sigmas make ist slow (especially when smaller than 0.1) since more steps necessary in numerical integration 



R_vec =np.linspace(minR,maxR,Rsteps)
#R_vec = [8.] #in case one wants to look at a specific R: comment this line in
maxR = max(R_vec)
minR = min(R_vec)
Rsteps = len(R_vec)
#print "R_vec: ", R_vec

sig_vec = np.linspace(minsig,maxsig,sigsteps)     #sig determines input noise strength
#sig_vec = [0.7]  #in case one wants to look at a specific sigma: comment this line in
maxsig = max(sig_vec)
minsig = min(sig_vec)
sigsteps = len(sig_vec)
#print "sig_vec: ", sig_vec

 
#optimization precision (ftol smaller than 1e-8 is useless here)
ftol = 1e-8 
xtol = 1e-5 


#initial conditions for minimization algo. when local maxima can be exluded: for speed purposes do a regular spacing of thresholds between t0min and t0max with some offset
t0min= -0.6    
t0max = 0.6     
t0off = 0.2
t0= np.linspace(t0min, t0max, N) +t0off
t0min, t0max = t0[0], t0[-1]
print "t0_vec: ",t0


#min and max stimulus values between which the integration over stimulus space is performed.
smin = -7  
smax = -smin

#generate string which incorporates the parameters used
parastr= 'N='+str(N)+'_Rsteps='+str(Rsteps)+'_maxR='+str(maxR)+'_sigsteps='+str(
                    sigsteps)+'_sigmin='+str(minsig)+'_sigmax='+str(maxsig)+'_ftol='+str(ftol)+'_xtol='+str(
                            xtol)
namestr= 'independent__' 
print namestr
print namestr+parastr 




'''information calculation as explained in the method section of the paper. #Eq. 12 of the paper'''
def Info_3N_ind(t,sig,R,smin,smax,Ntrap):  
    #t: threshold vector
    #R: output noise
    #sig: input noise
    #smin, smax: borders between which the integration is performed over stimulus space
    #Ntrap: number of steps in trapezoid integration
    
    ss=np.linspace(smin,smax,Ntrap)
    q = np.exp(-R)
    r = 1. - q

    H = np.zeros((len(ss),N)); 
    Q = np.zeros((len(ss),N));
    S = np.zeros((len(ss),N));
    Pk = np.zeros((len(ss),N,2));
    

    for n in range(0,N):   #for each neuron n: calculate H, Q, R; store Q and R in S
        
        '''modify here the input noise distribution by putting its integral (see Eq. 9 in the paper)'''    
        
        #sigmoidal: integral of Gaussian
        H[:,n] = 0.5*erfc((t[n]-ss)/sig/np.sqrt(2)) #Eq. 11 of the paper
        
        #generalized normal (beta determines the kurtosis), Eqs. 25-28 of paper
        #from scipy.special import gamma 
        #alpha = sig* np.sqrt(gamma(1./beta)/gamma(3./beta)) # variance of generalized normal 
        #H[:,n] =  0.5 - np.sign(t[n]-xx)*gamma(1./beta,(np.abs(t[n]-xx)/alpha))*gammainc(1./beta,(np.abs(t[n]-xx)/alpha)**beta)/(2.*gamma(1./beta))   #integral of generalized normal distribution
        
        #logistic 
        #H[:,n] = 1./(1.+np.exp((t[n]-ss)/sig))      
        

        Q[:,n] = 1. - r*H[:,n]                      #Eq. 13 from paper
        S[:,n] = r*H[:,n]                           #Eq. 14 from paper
        Pk[:,n,0] = Q[:,n]                          #Eq. 16 from paper
        Pk[:,n,1] = S[:,n]                          #Eq. 16 from paper
    
    
    '''stimulus distribution'''
    
    #Gaussian stimulus distribution
    Ps = np.exp(-0.5*ss**2)*(2*np.pi)**-0.5 
    
    #for generalized Normal
    #from scipy.stats import gennorm 
    #alpha = 1* np.sqrt(gamma(1./beta)/gamma(3./beta)) # see variance of generalized normal: https://en.wikipedia.org/wiki/Generalized_normal_distribution
    #Ps = gennorm.pdf(ss/alpha, beta)/alpha

    
 
    '''adding up the individual contributions of the information terms'''
    Itot = 0.      
    for i1 in range(0,2): #for each neuron N={1,2,3} all possible states i (zero spikes (i=0) and more than zero spikes(i=1)) are run through   
        for i2 in range(0,2):
            for i3 in range(0,2): 
                              
                P_comb = Pk[:,0,i1]*Pk[:,1,i2]*Pk[:,2,i3] #The part over which is summed on the left hand side of Eq. 8 from paper; number of factors and loops has to be adjusted with N
                B_ = np.trapz(P_comb*Ps,ss)                    
                I_ = np.trapz(P_comb*Ps*(np.log2(P_comb+eps) - np.log2(B_+eps)),ss) #The part over which is summed in Eq. 12 from paper
                Itot = Itot + I_
                
    return Itot*-1.



'''for each combination of input and oputput noise the following values are calculated:'''
Imatrix = np.zeros((sigsteps,Rsteps))       #for storing max information values
tmatrix = np.zeros((sigsteps,Rsteps,N))     #for storing opt threshold values
eval_matrix = np.zeros((sigsteps,Rsteps,N))  #for storing eigenvalues of hessian at opt thresholds
evec_matrix = np.zeros((sigsteps,Rsteps,N,N)) #for storing eigenvectors of hessian at opt thresholds


'''instead of running sequentially through all sigma, 
define a function which does the calculation for a single sigma 
and then use "Joblib" to do embarassingly parallel calculations for different sigma'''

def opt_Info_sig(sig): #do the optimization for a vector of output noises for a given sigma
    
    j=np.where(sig_vec==sig)[0]
    Ntrap = np.int(1.5/sig*1e2) #number of steps in the trapezoid integration. in order to be precise, number of steps have to be increased with smaller sigma 
    

    I_j = np.zeros(Rsteps)           #for storing max info values for this specific sigma
    t_j = np.zeros((Rsteps,N))      #for opt threshold values for this specific sigma
    
    eval_j = np.zeros((Rsteps,N))     #for storing eigenvalues of hessian at opt thresholds for this specific sigma
    evec_j = np.zeros((Rsteps,N,N))   #for storing eigenvectors of hessian at opt thresholds for this specific sigma
   
    #run a loop of R
    for i in range(0,len(R_vec)):
        R = R_vec[i]
        #print i, R

        '''doing the optimization'''
        #http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize
        t= scipy.optimize.minimize(Info_3N_ind,x0=t0,args=(sig,R,smin,smax,Ntrap), 
                                   method='nelder-mead', options={'xtol': xtol, 'ftol': ftol, 'disp': False})  #obtaining optimal trehsholds
      
        I_j[i] = Info_3N_ind(t.x,sig,R,smin,smax,Ntrap)  #max information
        t_j[i,:] = np.sort(t.x)                        #optimal thresholds 
        
        #calculate Hesse matrix of information landscape
        H= nd.Hessian(Info_3N_ind)(t.x,sig,R,smin,smax,Ntrap)
        
        #get eigenvalues and eigenvectors of Hessian
        ev = np.linalg.eig(H)
        eval_j[i,:] = ev[0]
        evec_j[i,:,:] = ev[1]
        
       
    
        i= i+1
        
    print "------------j_sig=", j+1,"of",len(sig_vec),", sig=",sig,"; ",  time.strftime("%d %b %Y %H:%M:%S"), "------------"    
    
    return I_j, t_j, eval_j, evec_j




if load_data_from_file == True:      #for plotting: load data from file if calculations had been run already once
    datafile = np.load(namestr+parastr+'_data_ind.npz')
    Imatrix = datafile['arr_0']
    tmatrix = datafile['arr_1']
    eval_matrix = datafile['arr_2']
    evec_matrix = datafile['arr_3']

    print "loaded data from file"
    
else:     #doing the acutal calculation using joblib.Parallel 
    calc_results=Parallel(n_jobs=cpus)(delayed(opt_Info_sig)(sig) for sig in sig_vec) #actual calculation
    
    for j in range(0,len(sig_vec)):
        Imatrix[j,:] = calc_results[j][0]
        tmatrix[j,:] = calc_results[j][1]
        eval_matrix[j,:] = calc_results[j][2]
        evec_matrix[j,:] = calc_results[j][3]  
    
    print "done calculating"
    
    #save info, opt thresholds, eig-values, and eig-vectors in one file
    np.savez(namestr+parastr+'_data_ind.npz',Imatrix, tmatrix, eval_matrix, evec_matrix)
    print "saved calculated values"




eval_sort= np.sort(eval_matrix, axis =2)  #for each i,j the eigenvalues are sorted



'''
###################
plotting 
######################
'''


#generate folder for saving plots
folder_overall = namestr+parastr
if not os.path.exists(folder_overall):
        os.makedirs(folder_overall)


'''plot heatmaps'''

print "plot heatmaps"

if Rsteps > 1:
    if sigsteps > 1:
        
        '''plot information depending on R and sigma as 2D heatmap '''
        fig = plt.figure(figsize=(4.0,3.5))
        
        #for heatmap with fixed (quadratic size)
        h = [Size.Fixed(.8), Size.Fixed(2.4)] #width: lowest and extend
        v = [Size.Fixed(0.65), Size.Fixed(2.4)] #height: lowest and extend
        divider = Divider(fig, (0.0, 0.0, .5, .5), h, v, aspect=False)
        ax = LocatableAxes(fig, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
        fig.add_axes(ax)      
        
        Z=ax.pcolormesh(R_vec, sig_vec,Imatrix*-1., cmap=plt.get_cmap('plasma'))

        cbticks=np.linspace(np.amin(Imatrix*-1.), np.amax(Imatrix*-1.), num=5, endpoint=True)
        fig.subplots_adjust(right=0.78) #makes extra subplot for colorbar (necessary because of twinaxes)
        cbar_ax = fig.add_axes([0.83, 0.65/3.5, 0.03, 2.4/3.5]) #position and size
        cb=fig.colorbar(Z, cax=cbar_ax)#, ticks=[cbticks])
        tick_locator = matplotlib.ticker.LinearLocator(numticks=5)	  #setting number of colorbar ticks	
        cb.locator = tick_locator
        cb.update_ticks()        
        
        #setting color bar ticks 
        cbticks = np.round(np.linspace(np.amin(Imatrix*-1.), np.amax(Imatrix*-1.),num=5, endpoint=True),1).tolist() 
        cbticks_new=['{:.1f}'.format(ticks) for ticks in cbticks]
        cb.ax.set_yticklabels(cbticks_new)
        cb.ax.set_title('$I_{\\mathrm{ind}}$', fontsize=16)
        
        CS = ax.contour(R_vec, sig_vec, Imatrix*-1., 6, colors='w',linewidths=0.75)
        #plt.clabel(CS, inline=1, fontsize=8) 
              
        ax.set_xlabel(r'$R$',fontsize=16)
        ax.set_ylabel(r'$\sigma$',fontsize=16)
        ax.set_title("Independent channel", fontsize=12)

        ax.set_ylim(minsig,maxsig)
        ax.set_xlim(maxR,minR)
        ax.locator_params(axis='y',nbins=5)
        ax.locator_params(axis='x',nbins=4)

        Z.set_rasterized(True) #so that content is rasterized (png) but text like axis labels are still vector graphics
        figname= namestr+'_heatmap_Info.pdf'
        fig.savefig(os.path.join(folder_overall,figname), dpi=900) 


        
        '''plot number of distinct thresholds values for each combination of sigma and R'''
        roundacc=2 #rounding accuray. necessary, since they are not perfectly the same (2 decimal digits are enough)
        no_dis_t= np.zeros((sigsteps,Rsteps)) #number of distinct thresholds for each sigma-R-combination
        for j in range(0,len(sig_vec)):
            for i in range(0,len(R_vec)):
                no_dis_t[j,i]= np.count_nonzero(np.around(np.diff(tmatrix[j,i,:]), decimals=roundacc))+1  
            
        
        '''plot number of disctinct thresholds for combinations and R and sigma as 2D heatmap'''
        figb = plt.figure(figsize=(4.0,3.5))
        
        #for heatmap with fixed (quadratic size)
        h = [Size.Fixed(.8), Size.Fixed(2.4)] #width: lowest and extend
        v = [Size.Fixed(0.65), Size.Fixed(2.4)] #height: lowest and extend
        divider = Divider(figb, (0.0, 0.0, .5, .5), h, v, aspect=False)
        axb = LocatableAxes(figb, divider.get_position())
        axb.set_axes_locator(divider.new_locator(nx=1, ny=1))
        figb.add_axes(axb)  
       
        levels = np.arange(0,N+1)
            
        if N==3:
            own_cmap = cmap = matplotlib.colors.ListedColormap(['#77ABD7','#E4E2E6','#A07178'])
            Zb = axb.contourf(R_vec,sig_vec, no_dis_t, levels, cmap=own_cmap)
        else: 
            Zb = axb.contourf(R_vec,sig_vec, no_dis_t, levels, cmap=plt.get_cmap('plasma'))


        figb.subplots_adjust(right=0.78) #makes extra subplot for colorbar (necessary because of twinaxes)
        cbar_axb = figb.add_axes([0.83, 0.65/3.5, 0.03, 2.4/3.5]) #position and size

        cbb = figb.colorbar(Zb, cax=cbar_axb, ticks=[0.5, 1.5,2.5])
        cbb.ax.set_yticklabels(['1', '2', '3'])
        cbb.set_label(r'$\#$ distinct $\theta_i$', labelpad=10, fontsize=14)#, y=0.45)
        
        axb.set_xlabel(r'$R$',fontsize=16)
        axb.set_ylabel(r'$\sigma$',fontsize=16)
        axb.set_title("Independent channel", fontsize=12)
        
        axb.set_ylim(minsig,maxsig)
        axb.set_xlim(maxR,minR)
        axb.locator_params(axis='y',nbins=5)
        axb.locator_params(axis='x',nbins=3)

        figname=namestr+'_distinct_thetas2_contour.png'
        figb.savefig(os.path.join(folder_overall,figname)) 
        figname=namestr+'_distinct_thetas2_contour.pdf'
        figb.savefig(os.path.join(folder_overall,figname)) 
        
            


'''PLotting: for a given sigma, plot max info, opt thresholds, eigenvalues and derivatives in depence of R ("fineR") '''

print "plot info, theta, eigvalues and derivatives for each sigma "
if Rsteps > 1:
    
    folder_fineR = 'fineR'
    if not os.path.exists(os.path.join(folder_overall,folder_fineR)):
        os.makedirs(os.path.join(folder_overall,folder_fineR))
    j=-1    
    for sig in sig_vec:    
        j= j+1
        print 'j_sig=', j


        '''Plot optimal Info(R), thetas(R), EVals(R) and grads(R) for only in one figure with shared x-axis'''
        fig1 = plt.figure(figsize=(4,9))
    
    
        '''info'''
        ax0 = plt.subplot(411)
        ax0.plot(R_vec,-1.0*Imatrix[j,:],  ls="-", linewidth=3)# marker='o', markersize=2)#, label='I1')
            
        ax0.set_ylabel(r'$\mathrm{Information}\,\,I_m\,\,\mathrm{(bits)}$',fontsize=14) 
        ax0.set_title(r'Independent, $N=%s,\,\,   \sigma=%s$'%(str(N),format(sig, '.3f')), fontsize=12)
        
        ax0.set_xlim(maxR,minR)
        ax0.locator_params(axis='y',nbins=5)
        ax0.set_ylim(bottom=0)
        plt.setp(ax0.get_xticklabels(), visible=False)        
    

        '''thetas'''
        ax1 = plt.subplot(412)
        for k in range(0,N):
             ax1.plot(R_vec,tmatrix[j,:,k],  ls='-',linewidth=2, marker='o', markersize=2, 
                      label=r'$\theta_%s$'%str(k+1))
        
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylabel(r"$\mathrm{opt.\,\,thresholds\,\,} \theta$", fontsize=14)
    
        ax1.locator_params(axis='y',nbins=5)
        plt.setp(ax1.get_xticklabels(), visible=False)        
    
    
        '''Eigenvalues'''
        ax2 = plt.subplot(413, sharex = ax1)
        for k in range(0,N):
            #ax2.semilogy(R_vec,eval_sort[j,:,k], ls=linestyles[k], marker=markers[k], markersize=2, label=r'$EV_%s$'%str(k+1))
            ax2.plot(R_vec,-eval_sort[j,:,k], ls='-', linewidth=2, marker='o', markersize=2, label=r'$\lambda_%s$'%str(N-k))
        
        ax2.legend(loc='best', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{Eigenvalues} \,\, \lambda_k$',fontsize=14)

        
        ax2.locator_params(axis='x',nbins=5)
        plt.setp(ax2.get_xticklabels(), visible=False)        
    
        
        #have y-axis such, that it is linear from 0 to 0.01, but then scales logarithmically
        plt.yscale('symlog', linthreshy=1e-2,subsy=[2, 3, 4, 5, 6, 7, 8, 9],linscaley=0.5)
        ax2.set_ylim(top=0)
        
        
        '''derivatives'''   
        #calculate derivate of I using np.diff
        dR = (maxR-minR) / Rsteps 
        d2dR =  np.gradient(-1.*Imatrix[j,:], dR) #old way of calculating Delta_I/Delta_R (advatnage: same length; disaadvantage: intermediate points at discontinuity)
        d2dRdiff =  np.diff(-1.*Imatrix[j,:]) /dR #new way of calculating Delta_I/Delta_R 
        d2dR2diff = np.diff(d2dR) /dR #second derivative

        
        ax3 = plt.subplot(414, sharex = ax1)
        
        ax3.plot(R_vec[1:],d2dRdiff, ls='-', linewidth=2, marker='o', markersize=2, color='C0',
                  label=r'1st derivative')
        ax3.plot(R_vec[2:-1], d2dR2diff[1:-1],  ls='-', linewidth=2, marker='o', markersize=2, color='C1',
                 label=r'2nd derivative') 

        ax3.legend(loc='best', fontsize=10)
        ax3.set_xlabel(r'$R$',fontsize=14)
        ax3.set_ylabel(r'$\mathrm{d}^n I/\mathrm{d}R^n$',fontsize=14)

        ax3.set_xlim(maxR,minR)
        #ax3.set_ylim(bottom=0)
        ax3.locator_params(axis='y',nbins=5)
    
            
        plt.tight_layout(pad=0.3)
        figname = namestr+'__sig='+format(sig, '.5f')+'_info+thetas+evs+grad.pdf'
        fig1.savefig(os.path.join(folder_overall,folder_fineR, figname)) 
    
        plt.close()



'''PLotting: for a given R, plot max info, opt thresholds, eigenvalues and derivatives in depence of sigma ("finesig") '''

print "plot info, theta, eigvalues and derivatives for each R "

if sigsteps > 1:
    folder_finesig = 'finesig' 
    if not os.path.exists(os.path.join(folder_overall,folder_finesig)):
        os.makedirs(os.path.join(folder_overall,folder_finesig))
    
    i=-1
    for R in R_vec:    
        i= i+1
        print 'i_R=', i 
        
        
        '''Plot optimal Info(sig), thetas(sig), EVals(sig) and grads(sig) for only in one figure with shared x-axis'''
        fig1 = plt.figure(figsize=(4,9))
    
    
        '''info'''
        ax0 = plt.subplot(411)
        ax0.plot(sig_vec,-1.0*Imatrix[:,i],  ls="-", linewidth=3)# marker='o', markersize=2)#, label='I1')
            
        ax0.set_ylabel(r'$\mathrm{Information}\,\,I_m\,\,\mathrm{(bits)}$',fontsize=14) 
        ax0.set_title(r'Independent, $N=%s,\,\,   R=%s$'%(str(N),format(R, '.3f')), fontsize=12)
        
        ax0.set_xlim(0.,maxsig) 
        ax0.locator_params(axis='y',nbins=5)
        ax0.set_ylim(bottom=0)
        plt.setp(ax0.get_xticklabels(), visible=False)        
    

        '''thetas'''
        ax1 = plt.subplot(412)
        for k in range(0,N):
             ax1.plot(sig_vec,tmatrix[:,i,k],  ls='-',linewidth=2, marker='o', markersize=2, 
                      label=r'$\theta_%s$'%str(k+1))
        
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylabel(r"$\mathrm{opt.\,\,thresholds\,\,} \theta$", fontsize=14)
    
        ax1.locator_params(axis='y',nbins=5)
        plt.setp(ax1.get_xticklabels(), visible=False)        
    
    
        '''Eigenvalues'''
        ax2 = plt.subplot(413, sharex = ax1)
        for k in range(0,N):
            #ax2.semilogy(sig_vec,eval_sort[j,:,k], ls=linestyles[k], marker=markers[k], markersize=2, label=r'$EV_%s$'%str(k+1))
            ax2.plot(sig_vec,-eval_sort[:,i,k], ls='-', linewidth=2, marker='o', markersize=2, label=r'$\lambda_%s$'%str(N-k))
        
        ax2.legend(loc='best', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{Eigenvalues} \,\, \lambda_k$',fontsize=14)

        
        ax2.locator_params(axis='x',nbins=5)
        plt.setp(ax2.get_xticklabels(), visible=False)        
    
        
        #have y-axis such, that it is linear from 0 to 0.01, but then scales logarithmically
        plt.yscale('symlog', linthreshy=1e-2,subsy=[2, 3, 4, 5, 6, 7, 8, 9],linscaley=0.5)
        ax2.set_ylim(top=0)
        
        
        '''calculate derivate of grad_matrix using np.diff'''
        dsig = (maxsig-minsig) / sigsteps #problem: dsig = 0 for sigsteps = 1
        dIdsig =  np.gradient(-1.*Imatrix[:,i], dsig) 
        dIdsigdiff =  np.diff(-1.*Imatrix[:,i]) /dsig 
        d2dsig2diff = np.diff(dIdsig)/ dsig 


        
        ax3 = plt.subplot(414, sharex = ax1)
        
        ax3.plot(sig_vec[1:],dIdsigdiff, ls='-', linewidth=2, marker='o', markersize=2, color='C0',
                  label=r'1st derivative')
        ax3.plot(sig_vec[2:-1], d2dsig2diff[1:-1],  ls='-', linewidth=2, marker='o', markersize=2, color='C1',
                 label=r'2nd derivative') 
        
        ax3.legend(loc='best', fontsize=10)
        ax3.set_xlabel(r'$\sigma$',fontsize=14)
        ax3.set_ylabel(r'$\mathrm{d}^n I/\mathrm{d}\sigma^n$',fontsize=14)

        ax3.set_xlim(0.,maxsig) 
        ax3.locator_params(axis='y',nbins=5)
          
        plt.tight_layout(pad=0.3)
        figname = namestr+'__R='+format(R, '.3f')+'_info+thetas+evs+grad.pdf'
        fig1.savefig(os.path.join(folder_overall,folder_finesig, figname)) 
    
        plt.close()
        


  
print "done ploting"


#   plt.close('all')


endtime = time.time()
timerun =  endtime - starttime
timeprint = "time run:  " + str(timerun/60.) +" min;     " + str(timerun/3600.)+" hours;   "+ str(timerun/3600./24.)+" days"
print timeprint
