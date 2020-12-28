#!/usr/bin/python

'''
As independent.py but for the lumped channel

With that the following figures can be generated:
    - Max. mutual information for each input and output noise combination (Fig 3B)
    - Optimal thresholds and optimal threshold diversity (Fig. 4F-J)
    - How the bifurcations of optimal thresholds are related to derivates of maximal information (Fig. 5 C+D)
    - How the bifurcations of optimal thresholds are related to the eigenvalues of the Hesse matrix of the information landscape (Fig. 6 E)



Tested and run with Python 2.7.
The Python version and all the required packages for running this code can be loaded with Anaconda by using the provided yml-file:
$ conda env create -f py2.yml

(For more info, see: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file )
    


'''

import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
plt.ioff()  #figure windows are only opened with plt.show
#plt.ion()  #figure windows pop up
from mpl_toolkits.axes_grid1 import Divider, LocatableAxes, Size #for heatmap with fixed (quadratic size)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #for insets
from mpl_toolkits.axes_grid1.inset_locator import mark_inset #for insets    
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

cpus = 30 #number of cpus used for parallel calculation
print "number of cpus:", cpus


'''parameters'''

N = 3     #number of neurons (also number of loops and factor have to be adjusted in the function Info_3N_loop  (in specific: P_comb = Pk[:,0,i1]*Pk[:,1,i2]*Pk[:,2,i3]) )


Rsteps= 200 #number of data points for output noise
maxR= 10.0 #maximum R
minR = 0.001  #standard 0.001, usually not touched

sigsteps = 200 #numer of data points for input noise
maxsig = 0.7 #maximum sigma
minsig = 0.01 #small sigmas make ist slow (especially when smaller than 0.1) since more steps necessary in numerical integration 




#optimozation algorithm
#algo='BFGS'            #fast, but problems at continuous threshold jumps
#algo='nelder-mead'     #slow, but no problems at continuous threshold jumps
algo='L-BFGS-B'         #fastest, but problems at continuous threshold jumps


#initial conditions for minimization algo, do a regular spacing of thresholds between t0min and t0max but with an offset of t0off
'''for N=3'''
t0min= -1.0    
t0max = 1.0
t0off = 0.1 #offset. reason: that it does not get stuck at local maxima for N uneven (happens regularly when initial middle threshold value is 0)
t0= np.linspace(t0min, t0max, N) +t0off
t0min, t0max = min(t0), max(t0)
print "t0_vec: ",t0

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

parastr= 'N='+str(N)+'_algo='+algo+'_ftol='+str(ftol)+'_minsig='+str(minsig)+'_maxsig='+str(maxsig)+'_sigsteps='+str(
        sigsteps)+'_maxR='+str(maxR)+'_Rsteps='+str(Rsteps)
namestr= 'lumped__' 
print namestr+parastr 



#min and max stimulus values between which the integration over stimulus space is performed.
smin = -7  
smax = -smin



def Poiss(k,n): #Poisson distribution
    return np.exp(-n)*n**k/scipy.math.factorial(k)


def T(N,k,ss,tt,R,sig,H):   #Eq. 17, 18 from paper
    #since indexing starts with 0: H[N] -> H[N-1]; but T(N-1) stays T(N-1)
    if k==0:
        return np.prod(1.-H[0:N,:]+np.exp(-R)*H[0:N,:],axis=0)  #Eq. 22 from paper
    elif N==1:
        return Poiss(k,R)*(H[0,:]) #Eq. 23 from paper
    else:
        summ0 = (1.-H[N-1,:] + np.exp(-R)*H[N-1,:])*T(N-1,k,ss,tt,R,sig,H) #according to Eq. 21 with k=0
        summ = summ0
        for kN in range(1,k+1):
            summ_kN = Poiss(kN,R)*H[N-1,:] * T(N-1,k-kN,ss,tt,R,sig,H) #according to Eq. 21 with k>0
            summ += summ_kN
            
        return summ

      
    

def Info_lump(t,N,R,sig,smin,smax,Ntrap):
    #t: threshold vector
    #N: number of neurons
    #R: output noise
    #sig: input noise
    #smin, smax: borders between which the integration is performed over stimulus space
    #Ntrap: number of steps in trapezoid integration

    tt = t.reshape(N,1)
    ss=np.linspace(smin,smax,Ntrap)

    Ps = np.exp(-0.5*ss**2)*(2*np.pi)**-0.5 #stimulus distribution
 
    H = 0.5 * erfc((tt-ss)/(np.sqrt(2.)*sig)) #%P_{1|x,i} from Eq. 4 in McDonnell et al., 2006
    
    Itot = 0.    
    k= -1
    I_k=-1.
    while I_k < -ftol: #sum up information contributions for each output value k, untill new contributions are smaller than ftol
        k += 1
        Pks = T(N,k,ss,tt,R,sig,H)  #Eq. 17, 18 from paper
        B2 = np.trapz(Ps*Pks,ss)                    
        I_k = -np.trapz(Ps*Pks*(np.log2(Pks+eps) - np.log2(B2+eps)),ss) #information contribution for each k
        Itot = Itot + I_k
        
    return Itot

    
#numerically calculate eigenvalues and eigenvectors of Hesse matrix at optimal thresholds (tmaxtrix1)
def hes(sig):
    Ntrap = np.int(2./sig*1e2) #number of steps in trapezoid integration
    i=np.where(sig_vec==sig)[0]

    #calculate Hesse matrix
    H= nd.Hessian(Info_lump)(tmatrix1[i[0],j],N,R,sig,smin,smax,Ntrap)
    
    #get EVals and EVecs of Hesse
    ev = np.linalg.eig(H)
    eval_ = ev[0]
    evec_ = ev[1]
  
    return eval_, evec_


'''perform optimization of thresholds by using optimal thresholds from slightly lower sigma as initial values for next optimization ("increasing sigma")
and by by using optimal thresholds from slightly higher sigma as initial values for next optimization ("decreasing sigma").
then compare these two values to distuinguish local from global optimum near critial noise levels'''

def opt_Info_R(R):  
    j=np.where(R_vec==R)[0]

    
    '''increasing sigma'''
    #adapting the initial conditions to the optimum of the previous sigma (be careful to avoid local maxima)
    t0= np.linspace(t0min, t0max, N) +t0off 
    Imm_low = np.zeros(sigsteps)
    tmm_low = np.zeros((sigsteps,N))
    
    #'''using the same initial conditions for all sigma (thus actually: neither increasing nor decreasing sigma)
    #(slowyly, but avoiding the possibility of getting stuck in local maxima due to search only near known maxima)'''
    t00 = np.linspace(t0min, t0max, N) +t0off 
    Imm0 = np.zeros(sigsteps)
    tmm0 = np.zeros((sigsteps,N))   
    
    for i,sig in enumerate(sig_vec): #increasing sigma
        
        #number of steps in the trapezoid integration. in order to be precise number of steps have to be increased with smaller sigma 
        Ntrap = np.int(np.max([2./sig*1e2,200]))
                
        

        '''optimization'''
        #http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

        t= scipy.optimize.minimize(Info_lump,x0=t0,args=(N,R,sig,smin,smax,Ntrap), 
             method=algo, options={'ftol': ftol, 'disp': False})    
        
        t0 = np.sort(t.x) 
        
        Imm_low[i] = Info_lump(t.x,N,R,sig,smin,smax,Ntrap)
        tmm_low[i,:] = np.sort(t.x)
        
    
    '''decreasing sigma '''
    t0= np.linspace(t0min, t0max, N) +t0off
    Imm_high = np.zeros(sigsteps)
    tmm_high = np.zeros((sigsteps,N))  
    
    for ss in range(0,len(sig_vec)): 
        i=len(sig_vec)-ss-1
        sig = sig_vec[i]; #decreasing sigma
        
        Ntrap = np.int(np.max([2./sig*1e2,200]))
        
        t= scipy.optimize.minimize(Info_lump,x0=t0,args=(N,R,sig,smin,smax,Ntrap), 
             method=algo, options={'ftol': ftol, 'disp': False})    
                          
        t0 = np.sort(t.x) 
        
        Imm_high[i] = Info_lump(t.x,N,R,sig,smin,smax,Ntrap)
        tmm_high[i,:] = np.sort(t.x)


        #print "sig_j=", j+1,"/",len(sig_vec),", sig=",sig,"; i=", i+1,"/",Rsteps,";  th=",t.x,'  ',  time.strftime("%d %b %Y %H:%M:%S")
      
    '''using the same initial conditions for all sigma (thus: neither increasing nor decreasing sigma)
    (slowyly, but avoiding the possibility of getting stuck in local maxima due to search only near known maxima)'''
    t00 = np.linspace(t0min, t0max, N) +t0off 
    Imm0 = np.zeros(sigsteps)
    tmm0 = np.zeros((sigsteps,N))   
    
    # for i,sig in enumerate(sig_vec):
        
    #     #number of steps in the trapezoid integration. in order to be precise number of steps have to be increased with smaller sigma 
    #     if sig < 1.0:
    #         Ntrap = np.int(2./sig*1e2)
    #     else:
    #         Ntrap = 200
                
        

    #     '''optimization'''
    #     #http://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/optimize.html#constrained-minimization-of-multivariate-scalar-functions-minimize

        
    #     tt= scipy.optimize.minimize(Info_lump,x0=t00,args=(N,R,sig,smin,smax,Ntrap), 
    #         method=algo, options={'ftol': ftol, 'disp': False})    
        
    #     Imm0[i] = Info_lump(tt.x,N,R,sig,smin,smax,Ntrap)
    #     tmm0[i,:] = np.sort(tt.x)
         
    
    print "------------R_j=", j+1,"of",len(R_vec),", R=",R,"; ",  time.strftime("%d %b %Y %H:%M:%S"), "------------"    
    
    #print Imm.shape    
    return Imm0, tmm0, Imm_low, tmm_low, Imm_high, tmm_high#, eval_, evec_

#for each combination of input and oputput noise the following values are calculated:
Imatrix0 = np.zeros((sigsteps,Rsteps))      #no adapted initial conditions, not calculated here
tmatrix0 = np.zeros((sigsteps,Rsteps,N))    #no adapted initial conditions, not calculated here
Imatrix_incr = np.zeros((sigsteps,Rsteps))   #from "increasing sigma"
tmatrix_incr = np.zeros((sigsteps,Rsteps,N)) #from "increasing sigma"
Imatrix_decr = np.zeros((sigsteps,Rsteps))  #from "decr sigma"
tmatrix_decr = np.zeros((sigsteps,Rsteps,N)) #from "decr sigma"
eval_matrix = np.ones((sigsteps,Rsteps,N))   #eigenvalues of Hessian around optimum
evec_matrix = np.ones((sigsteps,Rsteps,N,N)) #eigenvectors of Hessian around optimum


"""
instead of running sequentially through all R, 
define a function which does the calculation for a single R 
and then use embarassingly parallel to do calculations for different R parallely
(note that in def opt_Info_R(R) we avoid the local maximum by performing the calculation for increasing and decreasing sigma at a given R.
in order to avoid the local maximum when just having one sigma, one has to adjust the code in order to have increasing and decreasing R at a given simga
"""

if load_data_from_file == True: 
    #load data from file if calculations have been run before
    datafile = np.load(namestr+parastr+'_data.npz')
    Imatrix0 = datafile['arr_0']
    tmatrix0 = datafile['arr_1']
    Imatrix_incr = datafile['arr_2']
    tmatrix_incr = datafile['arr_3']
    Imatrix_decr = datafile['arr_4']
    tmatrix_decr = datafile['arr_5']
    eval_matrix=datafile['arr_6']
    evec_matrix=datafile['arr_7'] 
    Imatrix1 = datafile['arr_8'] 
    tmatrix1 = datafile['arr_9']
    Imatrix2 = datafile['arr_10'] 
    tmatrix2 = datafile['arr_11']
    
else:
    #doing the acutal calculation using joblib.Parallel. 
    calc_results=Parallel(n_jobs=cpus)(delayed(opt_Info_R)(R) for R in R_vec) #actual calculation
    
    for i in range(0,len(R_vec)):
        Imatrix0[:,i] = calc_results[i][0]
        tmatrix0[:,i] = calc_results[i][1]
        Imatrix_incr[:,i] = calc_results[i][2]
        tmatrix_incr[:,i] = calc_results[i][3]
        Imatrix_decr[:,i] = calc_results[i][4]
        tmatrix_decr[:,i] = calc_results[i][5]
    
    
    print "done calculating optimum"
    
    print time.strftime("%d %b %Y %H:%M:%S")
    
    
    Imatrix1 = np.copy(Imatrix_incr)
    tmatrix1 = np.copy(tmatrix_incr)
    Imatrix2 = np.copy(Imatrix_decr)
    tmatrix2 = np.copy(tmatrix_decr)
    
    for j,R in enumerate(R_vec):
            
        I_diff =  -1.*(Imatrix_incr[:,j] - Imatrix_decr[:,j])
        I_diff[np.abs(I_diff) < ftol*10 ] = 0. #set I_diff to 0 where difference is less than ftol*10
        pos_inds = np.where(I_diff>0.)[0]  #get indices where I_diff is positive (for these indices choose tmatrix_incr)
        neg_inds = np.where(I_diff<0.)[0]  #get indices where I_diff is positive (for these indices choose tmatrix_decr)
        
    
        Imatrix1[neg_inds,j] = Imatrix_decr[neg_inds,j]
        tmatrix1[neg_inds,j] = tmatrix_decr[neg_inds,j]
        
        #should be equivalent to Imatrix1, tmatrix1
        Imatrix2[pos_inds,j] = Imatrix_incr[pos_inds,j]
        tmatrix2[pos_inds,j] = tmatrix_incr[pos_inds,j]
    
    
        #calculate eigenvectors and eigenvalues
        calc_results_evs=Parallel(n_jobs=cpus)(delayed(hes)(sig) for sig in sig_vec) #actual calculation
        
        for i in range(0,len(sig_vec)):
            eval_matrix[i,j] = calc_results_evs[i][0]
            evec_matrix[i,j] = calc_results_evs[i][1]
            
        print "------------R_j=", j+1,"of",len(R_vec),", R=",R,"; ",  time.strftime("%d %b %Y %H:%M:%S"), "------------"    
    
    
    print "done calculating hesse"
    
    
    np.savez(namestr+parastr+'_data.npz',Imatrix0, tmatrix0, Imatrix_incr, tmatrix_incr, 
             Imatrix_decr, tmatrix_decr, eval_matrix, evec_matrix, Imatrix1, tmatrix1, Imatrix2, tmatrix2)
    print "saved calculated values"
    

tmatrix1 =  np.flip(tmatrix1,axis=2) #just that theta1 is lowest and theta3 is highest
#tcum_matrix1= 0.5*erfc(-tmatrix/np.sqrt(2)) #cumulative
eval_sort= np.sort(eval_matrix, axis =2)  #for each i,j the EVals are sorted




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
        
        #attention: colormap is normalized between 0 and 2
        Z=ax.pcolormesh(R_vec, sig_vec,Imatrix1*-1., vmin=0., vmax=2., cmap=plt.get_cmap('plasma'))
        
        cbticks=np.linspace(np.amin(Imatrix1*-1.), np.amax(Imatrix1*-1.), num=5, endpoint=True)
        fig.subplots_adjust(right=0.78) #makes extra subplot for colorbar (necessary because of twinaxes)
        cbar_ax = fig.add_axes([0.83, 0.65/3.5, 0.03, 2.4/3.5]) #position and size
        cb=fig.colorbar(Z, cax=cbar_ax)#, ticks=[cbticks])
        tick_locator = matplotlib.ticker.LinearLocator(numticks=5)	  #setting number of colorbar ticks	
        cb.locator = tick_locator
        cb.update_ticks()        
            
        cbticks = np.round(np.linspace(np.amin(Imatrix1*-1.), np.amax(Imatrix1*-1.),num=5, endpoint=True),1).tolist() 
        cbticks = np.round(np.linspace(0,2,num=5, endpoint=True),1).tolist() 
        cbticks_new=['{:.1f}'.format(ticks) for ticks in cbticks]
        cb.ax.set_yticklabels(cbticks_new)
        cb.ax.set_title('$I_{\\mathrm{lump}}$', fontsize=16)

        
        #contours
        v = np.linspace(0.3,1.8,6,endpoint=True)  #contours linearly spaced, take last 5 values
        #v = np.append([np.max(Imatrix1*-1.)*0.95],v)  #add an additional contour which is very close to the maximum
        CS = ax.contour(R_vec, sig_vec, Imatrix1*-1., v, colors='w',linewidths=0.75)

        #plt.clabel(CS, inline=1,inline_spacing=2, fontsize=8, fmt='%1.1f') 
        
        ax.set_xlabel(r'$R$',fontsize=16)
        ax.set_ylabel(r'$\sigma$',fontsize=16)
        ax.set_title("Lumped channel", fontsize=12)

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
                no_dis_t[j,i]= np.count_nonzero(np.around(np.diff(tmatrix1[j,i,:]), decimals=roundacc))+1   
            
        
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
        axb.set_title("Lumped channel", fontsize=12)
        
        axb.set_ylim(minsig,maxsig)
        axb.set_xlim(maxR,minR)
        axb.locator_params(axis='y',nbins=5)
        axb.locator_params(axis='x',nbins=3)

        figname=namestr+'_distinct_thetas2_contour.png'
        figb.savefig(os.path.join(folder_overall,figname)) 
        figname=namestr+'_distinct_thetas2_contour.pdf'
        figb.savefig(os.path.join(folder_overall,figname)) 
        
            


'''PLotting: for a given R, plot max info, opt thresholds, eigenvalues and derivatives in depence of sigma ("finesig") '''

print "plot info, theta, eigvalues and derivatives for each R"

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
        ax0.plot(sig_vec,-1.0*Imatrix1[:,i],  ls="-", linewidth=3)# marker='o', markersize=2)#, label='I1')
            
        ax0.set_ylabel(r'$\mathrm{Information}\,\,I_m\,\,\mathrm{(bits)}$',fontsize=14) 
        ax0.set_title(r'Lumped, $N=%s,\,\,   R=%s$'%(str(N),format(R, '.3f')), fontsize=12)
        
        ax0.set_xlim(0.,maxsig) 
        ax0.locator_params(axis='y',nbins=5)
        ax0.set_ylim(bottom=0)
        plt.setp(ax0.get_xticklabels(), visible=False)        
    

        '''thetas'''
        ax1 = plt.subplot(412)
        for k in range(0,N):
             ax1.plot(sig_vec,tmatrix1[:,i,k],  ls='-',linewidth=2, marker='o', markersize=2, 
                      label=r'$\theta_%s$'%str(k+1))
            #ax1.plot(sig_vec,tmatrix[:,k], ls="None", color='k', marker='o', markersize=2.0) #McD style
        
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylabel(r"$\mathrm{opt.\,\,thresholds\,\,} \theta$", fontsize=14)
    
        ax1.locator_params(axis='y',nbins=5)
        plt.setp(ax1.get_xticklabels(), visible=False)        
    
    
        '''Eigenvalues'''
        ax2 = plt.subplot(413, sharex = ax1)
        for k in range(0,N):
            #ax2.semilogy(sig_vec,eval_sort[j,:,k], ls='-', marker='o', markersize=2, label=r'$EV_%s$'%str(k+1))
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
        dIdsig =  np.gradient(-1.*Imatrix1[:,i], dsig) 
        dIdsigdiff =  np.diff(-1.*Imatrix1[:,i]) /dsig #new way of calculating Delta_I/Delta_R 
        d2dsig2diff = np.diff(dIdsig)/ dsig  #new way of calculating Delta_I/Delta_R 


        
        ax3 = plt.subplot(414, sharex = ax1)
        
        ax3.plot(sig_vec[1:],dIdsigdiff, ls='-', linewidth=2, marker='o', markersize=2, color='C0',
                  #label=r'$\frac{\Delta I}{\Delta R}$')
                  label=r'1st derivative')
#        ax3.plot(sig_vec[2:-1], d2dsig2diff[1:-1],  ls='-', linewidth=2, marker='o', markersize=2, color='C1',
#                 label=r'2nd derivative') #actually d2dR2diff[1:-1] is corect, but for presting purpose I want to have both derivatives the same sign

#        ax3.legend(loc='best', fontsize=10)
        ax3.set_xlabel(r'$\sigma$',fontsize=14)
        ax3.set_ylabel(r'$\mathrm{d}^n I/\mathrm{d}\sigma^n$',fontsize=14)

        ax3.set_xlim(0.,maxsig) 
        #ax3.set_ylim(bottom=0)
        ax3.locator_params(axis='y',nbins=5)
           
        plt.tight_layout(pad=0.3)

        figname = namestr+'__R='+format(R, '.3f')+'_info+thetas+evs+grad.pdf'
        fig1.savefig(os.path.join(folder_overall,folder_finesig, figname)) 
    
    
        plt.close()
        
        
        
        
'''PLotting: for a given sigma, plot max info, opt thresholds, eigenvalues and derivatives in depence of R ("fineN") '''

print "plot info, theta, eigvalues and derivatives for each sigma"


if Rsteps > 1:
    
    folder_fineR = 'fineR'
    if not os.path.exists(os.path.join(folder_overall,folder_fineR)):
        os.makedirs(os.path.join(folder_overall,folder_fineR))
    j=-1    
    
    #'''only plot for sig=0.1 (j=26)'''
    
    j=-1    
    for sig in sig_vec:    
        j= j+1
        print 'j_sig=', j


        '''Plot grad1(R)'''
    
        '''calculate derivate of Imatrix using np.gradient'''
        dR = (maxR-minR) / Rsteps 
        dNdmax =  np.gradient(-1.*Imatrix1[j,:], dR) 
        dNdRdiff =  np.diff(Imatrix1[j,:]) /dR 


        
        '''calculate derivate of grad_matrix using np.gradient'''
        dR = (maxR-minR) / Rsteps 
        d2dR2diff = np.diff(dNdmax) /dR
        d2dR2diff = np.diff(dNdRdiff) /dR


  
        '''Plot optimal Info(R), thetas(R), EVals(R) and grads(R) for only in one figure with shared x-axis'''
        fig1 = plt.figure(figsize=(4,9))
    
    
        '''info'''
        ax0 = plt.subplot(411)
        ax0.plot(R_vec,-1.0*Imatrix1[j,:],  ls="-", linewidth=3)# marker='o', markersize=2)#, label='I1')
            
        ax0.set_ylabel(r'$\mathrm{Information}\,\,I_m\,\,\mathrm{(bits)}$',fontsize=14) 
        ax0.set_title(r'Lumped, $N=%s,\,\,   \sigma=%s$'%(str(N),format(sig, '.3f')), fontsize=12)
        
        ax0.set_xlim(maxR,minR)
        ax0.locator_params(axis='y',nbins=5)
        ax0.set_ylim(bottom=0)
        plt.setp(ax0.get_xticklabels(), visible=False)        
    

        '''thetas'''
        ax1 = plt.subplot(412)
        for k in range(0,N):
             ax1.plot(R_vec,tmatrix1[j,:,k], ls='-', linewidth=2, marker='o', markersize=2, 
                      label=r'$\theta_%s$'%str(k+1))
            #ax1.plot(sig_vec,tmatrix[:,k], ls="None", color='k', marker='o', markersize=2.0) #McD style
        
        ax1.legend(loc='best', fontsize=12)
        ax1.set_ylabel(r"$\mathrm{opt.\,\,thresholds\,\,} \theta$", fontsize=14)
    
        ax1.locator_params(axis='y',nbins=5)
        plt.setp(ax1.get_xticklabels(), visible=False)        
    
    
        '''Eigenvalues'''
        ax2 = plt.subplot(413, sharex = ax1)
        for k in range(0,N):
            #ax2.semilogy(R_vec,eval_sort[j,:,k], ls='-', marker='o', markersize=2, label=r'$EV_%s$'%str(k+1))
            ax2.plot(R_vec,-eval_sort[j,:,k], ls='-', linewidth=2, marker='o', markersize=2, label=r'$\lambda_%s$'%str(N-k))
        
        ax2.legend(loc='best', fontsize=12)
        ax2.set_ylabel(r'$\mathrm{Eigenvalues} \,\, \lambda_k$',fontsize=14)
        
        ax2.locator_params(axis='x',nbins=5)
        plt.setp(ax2.get_xticklabels(), visible=False)        
    
        
        #have y-axis such, that it is linear from 0 to 0.01, but then scales logarithmically
        plt.yscale('symlog', linthreshy=1e-2,subsy=[2, 3, 4, 5, 6, 7, 8, 9],linscaley=0.5)
        ax2.set_ylim(top=0)
        
        
        '''derivatives'''   
        dR = (maxR-minR) / Rsteps 
        dNdmax =  np.gradient(-1.*Imatrix1[j,:], dR) 
        dNdRdiff =  np.diff(-1.*Imatrix1[j,:]) /dR #new way of calculating Delta_I/Delta_R 
            
        ax3 = plt.subplot(414, sharex = ax1)
        ax3.plot(R_vec[0:-1],dNdRdiff,  ls='-' ,linewidth=2, marker='o', markersize=2, label=r'$\frac{\Delta I}{\Delta R}$')
    
        ax3.set_xlabel(r'$R$',fontsize=14)
        ax3.set_ylabel(r'$\mathrm{d}I/\mathrm{d}R$',fontsize=14)
               
        ax3.set_xlim(maxR,minR)
        ax3.set_ylim(bottom=0)
        ax3.locator_params(axis='y',nbins=5)
    
        plt.tight_layout(pad=0.3)

        figname = namestr+'__sig='+format(sig, '.5f')+'_info+thetas+evs+grad.pdf'
        fig1.savefig(os.path.join(folder_overall,folder_fineR, figname)) 

        plt.close()



  
print "done ploting"


##     plt.close('all')

endtime = time.time()
timerun =  endtime - starttime
timeprint = "time run:  " + str(timerun/60.) +" min;     " + str(timerun/3600.)+" hours;   "+ str(timerun/3600./24.)+" days"
print
print timeprint
