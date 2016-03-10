import numpy
import l1_featuresign
import l2_learn_basis_dual
import getObjective2
from math import sqrt

def sparse_coding(X_total, num_bases, beta, sparsity_func, epsilon, num_iters, batch_size, fname_save, pars, Binit=[], resample_size=0):
# Fast sparse coding algorithms

#    minimize_B,S   0.5*||X - B*S||^2 + beta*sum(abs(S(:)))
#    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
#  The detail of the algorithm is described in the following paper:
# 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
# Advances in Neural Information Processing Systems (NIPS) 19, 2007

# Written by Honglak Lee <hllee@cs.stanford.edu>
# Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

# options:
# X_total: training set 
# num_bases: number of bases
# beta: sparsity penalty parameter
# sparsity_func: sparsity penalty function ('L1', or 'epsL1')
# epsilon: epsilon for epsilon-L1 sparsity
# num_iters: number of iteration
# batch_size: small-batch size
# fname_save: filename to save
# pars: additional parameters to specify (see the code)
# Binit: initial B matrix
# resample_size: (optional) resample size 

    if resample_size > 0 :
        assert (X_total.shape[1] > resample_size)
        X = X_total[:, numpy.random.choice(X_total.shape[1], resample_size,replace = False)]
    else:
        X = X_total
    

    patch_size = X.shape[0]
    num_patches = X.shape[1]
    num_bases = num_bases
    num_trials = num_iters
    filename = fname_save
    if batch_size <= 0 :
        batch_size = X.shape[1]/10
    

    
    noise_var = 1
    sigma = 1
    VAR_basis = 1

    if 'display_images' not in locals():
    	display_images = False	
    if 'display_every' not in locals():
       	display_every = 1
    if 'save_every' not in locals():
    	save_every = 1
    if 'save_basis_timestamps' not in locals():
    	save_basis_timestamps = True
    

# Sparsity parameters
    tol = 0.005


# L1 sparsity function
    if sparsity_func=='epsL1':
        reuse_coeff = False
    else:
    	epsilon = []
#--------- this empty [] is suspicious... 
        reuse_coeff = True


#--------- I deleted this line since it is only printing result.
#pars  

#--------- set path
# addpath sc2

# initialize basis
    if Binit == []:
        print('no binit here...')
        B = numpy.random.rand(patch_size,num_bases)-0.5
    	B = B - numpy.tile(numpy.mean(B,axis = 0), (X.shape[0],1))
        #print B.shape
        
        B = numpy.dot(B , numpy.diag(numpy.divide(1,numpy.sqrt(numpy.sum(B*B,axis=0)))))
    else:
        print('Using Binit...')
        B = Binit
    L = B.shape[0]
    M = B.shape[1]
    winsize = sqrt(L);

# initialize display
#    if pars.display_images:
#!!!!don't know about this line now!!!!
#        figure(1), display_network_nonsquare2(B);
        # figure(1), colormap(gray);

    S_all = numpy.zeros((M, num_patches))

    # initialize t only if it does not exist
    #not sure is t a local or not
    if 't' not in locals():
    	t = 0	
    	# statistics variable
    	fobj_avg = []
        fresidue_avg = []
        fsparsity_avg = []
        var_avg = []
        svar_avg = []
        var_tot = 0

        svar_tot = 0
        elapsed_time = 0
    else:
    	# make sure that everything is continuous
    	t = numpy.size(fobj_avg,0) - 1 



# optimization loop
    run = 1
    while t < num_trials:
        if t == 1:
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/B-1.txt',B)
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/S-1.txt',S_all)
        if t == 30:
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/B-30.txt',B)
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/S-30.txt',S_all)
        if t == 50:
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/B-50.txt',B)
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/S-50.txt',S_all)
        if t == 80:
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/B-80.txt',B)
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/S-80.txt',S_all)
        if t == 100:
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/B-100.txt',B)
            numpy.savetxt('/Users/weijian/Documents/Maverick/UCLA/senior/winter/math191/fast_sc_python/data/S-100.txt',S_all)

        t = t + 1

        ##!!! let's skip this right now:
        ##start_time = cputime;
        
        fobj_total = 0
        fresidue_total = 0
        fsparsity_total = 0
        var_tot = 0
        svar_tot = 0

        if resample_size > 0:
            print "resample X (%d out of %d)...\n" % (resample_size, X_total.shape[1]);
            X = X_total[:, numpy.random.choice(X_total.shape[1], resample_size,replace = False)];
        
        
        # Take a random permutation of the samples
        indperm = numpy.random.permutation(X.shape[1])
        print indperm.shape
        for batch in range(0,X.shape[1] / batch_size - 1):
            # Show progress in epoch
            print '.'
            if batch % 20 ==0:
                print '\n'

            # This is data to use for this step
            batch_idx = indperm[ batch_size * batch : batch_size * (batch + 1)]
            print batch_idx.shape
            Xb = X[:,batch_idx]
            
            # learn coefficients (conjugate gradient)
            if t == 1 or ~reuse_coeff:
                if sparsity_func == 'L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
                    #print B.shape
                    #print Xb.shape
                    #print beta/sigma*noise_var
                    S = l1_featuresign.l1ls_featuresign(B, Xb, beta/sigma*noise_var)
                #else:
                #    S = cgf_fitS_sc2(B, Xb, sparsity_func, noise_var, beta, epsilon, sigma, tol, False, False, False)
                #not sure about 0's type
                S = numpy.nan_to_num(S)
                print S.shape
                print S_all[:,batch_idx].shape
                S_all[:,batch_idx] = S
                #print S_all[:,batch_idx]
                
            else:
                if sparsity_func == 'L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
                    tic
                    S = l1_featuresign.l1ls_featuresign(B, Xb, beta / sigma * noise_var, S_all[:,batch_idx])
                    FS_time = toc
                #else:
                #    S = cgf_fitS_sc2(B, Xb, sparsity_func, noise_var, beta, epsilon, sigma, tol, False, False, False, S_all[:,batch_idx])

                S = numpy.nan_to_num(S)
                S_all[:,batch_idx] = S        
            if sparsity_func =='L1' or sparsity_func == 'LARS' or sparsity_func == 'FS':
                sparsity_S = numpy.count_nonzero(S)*1.0 /numpy.size(S)
                print "non-zero "
                print numpy.count_nonzero(S)
                print "size" 
                print numpy.size(S)
                print "sparsity_S = %g\n" % sparsity_S
        
            
            # get objective
            ###!!!!not sure how you translate return value
            [fobj, fresidue, fsparsity] = getObjective2.getObjective2(B, S, Xb, sparsity_func, noise_var, beta, sigma, epsilon)
            
            fobj_total      = fobj_total + fobj
            fresidue_total  = fresidue_total + fresidue
            fsparsity_total = fsparsity_total + fsparsity
          ###may have problem here?
            #print 'var_tot'
            #print numpy.isinf(S).any()
            #print numpy.isnan(S).any()
            
            var_tot = var_tot + numpy.sum(numpy.sum(S ** 2,axis=0),axis=0)*1.0 / S.shape[0]
            
            # update basis
            B = l2_learn_basis_dual.l2ls_learn_basis_dual(Xb, S, VAR_basis)
        
        
        # get statistics
        fobj_avg.append(fobj_total / num_patches)
        fresidue_avg.append(fresidue_total / num_patches)
        fsparsity_avg.append(fsparsity_total / num_patches)
        var_avg.append(var_tot / num_patches)
        svar_avg.append(svar_tot / num_patches)
        #!!!stat.elapsed_time(t)  = cputime - start_time
        
        # !!!!!!display
        #if ( display_images and t % display_every == 0) or t % save_every == 0 or t == num_trials:
            #display_figures(pars, stat, B, S, t)
        
        
     #   fprintf(['epoch= %d, fobj= %f, fresidue= %f, fsparsity= %f, took %0.2f ' ...
     #            'seconds\n'], t, stat.fobj_avg(t), stat.fresidue_avg(t), ...
     #           stat.fsparsity_avg(t), stat.elapsed_time(t));
        
        # save results
        #if t % save_every ==0 or t == num_trials:
        #    print "saving results ...\n"
        #    experiment = []
        #    matfname = "%s.mat" % filename
            
            #if display_images:
            #    save_figures(pars, t)
            
            #!!!save(experiment.matfname, 't', 'pars', 'B', 'stat');
        #    print "saved as %s\n" % experiment.matfname
        print "one more run turn:%d\n" % run
        run = run + 1
    print fobj_avg 
    print fresidue_avg
    print fsparsity_avg
    print var_avg
    print svar_avg

    return [B, S_all]

##don't need extea assert function, using python's own.

