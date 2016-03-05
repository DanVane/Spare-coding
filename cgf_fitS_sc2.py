import numpy as np



def cgf_fitS_sc2(A,X, sparsity, noise_var, beta, epsilon, sigma, tol, disp_ocbsol, disp_patnum, disp_stats, Sinit):
    maxiter = 100
    [L,M] = A.shape
    N = M #??

    if tol!=None: tol=0.001
    if disp_ocbsol != None      :   disp_ocbsol     = 0
    if disp_patnum != None      :   disp_patnum 	= 1
    if disp_stats != None       :   disp_stats 	    = 1
    if maxiter != None          :   maxiter 		= 8
    if 'reduction' in locals()  :   reduction 		= 8

    initiated = None

    # XXX: we don't use initialization for "log" sparsity function because of local optima
    if Sinit is None: #|| strcmp(sparsity, 'log') || strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
        Sinit=np.dot(np.transpose(A),X)
        normA2=sum(np.multiply(A,A)) #TODO:elementary wise square and then sum all elements in the same column
        # And then make it a row vector
        for i in range(N):
            # Seems like to be dividing each colum by the normA2 vector
            Sinit[:,i]=Sinit[:,i]./normA2;
        initiated = 0
    else:
        initiated = 1

    '''
    # TODO: sparsity is something special in Matlab, since l1_featuresign did not use any sparsity, maybe this step is
    # unnecessary
    if sparsity!= 'log' and  sparsity != 'huberL1' and sparsity != 'epsL1':
        print 'sparsity function is not properly specified!\n'
    '''
    m_lambda = 1.0/noise_var

    '''
    # Another one involving sparsity ?????
    if strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
	    if ~exist('epsilon','var') || isempty(epsilon) || epsilon==0
		    error('epsilon was not set properly!\n')
    '''

