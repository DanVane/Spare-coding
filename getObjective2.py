import numpy as np
from scipy import special

def getObjective2(A, S, X, sparsity, noise_var, beta, sigma, epsilon):
    #if ~strcmp(sparsity, 'log') && ~strcmp(sparsity, 'huberL1') && ~strcmp(sparsity,'epsL1') && ...
            #~strcmp(sparsity,'FS') && ~strcmp(sparsity, 'L1') && ~strcmp(sparsity,'LARS') && ...
            #~strcmp(sparsity, 'trueL1') && ~strcmp(sparsity, 'logpos')
    if (sparsity != 'log' and sparsity != 'huberL1' and sparsity != 'epsL1' and sparsity != 'FS' and sparsity != 'L1' and sparsity != 'LARS' and sparsity != 'trueL1' and sparsity !='logpos'):
        raise Exception('sparsity function is not properly specified!\n')

    #if strcmp(sparsity, 'huberL1') || strcmp(sparsity, 'epsL1')
    if (sparsity == 'huberL1' or sparsity == 'epsL1'):
        #if ~exist('epsilon','var') || isempty(epsilon) || epsilon==0
        if (not 'epsilon' in globals() or not epsilon or not np.any()):
            raise Exception('epsilon was not set properly!\n')

    #E = A*S - X;
    E = np.dot(A,S) - X
    ld=1/noise_var
    #fresidue  = 0.5*lambda*sum(sum(E.^2));
    fresidue  = 0.5*ld*((E**2).sum())

    #if strcmp(sparsity, 'log')
    if sparsity == 'log':
        #fsparsity = beta*sum(sum(log(1+(S/sigma).^2)));
        fsparsity = beta*(np.log(1+(S/sigma)**2)).sum()
    #elseif strcmp(sparsity, 'huberL1') 
    #elif sparsity == 'huberL1':
        ## what is huber_func? not sure if this is entirely correct
        ##fsparsity = beta*sum(sum(huber_func(S/sigma, epsilon)));
        #fsparsity = beta*(huber(S/sigma, epsilon).sum())
    #elseif strcmp(sparsity, 'epsL1')
    elif sparsity == 'epsL1':
        #fsparsity = beta*sum(sum(sqrt(epsilon+(S/sigma).^2)));
        fsparsity = beta*(np.sqrt((epsilon+(S/sigma)**2)).sum())
    #elseif strcmp(sparsity, 'L1') | strcmp(sparsity, 'LARS') | strcmp(sparsity, 'trueL1') | strcmp(sparsity, 'FS')
    # check the or's 
    elif (sparsity == 'L1' or sparsity == 'LARS' or sparsity == 'trueL1' or sparsity == 'FS'):
        #fsparsity = beta*sum(sum(abs(S/sigma)));
        fsparsity = beta*(np.absolute(S/sigma).sum())
    #elseif strcmp(sparsity, 'logpos')
    elif sparsity == 'logpos':
        #fsparsity = beta*sum(sum(log(1+(S/sigma))));
        fsparsity = beta*(np.log(1+(S/sigma)).sum())

    fobj = fresidue + fsparsity
    return (fobj, fresidue, fsparsity)
