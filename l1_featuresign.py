import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix

'''Assuming that all N-dimensional arrays are numpy arrays'''
# A: matrix
# Y: vector or matrix... will be transformed to matrix
#    If matrix, then # of rows = # of rows of A
# gamma: scalar
# Xinit: matrix or vector?
def l1ls_featuresign(A, Y, gamma, Xinit=np.array([])):

    use_Xinit = False
    # make Y and Xinit into a matrix (column vector) if not already
    Y = np.array([Y]).transpose() if len(Y.shape)==1 else Y
    Xinit = np.array([Xinit]).transpose()

    try:
        AtA = np.dot(A.transpose(), A)
        AtY = np.dot(A.transpose(), Y).ravel()
    except ValueError, e:
        print "ERROR: Dimension of A or Dimension of Y not valid"
        print "Y can be 1D numpy array, or 2D matrix with number or rows !== number of rows of A"
        print e

    #rankA is not actually the rank of A, just a limit on dimension
    #that will be applied later on the number of non zeros in x
    rankA = min(len(A)-1, len(A[0])-1)   #TODO: originally -10 but for testing we have rank=3 so

    #ret is the result (Xout) of our featuresign step
    #after minimizing 0.5*||y-A*x||^2 + gamma*||x||_1
    #Notice that x can have multiply columns depending on Y
    ret = []
    if len(Xinit)>0:
        use_Xinit = True

    for i in range(Y.shape[1]):
        if i%100 == 0:
            print '.'
        if use_Xinit:
            Xinit_col = Xinit[:,i].ravel()
            print "Xinit_col", Xinit_col
            idx1 = np.where(Xinit_col != 0)[0]

            print "idx1 is ", idx1
            maxn = min(len(idx1), rankA)
            print "maxn", maxn
            xinit = np.zeros(Xinit_col.shape)
            print "xinit is currently:", xinit
            print "the index is", idx1[0:maxn]
            xinit[idx1[0:maxn]] = Xinit_col[idx1[0:maxn]]
            print "xinit is now:", xinit
            print "You SHOULD RUN THESE INPUTS FOR ls_featuresign_sub"
            print [A,Y[:,i], AtA, AtY, gamma, xinit]

            # the index zero is because featuresign_sub returns 2 values,
            # we are only interested in the first one.
            ret.append(ls_featuresign_sub(A,Y[:,i], AtA, AtY, gamma, xinit)[0])
            print "YOU SHOULD GET THESE OUT PUTS"
            print ret
        else:

            print "You SHOULD RUN THESE INPUTS FOR ls_featuresign_sub"
            print [A,Y[:,i], AtA, AtY, gamma]
            ret.append(ls_featuresign_sub(A,Y[:,i], AtA, AtY, gamma)[0])
            print "YOU SHOULD GET THESE OUT PUTS"
            print ret
    return ret

'''
testing first function
'''




def ls_featuresign_sub(A, y, AtA, Aty, gamma, xinit = np.array([])):
    '''Step 1 initialize'''
    [L, M] = A.shape
    rankA = min(len(A)-1, len(A[0])-1)  #Why?
    usexinit = False
    if len(xinit)==0:
        xinit       = np.array([]) #TODO: I don't think this value is used
        x           = np.zeros(M) #TODO: What type of sparse matrix is best here?
        theta       = np.zeros(M)
        act         = np.zeros(M)
        allowZero   = False
        print "There was NO xinit provided"
    else:
        x           = xinit
        theta       = np.sign(x)
        act         = abs(theta)
        usexinit    = True
        allowZero   = True

    #TODO: not sure what this is
    #fname_debug = sprintf('../tmp/fsdebug_%x.mat', datestr(now, 30));

    fobj = 0

    ITERMAX = 3
    optimality1 = False
    print "AtA, x, Aty = ", AtA, x, Aty
    i = 0
    for i in range(ITERMAX):
        act_idx0 = np.where(act == 0)[0] #TODO: this does not work well when act is sparse
        print "act_idx0", act_idx0

        # Step 2:
        #TODO: Is this the Gradient of ||y-Ax||^2 wrt every x?
        grad = np.dot(AtA, x) - Aty
        print "grad =",grad
        theta = np.sign(x) #TODO: sin't this repeated?

        optimality0 = False
        noMx = False
        mx = -1
        try:
            mx = max(abs(grad[act_idx0]))
            mx_idx = np.where(abs(grad[act_idx0]) == mx)
            print "mx {0}and mx indx at{1}".format(mx, mx_idx)
        except Exception:
            noMx = True
            print "DID NOT FIND MAX GRADIENT"
        #This is the index in the act_idx0 array, NOT the grad array


        if not noMx and mx>=gamma and (i>0 or (not usexinit)):
            print "I'M IN HERE"

            act[act_idx0[mx_idx]] = 1
            print"act is", act
            theta[act_idx0[mx_idx]] = -np.sign(grad[act_idx0[mx_idx]])
            usexinit = False
        else:
            optimality0 = True #TODO: this condition is never checked by anything else...so we can maybe delete
            if optimality1:
                break

        # either optimality 0 is not satisfied or optimlaity 1 is not satisfied
        act_idx1 = np.where(act == 1)[0]
        print "act_idx1 is this", act_idx1
        print "rankA", rankA
        if len(act_idx1)>rankA:
            print "Sparsity penalty is too small: too many coefficients"

            return x, fobj

        if len(act_idx1) == 0:
            if allowZero:

                print "IM CHEKCING ALLOW zero"
                allowZero = False
                continue
            return

        k = 0
        while True:
            k += 1
            if k>ITERMAX:
                print "(b)Exceeding maximum number of iterations, solution might not be optimal"
                return x, fobj
            if len(act_idx1) == 0:
                if allowZero:
                    allowZero = False
                    break
                else:
                    return x, fobj
            print "Use These values for compute FS step input:"
            print [x, act, act_idx1,]

            [x, theta, act, act_idx1, optimality1, lsearch, fobj] = \
                compute_FS_step(x, A, y, AtA, Aty, theta, act, act_idx1, gamma)
            print "done with compute FS step and outputs are "
            print x, act_idx1, lsearch
            # print "you should get these outputs from compute FS step:"
            # print [x, theta, act, act_idx1, optimality1, lsearch, fobj]
            # Step 4: check optimality condition 1
            if optimality1:
                break
            if lsearch>0:
                continue #TODO: continue just goes to the next while iteration... does this even do anything?

    if i >= ITERMAX:
        print "(a)Exceeding maximum number of iterations, solution might not be optimal"



    print "CHECKING inputs values of fobj_featuresign within sub"
    print []
    fobj, objA = fobj_featuresign(x, A, y, AtA, Aty, gamma)
    return x, fobj


'''Testing ls_featuresign_sub'''








def compute_FS_step(x, A, y, AtA, Aty, theta, act, act_idx1, gamma):

    x = x.astype(float)
    x2 = x[act_idx1]
    AtA2 = AtA[:,act_idx1][act_idx1,:]
    theta2 = theta[act_idx1]
    x_new = np.linalg.solve(AtA2, Aty[act_idx1] - gamma*theta2)

    optimality1 = False
    # lsearch = 1
    # TODO: consider changing to fobj, objA = fobj_featuresign(...)
    fobj = 0


    # TODO: why does equal sign imply condition A is satisfied?
    # This step seems to be checking optimality1.. which seems to be condition a in the paper
    # But the conditions used seem to be quite different

    if all(np.sign(x_new) == np.sign(x2)):
        # I think optimality1 == condition A and is true when
        # objA == 0, or < eps
        optimality1 = True

        x[act_idx1] = x_new
        lsearch = 1.0
        # The following comment from author seems to be what the original condition should be
        # But for some reason we assumed this to be 0
        fobj = 0 # fob_featuresign(x, A, y, AtA, Aty, gamma)
        print "Optimality 1 is satisfied in compute_FS_step!"
        return [x, theta, act, act_idx1, optimality1, lsearch, fobj]
    


    ''' line search '''
    progress = np.true_divide((0-x2), x_new-x2)
    lsearch = 0.0
    a= 0.5*sum(np.power(np.dot(A[:, act_idx1],(x_new- x2)), 2))
    b= np.dot(x2,np.dot(AtA2,(x_new- x2))) - np.dot(np.transpose(x_new- x2),Aty[act_idx1])

    fobj_lsearch = gamma * sum(abs(x2))

    # TODO: chekc if progress is row vector!
    prog_with_idx = zip(np.hstack([progress,1]), range(len(progress)+1))

    [sort_lsearch, ix_lsearch] = zip(*sorted(prog_with_idx, key=lambda x: x[0]))

    remove_idx=[]
    for i in range(len(sort_lsearch)):
        t = sort_lsearch[i]
        if t<=0 or t>1:
            continue
        s_temp= x2+ (x_new- x2)*t
        fobj_temp = a*t**2 + b*t + gamma*sum(abs(s_temp)) # ======> pay attention here... what's going on?
            #What is fobj_temp? what is fobj?
            #I think it is the objective function
        if fobj_temp < fobj_lsearch:
            fobj_lsearch = fobj_temp
            lsearch = t
            if t<1:
                remove_idx.append(ix_lsearch[i]) # remove_idx can be more than two..
        elif fobj_temp > fobj_lsearch:
            break
        else:
            if (sum(x2==0)) == 0:
                lsearch = t
                fobj_lsearch = fobj_temp;
                # remove_idx can be more than two..
                if t<1:
                    remove_idx.append(ix_lsearch[i])

    # Updating the result of line search I THINK
    if lsearch>0:
        x_new = x2 + (x_new-x2)*lsearch
        x[act_idx1] = x_new
        theta[act_idx1] = np.sign(x_new)

    # removing the newly created zero entries.. mark them as inactive
    if lsearch < 1 and lsearch > 0:
        remove_idx = np.where(abs(x[act_idx1])<np.finfo(float).eps)[0]
        x[act_idx1[remove_idx]] = 0

        theta[act_idx1[remove_idx]] = 0
        act[act_idx1[remove_idx]] = 0
        # act_idx1 = act_idx1[remove_idx]
        act_idx1 = np.where(act != 0)[0]
    print "the type of x is ",type(x), type(x[0])
    return [x, theta, act, act_idx1, optimality1, lsearch, fobj]


def fobj_featuresign(x, A, y, AtA, Aty, gamma):
    f= 0.5*sum((y-np.dot(A,x))**2)
    f= f+ gamma*sum(abs((x)))

    # if nargout >1:  # Still not sure what this thing is doing
    g= np.dot(AtA,x) - Aty
    g= g+ gamma*np.sign(x)

    return f, g



