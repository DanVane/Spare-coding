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
        print "Y can be 1D numpy array, or 2D matrix with number or rows == number of rows of A"
        print e

    #rankA is not actually the rank of A, just a limit on dimension
    #that will be applied later on the number of non zeros in x
    rankA = min(len(A)-1, len(A[0])-1)     #TODO: originally -10 but for testing we have rank=3 so

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
            ret.append(ls_featuresign_sub(A,Y[:,i], AtA, AtY, gamma, xinit))
        else:
            ret.append(ls_featuresign_sub(A,Y[:,i], AtA, AtY, gamma))

    return ret

'''
testing first function
'''
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
y = np.array([1,5,5])
gamma = 0.2
X = np.array([[2,1,2]])

ret = l1ls_featuresign(A, y, gamma)



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

    ITERMAX = 1000
    optimality1 = False
    print "AtA, x, Aty = ", AtA, x, Aty
    i = 0
    for i in range(ITERMAX):
        act_idx0 = np.where(act == 0)[0] #TODO: this does not work well when act is sparse
        print "act_idx0", act_idx0
        #TODO: Is this the Gradient of ||y-Ax||^2 wrt every x?
        grad = np.dot(AtA, x) - Aty
        print "grad =",grad
        theta = np.sign(x) #TODO: sin't this repeated?

        optimality0 = False
        mx = max(abs(grad[act_idx0]))
        #This is the index in the act_idx0 array, NOT the grad array
        mx_idx = np.where(abs(grad[act_idx0]) == mx)
        print "mx {0}and mx indx at{1}".format(mx, mx_idx)

        if mx>=gamma and (i>0 or not usexinit):
            act[act_idx0[mx_idx]] = 1
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
                allowZero = False
                continue

        k = 0
        while True:
            k += 1
            if k>ITERMAX:
                print "Exceeding maximum number of iterations, solution might not be optimal"
                return x, fobj
            if len(act_idx1) == 0:
                if allowZero:
                    allowZero = False
                    break
                else:
                    return x, fobj

            x, theta, act, act_idx1, optimality1, lsearch, fobj = \
                compute_FS_step(x, A, y, AtA, Aty, theta, act, act_idx1, gamma)

            # Step 4: check optimality condition 1
            if optimality1:
                break
            if lsearch>0:
                continue #TODO: continue just goes to the next while iteration... does this even do anything?

    if i >= ITERMAX:
        print "Exceeding maximum number of iterations, solution might not be optimal"

    fobj = fobj_featuresign(x, A, y, AtA, Aty, gamma)
    return x, fobj


'''Testing ls_featuresign_sub'''








def compute_FS_step(x, A, y, AtA, Aty, theta, act, act_idx1, gamma):

    x2 = x[act_idx1]
    AtA2 = AtA[:,act_idx1][act_idx1,:]
    theta2 = theta[act_idx1]
    x_new = np.linalg.solve(AtA2, Aty[act_idx1] - gamma*theta2)

    optimality1 = False

    print "we did it"
    lsearch = 1
    fobj = 0

    '''THE PART DIEN IS WORKING ON... basically line search '''


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
        act_idx1 = act_idx1[remove_idx]re


    return [x, theta, act, act_idx1, optimality1, lsearch, fobj]


def fobj_featuresign(x, A, y, AtA, Aty, gamma):
    f= 0.5*sum(abs((y-A*x)))**2
    f= f+ gamma*sum(abs((x)))

    # if nargout >1:  # Still not sure what this thing is doing
    g= AtA*x - Aty
    g= g+ gamma*np.sign(x)

    return f, g




import timeit

s = """\
import numpy as np
y = np.array([1,2,3])
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
np.dot(A.transpose(),y)
"""

s2 = """\
import numpy as np
y = np.array([[1,2,3]]).transpose()
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
np.dot(A.transpose(),y)
"""

timeit.timeit(s, number=10000)
timeit.timeit(s2, number=10000)

