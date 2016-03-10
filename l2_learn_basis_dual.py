import numpy as np
from scipy import optimize as op
import time
import random 
from numpy import linalg as la

# don't know if fvals are useful or neccessary, probably not
def l2ls_learn_basis_dual(X, S, l2norm):
    #L = size(X,1);
    L = np.size(X, axis=0)
    #N = size(X,2);
    N = np.size(X, axis=1)
    #M = size(S, 1);
    M = np.size(S, axis=0)

    #tic
    t0 = time.time()
    #SSt = S*S';
    SSt = S.dot(S.conj().transpose())
    #XSt = X*S';
    XSt = X.dot(S.conj().transpose())


    #if exist('Binit', 'var')
    #if hasattr(Binit, 'attr_name'):
        ##dual_lambda = diag(Binit\XSt - SSt);
        #dual_lambda = diag(la.solve(Binit,XSt) - SSt)
    ##else
    #else:
        #dual_lambda = 10*abs(rand(M,1)); % any arbitrary initialization should be ok.
    #dual_lambda = 10 * abs(random.uniform(M,1)) #old
    #ld = 10 * abs(np.random.rand(M,1)) #old
    ld = 10 * abs(np.random.rand(M))
    dual_lambda = np.asarray(ld)
    #dual_lambda = np.array([1.,2.,3.])
    #end

    #c = l2norm^2;
    c = l2norm**2
    #trXXt = sum(sum(X.^2));
    XXt = X.dot(X.conj().transpose())
    trXXt = (XXt).trace(offset=0)

    #lb=zeros(size(dual_lambda));
    bd = []
    for i in range (0, M):
        bd.append((0,None))
    
    #options = optimset('GradObj','on', 'Hessian','on');

    #%  options = optimset('GradObj','on', 'Hessian','on', 'TolFun', 1e-7);

    #[x, fval, exitflag, output] = fmincon(@(x) fobj_basis_dual(x, SSt, XSt, X, c, trXXt), dual_lambda, [], [], [], [], lb, [], [], options);
    #output = op.fmin_ncg(f=fobj, x0=dual_lambda, fprime=fobj_grad, fhess=fobj_hess, args=(SSt, XSt, X, c, trXXt), maxiter=1000000, avextol=1e-8)
    #output = op.fmin_cg(f=fobj, x0=dual_lambda, fprime=fobj_grad, args=(SSt, XSt, X, c, trXXt), maxiter = 100, full_output = True)
    #output = op.fmin_bfgs(f=fobj, x0=dual_lambda, fprime=fobj_grad, args=(SSt, XSt, X, c, trXXt), gtol = 1e-07, maxiter = 100, full_output = True)
    #output = op.minimize(fun=fobj, x0=dual_lambda, method = 'SLSQP', jac=fobj_grad, bounds = [lb,None], tol = 1e-07, args=(SSt, XSt, X, c, trXXt), options = {'disp' : True, 'maxiter' : 7, 'return_all' : True})
    output = op.fmin_l_bfgs_b(func=fobj, x0=dual_lambda, fprime=fobj_grad, args=(SSt, XSt, X, c, trXXt),bounds = bd, maxiter = 100)
    
    
    #% output.iterations
    #fval_opt = -0.5*N*fval;
    #fval_opt = -0.5*N*output[1]
    dual_lambda = output[0]
    #dual_lambda = output

    #print output
    #print dual_lambda

    #Bt = (SSt+diag(dual_lambda)) \ XSt';
    Bt = la.solve(SSt+np.diag(dual_lambda), XSt.conj().transpose())
    #B_dual= Bt';
    B_dual = Bt.conj().transpose()
    #fobjective_dual = fval_opt;
    #fobjective_dual = fval_opt


    #B= B_dual;
    B = B_dual
    #fobjective = fobjective_dual;
    #fobjective = fobjective_dual
    #toc
    elapsed = time.time() - t0
    print 'time elapsed: ' + str(elapsed)

    return B
	

def fobj(dual_lambda, SSt, XSt, X, c, trXXt):
    #L= size(XSt,1);
    L = XSt.shape[0]
    #M= length(dual_lambda);
    M = np.size(dual_lambda, axis = 0)
    #M = shape[1]

    #SSt_inv = inv(SSt + diag(dual_lambda));
    #SSt_inv = la.inv(SSt + np.diag(dual_lambda[:,0])) #old
    SSt_inv = la.inv(SSt + np.diag(dual_lambda))

    #% trXXt = sum(sum(X.^2));
    if (L>M):
        #% (M*M)*((M*L)*(L*M)) => MLM + MMM = O(M^2(M+L))
        #f = -trace(SSt_inv*(XSt'*XSt))+trXXt-c*sum(dual_lambda);
        f = -(SSt_inv.dot(XSt.conj().transpose().dot(XSt))).trace(offset=0)+trXXt-c*dual_lambda.sum()
        
    else:
        #% (L*M)*(M*M)*(M*L) => LMM + LML = O(LM(M+L))
        #f = -trace(XSt*SSt_inv*XSt')+trXXt-c*sum(dual_lambda);
        f = -(XSt.dot(SSt_inv.dot(XSt.conj().transpose()))).trace(offset=0)+trXXt-c*dual_lambda.sum()
    f *= -1
    #print f
    return f

def fobj_grad(dual_lambda, SSt, XSt, X, c, trXXt):
    #M= length(dual_lambda);
    M = np.size(dual_lambda, axis = 0)
    #M = shape[1]

    #SSt_inv = inv(SSt + diag(dual_lambda));
    #SSt_inv = la.inv(SSt + np.diag(dual_lambda[:,0]))
    SSt_inv = la.inv(SSt + np.diag(dual_lambda))

    #% Gradient of the function evaluated at x
    #g = zeros(M,1);
    g = np.zeros((M,1))
    #temp = XSt*SSt_inv;
    temp = XSt.dot(SSt_inv)
    #g = sum(temp.^2) - c;
    g = (temp**2).sum(axis=0) -c
    g *= -1
    #print g
    return g

def fobj_hess(dual_lambda, SSt, XSt, X, c, trXXt):
    #SSt_inv = inv(SSt + diag(dual_lambda));
    #SSt_inv = la.inv(SSt + np.diag(dual_lambda[:,0]))
    SSt_inv = la.inv(SSt + np.diag(dual_lambda))

    #temp = XSt*SSt_inv;
    temp = XSt.dot(SSt_inv)

    #% Hessian evaluated at x
    #% H = -2.*((SSt_inv*XSt'*XSt*SSt_inv).*SSt_inv);

    # might not work due to numpy/regular array
    #H = -2.*((temp'*temp).*SSt_inv);
    H = -2*((temp.conj().transpose().dot(temp))*SSt_inv)
    H *= -1
    #print H
    return H

