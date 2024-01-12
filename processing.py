import numpy as np
from sklearn.linear_model import HuberRegressor
import config
from multiprocessing import Pool
from visualize import plot3d_fit
from scipy import linalg
import matplotlib.pyplot as plt
import os
import scipy.io as scio
from scipy import optimize



def processing(data, window, dist_x, dist_y, corr_res, sr, mode="processing"):
    """"
    Function for parallel array analysis.
    Input:
        data: traces (array)
        time: time (array)
        window: length of the window in samples (int)
        dist_x: matrix containing all distances between the stations in x direction (array)
        dist_y: matrix containing all distances between the stations in y direction (array)
        corr_res: resolution of the cross correlation (int)
        sr: sampling rate (int)
        mode: "processing", "testing_inversion", "testing all" (string)
    Output:
        v_app: apparent velocity (array)
        BAZ: Back azimuth (array) 
        rmse: Root mean square error (array) 
        max_d: Max value of each window (array)
        median_c: the median of the cross correlation matrix (array)
        
    """
    # Preallocating arrays
    step_length = np.arange(0, data.shape[1] - window + 1, corr_res)
    BAZ = np.zeros(len(step_length))
    v_app = np.zeros(len(step_length))
    rmse = np.zeros(len(step_length))
    median_c = np.zeros(len(step_length))
    max_d = np.zeros(len(step_length))
    
    # Creating input list for parallel processing function
    if mode == "processing":
        args_list = [(i,data, window, dist_x, dist_y, step_length, sr, mode) for i in range(0,len(step_length))]
    elif mode == "testing_inversion" or mode == "testing_all":
        args_list = [(i,data, window, dist_x, dist_y, step_length, sr, mode) for i in range(0,1)]

    # Parallel processing
    with Pool() as pool:
        results = pool.map(wrapper_function, args_list)

    # Unpack results from parallel processing output
    for i,entry in enumerate(results):
        v_app[i] = entry[0]
        BAZ[i] = entry[1]
        rmse[i] = entry[2]
        max_d[i] = entry[3]
        median_c[i] = entry[4]

    return median_c, max_d, rmse, v_app, BAZ

def processing_parallel(args):
    """"
    Main parallel processing function
    """
    i, data, window, dist_x, dist_y, step_length, sr, mode = args

    # Cut data and time to window size
    d = data[:,step_length[i]:step_length[i]+window]

    # Calculate cross correlation
    if mode == "processing":
        [c, median_c] = cross_corr(d, config.max_delay, sr)
    elif mode == "testing_inversion":
        [c, median_c] = generate_sythetic_delay(dist_x, dist_y)
    elif mode == "testing_all":
        [c_org, median_c] = generate_sythetic_delay(dist_x, dist_y)
        d = generate_synthetic_data(c_org, sr, window)
        [c, median_c] = cross_corr(d, config.max_delay, sr)
        print("median_c: " + str(median_c))

    # Calculate max value
    max_d = np.mean(np.max(d,1))

    if median_c > config.c_median_threshold and max_d > config.amp_threshold:

        # Prepare input arrays for Regression
        delays = -1.0 * c

        # Create a boolean matrix with the independent entries
        independent = np.tri(c.shape[0], dtype=bool, k=1)

        # Create a condition for the non-NaN delays
        valid_delays = ~np.isnan(delays)

        # Apply the condition to the delays and distances
        tau = delays[valid_delays & independent].T
        A = np.vstack([-dist_x[valid_delays & independent], -dist_y[valid_delays & independent]]).T

        # Start Regression
        if config.inversion_method == "Huber":
            mdl = HuberRegressor(epsilon=config.Huber_epsilon).fit(A, tau)
            rmse = np.sqrt(np.mean((tau - mdl.predict(A))**2))
            beta = [mdl.coef_[0], mdl.coef_[1]]
        elif config.inversion_method == "Tikhonov":
            mdl = TikhonovInversion(alpha=config.Tikhonov_alpha, k=config.Tikhonov_k)
            mdl.fit(A, tau)
            beta = mdl.invert(tau)
            rmse = np.sqrt(np.mean((tau - mdl.predict(A))**2))

        if mode == "testing_inversion" or mode == "testing_all":
            print("rmse: " + str(np.round(rmse,2)))
        
        if rmse < config.rmse_threshold:

            # Calculate apparent velocity and backazimuth
            v_app = np.sqrt(1.0 / (beta[0] ** 2 + beta[1] ** 2))
            BAZ = np.arctan2(beta[1], beta[0])
            if mode == "testing_inversion" or mode == "testing_all":
                print("v_app: " + str(np.round(v_app,2)) + " km/s")
                print("BAZ: " + str(np.round(np.rad2deg(BAZ),2)) + "°")
            if config.fit_plot_flag:
                plot3d_fit(A, tau, mdl, rmse, BAZ)
            
        else:
            v_app = np.nan
            BAZ = np.nan
    else:
        v_app = np.nan
        BAZ = np.nan
        rmse = np.nan
    return v_app, BAZ, rmse, max_d, median_c

def wrapper_function(args):
    return processing_parallel(args)

def cross_corr(A, maxlag, sr):
    num = np.size(A, 0)
    delay = np.zeros((num, num))
    c_max = np.zeros((num, num))

    for i in range(num):
        mean_A_i = np.mean(A[i, :])

        for k in range(num):
            mean_A_k = np.mean(A[k, :])

            # Calculate the cross-correlation using numpy's correlate function
            cross_corr_result = np.correlate(A[i, :] - mean_A_i, A[k, :] - mean_A_k, mode='full')

            # Calculate the normalization factors
            normalization = np.sqrt(np.sum((A[i, :] - mean_A_i)**2) * np.sum((A[k, :] - mean_A_k)**2))

            # Calculate the normalized cross-correlation
            normalized_corr = cross_corr_result / normalization

            # Cut the cross-correlation function at the maximum lag
            maxlag_samples = int(maxlag * sr)
            normalized_corr = normalized_corr[len(A[i, :]) - 1 - maxlag_samples:len(A[i, :]) + maxlag_samples]

            c_max[i, k] = np.max(normalized_corr)
            c_ind = np.argmax(normalized_corr) - (len(normalized_corr) - 1) / 2
            delay[i, k] = c_ind / sr

    c_max[c_max == 1] = np.nan
    c_median = np.nanmedian(c_max)

    return delay, c_median

def invert(A, b, k, l):

	u, s, v = linalg.svd(A, full_matrices=False) #compute SVD without 0 singular values

	#number of `columns` in the solution s, or length of diagnol
	
	S = np.diag(s)
	sr, sc = S.shape          #dimension of

	for i in range(0,sc-1):
		if S[i,i]>0.00001:
			S[i,i]=(1/S[i,i]) - (1/S[i,i])*(l/(l+S[i,i]**2))**k

	x1=np.dot(v.transpose(),S)    #why traspose? because svd returns v.transpose() but we need v
	x2=np.dot(x1,u.transpose())
	x3=np.dot(x2,b)

	return x3

def generate_sythetic_delay(dist_x, dist_y):
    """""
    Function for testing the inversion methods. Generates synthetic delay times.
    """""
    # Generate random backazimuth (-180-180°)
    BAZ = np.random.uniform(-180, 180)

    # Generate random apparent velocity (0-20 km/s)
    v_app = np.random.uniform(0, 20)

    print("Generating synthetic data with BAZ = " + str(np.round(BAZ,2)) + " ° and v_app = " + str(np.round(v_app,2)) + " km/s")

    # Calculate distances in BAZ direction
    dist = dist_x * np.cos(np.deg2rad(BAZ)) + dist_y * np.sin(np.deg2rad(BAZ))

    # Calculate delay times
    delay = dist / v_app

    return delay, 1

def generate_synthetic_data(delay, sr, window, noise_level=0.2):
    """
    Function for generating synthetic time series, with a given delay time.
    """
    # Calculate number of samples
    samples = int(window * sr)

    # Preallocate data array
    data = np.zeros((len(delay), samples))

    # Generate gaussian pulse
    amp = np.exp(-np.linspace(-25, 25, samples) ** 2)
    
    # Generate synthetic data
    noise_level = noise_level * np.max(amp)
    for i in range(len(delay)):
        data[i,:] = np.roll(amp, int(delay[i, 0] * sr)) + np.random.normal(0, noise_level, samples)

    return data

class TikhonovInversion:
    def __init__(self, alpha=1.0, k=1.0):
        self.alpha = alpha
        self.k = k
        self.u = None
        self.s = None
        self.v = None
        self.x = None
        self.max_s = None
        self.S_inv = None

    def fit(self, A, b):
        # Check input dimensions
        if A.shape[0] != b.shape[0]:
            raise ValueError("Matrix dimensions are not compatible for inversion.")

        self.u, self.s, self.v = self.csvd(A)

        # Compute the maximum singular value
        self.max_s = np.max(self.s)

        # Construct the diagonal matrix with modified singular values
        S = np.diag(self.s)
        sr, sc = S.shape 

        # Find the optimal regularization parameter (alpha)
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            self.alpha = self.alpha
        elif self.alpha == "gcv":
            self.alpha = self.gcv(A, b, alpha_values = np.logspace(-5, 3, 20))
        elif self.alpha == "l_curve":
            self.alpha = self.l_curve(self.u, self.s, b)

        for i in range(0,sc-1):
            if S[i,i]>0.00001:
                S[i,i]=(1/S[i,i]) - (1/S[i,i])*(self.alpha/(self.alpha+S[i,i]**2))**self.k

        self.S_inv = S

    def invert(self, b):
        """
        Predict the solution for the given data.
        """
        self.x = np.dot(self.v.T, np.dot(self.S_inv, np.dot(self.u.T, b)))
        return self.x
    
    def predict(self, A):
        """"
        solving Ax = b for b
        """ 
        return np.dot(A, self.x)
    
    
    def gcv(self, X, y, alpha_values):
        """
        Calculate the regularization parameter using Generalized Cross Validation (GCV).

        Parameters:
        - X: Input matrix (design matrix)
        - y: Target values
        - alpha_values: List of regularization parameter values to consider

        Returns:
        - Best regularization parameter according to GCV
        """

        n, p = X.shape
        best_alpha = None
        min_gcv = float('inf')

        for alpha in alpha_values:
            # Calculate the Tikhonov regularization matrix
            # regularization_matrix = alpha * np.dot(self.v.T / (self.s**2 + alpha), self.u.T)
            regularization_matrix = alpha * np.identity(p)

            # Calculate the coefficient matrix
            coefficient_matrix = np.linalg.inv(X.T @ X + regularization_matrix) @ X.T

            # Calculate the predicted values
            predicted_values = X @ coefficient_matrix @ y

            # Calculate the residual vector
            residual = y - predicted_values

            # Calculate the trace of the hat matrix
            trace_hat_matrix = np.trace(X @ coefficient_matrix)

            # Calculate the Generalized Cross Validation (GCV) score
            gcv = np.linalg.norm(residual) ** 2 / ((n - trace_hat_matrix) ** 2)

            # Update the best regularization parameter if needed
            if gcv < min_gcv:
                min_gcv = gcv
                best_alpha = alpha

        return best_alpha

    def curvature(self, lambd, sig, beta, xi):
        '''
        computes the NEGATIVE of the curvature. Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
        '''
        # Initialization.
        phi = np.zeros(lambd.shape)
        dphi = np.zeros(lambd.shape)
        psi = np.zeros(lambd.shape)
        dpsi = np.zeros(lambd.shape)
        eta = np.zeros(lambd.shape)
        rho = np.zeros(lambd.shape)
        if len(beta) > len(sig): # A possible least squares residual.
            LS = True
            rhoLS2 = beta[-1] ** 2
            beta = beta[0:-2]
        else:
            LS = False
        # Compute some intermediate quantities.
        for jl, lam in enumerate(lambd):
            f  = np.divide((sig ** 2), (sig ** 2 + lam ** 2)) # ok
            cf = 1 - f # ok
            eta[jl] = np.linalg.norm(f * xi) # ok
            rho[jl] = np.linalg.norm(cf * beta)
            f1 = -2 * f * cf / lam 
            f2 = -f1 * (3 - 4*f)/lam
            phi[jl]  = np.sum(f*f1*np.abs(xi)**2) #ok
            psi[jl] = np.sum(cf*f1*np.abs(beta)**2)
            dphi[jl] = np.sum((f1**2 + f*f2)*np.abs(xi)**2)
            dpsi[jl] = np.sum((-f1**2 + cf*f2)*np.abs(beta)**2) #ok

        if LS: # Take care of a possible least squares residual.
            rho = np.sqrt(rho ** 2 + rhoLS2)

        # Now compute the first and second derivatives of eta and rho
        # with respect to lambda;
        deta  =  np.divide(phi, eta) #ok
        drho  = -np.divide(psi, rho)
        ddeta =  np.divide(dphi, eta) - deta * np.divide(deta, eta)
        ddrho = -np.divide(dpsi, rho) - drho * np.divide(drho, rho)

        # Convert to derivatives of log(eta) and log(rho).
        dlogeta  = np.divide(deta, eta)
        dlogrho  = np.divide(drho, rho)
        ddlogeta = np.divide(ddeta, eta) - (dlogeta)**2
        ddlogrho = np.divide(ddrho, rho) - (dlogrho)**2
        # curvature.
        curv = - np.divide((dlogrho * ddlogeta - ddlogrho * dlogeta),
            (dlogrho**2 + dlogeta**2)**(1.5))
        return curv

    def l_corner(self, rho,eta,reg_param,u,sig,bm):
        '''
        computes the corner of the L-curve.
        Inputs:
            rho, eta, reg_param - computed in l_curve function
            u left side matrix computed from svd (size: Nm x Nm) - Nm is the number of measurement points
            sig is the singular value vector of A
            bm is the measured results
        A is of Nm x Nu, where Nm are the number of measurements and Nu the number of unknowns
        Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
        '''
        # Set threshold for skipping very small singular values in the
        # analysis of a discrete L-curve.
        s_thr = np.finfo(float).eps # Neglect singular values less than s_thr.
        # Set default parameters for treatment of discrete L-curve.
        deg   = 2  # Degree of local smooting polynomial.
        q     = 2  # Half-width of local smoothing interval.
        order = 4  # Order of fitting 2-D spline curve.
        # Initialization.
        if (len(rho) < order):
            print('I will fail. Too few data points for L-curve analysis')
        Nm, Nu = u.shape
        p = sig.shape
        # if (nargout > 0), locate = 1; else locate = 0; end
        beta = (np.conjugate(u)) @ bm
        beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
        b0 = (bm - (beta.T @ u).T)#u @ beta
        # s = sig
        xi = np.divide(beta[0:int(p[0])], sig)
        # Call curvature calculator
        curv = self.curvature(reg_param, sig, beta, xi) # ok
        # Minimize 1
        # reg_c = optimize.fmin(curvature, 0.0, args = (sig, beta, xi), full_output=False, disp=False)
        # Minimize 1
        curv_id = np.argmin(curv)
        x1 = reg_param[int(np.amin([curv_id, len(curv)]))]
        x2 = reg_param[int(np.amax([curv_id, 0]))]
        reg_c = optimize.fminbound(self.curvature, x1, x2, args = (sig, beta, xi), full_output=False, disp=False)
        kappa_max = - self.curvature(reg_c, sig, beta, xi) # Maximum curvature.
        if kappa_max < 0:
            lr = len(rho)
            reg_c = reg_param[lr]
            rho_c = rho[lr]
            eta_c = eta[lr]
        else:
            f = np.divide((sig**2), (sig**2 + reg_c**2))
            eta_c = np.linalg.norm(f * xi)
            rho_c = np.linalg.norm((1-f) * beta[0:len(f)])
            if Nm > Nu:
                rho_c = np.sqrt(rho_c ** 2 + np.linalg.norm(b0)**2)
        return reg_c

    # reg_param[int(np.amin([curv_id+1, len(curv)])])

        # reg_c = fminbnd('lcfun',...
        # reg_param(min(gi+1,length(g))),reg_param(max(gi-1,1)),...
        # optimset('Display','off'),s,beta,xi); % Minimizer.
        

    def csvd(self, A):
        '''
        computes the svd based on the size of A.
        Input:
            A is of Nm x Nu, where Nm are the number of measurements and Nu the number of unknowns
        Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
        '''
        Nm, Nu = A.shape
        if Nm >= Nu: # more measurements than unknowns
            u, sig, v = linalg.svd(A, full_matrices=False)
        else:
            v, sig, u = linalg.svd(np.conjugate(A.T), full_matrices=False)
        return u, sig, v

    def l_curve(self, u, sig, bm, plotit = False):
        '''
        Plot the L-curve and find its "corner".
        Adapted from Per Christian Hansen, DTU Compute, October 27, 2010.
        Inputs:
            u: left side matrix computed from svd (size: Nm x Nm) - Nm is the number of measurement points
            sig: singular values computed from svd (size: Nm x 1)
            bm: your measurement vector (size: Nm x 1)
        '''
        # Set defaults.
        npoints = 200  # Number of points on the L-curve
        smin_ratio = 16*np.finfo(float).eps  # Smallest regularization parameter.
        # Initialization.
        Nm, Nu = u.shape
        p = sig.shape
        print('p {}'.format(p))
        print('bm {}'.format(bm.shape))
        print('u {}'.format(u.shape))
        # if (nargout > 0), locate = 1; else locate = 0; end
        beta = np.conjugate(u) @ bm
        beta2 = np.linalg.norm(bm) ** 2 - np.linalg.norm(beta)**2
        # if ps == 1:
        s = sig
        beta = np.reshape(beta[0:int(p[0])], beta.shape[0])
        xi = np.divide(beta[0:int(p[0])],s)
        xi[np.isinf(xi)] = 0

        eta = np.zeros((npoints,1))
        # print('eta {}'.format(eta.shape))
        rho = np.zeros((npoints,1)) #eta
        reg_param = np.zeros((npoints,1))
        s2 = s ** 2
        reg_param[-1] = np.amax([s[-1], s[0]*smin_ratio])
        ratio = (s[0]/reg_param[-1]) ** (1/(npoints-1))
        # print('ratio {}'.format(ratio))
        for i in np.arange(start=npoints-2, step=-1, stop = -1):
            reg_param[i] = ratio*reg_param[i+1]
        for i in np.arange(start=0, step=1, stop = npoints):
            f = s2 / (s2 + reg_param[i] ** 2)
            eta[i] = np.linalg.norm(f * xi)
            rho[i] = np.linalg.norm((1-f) * beta[:int(p[0])])
        if (Nm > Nu and beta2 > 0):
            rho = np.sqrt(rho ** 2 + beta2)
        # want to plot the L curve?
        if plotit:
            plt.loglog(rho, eta)
            plt.xlabel('Residual norm ||Ax - b||')
            plt.ylabel('Solution norm ||x||')
            plt.savefig(os.path.join(config.folder_results, 'L_curve.png'))
        # Compute the corner of the L-curve (optimal regularization parameter)
        lam_opt = self.l_corner(rho,eta,reg_param,u,sig,bm)
        return lam_opt 
