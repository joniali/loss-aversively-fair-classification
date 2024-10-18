import os
import sys
import dccp
import traceback
import importlib
import matplotlib
import numpy as np
import cvxpy as cvx 
import loss_funcs as lf # our implementation of loss funcs
from matplotlib import cm 
from copy import deepcopy
matplotlib.interactive = True 
import matplotlib.pyplot as plt
from dccp.problem import is_dccp
from scipy.optimize import minimize # for loss func minimization
from collections import defaultdict
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, Process, Queue
importlib.import_module('mpl_toolkits.mplot3d').Axes3D

### GTK is X 11 supported backend
#matplotlib.use('GTK') ## otherwise get error of QXC connection.. i.e. cannot find appropriate QT so files 
#import math 



def get_unfairness_proxy(w, x, y, x_control,apply_dmt_constraints):
	# returns  z - z_bar * d_theta ; using full or only positive data depending on the definition of fairness. 
	if apply_dmt_constraints:
		unfairness_idx = np.where(y == 1)
	else:
		unfairness_idx = (np.arange(y.shape[0]),)
	

	unfairness_proxy = cvx.abs(cvx.sum(cvx.multiply(x_control[unfairness_idx] - np.mean(x_control[unfairness_idx]), x[unfairness_idx] * w ) ) / unfairness_idx[0].shape[0] )

	return unfairness_proxy

def get_unfairness_thresh( sensitive_attrs_to_cov_thresh, apply_dmt_constraints, const_type):
	
	'''
		Assumes one sensitive attribute 
	'''

	assert(len(list(sensitive_attrs_to_cov_thresh.keys())) == 1)

	if apply_dmt_constraints:
		for sensitive_attr, constraints in sensitive_attrs_to_cov_thresh.items():
			
			unfairness_thresh = constraints[const_type][0]
	else:
		for sensitive_attr, threshold in sensitive_attrs_to_cov_thresh.items():
			unfairness_thresh = threshold

	return unfairness_thresh

def get_hinge_loss(w, x, y, num_points):

	return cvx.sum_entries(cvx.pos(1 - cvx.mul_elemwise(y, x * w))) / num_points

def train_model_more_benefits_both_groups(x, y, x_control, w_unfair, lamb, sensitive_attrs_to_cov_thresh, apply_dmt_constraints,const_type, gamma ): 
	'''
	Assumes 1 binary sensitive feature with multiple 
	min Loss + lamb * w 
	st. 
		B0 = B1 (equal benefits for both values of sensitive feature) 	
		d_theta_men - d_theta_old_men > gamma
		d_theta_women - d_theta_old_women > gamma


	
	'''
	num_points, num_features = x.shape

	w = cvx.Variable(num_features)

	w.value = w_unfair

	constraints = []
	
	
	## Ading following constraints
	## d_theta_men - d_theta_old_men > gamma
	## d_theta_women - d_theta_old_women > gamma

	cov_sum_dict = {}
	cov_sum_dict_old = {}

	for v in set(x_control):
		
		if apply_dmt_constraints:
			# positive incides for FNR 
			idx = np.where(np.logical_and(x_control == v,  y == 1))
		else:
			idx = np.where(x_control == v)
		
		dist_bound_old = np.dot(x[idx], w_unfair)
		dist_bound =  x[idx] * w
		
		cov_sum_dict[v] = cvx.sum(  dist_bound ) / idx[0].shape[0]
		cov_sum_dict_old[v] = np.sum(  dist_bound_old ) / idx[0].shape[0]
	
	for v in set(x_control):

		constraints.append( cov_sum_dict[v] - cov_sum_dict_old[v] >= gamma)


	# adding B0 - B1 <= thresh 
	unfairness_proxy = get_unfairness_proxy(w, x, y, x_control, apply_dmt_constraints)
	unfairness_thresh = get_unfairness_thresh(sensitive_attrs_to_cov_thresh, apply_dmt_constraints, const_type)
	constraints.append( unfairness_proxy <= unfairness_thresh ) 
	
	
	# logistic loss + lamb * || w ||

	loss = cvx.sum( cvx.logistic( cvx.multiply(-y, x * w ))) / num_points
	
	objective = cvx.Minimize(loss + cvx.sum_squares(w[1:]) * lamb) 
	
	prob = cvx.Problem(objective, constraints)

	#print(" Problem is DCP :  " , prob.is_dcp() )
	#print(" Problem is DCCP :  " , is_dccp(prob) )
	try:
		assert(prob.is_dcp() or is_dccp(prob))
		#print(installed_solvers())
		max_iters = 100000
		EPS = 1e-3
		prob.solve (
			#solver = cvx.SCS, verbose=False, use_indirect=True, eps = EPS)
		 solver=cvx.ECOS, verbose= False, 
            feastol=EPS, abstol=EPS, reltol=EPS,feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
             max_iters=max_iters) 
		# #print( " constraints are : ", constraints[0].value) 
		#print( " constraints are : ", constraints[1].value) 

		assert(constraints[0].value(EPS) and constraints[1].value(EPS))

		assert(prob.status == "Converged" or prob.status == "optimal")
	except:
		traceback.print_exc()
		print(prob.is_dcp() , is_dccp(prob) )
		print("loss averse constraint ", constraints[0].value, " fairness ",constraints[1].value )

		#sys.stdout.flush()
		#sys.exit(1)
		print ( "Didn't Converge or Problem isn't DCCP check the traceback" )

		#w.value = np.zeros(x.shape[1])

	
	return np.array(w.value).flatten(), prob.status

def train_model_one_shot(params):

	'''
	The function solves 
	min L(w) st: sum ( z - z_bar) < thresh 
	'''

	x, y, x_control, apply_fairness_constraints, apply_dmt_constraints, sensitive_attrs_to_cov_thresh, const_type, w_init, max_iters, EPS, lam = params 
	num_points, num_features = x.shape

	#print( "   in cvxpy file x.shape: ", num_points , num_features) 

	w = cvx.Variable(num_features)
	
	#np.random.seed(1234)
	w.value = np.random.rand(x.shape[1])

	loss = 0
	
	loss += cvx.sum_squares(w[1:]) * lam

	loss += cvx.sum(cvx.logistic(cvx.multiply(-y, x * w))) / num_points

	#loss += get_hinge_loss(w, x, y, num_points)
	
	#loss += lam * cvx.norm(w[1:], 2)
	
	#loss += cvx.sum_squares(w[1:]) * lam
	
	objective = cvx.Minimize(loss)
	
	constraints = []

	dual_value = None
	
	# original objective should be retrievable 
	
	if apply_fairness_constraints == 1:

		print( " Training with fairness constraints started... " )
		attr = "s1"
		#attr_arr = x_control[attr] #iterate over attributes when there are more than just s1 
		
		# B0 - B1 <= thresh 

		unfairness_proxy = get_unfairness_proxy(w, x, y, x_control[attr],apply_dmt_constraints)

		unfairness_thresh = get_unfairness_thresh ( sensitive_attrs_to_cov_thresh, apply_dmt_constraints, const_type)
		
		print( "unfairness thresh " , unfairness_thresh)

		constraints.append(unfairness_proxy <= unfairness_thresh)
					
		prob = cvx.Problem(objective, constraints)

	else:	
		print( ".........Training without fairness constraints started.........."  ) 
		prob = cvx.Problem(objective)

	#print(" Problem is DCCP :  " , is_dccp(prob) )
	try:
		assert(prob.is_dcp()) # or is_dccp(prob) )
		
		# have to specify dccp 
		tau, mu = 0.5, 1.2
		
		prob.solve(
			solver=cvx.ECOS, verbose= False, 
            feastol=EPS, abstol=EPS, reltol=EPS,feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
            max_iters=max_iters)

		for f_c in constraints:
			try:
				# print(f_c.value, cvx.abs(cvx.sum(cvx.multiply(attr_arr - np.mean(attr_arr), x * w ) ) / num_points ).value)
				# print("proxy value", f_c.value(EPS), unfairness_proxy.value)
				assert(f_c.value(EPS) == True)
				
				#dual_value = 1 / constraints[0].dual_value
				#print( " dual value: " , constraints[0].dual_value)
			except:
				print("Assertion failed. Fairness constraints not satisfied.")
				print(traceback.print_exc())
				sys.stdout.flush()
				return
		
		#sys.stdout.flush()
		#print("   w value is {}  \n\n\n\n\n".format(w.value))	
	
		assert(prob.status == "Converged" or prob.status == "optimal")

	except:
		traceback.print_exc()
		#sys.stdout.flush()
		#sys.exit(1)
		print(" Problem is DCP :  " , prob.is_dcp() )
		print(" Problem is DCCP :  " , is_dccp(prob) )
		#print( sensitive_attrs_to_cov_thresh["s1"], constraints[0].value ) 
		print ( "Didn't Converge or Problem isn't DCCP check the traceback" )
		w.value = np.zeros(x.shape[1])

	

	return np.array(w.value).flatten(), prob.status


def get_data_indices(x_control, y, apply_dmt_constraints ):
	## Returns indicies of data based on sensitive attributes and 
	## fairness notions. 
	##

	if apply_dmt_constraints:
		pos_idx_men =   np.where(np.logical_and(x_control == 1.0, y == 1))
		pos_idx_women = np.where(np.logical_and(x_control == 0.0, y == 1))
	else:
		pos_idx_men =   np.where(x_control == 1.0)
		pos_idx_women = np.where(x_control == 0.0)

	return pos_idx_men, pos_idx_women

def calculate_statistics(w, x, y, x_control, apply_dmt_constraints):

	'''
	Calculates statistics based on fairness notion. Assumes binary sensitive features 
	Returns benefits to men, benefits to women, and accuracy of the clf. 
	'''

	empty_arr = np.array([])
	
	benefits_men = 0 
	benefits_women = 0
	accuracy = 0 

	if w.size == 0 :
		print( " In calculate statistics.. w size is zero " ) 
		
		accuracy = 0 

		return benefits_men, benefits_women , accuracy
	
	#sensitive_attrs = list(x_control.keys())

	#for s  in sensitive_attrs
	if apply_dmt_constraints: 

		s_attr_to_fp_fn = get_fpr_fnr_sensitive_features(w, x, y, x_control, ["s1"]) # assuming s1 to be sensitve attr 
 
		benefits_men = 1.0 - s_attr_to_fp_fn["s1"][1]["fnr"] # assuming binary values of s
		benefits_women = 1.0 - s_attr_to_fp_fn["s1"][0]["fnr"]

	else:

		accepted = get_accepted_groups(w, x, y, x_control)
		benefits_men = accepted[1]
		benefits_women = accepted[0]

	
	return benefits_men, benefits_women, get_accuracy(w, x, y)

def get_accepted_groups(w, x, y, x_control):
	# Assumes binary variables
	#
	accep = {}
	for v in set(x_control):
		idx = np.where(x_control == v )
		accep[v] = np.where(np.dot(x[idx], w) > 0 )[0].shape[0]/idx[0].shape[0]

	return accep 	

def validate_gamma(uncons_men_pos, uncons_women_pos, gammas, accuracy, 
		model_men_pos, model_women_pos, slack = 0.0) :
		'''
			Helper function: given unconstrained benefits and lists of gammas with their corresponding accuracies and benefits, it results gamma which has more benefits than uncons clf and highest accuracy. 
		'''
		
		assert( len( gammas) == len(accuracy) == len( model_men_pos) == len ( model_women_pos)) 

		max_accu = -100

		gamma_validated = None
		#slack = 0.0

		for i in range(len(gammas)):
			#print("g: {}, men_tpr: {}, women_tpr: {}, old_men_tpr: {}, old_women_tpr: {}".format(gammas[i], model_men_pos[i], model_women_pos[i], uncons_men_pos, uncons_women_pos))

			if model_men_pos[i] >= uncons_men_pos and  model_women_pos[i]  >= uncons_women_pos and accuracy[i] >= max_accu :
				print ( " Found a gamma ", gammas[i], accuracy[i] ) 
				gamma_validated = gammas[i]
				max_accu = accuracy[i]

		if gamma_validated is None and False: # if gamma without slack is not found 
			for i in range(len(gammas)):
				#print("g: {}, men_tpr: {}, women_tpr: {}, old_men_tpr: {}, old_women_tpr: {}".format(gammas[i], model_men_pos[i], model_women_pos[i], uncons_men_pos, uncons_women_pos))

				if model_men_pos[i] >= uncons_men_pos - slack and  model_women_pos[i]  >= uncons_women_pos - slack and accuracy[i] >= max_accu :
					print ( " Found a gamma ", gammas[i], accuracy[i] ) 
					gamma_validated = gammas[i]
					max_accu = accuracy[i]

		#if gamma_validated is None:
		#	raise Exception("No gamma was found, Please change the range")
		#else:
		#	print(" gamma: ", gamma_validated)
		return gamma_validated 

def get_fpr_fnr_sensitive_features(w, x, y_true, x_control, sensitive_attrs, verbose = False):



    # we will make some changes to x_control in this function, so make a copy in order to preserve the origianl referenced object
    x_control_internal = deepcopy(x_control)

    s_attr_to_fp_fn = {}
    y_pred = np.sign(np.dot(x, w))
    
    for s in sensitive_attrs:
	    s_attr_to_fp_fn[s] = {}
	    s_attr_vals = x_control_internal # assumes just on sens attr whose control is given 
	    if verbose == True:
	        print("||  s  || FPR. || FNR. || OMR ||")
	    for s_val in sorted(list(set(s_attr_vals))):
	        s_attr_to_fp_fn[s][s_val] = {}
	        y_true_local = y_true[s_attr_vals==s_val]
	        
	       # print( "s ",  s_val , " num " , len(y_true_local))

	        y_pred_local = y_pred[s_attr_vals==s_val]

	        

	        acc = float(sum(y_true_local==y_pred_local)) / len(y_true_local)

	        fp = sum(np.logical_and(y_true_local == -1.0, y_pred_local == +1.0)) # something which is -ve but is misclassified as +ve
	        fn = sum(np.logical_and(y_true_local == +1.0, y_pred_local == -1.0)) # something which is +ve but is misclassified as -ve
	        tp = sum(np.logical_and(y_true_local == +1.0, y_pred_local == +1.0)) # something which is +ve AND is correctly classified as +ve
	        tn = sum(np.logical_and(y_true_local == -1.0, y_pred_local == -1.0)) # something which is -ve AND is correctly classified as -ve

	        all_neg = sum(y_true_local == -1.0)
	        all_pos = sum(y_true_local == +1.0)

	       # print("s ", s_val,  "fp: ", fp, "fn: ", fn, "tp: ", tp,"tn: ", tn )

	        fpr = float(fp) / float(fp + tn)
	        fnr = float(fn) / float(fn + tp)
	        tpr = float(tp) / float(tp + fn)
	        tnr = float(tn) / float(tn + fp)
	        omr = float(fp + fn) / float(tp + fp + fn + tn)


	        s_attr_to_fp_fn[s][s_val]["fp"] = fp
	        s_attr_to_fp_fn[s][s_val]["fn"] = fn
	        s_attr_to_fp_fn[s][s_val]["fpr"] = fpr
	        s_attr_to_fp_fn[s][s_val]["fnr"] = fnr
	        s_attr_to_fp_fn[s][s_val]["omr"] = omr

	        s_attr_to_fp_fn[s][s_val]["acc"] = (tp + tn) / (tp + tn + fp + fn)
	        if verbose == True:
	            if isinstance(s_val, float): # print the int value of the sensitive attr val
	                s_val = int(s_val)
	            print("||  %s  || %0.2f || %0.2f || %0.2f || " % (s_val, fpr, fnr, omr))

        
    return s_attr_to_fp_fn

def get_accuracy(w, x, y):


	"""
	returns the train/test accuracy of the model
	we either pass the model (w)
	else we pass y_predicted
	"""
	
	#if model is not None:
	assert( w.size !=0 ) 

	y_predicted = np.sign(np.dot(x, w))
		
	correct_answers = np.where(y_predicted == y)

	accuracy = float(correct_answers[0].shape[0]) / float(len(y))
	
	#print (" Accuracy ", accuracy)

	return accuracy #, correct_answers

def calculate_covariance(w, x_control, x, y, apply_dmt_constraints):
	##
	## Calculates the covariance z - z_bar * d_theta / num_points 
	## operates on difference indices for dmt assumes fnr 

	if apply_dmt_constraints: # FNR only 

		data_idx = np.where(y == 1) 

	else:

		data_idx = (np.arange(y.shape[0]),) # making a tuple as returned by np.where

	#print( data_idx[0].shape)
	return abs(np.sum((x_control[data_idx] - np.mean(x_control[data_idx]) ) * np.dot(x[data_idx], w))  / data_idx[0].shape[0])

def get_changed_y(y, percent_change):

	y_copy = y.copy()
	length = y_copy.shape[0]
	num_changes = int(length * percent_change / 100 ) 
	print( "lenght ", length, " num changes ", num_changes)
	indices = np.random.randint(length, size = num_changes)
	y_copy[indices] = y_copy[indices] * -1 
	return y_copy , indices

def logistic_loss_l2_reg(w, X, y, lam=None):

	if lam is None:
		lam = 1.0

	yz = y * np.dot(X,w)
	# Logistic loss is the negative of the log of the logistic function.
	logistic_loss = -np.sum(log_logistic(yz))
	l2_reg = (float(lam)/2.0) * np.sum([elem*elem for elem in w])
	out = logistic_loss + l2_reg
	return out

def log_logistic(X):

	""" This function is used from scikit-learn source code. Source link below """

	"""Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
	This implementation is numerically stable because it splits positive and
	negative values::
	    -log(1 + exp(-x_i))     if x_i > 0
	    x_i - log(1 + exp(x_i)) if x_i <= 0

	Parameters
	----------
	X: array-like, shape (M, N)
	    Argument to the logistic function

	Returns
	-------
	out: array, shape (M, N)
	    Log of the logistic function evaluated at every point in x
	Notes
	-----
	Source code at:
	https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
	-----

	See the blog post describing this implementation:
	http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
	"""
	if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
	out = np.empty_like(X) # same dimensions and data types

	idx = X>0
	out[idx] = -np.log(1.0 + np.exp(-X[idx]))
	out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
	return out

def train_sklearn_svm(x, y, x_test = None, y_test= None , x_val = None, y_val= None):
	
	val_range = [.01, .05, .1, .2, .5, 1., 2., 5., 10., 20., 50., 100.]
	acc_max = -1
	w_return = None
	acc_test = -1 
	acc_train = -1 
	for penalty in val_range:
		if True:
			clf = LinearSVC(C=penalty, class_weight=None, dual=True, fit_intercept=True, intercept_scaling=1, loss='hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=0, tol=0.0001,verbose=0)
		if False: 
			clf = SVC(C = penalty, tol = 1e-6)

		clf.fit(x, y)
		clf_acc_train = clf.score(x,y)
		clf_acc_test = clf.score(x_test,y_test)
		clf_acc_val = clf.score(x_val,y_val)
		if clf_acc_val > acc_max :
			acc_max = clf_acc_val
			acc_train = clf_acc_train 
			acc_test = clf_acc_test 
			#w_return = np.array(clf.coef_[0])


	
	
	
	#print(clf.coef_[0].shape)
	#print(" svm accuracy : ", get_accuracy(np.array(clf.coef_[0]), x, y), clf.score(x,y))
	#print(" svm accuracy : ", clf.score(x,y))
	#print(" svm accuracy : ", clf2.score(x,y))
	#return w_return #np.array(clf.coef_[0])
	#return clf.score(x,y), clf.score(x_test,y_test), clf.score(x_val,y_val)
	return acc_train, acc_test , acc_max

def update_covariance_threshold( threshold, sensitve_attr, apply_dmt_constraints):
	# 
	# gives dict of sensitive variable to cov thresh based on 
	# type of fair system. 
	assert(len(sensitve_attr) == 1)
	if apply_dmt_constraints:
				
		return {sensitve_attr[0]: {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:threshold, 1:threshold}}}
	else:

		return {sensitve_attr[0]: threshold }

def add_intercept(x):

	""" Add intercept to the data before linear classification """
	m,n = x.shape
	intercept = np.ones(m).reshape(m, 1) # the constant b
	return np.concatenate((intercept, x), axis = 1)

def split_into_train_test(x_all, y_all, x_control_all, train_fold_size):

	
	split_point = int(round(float(x_all.shape[0]) * train_fold_size))
	x_all_train = x_all[:split_point]
	x_all_test = x_all[split_point:]
	y_all_train = y_all[:split_point]
	y_all_test = y_all[split_point:]
	x_control_all_train = {}
	x_control_all_test = {}
	for k in x_control_all.keys():	
		x_control_all_train[k] = x_control_all[k][:split_point]
		x_control_all_test[k] = x_control_all[k][split_point:]
	
	return x_all_train, y_all_train, x_control_all_train, x_all_test, y_all_test, x_control_all_test

def get_line_coordinates(w, x1, x2):

	 y1 = (-w[0] - (w[1] * x1)) / w[2]
	 y2 = (-w[0] - (w[1] * x2)) / w[2]
	 return y1,y2


def plot_shared ( Y, x_label, fname ):
	
	'''
	arguments:
	plot_labels: of different types of plots ( e.g. diff lambda) 
	X : a shared X -axis 
	Y : a dictionary of dictionaries first key corresponds to the plot labels whose keys are used as the labels of y-axis and value are list of numbers to be ploted 
	x_label: 
	fname : to save 
	'''
	
	#lamb_dict = sorted(list(Y.keys())
	plot_labels = list(Y.keys()) # number of different labels: labels are lambda = xxx

	#print(plot_labels)

	Z = Y[plot_labels[0]]
	
	
	key_list  =  list(Z.keys()) # as dictionary access might give different key everytime we do key in Y so we make a list and access in same order
	num_plots = len(key_list)   # different types of plots 

	print(" key list ", key_list) 

	#assert( num_plots == len(plot_labels))

	f, ax = plt.subplots(num_plots -1,sharex=True,figsize = (8, 6.5), squeeze = False) # num_plots - x_axis  squeez false such that when axis = 1 it still allows indexing
	
	min_z =  1000 
	max_z = -1000
	Z_min = {key:  1000 for key in key_list}
	Z_max = {key: -1000 for key in key_list}

	for pl in plot_labels: 

		Z = Y[pl]
		c = np.random.rand(3,)
		i = 0

		for key in key_list:
			
			if key == x_label:
				#print ("skipping",  key ) 
				continue
			if key not in Z:
				i+=1 
				continue 
			#print( key ) 
			min_curr = min(Z[key])
			max_curr = max(Z[key])

			if min_curr < Z_min[key]:
				Z_min[key] = min_curr

			if max_curr > Z_max[key]:
				Z_max[key] = max_curr

			
			if i == 0:
				ax[i,0].plot(Z[x_label], Z[key]  , 'o', c = c, label = pl, alpha = 0.5 )
			else:
				ax[i,0].plot(Z[x_label], Z[key]  , 'o', c = c, alpha = 0.5 )
			i+=1

	ax[0,0].legend( loc= 'upper left', fontsize=15, bbox_to_anchor = (0,0.9 + num_plots * 0.2)) 
	i = 0
	plt.xlabel(x_label)
	for key in key_list:

		if key == x_label:
			#print ("skipping",  key ) 
			continue

		ax[i,0].set_ylabel(key)
		#print ( key , Z_min[key], Z_max[key])
		if Z_max[key] - Z_min[key] == 0:
			i+=1
			continue			

		ax[i,0].set_yticks(np.linspace( Z_min[key] , Z_max[key] , 5))
		#ax[i].set_yticks(np.arange(0,100))
		i+=1

	f.tight_layout()
	
	#plt.savefig(fname, dpi = 400, bbox_inches = "tight") #/compare_loose_fixed_more. No way of making the show() method tight .. have to manually add space for the legend 
	
	#plt.close()
	
	return ax 

def plot_3D ( data, grid_label ):
	
	'''
	arguments:
	grid_label: x, y string labels 
	
	data : dictionary with key as labels and values list of data 
	
	fname : to save 
	'''
	
	#lamb_dict = sorted(list(Y.keys())
	

	x_axis = data[grid_label[0]]
	y_axis = data[grid_label[1]]
	#X, Y  = np.meshgrid(x_axis, y_axis) There might of epsilon for some lambda that might not converge 
	data.pop(grid_label[0],None)
	data.pop(grid_label[1],None)
	
	plot_labels = list(data.keys()) # number of different labels 

	print(plot_labels)

	num_plots = len(plot_labels)   # number of plots - 2 D grid 

	
	for key in data:
		c = np.array([1,0,0])#np.random.rand(3,)
		fig = plt.figure()
		ax = fig.gca(projection= '3d') 
		ax.set_xlabel(grid_label[0])
		ax.set_ylabel(grid_label[1])
		ax.set_zlabel(key)
		
		Z = data[key]
		
		#surf = ax.plot_surface(X,Y,
		ax.scatter(x_axis, y_axis, Z, c = c) 

	plt.show()
	print( " \n\n After plt Show \n\n" ) 
	plt.close()
	plt.close("all")
	#close('all')
	#return ax 


