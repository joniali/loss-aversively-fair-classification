from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
# from load_credit_card_default_data import *
from generate_synthetic_data_dmt import *
from generate_synthetic_data import *
# from load_german_credit_data import *
from prepare_adult_data import *
from collections import Counter
import matplotlib.pyplot as plt
# from load_compas_data import *
import multiprocessing as mp
from load_sqf_data import *
from config import Config
import loss_funcs as lf
import seaborn as sns
import pandas as pd
import numpy as np
import utils as ut
import matplotlib
import traceback
import random
import os,sys
import time

class practicallyFair(Config):

	def __init__(self, SEED, dataset = None, apply_dmt_constraints = None):

		'''
			Downloads data, initializes x, y, x_control (values of sensitive features), 

		'''
		#datasets = {'adult': False, 'synthetic':False, 'compas':False, 'synthetic_dmt': False} set datasets through arguments 
		#if dataset in datasets:


		super(practicallyFair,self).__init__()
		

		if dataset is not None:
			temp = "self." + dataset
			exec("%s = %d" % (temp,True))
				
		random.seed(SEED) # set the random seed so that the random permutations can be reproduced again
		np.random.seed(SEED)
		self.X_raw = np.array([])
		try:	
			
			data = np.load(self.fname + "{}_.npz".format(SEED))
			self.x_control_train ={}
			self.x_control_test = {}
			self.x_control_validate = {}
			self.X_raw = np.array([])
			self.x_train, self.y_train, self.x_control_train["s1"], self.x_test, self.y_test, self.x_control_test["s1"], self.x_validate, self.y_validate, self.x_control_validate["s1"] = data['x_train'], data['y_train'], data['x_control_train'], data['x_test'],data['y_test'], data['x_control_test'] , data['x_validate'], data['y_validate'], data['x_control_validate'] 
			#print( self.x_train[150] ) 
		
			print ( " *** Loaded data from npz file !!! ") 

		except IOError: 
			'''
				read data and save x_train, y_train, ... in npz file
			'''
			print( "Didn't find saved npz so reading data .. ")

			if self.adult:

				X,y, x_control,self.X_raw = load_adult_data() #
				self.X_raw = np.array(self.X_raw)

				x_control["s1"] = x_control["sex"]
				x_control.pop("sex",None)
							

			elif self.saf: 
				X,y, x_control,self.X_raw = fetch_data_from_db([2012], False, True) 

				x_control["s1"] = x_control["race"]
				x_control.pop("race",None)
				
		
				
			elif self.synthetic:
				X, y, x_control = generate_synthetic_data(plot_data = False) # set plot_data to False to skip the data plot


			elif self.synthetic_dmt: 
				X, y, x_control = generate_synthetic_data_dmt(2, plot_data = False)		
				y = -1 * y # Making FPR -> FNR 
				
			#ut.compute_p_rule(x_control["s1"], y) # compute the p-rule in the original data
			 

			""" Split the data into train and test """
			X = ut.add_intercept(X) # add intercept to X before applying the linear classifier
			
			train_fold_size = 0.7 
			self.x_train, self.y_train, self.x_control_train, self.x_test, self.y_test, self.x_control_test = ut.split_into_train_test(X, y, x_control, train_fold_size)

			train_fold_size = 0.7
			self.x_train, self.y_train, self.x_control_train, self.x_validate, self.y_validate, self.x_control_validate = ut.split_into_train_test(self.x_train, self.y_train, self.x_control_train, train_fold_size)
			
						
			print( "  x_train before saving ", self.x_train.shape) 
			np.savez(self.fname + "{}_.npz".format(SEED), x_train = self.x_train, y_train = self.y_train, x_control_train = self.x_control_train["s1"], x_test = self.x_test, y_test = self.y_test, x_control_test =  self.x_control_test["s1"], x_validate = self.x_validate, y_validate = self.y_validate, x_control_validate = self.x_control_validate["s1"])

			if self.X_raw.size > 0 :
				print ( "saving raw ") 
				np.savez(self.fname + "raw_", X_raw = self.X_raw)
		
				
		self.length = self.x_train.shape[1]
		
		
		self.w_init_sing = np.random.rand(self.length,1)
		
		self.num_points_train, self.num_features = self.x_train.shape
		self.num_points_test, self.num_features = self.x_test.shape
		self.num_points_validate, self.num_features = self.x_validate.shape
		
		self.num_points = self.num_points_train + self.num_points_test + self.num_points_validate
		self.num_points_men = np.sum(self.x_control_train["s1"]) + np.sum(self.x_control_test["s1"]) +  np.sum(self.x_control_validate["s1"])
		self.num_points_women = self.num_points - self.num_points_men

		print( " Y train size :" , self.y_train.shape )
		print(" Number of points in training set are " , self.num_points_train)
		print(" Number of points in validate set are " , self.num_points_validate)
		print(" Number of points in test set are " , self.num_points_test)
		print(" Number of features are " , self.num_features)
	
		print( " men in train: {}".format(np.sum(self.x_control_train["s1"]))) 
		print( " men in test: {}".format(np.sum(self.x_control_test["s1"]))) 
		print( " men in validate: {}".format(np.sum(self.x_control_validate["s1"]))) 

		self.pos_men_train = np.where(np.logical_and(self.x_control_train["s1"] == 1.0,  self.y_train == 1.0))[0].shape[0]
		self.pos_men_test = np.where(np.logical_and(self.x_control_test["s1"] == 1.0,  self.y_test == 1.0))[0].shape[0]
		self.pos_men_validate = np.where(np.logical_and(self.x_control_validate["s1"] == 1.0,  self.y_validate == 1.0))[0].shape[0]
		self.pos_women_train = np.where(np.logical_and(self.x_control_train["s1"] == 0.0,  self.y_train == 1.0))[0].shape[0]
		self.pos_women_test = np.where(np.logical_and(self.x_control_test["s1"] == 0.0,  self.y_test == 1.0))[0].shape[0]
		self.pos_women_validate = np.where(np.logical_and(self.x_control_validate["s1"] == 0.0,  self.y_validate == 1.0))[0].shape[0]
		print (" pos men in train ", self.pos_men_train ) 
		print (" pos men in test ", self.pos_men_test)
		print (" pos men in validate ", self.pos_men_train)

		print (" pos women in train ", self.pos_women_train)
		print (" pos women in test ",  self.pos_women_test)
		print (" pos women in validate ", self.pos_women_validate)
	

		print( " Total pos points ", np.where(self.y_train == 1.0)[0].shape[0] + np.where(self.y_test == 1.0)[0].shape[0] + np.where(self.y_validate == 1.0)[0].shape[0])
		
		
		#print(" Number of women are: ", total_women)
				
		pos_men =   np.where(np.logical_and(self.x_control_train["s1"] == 1.0,  self.y_train == 1.0))[0].shape[0] + np.where(np.logical_and(self.x_control_test["s1"] == 1.0,   self.y_test == 1.0))[0].shape[0] + np.where(np.logical_and(self.x_control_validate["s1"] == 1.0,  self.y_validate == 1.0))[0].shape[0] 

		pos_women = np.where(np.logical_and(self.x_control_train["s1"] == 0.0,  self.y_train == 1.0))[0].shape[0] + np.where(np.logical_and(self.x_control_test["s1"] == 0.0,  self.y_test == 1.0))[0].shape[0] + np.where(np.logical_and(self.x_control_validate["s1"] == 0.0,  self.y_validate == 1.0))[0].shape[0]

				 			
		self.lamb_validated = self.validate_lamb() 


		print( "validated lambda is : ", self.lamb_validated) 
			
	def train_normal_classifier(self,lamb, apply_fairness_constraints):
		
		#assert (lamb_validated is not None )
		if self.classification:
			w, status = ut.train_model_one_shot((self.x_train, self.y_train, self.x_control_train, apply_fairness_constraints, self.apply_dmt_constraints, self.sensitive_attrs_to_cov_thresh, self.const_type, self.w_init_sing, 5000, 1e-6, lamb))
		else:
			w, status = ut.train_model_one_shot_reg((self.x_train, self.y_train, self.x_control_train, apply_fairness_constraints, self.apply_dmt_constraints, self.sensitive_attrs_to_cov_thresh, self.const_type, self.w_init_sing, 5000, 1e-6, lamb))
		
		if apply_fairness_constraints:
			print( " Training with constraints Finished status : ", status) 
		else:
			print( " Training without constraints Finished : ", status) 


	
		return w, status
	
	def train_unfair_fair_clfs(self):

		sensitive_attrs = list(self.x_control_test.keys())
		
		assert( self.lamb_validated is not None)

		print("**************  UNCONSTRAINED  ******")

		apply_fairness_constraints = 0		
		#lamb_validated = 0 
		w_init, status_init = self.train_normal_classifier(self.lamb_validated, apply_fairness_constraints) # w_init is used as the existed classifier 
		

		benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_init, self.x_test, self.y_test, self.x_control_test["s1"], self.apply_dmt_constraints)

		cov_uncons =  ut.calculate_covariance(w_init , self.x_control_train["s1"], self.x_train, self.y_train, self.apply_dmt_constraints)

		
		print ( " On Test data " )
		print()
		print( " Accuracy test ",  accuracy)
		print( " Covariance train " ,  cov_uncons)
		print( " benefits_women ", benefits_women)
		print( " benefits_men ", benefits_men)
		print( " L1 Norm ", np.sum(np.absolute(w_init)))
		print()
		print("STATS on training data")
		benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_init, self.x_train, self.y_train, self.x_control_train["s1"], self.apply_dmt_constraints)
		print("Accuracy : " , accuracy)
		print( " hinge loss", lf._hinge_loss(w_init,self.x_train,self.y_train, self.lamb_validated))
		print()
			
		##### Training Fair classifier ########
		print("**************  CONSTRAINED  ******")

		apply_fairness_constraints = 1				
		
		
		self.sensitive_attrs_to_cov_thresh = ut.update_covariance_threshold(0, sensitive_attrs, self.apply_dmt_constraints)
		


		self.const_type = 2
 		
		w_cons, status_cons = self.train_normal_classifier(self.lamb_validated, apply_fairness_constraints) # change the benefits to accomodate tpr 
		
		benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_cons, self.x_test, self.y_test, self.x_control_test["s1"], self.apply_dmt_constraints)

		cov_cons = ut.calculate_covariance(w_cons , self.x_control_train["s1"], self.x_train, self.y_train, self.apply_dmt_constraints)
		

		print()
		print( " L2 Norm of W ",  np.sqrt(np.dot(w_cons,w_cons)))
		print( " L1 Norm ", np.sum(np.absolute(w_cons)))
		print()
		print( " Accuracy test ",  accuracy)
		print( " Covariance train " ,  cov_cons)
		print( " benefits_women ", benefits_women)
		print( " benefits_men ", benefits_men)

		
		# print(" % decicions changed ", ut.get_percent_decions_changed(self.x_test, w_cons, w_init))

		print()
		print("STATS on training data")
		benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_cons, self.x_train, self.y_train, self.x_control_train["s1"], self.apply_dmt_constraints)
		print("Accuracy : " , accuracy)
		print()
		print( " hinge loss", lf._hinge_loss(w_cons,self.x_train,self.y_train, self.lamb_validated))


		return w_init, status_init, w_cons, status_cons 

	def validate_lamb(self):
		'''
			Helper Function: Solves 
				L + lamb * w 
			for several lamb's and returns one with highest accuracy

		'''
		
		print ( " \n\n  Validating Lambda \n\n" ) 
		
		apply_fairness_constraints = 0
		
				 
		lambs =  np.logspace(np.log10(1e-5), np.log10(1e-2), 5)
		#lambs =  np.linspace(1e-5, 2, 10)
		#lambs = np.append(lambs, [.01, .05, .1, .2, .5, 1., 2., 5., 10., 20., 50., 100. ])
		lambs = np.append(lambs,[0])
		
		max_acc = -1000
		
		sensitive_attrs = list(self.x_control_train.keys())

		for lamb in lambs:
			
						
			w, status = self.train_normal_classifier(lamb, apply_fairness_constraints)
			if self.classification:
				benefits_men, benefits_women , accuracy = ut.calculate_statistics(w, self.x_validate, self.y_validate, self.x_control_validate["s1"], self.apply_dmt_constraints)			
			else:
				accuracy = - ut.get_mse(self.y_validate, np.dot(self.x_validate, w))

			if accuracy > max_acc:
				max_acc = accuracy
				lamb_validated = lamb


			print( " lamb ", lamb , accuracy)

			#benefits_men, benefits_women , accuracy = ut.calculate_statistics(w, self.x_test, self.y_test, self.x_control_test["s1"], self.apply_dmt_constraints)
			#print( " test data accuracy {} \n".format(accuracy))
		return lamb_validated

	## Functions for More benefits
	def search_gamma(self, w_init, SEED):

		'''
		min L
		s.t 
			d-theta_old >= d-theta_new + gamma
			(z- z_bar) d-theta-new <= cov-threshold

		This method searches list of gammas set in config file, for a set covariance threshold. Then from the list of gammas clf which gives us maximum accuracy and whose benefits are higher than the previous clf on validation data, is returned. 
		'''
		print ( " ************* Searching Gamma on validation set ************" )
		
		assert( self.lamb_validated is not None)

		#if self.lamb_validated is None:
		#	lamb_validated = self.validate_lamb()
		#print(self.gammas)
		
		lambs = np.array([self.lamb_validated])
	

		sensitive_attrs = list( self.x_control_validate.keys() ) 

				
		
		cov_thresh = ut.get_unfairness_thresh(self.sensitive_attrs_to_cov_thresh, self.apply_dmt_constraints, self.const_type)

		print( " cov threshold : " , cov_thresh) 
		
		Y = {}
		
		for lam in lambs:

			Z = {}
			try:
				print ( " lam :", lam ) 
				accuracy_list = []
				norm_list = []
				gammas_valid = []
				men_benefits_list = []
				women_benefits_list = []

				data = np.load("img/{}gammas_more_benefits_{}_{}.npz".format(self.fname, cov_thresh, SEED))


				gammas_valid, accuracy_list, norm_list,men_benefits_list, women_benefits_list = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"]
				
			
			except IOError: 
				'''
					read data and save x_train, y_train, ... in npz file
				'''
				print( "Didn't find saved npz so computing data .. ")
				 
				for gamma in self.gammas:

					w_benefits, status = ut.train_model_more_benefits_both_groups(self.x_train, self.y_train, self.x_control_train["s1"],  w_init, lam, self.sensitive_attrs_to_cov_thresh , self.apply_dmt_constraints, self.const_type, gamma)
				
 
					if status == "Converged" or status == "optimal":

						benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_benefits, self.x_validate, self.y_validate, self.x_control_validate["s1"], self.apply_dmt_constraints)	
		
						norm = np.sqrt(np.dot(w_benefits, w_benefits))

						gammas_valid.append(gamma)
						accuracy_list.append(accuracy) 
						norm_list.append(norm)
						men_benefits_list.append(benefits_men)
						women_benefits_list.append(benefits_women)
						
								
				np.savez("img/{}gammas_more_benefits_{}_{}".format(self.fname,cov_thresh, SEED), gammas_valid,  accuracy_list, norm_list, men_benefits_list, women_benefits_list)
				
								
			#print ( Z ) 
			# Z["Gamma"] = gammas_valid 
			# Z["Accuracy"] = accuracy_list
			# Z["Norm"] = norm_list
			# Z["Benefits men"] = men_benefits_list
			# Z["Benefits Women"] = women_benefits_list 
			# Y[" lambda {}".format(lam)] = 	Z

		#print(Z)
		'''
		x_label = "Gamma"
		name = "img/{}gammas_more_benefits_.png".format(self.fname)
		
		ax = ut.plot_shared( Y, x_label,  name )
		'''
		benefits_uncons_men, benefits_uncons_women , accuracy_uncons = ut.calculate_statistics(w_init, self.x_validate, self.y_validate, self.x_control_validate["s1"], self.apply_dmt_constraints)
		'''
		ax[2,0].axhline( y = benefits_uncons_men) 
		ax[3,0].axhline( y = benefits_uncons_women) 
		
		plt.savefig(name, dpi = 400, bbox_inches = "tight") #/compare_loose_fixed_more. No way of making the show() method tight .. have to manually add space for the legend 
		plt.close()
		'''

		return ut.validate_gamma(benefits_uncons_men, benefits_uncons_women, gammas_valid, accuracy_list, men_benefits_list, women_benefits_list, self.benefit_slack)

	def train_more_benefits(self, SEED):

		'''
		min L
		s.t 
			d-theta_old >= d-theta_new + gamma
			(z- z_bar) d-theta-new <= cov-threshold

		second constraint's domain uses only positive samples in the GT for equality of TPR 
		'''
		
		if self.apply_dmt_constraints:
			print(" \n\n\n ********** Training More TPR on {} ********** \n\n\n".format(self.fname))
		else :
			print(" \n\n\n ********** Training More AR on {} ********** \n\n\n".format(self.fname))		


		print ( "\n\n ******* validated lambda is {} \n\n".format(self.lamb_validated))


		w_init, status_init, w_cons, status_cons = self.train_unfair_fair_clfs() 
		
		cov_uncons =  ut.calculate_covariance(w_init , self.x_control_train["s1"], self.x_train, self.y_train, self.apply_dmt_constraints)

		sensitive_attrs = list(self.x_control_test.keys())
		
		apply_fairness_constraints = True

		it = 0.05
		cov_range = np.arange(1.0, 0.0-it, -it).tolist()
		unfairness_list = []
		acc_list = []
		cov_fact_list  = []
		men_benefits_list = []
		women_benefits_list = []
		
		unfairness_list_our = []
		acc_list_our = []
		men_benefits_list_our = []
		women_benefits_list_our = []
		
		cov_fact_list_our = []	
				
		Y = {}
	
		try:
				
			data = np.load("img/{}_{}_more_benefits.npz".format(self.fname, SEED))
			print("Found ", "img/{}_{}_more_benefits.npz".format(self.fname, SEED))

			unfairness_list, acc_list, unfairness_list_our, acc_list_our, cov_fact_list, cov_fact_list_our, men_benefits_list, men_benefits_list_our, women_benefits_list, women_benefits_list_our = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"], data["arr_4"], data["arr_5"], data["arr_6"], data["arr_7"], data["arr_8"], data["arr_9"]
				
		except IOError: 
		
			for factor in cov_range:
				
				print ( "\n\n factor covariance  " , factor ) 
				
				self.sensitive_attrs_to_cov_thresh = ut.update_covariance_threshold(factor*cov_uncons, sensitive_attrs, self.apply_dmt_constraints)

				w_cons, status = self.train_normal_classifier(self.lamb_validated, apply_fairness_constraints)

				
				gamma = self.search_gamma(w_init, SEED)

				if gamma is None:
					continue



				if status == "Converged" or status == "optimal":
					benefits_men, benefits_women , accuracy = ut.calculate_statistics(w_cons, self.x_test, self.y_test, self.x_control_test["s1"], self.apply_dmt_constraints)
					
					unfairness_list.append(abs(benefits_women - benefits_men))
					acc_list.append(accuracy)
					cov_fact_list.append(factor)
					men_benefits_list.append(benefits_men)
					women_benefits_list.append(benefits_women)
								
					
			#for factor in cov_range:
				

					w_more_benefits, status = ut.train_model_more_benefits_both_groups(self.x_train, self.y_train, self.x_control_train["s1"],  w_init, self.lamb_validated, self.sensitive_attrs_to_cov_thresh, self.apply_dmt_constraints, self.const_type,  gamma)
					
					
					if status == "Converged" or status == "optimal":
		
						benefits_men_ours, benefits_women_ours , accuracy_ours = ut.calculate_statistics(w_more_benefits, self.x_test, self.y_test, self.x_control_test["s1"], self.apply_dmt_constraints)

				
						unfairness_list_our.append(abs(benefits_women - benefits_men))
						cov_fact_list_our.append(factor)
						acc_list_our.append(accuracy_ours)
						men_benefits_list_our.append(benefits_men_ours)
						women_benefits_list_our.append(benefits_women_ours)
					
						
						print("Just fairness:  a {}, unfairness {}, Fairness + tpr a: {} unfairness: {} gamma: {}".format(accuracy, abs(benefits_men_ours - benefits_women_ours), accuracy_ours, abs(benefits_women - benefits_men), gamma )) 

				#np.savez("img/{}more_benefits".format(self.fname), unfairness_list, acc_list, unfairness_list_our, acc_list_our, cov_fact_list, cov_fact_list_our, men_benefits_list, men_benefits_list_our, women_benefits_list, women_benefits_list_our) 


		data = np.vstack((cov_fact_list, acc_list , men_benefits_list, women_benefits_list, unfairness_list, cov_fact_list_our, acc_list_our, men_benefits_list_our, women_benefits_list_our, unfairness_list_our))

		print ( data.shape) 
		df = pd.DataFrame(data.T)
		df.to_csv("img/{}_{}_more_benefits.csv".format(self.fname, SEED), sep = '\t', index = False)	
		print ( " saved to csv " ) 	


		## plotting results using function in Utils 
		if False:	
		
			Z = {}	
			Z["Cov_factor"] = cov_fact_list
			Z["Accuracy"] = acc_list
			Z["Men Benefits"] = men_benefits_list
			Z["Women Benefits"] = women_benefits_list 
			
			#Z["Loss"] = loss_list
			#Z["Constraint"] = constraint_list
			Z["Unfairness"] = unfairness_list
			Y["Only Fairness"] = Z	
			Z = {}
			Z["Cov_factor"] = cov_fact_list
			Z["Accuracy"] = acc_list_our
			Z["Men Benefits"] = men_benefits_list_our
			Z["Women Benefits"] = women_benefits_list_our
			Z["Unfairness"] = unfairness_list_our
			Y["Fairness more Benefits"] = Z

			x_label = "Cov_factor"
			name = "img/{}more_benefits_detail.png".format(self.fname)
			ax = ut.plot_shared ( Y, x_label,  name )
		
			plt.gca().invert_xaxis()		
			plt.savefig(name, dpi = 400, bbox_inches = "tight") #/compare_loose_fixed_more. No way of making the show() method tight .. have to manually add space for the legend 
			plt.close()	

		return data.T
	
	
	
		
def train_average_more_benefits():

	# seeds for AR 
	# SEEDS = [11223344, 22334455, 33445566, 44556677, 55667788 ] #These were used for adult and compas 
	# SEEDS = [1125253344, 1122334455,  1122335544, 1122553344, 1155223344] # used for synth and stop question frisk 
	# seeds for dmt 
	# SEEDS = [1125253344, 1122334455,  1122335544, 1122553344, 1155223344, 2233112, 1212, 11233, 344552, 112234, 11455, 55667, 44556, 567889] # seeds synth dmt 
	data_averaged = None
	data_list = []
	data_shape = None
	data_largest = None

	conf = Config()
	SEEDS = conf.SEEDS
	results_count = 0 
	SEEDS_used = []
	for SEED in SEEDS:

		#if os.path.isfile(conf.fname+".npz"):
		#		os.remove(conf.fname + ".npz")

		prac_fair = practicallyFair(SEED)
		data = prac_fair.train_more_benefits(SEED)
		
		
		
		if data_largest is None :
			data_shape = data.shape
			data_largest = data
		elif data_shape[0] <= data.shape[0]:
		 	data_shape = data.shape
		 	data_largest = data

		print(" data shape,", data.shape[0])
		if data.shape[0] == 21:
			results_count+=1
			SEEDS_used.append(SEED)
			data_list.append(data)

		if results_count == 5 :
			break

	data_averaged = np.zeros_like(data_largest)
	#print(" averaged data shape ", data_averaged.shape)
	#print(" list elem shape ", data_list[0].shape)
	index = 0 
	print(SEEDS_used)
	for cov_fact,cov_fact_our in zip(data_largest[:,0],data_largest[:,5]):
		
		num_cov_fact = 0
		print(cov_fact, cov_fact_our)
		
		for j in range(len(data_list)):

			idx = np.where(data_list[j][:,0] == cov_fact) # check in cov list 
			idx_our = np.where(data_list[j][:,5] == cov_fact_our) # check in cov our list
			
			if idx[0].shape[0] == 1  and idx_our[0].shape[0] == 1:
				num_cov_fact += 1 
			elif idx[0].shape[0] == 0  and idx_our[0].shape[0] == 0:
				continue;
				print(idx[0], idx_our[0])
			else:
				raise ValueError(" Something wrong, cov and cov_our are not unique")
			
			assert(idx[0][0] == idx_our[0][0])
			data_averaged[index] += data_list[j][idx][0]

		print(num_cov_fact)
		data_averaged[index] /= num_cov_fact
		print(data_averaged[index])
		index += 1

	df = pd.DataFrame(data_averaged)
	
	name = "img/average_{}_SEEDS_{}_more_benefits.csv".format(conf.fname, SEEDS_used)
	name = name.replace(" ","")
	name = name.replace("[","_")
	name = name.replace("]","_")
	name = name.replace(",","_")
	
	df.to_csv(name, sep = '\t', index = False)	

	if prac_fair.apply_dmt_constraints:
		ylabel = "True positive rate"
		label_text = "EOP"
	else:
		ylabel = "Acceptance rate"
		label_text = "SP"

	print ( " saved to csv " ) 	

	sns.set_style("darkgrid")
	csv_file = pd.read_csv(name, sep = "\t")

	# generate plots might have to change them for different datasets 
	plt.rcParams.update({'font.size': 15})
	# figure(figsize=(8, 6), dpi=80)
	plt.gca().invert_xaxis()
	plt.plot(csv_file["0"],csv_file["2"], color = "orange", lw = 2, linestyle = "-")
	plt.plot(csv_file["0"],csv_file["3"],  color = "blue", lw = 2,linestyle = "-")
	plt.plot(csv_file["5"],csv_file["7"],  color = "orange",lw = 2, linestyle = "--")
	plt.plot(csv_file["5"],csv_file["8"],  color = "blue",lw = 2, linestyle = "--")
	plt.bar(-1,[-0.1],lw = 0, color = "orange", label = "Non-Pro")#, alpha = 0.0)
	plt.bar(-1,[-0.1],lw = 0, color = "blue", label = "Pro")
	plt.plot(csv_file["0"][0],[0.1], color = "black", lw = 2, linestyle = "-", label = label_text)
	plt.plot(csv_file["0"][0],[0.1], color = "black", lw = 2, linestyle = "--", label = f"loss Aversive + {label_text}")


	plt.legend(bbox_to_anchor=(-0.01, 1.28),loc='upper left', fontsize=15, ncol=2)
	# 0.3, 0.9
	cols = ["2","3","7","8"]
	plt.ylim(csv_file[cols].min().min() - 0.05, csv_file[cols].max().max() + 0.05)
	plt.xlim(1.01, -0.05)

	plt.xlabel("Cov. Multiplicative Factor")
	

	plt.ylabel(ylabel)
	name = f'img/{prac_fair.fname}benefits.pdf'
	plt.savefig(name, format="pdf", bbox_inches="tight")
	plt.close()

	plt.rcParams.update({'font.size': 15})
	# figure(figsize=(8, 6), dpi=80)

	plt.gca().invert_xaxis()


	plt.plot(csv_file["0"],csv_file["1"], color = "orange", lw = 2, linestyle = "-", label = label_text)
	plt.plot(csv_file["5"],csv_file["6"],  color = "blue", lw = 2,linestyle = "--", label = f"loss Aversive + {label_text}")

	cols = ["1","6"]

	plt.legend(bbox_to_anchor=(0.07, 1.18),loc='upper left', fontsize=15, ncol=2)
	plt.ylim(csv_file[cols].min().min() - 0.05, csv_file[cols].max().max() + 0.05)
	plt.xlim(1.01, -0.05)

	plt.xlabel("Cov. Multiplicative Factor")
	ylabel = "Accuracy"
	plt.ylabel(ylabel)
	name = f'img/{prac_fair.fname}accuracy.pdf'
	plt.savefig(name, format="pdf", bbox_inches="tight")

if __name__ == '__main__':


	if False: # more Benefits 
		# To select the dataset switch appropriate 
		# boolean values in the config files 
		#  
		SEED = 1122334455
		prac_fair = practicallyFair(SEED)
		prac_fair.train_more_benefits(SEED)

			

	if True:
		# to produce aggregate results over multiple seeds 
		train_average_more_benefits()


	


