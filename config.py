import numpy as np

class Config(object):
	def __init__(self):

		# set one fo the datasets = True and run practically_fair_classifier

		self.adult =			False  # used for DI 
		self.synthetic = 		False  # used for DI 
		self.saf = 				False # used for DMT
		self.synthetic_dmt =	True  # used for DMT 
		self.synthetic_reg =    False
		self.lamb_validated =   None
		self.classification =   True

		if self.adult:	# By default it is used to show results for DI
			self.fname = '_adult_'
			self.gammas = np.linspace( 0.5, 3, 10)
			self.gammas = np.append(self.gammas, [0]) 
			self.apply_dmt_constraints = 0 # Don't apply DMT 
			self.sensitive_attrs_to_cov_thresh = {"s1":0}
			## for incremental fairness
			#self.benefits_diff_thresh_inc = np.linspace(-0.15, 1, 70) # for 0.5 % 
			# run it with smaller range just negative -0.5 0
			#self.benefits_diff_thresh_inc = np.linspace(-1, 1, 50) # for 0.5 % 
			self.benefits_diff_thresh_inc = np.linspace(-0.5, 0, 30) # for 0.5 % 
			self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.000001, 0.2, 30) #1.5 2.5 3.5 4.5
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.1, 70, 50) # for 3.5 and 4.5 % 
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.001, 10, 30) # for 0.5 % 
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.1, 150, 50) # for 5.5 % 
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.1, 350, 50) # for 6.5 % 
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(250, 500, 50) # for 7.5 %
			#self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.1, 200, 100) # new formulation 200 for 6.5

			self.benefit_slack = 0.0
			self.const_type = -1 
			self.SEEDS = [11223344, 22334455, 33445566, 44556677, 55667788 ]
		
		elif self.synthetic: # used to show results for DI 
			self.fname = "_synthetic_"
			#self.gammas = np.linspace( -1.5, 0.0, 10)
			self.gammas = np.linspace(0.0, 2, 10)
			self.apply_dmt_constraints = 0
			self.sensitive_attrs_to_cov_thresh = {"s1":0}
			#self.benefits_diff_thresh_inc = -1 * np.logspace(np.log10(1e-7), np.log10(0.5), 35)
			#self.benefits_diff_thresh_inc = np.append(self.benefits_diff_thresh_inc,-1 * (1.5 -np.logspace(np.log10(0.5), np.log10(1.0), 35)))
			self.benefits_diff_thresh_inc = -1 * (1.6 -np.logspace(np.log10(0.7), np.log10(1.0), 70))
			
			self.benefits_diff_thresh_inc = np.append(self.benefits_diff_thresh_inc, [0])
			 #np.linspace(-1.0, 0., 70)
			#self.decisions_changed_proxy_params_inc = -1 * np.logspace(np.log10(1e-7),np.log10(0.2), 50)
			self.decisions_changed_proxy_params_inc = -1 * np.logspace(np.log10(0.1),np.log10(.4), 70)
			#np.linspace(0.0,  0.00000001, 50)

			self.const_type = -1
			self.benefit_slack = 0.0
			self.SEEDS = [1125253344, 1122334455,  1122335544, 1122553344, 1155223344]
		
				
		elif self.saf: 
			self.fname = '_saf_'
			
			self.gammas = np.logspace(-2, 0, 20)#np.linspace( 0, 0.5,20)
			self.gammas = np.append(self.gammas, [0]) 
			self.apply_dmt_constraints = 1 # apply DMT constraint. This controls when fairness definition is DI or DMT-FNR
			self.const_type = 2 # 0 = OMR , 1 = FPR, 2 = FNR  only show results for FNR or TPR i.e. 1 - FNR 
			self.sensitive_attrs_to_cov_thresh = {"s1": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}}
			self.benefits_diff_thresh_inc = np.linspace(-.04,0.,100)
			self.decisions_changed_proxy_params_inc = -1 * np.linspace(0,  0.5, 100)
			self.benefit_slack = 0.0
			# for just change 0,  1, 100 for 2;  0,  1, 100 for 3;  0,  1, 100 for 5
			# 0  3 100 for 7  0  10 150 for 9  0  20 250 for 11; 13;25 0,  40, 250 
			self.SEEDS = [1125253344, 1122334455,  1122335544, 1122553344, 1155223344]
			
		
		elif self.synthetic_dmt: # used to show results for DMT

			self.fname = "_synthetic_dmt_"
			#self.gammas = np.logspace( np.log10(0.01), np.log10(300), 40)
			self.gammas = np.logspace(np.log10(0.001), np.log10(30), 35)
			self.gammas = np.append(self.gammas, 330 - np.logspace(np.log10(30), np.log10(300), 35))
			#self.benefits_diff_thresh_inc = np.append(self.benefits_diff_thresh_inc,-1 * (1.5 -np.logspace(np.log10(0.5), np.log10(1.0), 35)))
			#self.gammas = np.linspace( 0,20,30)
			self.gammas = np.append( self.gammas, [0]) 
			self.apply_dmt_constraints = 1
			self.const_type = 2 # 0 = OMR , 1 = FPR, 2 = FNR 
			self.sensitive_attrs_to_cov_thresh = {"s1": {0:{0:0, 1:0}, 1:{0:0, 1:0}, 2:{0:0, 1:0}}}
			self.benefits_diff_thresh_inc = np.linspace(-0.5, 0, 50)
			self.decisions_changed_proxy_params_inc = -1 * np.linspace(0.0,  0.5, 60)
			self.benefit_slack = 0.5 # slack for benefits difference i.e. in loss averse case for some covariance values we cannot solve the problem where benefits increase. 
			self.SEEDS = [1125253344, 1122334455,  1122335544, 1122553344, 1155223344, 2233112, 1212, 11233, 344552, 112234, 11455, 55667, 44556, 567889]
		