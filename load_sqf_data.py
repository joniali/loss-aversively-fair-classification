import numpy as np
from random import shuffle, seed
from collections import defaultdict, Counter
import sqlite3
from sklearn import preprocessing
import sys
from sklearn.preprocessing import MaxAbsScaler

#SEED = 1234
#seed(SEED)
#np.random.seed(SEED)

def fetch_data_from_db(years, USE_SENS_CLASSIFICATION, USE_BALANCED_CLASSES):

    SQLITE_FILES_DIR = "/NS/twitter-6/work/mzafar/ComputationalDiscrimination/SQF_analysis/sqlite_data/"
    print("\n\n\n")

    filter_str = """ (lower(crimsusp) LIKE '%cpw%' OR lower(crimsusp) LIKE '%c.p.w.%' OR lower(crimsusp) like '%weapon%' OR replace(detailcm, ' ', '')='20') and race in ('B', 'W')"""
    

    REASONS_STOP = ["cs_objcs" , "cs_descr" , "cs_casng" , "cs_lkout" , "cs_cloth" , "cs_drgtr" , "cs_furtv" , "cs_vcrim" , "cs_bulge" , "cs_other"]


    # need to do one hot encoding for these
    # CATEGORICAL_VARIABLES = ['sex', 'race', 'build', 'trhsloc', 'inout', 'datestop', 'timestop', 'radio', 'offunif', 'pct'] + REASONS_STOP
    # CATEGORICAL_VARIABLES = ['sex', 'race', 'build', 'trhsloc', 'inout', 'radio', 'offunif', 'pct'] + REASONS_STOP
    CATEGORICAL_VARIABLES = ['sex', 'race', 'build', 'trhsloc', 'radio', 'offunif'] + REASONS_STOP
    # CATEGORICAL_VARIABLES = ['sex', 'race', 'build', 'trhsloc', 'radio', 'offunif'] + ['cs_bulge', 'cs_descr', 'cs_objcs', 'cs_other', 'cs_vcrim']

    # need to to mean/median normalization for these
    # CONT_VARIABLES = ['year', 'ht_feet', 'ht_inch', 'age', 'perobs']
    CONT_VARIABLES = ['ht_feet', 'ht_inch', 'age', 'perobs']
    DECISION_VARIABLES = ['pistol', 'riflshot', 'asltweap', 'knifcuti', 'machgun', 'othrweap']
    ALL_VARIABLES = CATEGORICAL_VARIABLES + CONT_VARIABLES + DECISION_VARIABLES

    VAL_CONSTRAINTS = {}
    VAL_CONSTRAINTS["datestop"] = ["datestop not like '1900-%'"]
    

    SENSITIVE_ATTRS = ['race']


    num_all_stopped = 0
    num_cpw_susp = 0
    num_cpw_complete_fields = 0
    features_to_vals = defaultdict(list)
   
    for year in years:
        
        # open the DB connection
        conn = sqlite3.connect("%s/%d.sqlite3" % (SQLITE_FILES_DIR, year))
        cursor = conn.cursor()

        # all the stops made in this period
        query = "SELECT count(*) FROM sqf_data_%d;" % (year)
        cursor.execute(query)
        res = cursor.fetchone()
        num_all_stopped += res[0]

        # all the stops made in this period
        query = """SELECT count(*) FROM sqf_data_%d WHERE %s;""" % (year, filter_str)
        cursor.execute(query)
        res = cursor.fetchone()
        num_cpw_susp += res[0]
        

        attr_list = ",".join(CATEGORICAL_VARIABLES + CONT_VARIABLES)
        empty_constraint = ""
        empty_constraint += " != '' AND ".join(ALL_VARIABLES) + " != ''"
        empty_constraint += " AND "
        empty_constraint += " != ' ' and ".join(ALL_VARIABLES) + " != ' '"
        empty_constraint += " AND "
        empty_constraint += " != 'X' and ".join(ALL_VARIABLES) + " != 'X'" # unknown values
        empty_constraint += " AND "
        empty_constraint += " != 'Z' and ".join(ALL_VARIABLES) + " != 'Z'" # other values
        empty_constraint += " AND "
        empty_constraint += " != 'U' and ".join(ALL_VARIABLES) + " != 'U'" # other values

        query = "SELECT %s FROM sqf_data_%d WHERE %s and %s" % (",".join(ALL_VARIABLES), year, filter_str, empty_constraint)
        cursor.execute(query)
        res = cursor.fetchall()
        num_cpw_complete_fields += len(res)
        for r in res:
            for i in range(0,len(ALL_VARIABLES)):
                attr = ALL_VARIABLES[i]
                val = r[i]
                features_to_vals[attr].append(val)

       
        conn.close()

    print("\n\n")
    print("All people stopped from %d to %d: %d" % (years[0], years[-1], num_all_stopped))
    print("# CPW suspects stopped from %d to %d (including records with missing fields): %d" % (years[0], years[-1], num_cpw_susp))
    print("# CPW suspects stopped from %d to %d (excluding records with missing fields): %d" % (years[0], years[-1], num_cpw_complete_fields))
    

    features_to_vals = dict(features_to_vals)

    """ process features here """

    # process the two height variables
    height_arr = []
    for i in range(0,num_cpw_complete_fields):
        ht_f = int(features_to_vals["ht_feet"][i])
        ht_i = int(features_to_vals["ht_inch"][i])
        ht = ( ht_f * 12 )+ ht_i
        height_arr.append(ht)

    # convert height to one continuous variable
    del features_to_vals["ht_feet"]
    del features_to_vals["ht_inch"]
    ALL_VARIABLES.remove("ht_feet")
    CONT_VARIABLES.remove("ht_feet")
    ALL_VARIABLES.remove("ht_inch")
    CONT_VARIABLES.remove("ht_inch")

    features_to_vals["height"] = height_arr
    ALL_VARIABLES.append("height")
    CONT_VARIABLES.append("height")


    # process race here
    for i in range(0,num_cpw_complete_fields):
        # if features_to_vals["race"][i] in ['B', 'P', 'Q']: # black, black hispanic, white hispanic
        if features_to_vals["race"][i] in ['B']: # black, black hispanic, white hispanic
            features_to_vals["race"][i] = '0'
        # elif features_to_vals["race"][i] in ['W', 'A', 'I']: # white, asian, native american
        elif features_to_vals["race"][i] in ['W']: # white, asian, native american
            features_to_vals["race"][i] = '1'
        else:
            raise Exception('Unknown value for attr: %s' % race)

        if features_to_vals["sex"][i] == 'F': # black, black hispanic, white hispanic
            features_to_vals["sex"][i] = '0'
        elif features_to_vals["sex"][i] == 'M': # white, asian, native american
            features_to_vals["sex"][i] = '1'
        else:
            raise Exception('Unknown value for attr: %s' % race)



    """ make a decision variable here"""
    y = np.ones(num_cpw_complete_fields)  # by default, all labels will be +1
    for i in range(0,num_cpw_complete_fields):
        for attr in DECISION_VARIABLES:
            if features_to_vals[attr][i] == "Y":
                y[i] = -1.0
                break
            elif features_to_vals[attr][i] == "N":
                pass
            else:
                raise Exception("Unexpected value of the decision variable (Can only be y or N)")
    print("Total suspects with no gun found on them: %d (%0.1f%%)" % (y.tolist().count(1.0), y.tolist().count(1.0) * 100.0 / num_cpw_complete_fields))
    y = np.array(y)



    """ feature normalization here """
    X = np.array([]).reshape(len(y), 0) # empty array with num rows same as num examples, will hstack the features to it
    x_control = defaultdict(list)
    X_raw = np.array([])

    feature_names = []

    # for attr,vals in features_to_vals.items():
    for attr in sorted(features_to_vals.keys()):
        vals = features_to_vals[attr]
        vals_raw = features_to_vals[attr]
        if attr in DECISION_VARIABLES:
            continue # decision variables have already been taken care of

        elif attr in CONT_VARIABLES:
            vals = [float(v) for v in vals]
            vals = np.array(vals)
            vals = np.reshape(vals, (num_cpw_complete_fields, -1)) # convert from 1-d arr to a 2-d arr with one col

        elif attr in CATEGORICAL_VARIABLES: # for binary categorical variables, the label binarizer uses just one var instead of two
            lb = preprocessing.LabelBinarizer()
            lb.fit(vals)
            vals = lb.transform(vals)

        else:
            raise Exception("Unknown feature type (neither categorical, not cont):", attr)

        if attr in SENSITIVE_ATTRS:
            x_control[attr] = vals

        if attr in SENSITIVE_ATTRS and USE_SENS_CLASSIFICATION == False: # if we have decided not to use sensitive features
            pass
        else:
            X = np.hstack((X, vals))
            X_raw = np.hstack((X_raw, vals_raw))

            if attr in CONT_VARIABLES: # continuous feature, just append the name
                feature_names.append(attr)
            else: # categorical features
                if vals.shape[1] == 1: # binary features that passed through lib binarizer
                    feature_names.append(attr)
                else:
                    for k in lb.classes_: # non-binary categorical features, need to add the names for each cat
                        feature_names.append(attr + "_" + str(k))


    x_control = dict(x_control)
    
    for k in x_control.keys():
        assert(x_control[k].shape[1] == 1) # make sure that the sensitive feature is binary after one hot encoding
        x_control[k] = np.array(x_control[k]).flatten()
        
    

    """permute the date randomly"""
    perm = np.arange(0,X.shape[0])
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    print(x_control)
    print(X.shape)
    print(y.shape)
    

    if USE_BALANCED_CLASSES == True:

        n_min = min(dict(Counter(y)).values()) # smallest number of examples in a class. these many values will be selected from each class
        print(n_min)
        # initialize new empty arrays
        X_new = []
        y_new = []
        x_control_new = {}
        for k in x_control.keys():
            x_control_new[k] = []

        for v in set(y):
            idx = np.where(y == v)[0]
            shuffle(idx) # we will pick random entries having this class label
            idx = idx[:n_min]
            
            
            X_new += X[idx].tolist()
            y_new += y[idx].tolist()
            for k in x_control.keys():
                x_control_new[k] += x_control[k][idx].tolist()


        X = np.array(X_new)
        y = np.array(y_new)
        for k in x_control_new.keys():
            x_control[k] = np.array(x_control_new[k])

    scaler = MaxAbsScaler()#MinMaxScaler()
    X = scaler.fit_transform(X)
    
    """permute the data randomly"""
    perm = np.arange(0,X.shape[0])
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    
    assert(len(feature_names) == X.shape[1])
    
    x_sensitive = x_control["race"]
    #np.savez("sqf", X, y, x_sensitive)

    return X, y, x_control, X_raw
   


# fetch_data_from_db([2012], USE_SENS_CLASSIFICATION=False, USE_BALANCED_CLASSES=True)
