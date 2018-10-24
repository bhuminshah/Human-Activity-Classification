
# coding: utf-8

# In[84]:


import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
import glob, os
import scipy
import numpy 
import matplotlib
import sklearn
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer

# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler


def prc_preprocess(data_frame):
    #data_frame.fillna(0.0,inplace=True)
    #print(data_frame['var_rss12'].value_counts())
    #data_frame.drop_outliers(3)
    print(data_frame.head(5))
    print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
    del data_frame['var_rss23']
    del data_frame ['time']
    
    dataset = data_frame.loc[:, 'avg_rss12': 'avg_rss23']
    #dataset = dataset.replace(0,numpy.NaN)
    #transformed_data = dataset.fillna(dataset.mean(), inplace=True)


    '''
    scaler = StandardScaler().fit(dataset)
    dataset = scaler.transform(dataset)

    scaler = Normalizer().fit(dataset)
    dataset = scaler.transform(dataset)
    '''
    print(dataset.head(5))

    low = .2
    high = .80
    quant_df = dataset.quantile([low, high])

    dataset = dataset.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                    (x < quant_df.loc[high,x.name])], axis=0)

    #dataset.fillna(0.0, inplace=True)  # Added by Pratik, as data had NaNs



   # scaler = MinMaxScaler(feature_range=(0.1, 10.0))
    #dataset = scaler.fit_transform(dataset)

    dataset = pd.concat([data_frame.loc[:,'Activity'], dataset], axis=1)

    #dataset = dataset.replace(0.00, numpy.NaN) #Added by Pratik
    dataset.fillna(0.0, inplace=True)
    print(dataset.head(5))

    return dataset
    #return transformed_data
    pass

def prc_data_visualization(data_frame):
    #data_frame.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    #data_frame.boxplot(['avg_rss12','var_rss12','avg_rss13','var_rss13','avg_rss23', 'var_rss23'], by='Activity')
    #data_frame.boxplot(['avg_rss12'], by='Activity')


    #data_frame.boxplot(['var_rss23'], by='Activity')
    data_frame.boxplot(['avg_rss12'], by='Activity')
    data_frame.boxplot(['avg_rss13'], by='Activity')
    data_frame.boxplot(['var_rss12'], by='Activity')
    data_frame.boxplot(['var_rss13'], by='Activity')
    data_frame.boxplot(['avg_rss23'], by='Activity')
    #data_frame.hist()
   # scatter_matrix(data_frame)
    plt.show()
    pass

def prc_modelling(data_frame):
    print("In modelling")
    #print(data_frame[1:])
    #array = data_frame.values
    #print(array[:,0])
    X = data_frame.iloc[:, 1:5]
    Y = data_frame.iloc[:, 0]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                    random_state=seed)

    # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
   # models.append(('SVM', SVC()))
    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()



def prc_loadData():
    #Directory to the dataset
    #os.chdir(r"/Users/bhumin/Downloads/dataset/") Commented by Pratik
    os.chdir(r"C:\Users\Pratik\PycharmProjects\ML\dataset")
    results = []
    bending=pd.DataFrame([])
    
    #Bending1
    #Reading attributes from all the files in bending1 activity
    for counter, file in enumerate(glob.glob(r"bending1\dataset*")):
        namedf = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)


    #concatinating all the values read from each excel file
    bending=pd.concat(results)
    print("hi")
    print(bending.head(5))
    
    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = bending.replace(0,numpy.NaN)
    bending = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    bending['Activity'] = "Bending"
    
    
    #Bending2
    #Reading attributes from all the files in bending2 activity
    results = []
    bending2 = pd.DataFrame([])
    for counter, file in enumerate(glob.glob(r"bending2\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)
    
    #concatinating all the values read from each excel file
    bending2=pd.concat(results)
    
    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = bending2.replace(0,numpy.NaN)
    bending2 = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    bending2['Activity'] = "Bending"
   
    #appending both the bending into bending
    bending=bending.append(bending2)
    

    #Cycling
    #Reading attributes from all the files in cycling activity
    results = []
    for counter, file in enumerate(glob.glob(r"cycling\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)

    #concatinating all the values read from each excel file
    cycling = pd.concat(results)
    
    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = cycling.replace(0,numpy.NaN)
    cycling = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    cycling['Activity'] = "Cycling"
    
    
    #Lying
    #Reading attributes from all the files in lying activity
    results = []
    for counter, file in enumerate(glob.glob(r"lying\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)

    #concatinating all the values read from each excel file
    lying = pd.concat(results)

    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = lying.replace(0,numpy.NaN)
    lying = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    lying['Activity'] = "Lying"
   

    #Sitting
    #Reading attributes from all the files in sitting activity
    results = []
    for counter, file in enumerate(glob.glob(r"sitting\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)

    #concatinating all the values read from each excel file
    sitting = pd.concat(results)

    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = sitting.replace(0,numpy.NaN)
    sitting = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    sitting['Activity'] = "Sitting"
    
    
    #Standing
    #Reading attributes from all the files in standing activity
    results = []
    for counter, file in enumerate(glob.glob(r"standing\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)

    #concatinating all the values read from each excel file
    standing = pd.concat(results)

    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = standing.replace(0,numpy.NaN)
    standing = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    standing['Activity'] = "Standing"
   

    #Walking
    #Reading attributes from all the files in walking activity
    results = []
    for counter, file in enumerate(glob.glob(r"walking\dataset*")):
        namedf = pd.read_csv(file, skiprows=0, usecols=[1, 2, 3, 4, 5, 6, 7])
        results.append(namedf)
        
    #concatinating all the values read from each excel file
    walking = pd.concat(results)
    
    #Replacing 0 with NaN's and then replacing with mean of each column 
    dataset = walking.replace(0,numpy.NaN)
    walking = dataset.fillna(dataset.mean(), inplace=True)
    
    #adding the label 
    walking['Activity'] = "Walking"

    final_Dataset=pd.DataFrame([])
    final_Dataset=pd.concat([walking,standing,sitting,lying,cycling,bending],ignore_index=True)

    
    return final_Dataset

result=prc_loadData()
print(result.head(0))

processed_data = prc_preprocess(result)
#processed_data.describe()
#print(processed_data.describe())
#print(processed_data.sample(frac=1))

#print(processed_data.groupby('Activity').size())
#prc_data_visualization(processed_data)
prc_modelling(processed_data)
