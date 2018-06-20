import pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
import numpy as np
from sklearn.tree import export_graphviz
from sklearn.feature_extraction import DictVectorizer
import scikitplot as skplt
import matplotlib.pyplot as plt

#Adjusting options to view full data
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
data =  pd.read_csv('IBM Attrition.csv')
data.info()
data.head()






"""PREPARING DATA FOR DECISION TREE MODEL"""



#Transforming and Ordering Variables that are Ordinal and Categorical
data.Education = data.Education.astype('category')
data.Education =data.Education.cat.reorder_categories(['Below College' , 'College', 'Bachelor', 'Master', 'Doctor'])
data.Education =data.Education.cat.codes


data.EnvironmentSatisfaction = data.EnvironmentSatisfaction.astype('category')
data.EnvironmentSatisfaction = data.EnvironmentSatisfaction.cat.reorder_categories(['Low', 'Medium' , 'High', 'Very High'])
data.EnvironmentSatisfaction = data.EnvironmentSatisfaction.cat.codes


data.JobInvolvement = data.JobInvolvement.astype('category')
data.JobInvolvement = data.JobInvolvement.cat.reorder_categories(['Low', 'Medium' , 'High', 'Very High'])
data.JobInvolvement = data.JobInvolvement.cat.codes

data.JobSatisfaction = data.JobSatisfaction.astype('category')
data.JobSatisfaction = data.JobSatisfaction.cat.reorder_categories(['Low', 'Medium' , 'High', 'Very High'])
data.JobSatisfaction = data.JobSatisfaction.cat.codes

data.JobLevel = data.JobLevel.astype('category')
data.JobLevel = data.JobLevel.cat.reorder_categories([1, 2, 3, 4,5])
data.JobLevel = data.JobLevel.cat.codes


data.PerformanceRating  = data.PerformanceRating.astype('category')
data.PerformanceRating  = data.PerformanceRating.cat.reorder_categories(['Excellent', 'Outstanding'])
data.PerformanceRating  = data.PerformanceRating.cat.codes


data.RelationshipSatisfaction   = data.RelationshipSatisfaction.astype('category')
data.RelationshipSatisfaction   = data.RelationshipSatisfaction.cat.reorder_categories(['Low', 'Medium' , 'High', 'Very High'])
data.RelationshipSatisfaction   = data.RelationshipSatisfaction.cat.codes


data.WorkLifeBalance = data.WorkLifeBalance.astype('category')
data.WorkLifeBalance = data.WorkLifeBalance.cat.reorder_categories(['Bad', 'Good' , 'Better', 'Best'])
data.WorkLifeBalance = data.WorkLifeBalance.cat.codes


data.StockOptionLevel = data.StockOptionLevel.astype('category')
data.StockOptionLevel = data.StockOptionLevel.cat.reorder_categories([0, 1 , 2, 3])
data.StockOptionLevel = data.StockOptionLevel.cat.codes


##Creating Dummy variables for Nominal Categories

Departments = pd.get_dummies(data.Department, prefix='Dept')
Departments = Departments.drop('Dept_Sales', axis=1)
data = data.drop('Department', axis=1)
data = data.join(Departments)

Churn = pd.get_dummies(data.Attrition, prefix= 'Attr', drop_first=True)
data = data.drop('Attrition', axis=1)
data = data.join(Churn)

Education_Field = pd.get_dummies(data.EducationField, prefix= 'EF', drop_first=True)
data = data.drop('EducationField', axis=1)
data = data.join(Education_Field)

Gend= pd.get_dummies(data.Gender)
Gend = Gend.drop('Male', axis=1)
data = data.drop('Gender', axis=1)
data = data.join(Gend)


Role = pd.get_dummies(data.JobRole)
Role = Role.drop('Human Resources', axis=1)
data = data.drop('JobRole', axis=1)
data = data.join(Role)


Mstat =pd.get_dummies(data.MaritalStatus)
Mstat = Mstat.drop('Single', axis=1)
data = data.drop('MaritalStatus', axis=1)
data = data.join(Mstat)

Overtime = pd.get_dummies(data.OverTime, prefix='Ovt')
Overtime = Overtime.drop('Ovt_No', axis=1)
data = data.drop('OverTime', axis=1)
data = data.join(Overtime)


Travel = pd.get_dummies(data.BusinessTravel)
Travel= Travel.drop('Non-Travel', axis=1)
data = data.drop('BusinessTravel', axis=1)
data = data.join(Travel)

Over18 = pd.get_dummies(data.Over18) 
Over18 = Over18.drop('Y', axis =1)
data = data.drop('Over18', axis=1)
data = data.join(Over18)



#################           MODELING                #################

# CALCULATING ATTRITION RATE
n_employees = data['Attr_Yes'].count()
print(data.Attr_Yes.value_counts())
print(data.Attr_Yes.value_counts()/n_employees*100)


"""SPLITTING THE DATA INTO TARGET AND FEATURE VARIABLES"""

target = data.Attr_Yes
features = data.drop('Attr_Yes',axis=1).drop('EmployeeNumber',axis=1).drop('DailyRate',axis=1).drop('HourlyRate',axis=1).drop('MonthlyRate',axis=1)
target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)

"""INITIAL MODEL"""
#Creating Model
model = DecisionTreeClassifier(random_state=42)
model.fit(features_train,target_train)

# Check the accuracy of the prediction (in percentage points) for the training set and test set
model.score(features_train,target_train)*100
model.score(features_test,target_test)*100

"""
# Import the tree graphical visualization export function
from sklearn.tree import export_graphviz
 Export the tree to a dot file
export_graphviz(model,"tree.dot", features = data.feature_names )
"""


# Predict whether employees will churn using the test set
prediction = model.predict(features_test)

# Calculate precision score by comparing target_test with the prediction
precision_score(target_test, prediction)


# Use the initial model to predict churn
prediction = model.predict(features_test)

# Calculate recall score by comparing target_test with the prediction
recall_score(target_test, prediction)


# Use initial model to predict churn (based on features of the test set)
prediction = model.predict(features_test)

# Calculate ROC/AUC score by comparing target_test with the prediction
roc_auc_score(target_test, prediction)

print(metrics.confusion_matrix(target_test, prediction))


"""MODEL WITH BALANCED CLASS WEIGHT"""

# Create model with 'balanced' classes of people leaving and staying 
model_depth_b = DecisionTreeClassifier(class_weight='balanced',random_state=42)
model_depth_b.fit(features_train,target_train)
print(model_depth_b.score(features_train,target_train)*100)
print(model_depth_b.score(features_test,target_test)*100)

prediction = model_depth_b.predict(features_test)
precision_score(target_test, prediction)
recall_score(target_test, prediction)
roc_auc_score(target_test, prediction)
print(metrics.confusion_matrix(target_test, prediction))

# Use that function to print the cross validation score for 10 folds
print(cross_val_score(model,features,target,cv=10))


"""MODEL WITH OPTIMIZED PARAMETERS USING GRIDSEARCHCV""" 
"""
#Parameter Loops
work = range(5,21,1)
depth = [i for i in work]
samples = [i for i in range(10,100,10)]


# Create the dictionary with parameters to be checked
parameters = dict(max_depth=depth,min_samples_leaf=samples)


# initialize the param_search function using the GridSearchCV function, initial model and parameters above
param_search = GridSearchCV(model_depth_b, parameters,cv=10)

# fit the param_search to the training dataset
param_search.fit(features_train, target_train)

# print the best parameters found
b= param_search.best_params_
print(b)

# initialize the best model using parameters provided in description
model_prune= DecisionTreeClassifier(max_depth = b['max_depth'],min_samples_leaf=b['min_samples_leaf'],class_weight = "balanced",random_state=42)
model_prune.fit(features_train ,target_train)
prediction_prune = model_prune.predict(features_test)

# Accuracy of the model
print(model_prune.score(features_train ,target_train)*100)
print(model_prune.score(features_test ,target_test)*100)
print(recall_score(target_test,prediction_prune)*100)
print(roc_auc_score(target_test,prediction_prune)*100)
print(metrics.confusion_matrix(target_test,prediction_prune))
"""


'''MODEL USING FEATURES WITH RELATIVE IMPORTANCE'''

# Calculate feature importances
feature_importances = model.feature_importances_

# Create a list of features: done
feature_list = list(features)

# Save the results inside a DataFrame using feature_list as an indnex
relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])

# Sort values to learn most important features
relative_importances.sort_values(by="importance", ascending=False)


# select only features with relative importance higher than 1%
selected_features = relative_importances[relative_importances.values>0.01]

# create a list from those features: done
selected_list = selected_features.index

# transform both features_train and features_test components to include only selected features
features_train_selected = features_train[selected_list]
features_test_selected = features_test[selected_list]




#max_depth=4 gave me best performance, while limiting min_samples_leaf lowered my prediction and AUC
model_best = DecisionTreeClassifier(max_depth = 4,class_weight = "balanced",random_state=42)
model_best.fit(features_train_selected,target_train)
prediction_best = model_best.predict(features_test_selected)




# Accuracy of the model
print(model_best.score(features_train_selected,target_train)*100)
print(model_best.score(features_test_selected,target_test)*100)
print(recall_score(target_test,prediction_best)*100)
print(roc_auc_score(target_test,prediction_best)*100)
print(metrics.confusion_matrix(target_test,prediction_best))
print(cross_val_score(model_best,features_test_selected,target_test,cv=10))


fpr, tpr, threshold = metrics.roc_curve(target_test, prediction_best)
roc_auc = metrics.auc(fpr, tpr)
print(fpr)
print(tpr)
print(threshold)
plt.plot(fpr,tpr)
plt.show()

#Dot File for external Decision Tree Visual
export_graphviz(model_best,"tree2.dot",feature_names = selected_list)

#Saving Data
data.to_csv('Attrition Model')




# If another year passes which employees are predicted to leave if everything remains constant
next_data = pd.read_csv('One Year After.csv')
next_data.head()
predict_one_year= model_best.predict(next_data[selected_list])
values, count = np.unique(predict_one_year, return_counts=True)
print(predict_one_year)
df= pd.DataFrame(count,values)
next_data['Prediction']= np.array(predict_one_year)
next_data.to_csv('Year After')

print(df)


perct = np.sum(predict_one_year==1)/len(predict_one_year)

print(perct)
'''



'''




#data.to_csv('predicted set')


'''LOOP TO FIND OPTIMAL TUNING PARAMETERS'''

'''
#Prediction of training vs testing set
max_depths= list(range(1,10))
train_score=[]
test_score=[]
ind=[]

for max_depth in max_depths:
    model_depth = DecisionTreeClassifier(max_depth = max_depth,class_weight='balanced', random_state=42)
    model_depth.fit(features_train_selected,target_train)
    m = (model_depth.score(features_train_selected,target_train)*100)
    d = (model_depth.score(features_test_selected,target_test)*100)
    train_score.append(m)
    test_score.append(d)
    ind.append(max_depth)

d={'max_depth':ind, 'Train':train_score, 'Test':test_score}
print(pd.DataFrame(d))

import matplotlib.pyplot as plt
plt.plot(max_depths, train_score, color='blue', marker=".", label='train')
plt.plot(max_depths, test_score, color='red', marker='.', label='test')
plt.xlabel('max_depths')
plt.legend(['train', 'test'])
plt.show()





#prunining the Decision Tree
max_depths= list(range(1,10))
train_score=[]
test_score=[]
ind=[]

for max_depth in max_depths:
    model_depth = DecisionTreeClassifier(max_depth = max_depth,class_weight='balanced', random_state=42)
    model_depth.fit(features_train_selected,target_train)
    m = (model_depth.score(features_train_selected,target_train)*100)
    d = (model_depth.score(features_test_selected,target_test)*100)
    train_score.append(m)
    test_score.append(d)
    ind.append(max_depth)

d={'max_depth':ind, 'Train':train_score, 'Test':test_score}
print(pd.dataframe(d, index='max_depth'))


plt.plot(max_depths, train_score, color='blue', marker=".", label='train')
plt.plot(max_depths, test_score, color='red', marker='.', label='test')
plt.xlabel('max_depths')
plt.legend(['train', 'test'])
plt.show()

min_leafs= list(range(0,210,10))
trainl_score=[]
testl_score=[]

for min_samples_leaf in min_leafs:
    model_leaf = DecisionTreeClassifier(max_depth=6, min_samples_leaf= min_samples_leaf,class_weight='balanced',random_state=42)
    model_leaf.fit(features_train,target_train)
    l = (model_leaf.score(features_train,target_train)*100)
    p = (model_leaf.score(features_test,target_test)*100)
    trainl_score.append(l)
    testl_score.append(p)
print(trainl_score,testl_score)

plt.plot(min_leafs, trainl_score, color='blue', marker=".", label='train')
plt.plot(min_leafs, testl_score, color='red', marker='.', label='test')
plt.xlabel('min_leaf')
plt.legend(['train', 'test'])
plt.show()


#Looking at Max depths effect on AUC
max_depths= list(range(1,10))
train_score=[]
test_score=[]

for max_depth in max_depths:
    model_depth = DecisionTreeClassifier(max_depth = max_depth,class_weight='balanced', random_state=42)
    model_depth.fit(features_train_selected,target_train)
    ploop= model_depth.predict(features_test_selected)   
    fpr, tpr, threshold = metrics.roc_curve(target_test, ploop)
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=max_depth)
plt.title('AUC curve by Max Depth')
plt.legend()
plt.show()



'''



