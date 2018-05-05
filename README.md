# ml_stem_analysis

## Dependencies:
1. numpy
2. pandas
3. scipy
4. xgBosst
5. sklearn
6. ipytest
7. pytest

## Walkthrough

### plot()
This function takes in the dataframe and goes over all the non- funding variables and makes a subplot.
The plot in the notebook is very big `figsize = (400, 400)` for which it doesn't appear at first, due to huge number of features.
So after downloading double-clicking the image opens the plots.

### algorithm()
Takes in `X_train, X_test, y_train, y_test` as input and output returned is the mutual and roc score using the required algorithm for the feature detection i.e `DecisionTreeClassifier()`

### xgboost_()
Takes in `X_train, X_test, y_train, y_test` as input and output returned is the mutual and roc score using the xgboost algorithm for the feature detection i.e `XGBRegressor()`.    
`
  gs = GridSearchCV(xr, params, n_jobs=1,verbose=1)       
  gs.fit(X_train, y_train)      
  gs.best_params_     
  `
  used to find best params.
  
  ### work_()
  The driver function. Takes in the data as input and pre-processes it before calling the functions plot(), algorithm() and xgboost_().
  
  Output is the overall result.
  
  #### Univariate graphs
  ##### Frequency of Agency
  ![agency](https://github.com/ASH1998/ml_stem_analysis/blob/master/Image/types%20of%20agency.PNG)
  
  ##### Frequrency of sub-agency
  ![sub](https://github.com/ASH1998/ml_stem_analysis/blob/master/Image/types%20of%20subagency.PNG)
  
  ##### Year wise established
  ![year](https://github.com/ASH1998/ml_stem_analysis/blob/master/Image/year.PNG)
  
  ##### Final scores
  ![final](https://github.com/ASH1998/ml_stem_analysis/blob/master/Image/final.PNG)
  
  ### test_work_()
  Test case for the whole script.
  
  ![test case](https://github.com/ASH1998/ml_stem_analysis/blob/master/Image/test.PNG)
