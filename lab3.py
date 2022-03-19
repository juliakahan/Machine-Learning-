#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y}) 
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')


# In[115]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[116]:


X_train = X_train.reshape(-1,1)
X_test = X_test.reshape(-1,1)


# In[117]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)


# In[118]:


print(lin_reg.intercept_, lin_reg.coef_)


# In[119]:


y_pred_lin_regression_train = lin_reg.predict(X_train)


# In[120]:


y_pred_linear_regression_test = lin_reg.predict(X_test)


# In[121]:


from sklearn.metrics import mean_squared_error


# In[122]:


train_mse_lin = mean_squared_error(y_train,y_pred_lin_regression_train)
print(train_mse_lin)


# In[123]:


test_mse_lin = mean_squared_error(y_test,y_pred_linear_regression_test)


# In[124]:


print(test_mse_lin)


# In[125]:


import sklearn.neighbors


# In[126]:


knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)


# In[127]:


y_pred_knn_3_train = knn_3_reg.predict(X_train)
y_pred_knn_3_test = knn_3_reg.predict(X_test)


# In[128]:


train_mse_knn_3 = mean_squared_error(y_train, y_pred_knn_3_train)
print(train_mse_knn_3)
test_mse_knn_3 = mean_squared_error(y_test, y_pred_knn_3_test)
print(test_mse_knn_3)


# In[ ]:





# In[ ]:





# In[ ]:





# In[129]:


knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors = 5)
knn_5_reg.fit(X_train, y_train)


# In[130]:


y_pred_knn_5_train =knn_5_reg.predict(X_train)
y_pred_knn_5_test = knn_5_reg.predict(X_test)


# In[131]:


train_mse_knn_5 = mean_squared_error(y_train, y_pred_knn_5_train)
print(train_mse_knn_5)
test_mse_knn_5 = mean_squared_error(y_test, y_pred_knn_5_test)
print(test_mse_knn_5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[132]:


from sklearn.preprocessing import PolynomialFeatures
poly_feature_2 = PolynomialFeatures(degree = 2, include_bias = False)
X_2_poly_train = poly_feature_2.fit_transform(X_train)
X_2_poly_test = poly_feature_2.fit_transform(X_test)
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_2_poly_train, y_train)


# In[133]:


y_pred_poly_2_train = poly_2_reg.predict(X_2_poly_train)
y_pred_poly_2_test = poly_2_reg.predict(X_2_poly_test)


# In[134]:


train_mse_poly_2 = mean_squared_error(y_train, y_pred_poly_2_train)
test_mse_poly_2 = mean_squared_error(y_test, y_pred_poly_2_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[135]:


poly_feature_3 = PolynomialFeatures(degree = 3, include_bias = False)
X_3_poly_train = poly_feature_3.fit_transform(X_train)
X_3_poly_test = poly_feature_3.fit_transform(X_test)
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_3_poly_train, y_train)


# In[136]:


y_pred_poly_3_train = poly_3_reg.predict(X_3_poly_train)
y_pred_poly_3_test = poly_3_reg.predict(X_3_poly_test)


# In[137]:


train_mse_poly_3 = mean_squared_error(y_train, y_pred_poly_3_train)
test_mse_poly_3 = mean_squared_error(y_test, y_pred_poly_3_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[138]:


poly_feature_4 = PolynomialFeatures(degree = 4, include_bias = False)
X_4_poly_train = poly_feature_4.fit_transform(X_train)
X_4_poly_test = poly_feature_4.fit_transform(X_test)
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_4_poly_train, y_train)


# In[139]:


y_pred_poly_4_train = poly_4_reg.predict(X_4_poly_train)
y_pred_poly_4_test = poly_4_reg.predict(X_4_poly_test)


# In[140]:


train_mse_poly_4 = mean_squared_error(y_train, y_pred_poly_4_train)
test_mse_poly_4 = mean_squared_error(y_test, y_pred_poly_4_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[141]:


poly_feature_5 = PolynomialFeatures(degree = 5, include_bias = False)

X_5_poly_train = poly_feature_5.fit_transform(X_train)
X_5_poly_test = poly_feature_5.fit_transform(X_test)

poly_5_reg = LinearRegression()
poly_5_reg.fit(X_5_poly_train, y_train)


# In[142]:


y_pred_poly_5_train =poly_5_reg.predict(X_5_poly_train)
y_pred_poly_5_test = poly_5_reg.predict(X_5_poly_test)


# In[143]:


train_mse_poly_5 = mean_squared_error(y_train, y_pred_poly_5_train)
test_mse_poly_5 = mean_squared_error(y_test, y_pred_poly_5_test)


# In[ ]:





# In[ ]:





# In[144]:


mse = [ [train_mse_lin, test_mse_lin], [train_mse_knn_3, test_mse_knn_3], [train_mse_knn_5, test_mse_knn_5], [train_mse_poly_2, test_mse_poly_2], [train_mse_poly_3, test_mse_poly_3], [train_mse_poly_4, test_mse_poly_4], [train_mse_poly_5, test_mse_poly_5]]


# In[145]:


df_mse = pd.DataFrame(mse, index=["lin_reg", "knn_3_reg", "knn_5_reg", "poly_2_reg", "poly_3_reg", "poly_4_reg", "poly_5_reg"], columns = ["train_mse", "test_mse"])
df_mse


# In[147]:


reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg, poly_feature_2), (poly_3_reg, poly_feature_3), (poly_4_reg, poly_feature_4), (poly_5_reg, poly_feature_5)]
reg


# In[148]:


import pickle
with open('mse.pkl', 'wb') as f:
    pickle.dump(df_mse, f)


# In[149]:


import pickle
with open('reg.pkl', 'wb') as f:
    pickle.dump(reg, f)


# In[ ]:




