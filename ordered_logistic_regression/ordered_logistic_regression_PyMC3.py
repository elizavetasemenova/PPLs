import pandas as pd 
import numpy as np
import pymc3 as pm
from sklearn.metrics import confusion_matrix, accuracy_score


# ### Read data

# In[103]:


df  = pd.read_csv("WVS.csv") # exported the dataset from R
df.drop(df.columns[[0]], axis=1, inplace=True)
y = df.iloc[:,-1]
X = X.values
y = y.values
X = df.iloc[:, df.columns != 'y' ]


# ### Model specification

# In[131]:


print('Running on PyMC3 v{}'.format(pm.__version__))


# In[105]:


N = X.shape[0]
D = X.shape[1]


# In[106]:


with pm.Model() as mod:
    
    # Priors
    sigma = pm.HalfNormal('sigma', sd = 1)   
    
    beta = pm.Normal('beta', mu=0, sd=sigma, shape=D) 
    
    lp = pm.math.dot(X, beta)
    
    cutpoints = pm.Normal("cutpoints", mu=[-0.01,0], sd=20, shape=2,
                           transform=pm.distributions.transforms.ordered)
    
    # Likelihood 
    y_obs = pm.OrderedLogistic("y_obs", eta=lp, cutpoints=cutpoints, observed=y-1)


# ### Sampling

# In[153]:


with mod:
    # draw posterior samples
    trace = pm.sample(5000, tune=5000, nuts_kwargs=dict(target_accept=.85))


# ### Parameter estimates

# In[154]:


pm.summary(trace).round(2)


# ### Predictions

# In[155]:


ppc = pm.sample_ppc(trace, samples=5000, model=mod, size=1)


# In[192]:


y_pred_samps = ppc['y_obs']
y_pred = np.zeros(y_pred_samps.shape[1])

for i in range(0,len(pred)):

    p1 = np.mean(y_pred_samps[:,i] == 0)
    p2 = np.mean(y_pred_samps[:,i] == 1)
    p3 = np.mean(y_pred_samps[:,i] == 2)
    probs = [p1, p2, p3]
    
    y_pred[i] = probs.index(max(probs)) + 1


# ### Accuracy, confusion matrix

# In[193]:


confusion_matrix(y-1, y_pred-1)


# In[194]:


round(accuracy_score(y_true=y, y_pred=y_pred),2)


# In[ ]:
