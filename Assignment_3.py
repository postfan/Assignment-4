
# coding: utf-8

# In[4]:



#get_python().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
def spring_motion(x_0, v_0, h, N):
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.arange(0,N+1)*h
    x[0] = x_0
    v[0] = v_0
    
    x_i = x_0
    v_i = v_0
    
    for n in range(N):
        x_j = x_i + h * v_i
        v_j = v_i - h * x_i
        
        x[n+1] = x_j
        v[n+1] = v_j
        
        x_i = x_j
        v_i = v_j
        
    return (x,v,t)
    


# In[21]:


(x,v,t) = spring_motion(3,5,.01,10000)
plt.plot(t,x)
plt.xlabel("time")
plt.ylabel("position")
plt.savefig("position.pdf")
plt.clf()


# In[6]:


(x,v,t) = spring_motion(3,5,.01,1000)
plt.plot(t,v)
plt.xlabel("time")
plt.ylabel("velocity")
plt.savefig("velocity.pdf")
plt.clf()


# In[7]:


def exact_spring (x_0,v_0,h,N):
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.arange(0,N+1) * h
    
    x[0] = x_0
    v[0] = v_0
    
    for n in range(N):
        x[n+1] = x_0 * (np.cos(t[n+1])) + v_0 * (np.sin(t[n+1]))
        v[n+1] = -x_0 * (np.sin(t[n+1])) + v_0 * (np.cos(t[n+1]))
    
    return (x,v,t)


# In[8]:


xvt1 = spring_motion(3,5,.01,10000)
xvt2 = exact_spring(3,5,.01,10000)
x = xvt2[0] - xvt1[0]
t = xvt1[2]
plt.plot(t,x, label = "x")
plt.xlabel("time")
plt.ylabel("error")
plt.legend()
plt.savefig("position_error.pdf")
plt.clf()


# In[9]:


xvt1 = spring_motion(3,5,.01,10000)
xvt2 = exact_spring(3,5,.01,10000)
v = xvt2[1] - xvt1[1]
t = xvt1[2]
plt.plot(t,v)
plt.xlabel("time")
plt.ylabel("error")
plt.savefig("velocity_error.pdf")
plt.clf()


# In[17]:


def h_prop(x_0,v_0,t,N):
    x_diff = np.zeros(6)
    h_values = np.zeros(6)
    for n in range(6):
        t_0 = t/2**(n)
        h = t_0/N 
        xvt1 = spring_motion(x_0,v_0,h,N)
        xvt2 = exact_spring(x_0,v_0,h,N)
        x = xvt2[0] - xvt1[0]
        x_diff[n] = np.amax(np.abs(x))
        h_values[n] = t_0
        
    plt.plot(h_values,x_diff)
    plt.xlabel("h_values")
    plt.ylabel("x_diff")
    plt.savefig("truncation_error.pdf")
    plt.clf()


# In[20]:


h_prop(3,5,.001,100000)


# In[46]:


(x,v,t) = spring_motion(3,5,.01,1000)
E = x**(2)+v**(2)
plt.plot(t,E)
plt.savefig("energy.pdf")
plt.clf()


# In[12]:


# E increases over a long time frame. This corresponds with the error
# in position and velocity above.


# In[23]:


def implicit(x_0, v_0, h, N):
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.arange(0,N+1)*h
    x[0] = x_0
    v[0] = v_0
    
    x_i = x_0
    v_i = v_0
    
    for n in range(N):
        x_j = (x_i + h * v_i) / (h**2 + 1)
        v_j = (v_i - h * x_i) / (h**2 + 1)
        
        x[n+1] = x_j
        v[n+1] = v_j
        
        x_i = x_j
        v_i = v_j
        
    return (x,v,t)


# In[24]:


(x,v,t) = implicit(3,5,.01,10000)
plt.plot(t,x)
plt.xlabel("time")
plt.ylabel("position")
plt.savefig("implicit_position.pdf")
plt.clf()


# In[58]:


xvt1 = implicit(3,5,.01,10000)
xvt2 = exact_spring(3,5,.01,10000)
x = xvt2[0] - xvt1[0]
t = xvt1[2]
plt.plot(t,x)
plt.xlabel("time")
plt.ylabel("error")
plt.savefig("implicit_error.pdf")
plt.clf()


# In[59]:


(x,v,t) = implicit(3,5,.01,10000)
E = x**(2)+v**(2)
plt.plot(t,E)
plt.savefig("implicit_energy.pdf")
plt.clf()


# In[30]:


# The error with the implicit Euler function is the opposite compared to the
# explicit Euler function. With the explicit the energy was increasing when it
# was supposed to be constant, while with the implicit the energy is 
# decreasing when it's still supposed to be constant.


# In[60]:


(x,v,t) = spring_motion(3,5,.01,10000)
plt.plot(x,v)
plt.xlabel("position")
plt.ylabel("velocity")
plt.savefig("explicit_phasespace.pdf")
plt.clf()


# In[61]:


(x,v,t) = implicit(3,5,.01,10000)
plt.plot(x,v)
plt.xlabel("position")
plt.ylabel("velocity")
plt.savefig("implicit_phasespace.pdf")
plt.clf()


# In[62]:


(x,v,t) = exact_spring(3,5,.01,10000)
plt.plot(x,v)
plt.xlabel("position")
plt.ylabel("velocity")
plt.savefig("exact_phasespace.pdf")
plt.clf()


# In[34]:


def symplectic(x_0, v_0, h, N):
    x = np.zeros(N+1)
    v = np.zeros(N+1)
    t = np.arange(0,N+1)*h
    x[0] = x_0
    v[0] = v_0
    
    x_i = x_0
    v_i = v_0
    
    for n in range(N):
        x_j = x_i + h * v_i
        v_j = v_i - h * x_j
        
        x[n+1] = x_j
        v[n+1] = v_j
        
        x_i = x_j
        v_i = v_j
        
    return (x,v,t)


# In[63]:


(x,v,t) = symplectic(3,5,.01,10000)
plt.plot(x,v)
plt.xlabel("position")
plt.ylabel("velocity")
plt.savefig("symplectic_phasespace.pdf")
plt.clf()


# In[64]:


(x,v,t) = symplectic(3,5,.01,10000)
E = x**(2)+v**(2)
plt.plot(t,E)
plt.savefig("symplectic_energy.pdf")
plt.clf()


# In[ ]:


# The average energy over the long term is the same, which corresponds
# to the conservation of energy implied by the phase space diagram above.

