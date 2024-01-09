#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[3]:


# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])
# Perform an operation (e.g., adding 10 to each element)
new_arr = arr + 10
print(new_arr)  # Output: [11 12 13 14 15]


# In[5]:


import pandas as pd
# Create a simple DataFrame
data = {
    'Name': ['John', 'Anna', 'Peter', 'Linda'],
    'Age': [28, 23, 34, 29],
    'City': ['New York', 'Paris', 'Berlin', 'London']
}
df = pd.DataFrame(data)
# Display the DataFrame
print(df)


# In[7]:


#Data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
# Create a line plot
plt.plot(x, y)
# Add labels and title
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Plot')
# Display the plot
plt.show()


# In[77]:


import numpy as np
from scipy.stats import norm
def black_scholes_call(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_price
def black_scholes_call_vega(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega
def implied_volatility_newton_raphson(call_market, S, X, T, r, initial_volatility=0.2, tolerance=1e-6, max_iterations=100):
    sigma = initial_volatility
    for i in range(max_iterations):
        call_bs = black_scholes_call(S, X, T, r, sigma)
        vega = black_scholes_call_vega(S, X, T, r, sigma)
        f = call_bs - call_market
        f_prime = vega
        sigma = sigma - f / f_prime
        if abs(f) < tolerance:
            break
    return sigma
# Example usage:
call_market_price = 10.0
underlying_price = 100.0
strike_price = 100.0
time_to_expiry = 1.0
interest_rate = 0.05
implied_vol = implied_volatility_newton_raphson(call_market_price, underlying_price, strike_price, time_to_expiry, interest_rate)
print(f"Implied Volatility: {implied_vol}")


# In[78]:


import yfinance as yf
# Get the data for the stock AMZN
data = yf.download('AMZN', start='2023-09-01', end='2024-01-02')
# Print the first 5 rows of the data
print(data.head())


# In[79]:


S = Amzn['Adj Close'][-1]
print('The spot price is $', round(S,2), '.')


# In[84]:


Amzn = yf.Ticker("AMZN")
opt = Amzn.option_chain('2024-01-19')
opt.calls


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as si
import yfinance as yf
import os 


# In[54]:


AM_opt= yf.Ticker("AAPL")
opt= Apple_opt.option_chain('2024-01-19')
opt.calls


# In[90]:


pip install mibian


# In[91]:


from mibian import BS


# In[94]:


def newton_vol_call(S, K, T, C, r):
    
    #S: spot price
    #K: strike price
    #T: time to maturity
    #C: Call value
    #r: risk free rate
    #sigma: volatility of underlying asset
   
    MAX_ITERATIONS = 1000
    tolerance = 0.000001
    
    sigma = 0.25
    
    for i in range(0, MAX_ITERATIONS):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        vega = S * np.sqrt(T) * si.norm.pdf(d1, 0.0, 1.0)

        diff = C - price

        if (abs(diff) < tolerance):
            return sigma
        else: 
            sigma = sigma + diff/vega
        
        # print(i,sigma,diff)
        
    return sigma


# In[96]:


impvol = newton_vol_call(S, 165, 17/366, float(opt.calls.lastPrice[opt.calls.strike == 165]), 0.0547)
print('The implied volatility is', round(impvol*100,2) , '% for the one-month call with strike $ 165.00') 


# In[100]:


Amzn = yf.download("AMZN", start="2022-12-31", end="2023-12-31")


# In[101]:


S = Amzn['Adj Close'][-1]
print('The spot price is $', round(S,2), '.')


# In[102]:


log_return = np.log(Amzn['Adj Close'] / Amzn['Adj Close'].shift(1))
vol_h = np.sqrt(252) * log_return.std()
print('The annualised volatility is', round(vol_h*100,2), '%')


# In[57]:


import yfinance as yf
import pandas as pd

# Download historical stock prices for Amazon
Amazon = yf.download("AMZN", start="2023-01-02", end="2024-01-01")

# Calculate daily returns
Amazon['Daily_Return'] = Amazon['Close'].pct_change()

# Calculate historical volatility
historical_volatility = Amazon['Daily_Return'].std() * np.sqrt(252)  # Assuming 252 trading days in a year

print(f"The historical volatility for Amazon stock is: {historical_volatility:.4f}")


# In[34]:


import math
import numpy as np
import os

# Given values
S0 = 100  #spot price
K = 100   #strike prie
sigma = 0.20 #volatility
r = 0.05 #risk free interest rate
T = 1  #time to maturity 
N = 4  #number of periods
u=1.1 #Up factor 
d= 0.9 #Down factor 
p= 0.54 #risk neutral probability
q= 1-p #Probability of down movement 
payoff= "call"


# In[35]:


dT= float(T)/N                                     #Delta t
u=np.exp(sigma * np.sqrt(dT))                       #up factor 
d=1.0/u                                         #down factor
round(u,2), round(d,2)


# In[36]:


S=np.zeros ((N+1, N+1))
S[0,0]= S0
z=1
for t in range (1, N+1):
    for i in range(z):
        S[i,t]= S[i, t-1]*u
        S[i+1,t]= S[i, t-1]*u
    z+= 1


# In[28]:


S


# In[37]:


a=np.exp(r*dT)
p=(a-d)/(u-d)
q=1.0-p
round(p,2)


# In[48]:


a=np.exp(r*dT)
p=(a-d)/(u-d)
q=1.0-p
round(p,2)


# In[49]:


round(1-p,2)


# In[46]:


import math
# Given parameters
spot_price = 100  # Spot price of the stock
strike_price = 100  # Strike price of the European call option
num_periods = 4  # Number of periods in the binomial tree
u = 1.11  # Up factor
d = 0.9  # Down factor
p = 0.54  # Risk-neutral probability
q = 1 - p  # Probability of a down movement
risk_free_rate = 0.05  # Risk-free interest rate
time_to_maturity = 1  # Time to maturity in years
# Function to calculate the option value at each node
def european_call_option_value(spot, u, d, p, q, strike, num_periods):
    # Calculate the time step
    delta_t = time_to_maturity / num_periods
    # Calculate terminal stock prices
    terminal_stock_prices = [spot * (u ** up_moves) * (d ** (num_periods - up_moves)) 
                             for up_moves in range(num_periods + 1)]
    # Calculate terminal option values
    terminal_option_values = [max(0, price - strike) for price in terminal_stock_prices]
    # Backward induction to calculate the option value at each node
    for step in range(num_periods - 1, -1, -1):
        terminal_option_values = [
            math.exp(-risk_free_rate * delta_t) * 
            (p * terminal_option_values[up_moves + 1] + 
             q * terminal_option_values[up_moves]) 
            for up_moves in range(step + 1)]
    # Return the option value at the initial node
    return terminal_option_values[0]
# Calculate the value of the European call option
european_call_value = european_call_option_value(
    spot_price, u, d, p, q, strike_price, num_periods)
# Print the results
print(f"European call option value: ${european_call_value:.2f}")


# In[47]:


import networkx as nx
import matplotlib.pyplot as plt
# Define the node information
nodes_info = {
    'A': {'stock_price': 100, 'option_price': 10.55},
    'B': {'stock_price': 110.52, 'option_price': 5.41},
    'C': {'stock_price': 90.48, 'option_price': 11.26},
    'D': {'stock_price': 122.14, 'option_price': 11.26},
    'E': {'stock_price': 100, 'option_price': 0},
    'F': {'stock_price': 81.87, 'option_price': 0}}
# Create a directed graph
G = nx.DiGraph()
# Add nodes with attributes
for node, info in nodes_info.items():
    G.add_node(node, label=f"{node}\nStock: ${info['stock_price']:.2f}\nOption: ${info['option_price']:.2f}")
# Add edges to represent connections
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'F')])
# Set up plot
fig, ax = plt.subplots(figsize=(10, 6))
# Draw the graph
pos = {'A': (0, 0), 'B': (1, 1), 'C': (1, -1), 'D': (2, 2), 'E': (2, 0), 'F': (2, -2)}
labels = nx.get_node_attributes(G, 'label')
nx.draw(G, pos, with_labels=True, labels=labels, node_size=1500, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', arrowsize=20, ax=ax)
# Add axes and title
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim([-1, 3])
ax.set_ylim([-3, 3])
plt.title("Binomial Tree for Stock and Option Prices")
plt.show()

