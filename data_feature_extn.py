import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
from sklearn.preprocessing import StandardScaler




file_path = "/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset"
file_list = os.listdir(file_path)
file_list.remove('.DS_Store')


# selecting earning calls from the year 2018
subset=[]
companies=[]

start = 2018
end = 2018

# Only considering a subset of data
for file in file_list:
    
    year= int(file[:4])
    
    if year >= start and year < (end+1):
        subset.append(file)
        
    company = file[9:]
    if company not in companies:
        companies.append(company)
        

# sort according to dates
earning_calls = sorted(subset)

# to check features used 
print(df.columns)
len(df.columns)
len(earning_calls)


# ## Scraping volatility data

import re
from io import StringIO
from datetime import datetime, timedelta

import requests
import pandas as pd


# In[14]:


# Our data spans these dates
# 1 st Jan 2018 to 31 st Dec 2018 
# yahoo finance historical data

def get_yahoofinance_hist(company_idx):
    session = requests.Session()
    # period1 and period2 for max and min date = company value can be formatted
    download_link = 'https://query1.finance.yahoo.com/v7/finance/download/{company}?period1=1514764800&period2=1546214400&interval=1d&events=history&includeAdjustedClose=true'
#https://query1.finance.yahoo.com/v7/finance/download/GPN?period1=1514764800&period2=1546214400&interval=1d&events=history&includeAdjustedClose=true
    try:
        url = download_link.format(company=company_idx)
        response = session.get(url)
        response.raise_for_status()
        
        if response.status_code != 404:
            df = pd.read_csv(StringIO(response.text), parse_dates=['Date'])
    
            # filtering as we only need close price
            df_fil = df[['Date','Close']]
            
            return df_fil
        else:
            #print("exc1")
            return -1
    
    except:
        #print("exc2")
        #error_info.append(company_idx)
        return -1
    

closePrice={}
error_info=[]
error_info_idx=[]

# scraping data for all the companies and removing those for which data could not be scraped
for i in range(len(earning_calls)):
    company_idx = earning_calls[i][9:]
    
    if company_idx not in closePrice.keys():
        df = get_yahoofinance_hist(company_idx)
        if type(df) == int: # in case when the url was not accessible
            error_info.append(earning_calls[i])
            error_info_idx.append(i)
        else:    
            closePrice[company_idx] = df
            


# In[18]:


len(earning_calls)


# In[19]:


# dictionary of closing price for various companies
len(closePrice.keys())


# In[21]:


fil_calls = earning_calls


# In[22]:


# removing all calls for which data was unavailable
for i in sorted(error_info_idx, reverse=True):
    #print(i)
    del fil_calls[i]


# Arranging the close price values for 3 days before and after earning call

indices=[]
tou = 3 # timesteps
grdVals =np.zeros((len(fil_calls),2*tou+2))
idx =0
# there will be 8 timesteps as we need p(i-1) and p(i) to calculate the return price

#for i in range(1):
for call in fil_calls:
    #call = fil_calls[i]
    comp = call[9:]
    df =  closePrice[comp]
    
    date = call[:4]+'-'+call[4:6]+'-'+ call[6:8]
    
    # idx in the dataframe
    dt =-1
    
    for i in range(df.shape[0]):
        
        if str(df.iloc[i]['Date'])[:10]==date:
            dt =i
            indices.append(dt)
    
    if dt == -1:
        print("error")
    
    # taking 3 values before and 3 values after - tou = 3 for window
    vals=[]
    
    if (dt-tou-1)<0:
        # use the same value as close price day
        pre = [df.iloc[dt]['Close']]*(tou+1)
        vals = vals + pre
    else:
        # prev 4 days
        vals = vals + list(df.iloc[dt-tou-1:dt]['Close'])
    
    vals = vals+ [df.iloc[dt]['Close']]
    
    if (dt+tou)>(df.shape[0]):
        # use the same value as close price day
        post = [df.iloc[dt]['Close']]*(tou)
        vals = vals + post
    else:
        vals= vals + list(df.iloc[dt+1:dt+tou+1]['Close'])
    
    #print(idx)
    grdVals[idx]= vals
    idx+=1
    

grdVals = np.array(grdVals)
grdVals_df= pd.DataFrame(grdVals)
grdVals_df.to_csv(r'/Users/vishwa/Desktop/grdVals.csv')


# ## Approach - Getting Direct sentence embeddings from SentenceBERT

get_ipython().system('pip install sentence-transformers')


# In[411]:


from sentence_transformers import SentenceTransformer

# to get the sentence_model
sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')


# In[412]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np


# In[390]:


document = ["I ate dinner.", 
       "Bedford is an existing site that we have."]


# In[393]:


def pad_sent_embed(embed,max_len):
    
    # as we dont want to lose context, we will add blank sentence as a prembedding
    #if len(embed)>120:
        # take first 120 lines
        #lines=lines[:120]
    #else:
        # zero vector
    dim_size = 768
    zer_vec = np.zeros((max_len-len(embed),dim_size))
    embed = np.concatenate((zer_vec,embed),axis=0)
        
    return embed


# In[394]:


token_sent = []
for sentence in document:
    token_sent.append(word_tokenize(sentence.lower()))


# In[ ]:


token_sent


# In[417]:


embeddings = sentence_model.encode(document)
import pickle

#Store sentences & embeddings on disc
with open('/Users/vishwa/Desktop/embeds/embeddings.pkl', "wb") as fOut:
    pickle.dump({'sentences': document, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
with open('/Users/vishwa/Desktop/embeds/embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data['sentences']
    stored_embeddings = stored_data['embeddings']


# In[420]:


stored_embeddings.shape


# In[ ]:


sentence_embeddings = sentence_model.encode(document)


# In[ ]:


len(sentence_embeddings[0])


# In[ ]:


sentence_embeddings


# In[ ]:


sentence_embeddings=pad_sent_embed(sentence_embeddings)
sentence_embeddings


# In[554]:


cnt_ineq=0
file_ineq=0
max_len=0
min_date=30000000
max_date=0
tot_len=0



for i in range(len(fil_calls)):
#for i in range(5): 
    file =fil_calls[i]
          
    text_file_path = r'/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset/'+ file +'/text.txt'
    f = open(text_file_path, "r")
    # to display file content
    content =f.read()
    lines = content.split("\n") # split at new line character
    
    if lines[-1]=='':
        lines = lines[:-1] # last line blank removal
    
    aud_file_path = r'/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset/'+ file +'/features.csv'
    df = pd.read_csv(aud_file_path)
    
    # each sentence is mapped with the corresponding audio features
    # number of sentences in an earning call
    num_of_sent= len(lines)
    
    if len(lines)!=df.shape[0]:
        print("error")
        file_no=i
        cnt_ineq += 1
        
    if num_of_sent > max_len:
        max_len=num_of_sent
        
    tot_len+= len(lines)
        
    val = int(file[:8])

print("Max number of sentences in a call: {}".format(max_len))
print("Avg number of sentences in a call: {}".format(tot_len/len(fil_calls)))

text_data = np.zeros((len(fil_calls),max_len,768))

for i in range(len(fil_calls)):
#for i in range(201,298):
#for i in range(200,201):
    file =fil_calls[i]
    
    text_file_path = r'/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset/'+ file +'/text.txt'
    f = open(text_file_path, "r")
    # to display file content
    content =f.read()
    lines = content.split("\n") # split at new line character
    
    if lines[-1]=='':
        lines = lines[:-1] # last line blank removal
        
    sentence_embeddings = sentence_model.encode(lines)
    sentence_embeddings_padded = pad_sent_embed(sentence_embeddings,max_len)
    
    text_data[i] = np.array(sentence_embeddings_padded)
    
    with open('/Users/vishwa/Desktop/embeds/'+ file +'.pkl', "wb") as fOut:
        pickle.dump({'embeddings': sentence_embeddings_padded}, fOut, protocol=pickle.HIGHEST_PROTOCOL)
    
    if i%10==0:
        print("Iterations complete {}".format(i))


# In[27]:


import pickle
with open('/Users/vishwa/Desktop/embeds/20180122_SFBS.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_embeddings = stored_data['embeddings']


# In[555]:


file_path = "/Users/vishwa/Desktop/embeds"
file_list = os.listdir(file_path)
file_list.remove('.DS_Store')
file_list = sorted(file_list)


# In[28]:


textData = np.zeros((len(fil_calls),max_len,768))

for i in range(len(fil_calls)):
    
    with open('/Users/vishwa/Desktop/embeds/'+fil_calls[i]+'.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_embeddings = stored_data['embeddings']
    
    textData[i] = np.array(stored_embeddings)


# In[29]:
# to pad audio segment data or truncate if necessary
def pad_aud_seg(audio_features):
    # as we dont want to lose context, we will add blank sentence as a prembedding
    # as the last hidden state captures context
    num_features = audio_features.shape[1] #29
    
    # 120 audio segments ~ avg 114
    # to avoid high dimensionality
    max_seg_len = max_len
    seg = audio_features.shape[0]
    if seg >= max_seg_len:
        # take first 120 segments - truncate
        padded_audio = np.array(audio_features[:max_seg_len,:])
    else:
        zero_padding = [[0]*num_features]*(max_seg_len-seg)
        padded_audio = np.concatenate((zero_padding,audio_features),axis=0)
        
    return padded_audio


import math
from sklearn.preprocessing import MinMaxScaler

idx = 0

for i in range(len(fil_calls)):
#for i in range(5): 
    file =fil_calls[i]
        
    text_file_path = r'/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset/'+ file +'/text.txt'
    f = open(text_file_path, "r")
    # to display file content
    content =f.read()
    lines = content.split("\n") # split at new line character
    
    if lines[-1]=='':
        lines = lines[:-1] # last line blank removal
    
    aud_file_path = r'/Users/vishwa/Desktop/MAEC-master/MAEC_Dataset/'+ file +'/features.csv'
    df = pd.read_csv(aud_file_path)
    
    # each sentence is mapped with the corresponding audio features
    # number of audio utterance segments in an earning call
    num_of_seg= df.shape[0]
    
    if len(lines)!=df.shape[0]:
        print("error")
        file_no=i
        cnt_ineq += 1
        
    #print(idx)
    #print(num_of_seg)
    
    # replacing undefined str type
    df = df.replace('--undefined--', float('nan'))
    df = df.replace('--undefined-', float('nan'))
    df = df.replace('--undefined-- ', float('nan'))
    # fill forward to fill these values
    for col in df.columns:
        #try:
           # df[col].fillna( method ='ffill', inplace = True) 
        # to catch exception for all nan values
        #except RuntimeWarning as e:
            #print("error")
        df[col].fillna(0, inplace = True) 
    
    
    audio_feat = np.array(df)
    # pad the audio segment for efficient batching
    padded_values = pad_aud_seg(audio_feat)
    
    audioData[idx] = padded_values
    
    if idx%100==0:
        print("Files completed : {}".format(idx))
    
    idx +=1 


# In[37]:


audioData.shape


# # Computing Past 3 day volatility and next 3 day volatility

# In[390]:


# sending a list of closing price
def compute_volatility(close_pr,close_pr_prev):
    
    return_pr= [pr/pr_prev for pr,pr_prev in zip(close_pr,close_pr_prev)]
    #return_pr = [ (pr)/(pr-1) for pr in close_pr ]
    
    mean_rt = np.mean(return_pr)
    diff_rt = (return_pr-mean_rt)
    if len(close_pr)==0:
        print("error")
    vol = np.log(np.sqrt(sum(np.multiply(diff_rt,diff_rt))/(len(close_pr))))
    
    return vol


# In[391]:


# scraped and processed data
vol_file_path = r'/Users/vishwa/Desktop/grdVals.csv'
closing_pr = pd.read_csv(vol_file_path)
closing_pr= closing_pr.drop(['Unnamed: 0'],axis=1)
closing_pr.columns



# volitility of prev 3 days
pre_vol=[]

# volitility of next 3 days -- var to be predicted
post_vol=[]

for i in range(closing_pr.shape[0]):
    pre_curr = [closing_pr.iloc[i]['1']]+[closing_pr.iloc[i]['2']]+[closing_pr.iloc[i]['3']]
    pre_past = [closing_pr.iloc[i]['0']]+[closing_pr.iloc[i]['1']]+[closing_pr.iloc[i]['2']]
    
    post_curr = [closing_pr.iloc[i]['5']]+[closing_pr.iloc[i]['6']]+[closing_pr.iloc[i]['7']]
    post_past = [closing_pr.iloc[i]['4']]+[closing_pr.iloc[i]['5']]+[closing_pr.iloc[i]['6']]
    
    pre_vol.append(compute_volatility(pre_curr,pre_past))
    post_vol.append(compute_volatility(post_curr,post_past))


# In[393]:


corr=[]
for i in range(len(pre_vol)):
    if pre_vol[i]!=np.float('-inf'):
        corr.append(pre_vol[i])


# In[394]:


for i in range(len(pre_vol)):
    if pre_vol[i]==np.float('-inf'):
        pre_vol[i]= np.mean(corr)


# In[395]:


for i in range(len(post_vol)):
    if post_vol[i]==np.float('-inf'):
        #print("here")
        #print(i)
        post_vol[i]= np.mean(corr)



