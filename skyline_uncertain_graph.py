#!/usr/bin/env python
# coding: utf-8

# ### Author: Bithika (Date-12-AUG-2019)

# In[1]:
from matplotlib import pyplot as plt

import sys as sys
import networkx as nx
import pandas as pd
import random as r
import math as m
import time as t
import pypref as p
import csv as csv
import matplotlib.pyplot as plt
from decimal import Decimal
from tqdm import tqdm
from functools import reduce
from collections import defaultdict


# In[64]:


dataset_name = str(sys.argv[1])
distance_type = str(sys.argv[2])  ## ALL, MD, ED
num_query_nodes = int(sys.argv[3])  ## 2, 3, 5, 8, 10, 15, 20
run = int(sys.argv[4])  ###  cross-fold  5 or 10
global query_sel_strategy
query_sel_strategy = str(sys.argv[5])  ### RAND, HDEG, CLUST -- all are boubnded distance with two hop


# In[65]:


global MyTime , high_degree_threshold, clust_threshold, result_path
result_path = './results/'
MyTime = []
high_degree_threshold = 2  ##### dataset dependent p2p8k -- 15 (440), road129k -- 3 (20597) , 2 for (road-minnesota -1107)
clust_threshold = 0 ##### dataset dependent  p2p8k -- 0 (1349), road129k -- 0 (11654),  0for (road-minnesota - 354)
samples = 100



# In[66]:


print("****************************************************************")
print("****************************************************************")
print("Data Set: ", dataset_name)
print("distance_type: ", distance_type)
print("Number of Query Nodes: ", num_query_nodes)
print("Query Selection Strategy: ", query_sel_strategy)
print("Cross-fold Run: ", run)
print("Sample world:", samples)
print("****************************************************************")
print("****************************************************************")



# In[67]:


# Graph Generation
print("************    Graph Generation from data  *******************")
start_time = t.time() # START TIME
G = nx.Graph()       
MyFile = "./datasets/" + dataset_name + ".csv"

MyData = pd.read_csv(MyFile)
print(MyData)
edge_att = {}
for i in range(len(MyData)):
    a = int(MyData.iloc[i,0])
    b = int(MyData.iloc[i,1])
    c = Decimal(MyData.iloc[i,2])
    d = int(MyData.iloc[i,3])
    G.add_edge(a, b)
    edge_att[(a,b)] = {
        "p": c,
        "w": d
    }
nx.set_edge_attributes(G, edge_att)
MyTime.append(("Graph Generation:", t.time() - start_time)) # END TIME
print(MyTime[-1])

if dataset_name == 'test_data':
	samples = 2**G.number_of_nodes()
	print("samples", samples)

print("Nodes:",G.number_of_nodes(), "edges:",G.number_of_edges())
# In[68]:


# Hop Tree (Upto 2 hop Neighbors) Generation
def CreateHopDic_hop2(MyGraph):
    HopDic = {}
    for i in tqdm(list(MyGraph.nodes)):
        HopDic[i] = {}
        temp_i = set(nx.neighbors(MyGraph, i)) | {i}
        for j in set(nx.neighbors(MyGraph, i)):
            val_j = set(nx.neighbors(MyGraph, j)) - temp_i
            HopDic[i][j] = val_j
    return HopDic
print("Hop Tree Generation")
start_time = t.time() # START TIME
HopDic_G = CreateHopDic_hop2(G)
MyTime.append(("HopDic_2 creation:",t.time() - start_time)) # END TIME
print(MyTime[-1])


# In[69]:


#HopDic_G
def HopTreeNodes(num):
    neig = nx.neighbors(G, num)
    all_des = reduce(set.union, HopDic_G[num].values())
    return set(neig)|set(all_des)


# In[70]:


def initial_query_selection(G, query_sel_strategy):
    if query_sel_strategy.upper() == 'RAND':
        q1 = r.randint(1,len(G.nodes))
        return q1
    elif query_sel_strategy.upper() == 'HDEG':
        query_list = [i for (i,j) in sorted(list(G.degree()), key=lambda x: x[1],  reverse=True)                       if j >high_degree_threshold]
        q1 = r.choice(list(query_list))
        return q1
    elif query_sel_strategy.upper() == 'CLUST':
        clust_G = nx.clustering(G)
        query_list = [i for i in clust_G.keys()                       if clust_G[i] > clust_threshold]
        q1 = r.choice(list(query_list))
        return q1
    


# In[71]:


#######  query vertex generation
def get_query(G, HopDic_G, num_query_nodes, query_sel_strategy):
    Q = []
    Q.append(initial_query_selection(G, query_sel_strategy))
    temp = set()
    i = 0
    while i < (num_query_nodes - 1):
        temp = temp|HopTreeNodes(Q[i])
        temp = temp - set(Q)
        
        if len(temp) == 0:
            Q = []
            Q.append(initial_query_selection(G, query_sel_strategy))
            i = 0
        else:
            Q.append(r.choice(list(temp)))
            i  = i + 1
    print("***********  Query Points ***********")  
    print(Q)
    print("*************************************") 
    return Q


# In[72]:


######  BFS Prunning   #####
def candidate_selection(G, Q):
    #### BFS prunning 
    new_hop_dic_G=defaultdict(set)
    for q in Q:
        T=nx.bfs_tree(G, q, reverse=False)
        for i in T.nodes():
            new_hop_dic_G[i].add(q)
        
    #print(len(new_hop_dic_G))
    for i in new_hop_dic_G.keys():
        if set(Q) != new_hop_dic_G[i] :
            del new_hop_dic_G[i]
    
    for q in Q:
        try:
            del new_hop_dic_G[q]
        except:
            pass
    CL = list(new_hop_dic_G.keys())
    
    ####  weight based -- expected distance bound (here 4 hop)
    CL_m = CL.copy()
    for d in CL:
        for q in Q:
            try:
                dis, path = nx.single_source_dijkstra(G, d, target=q, cutoff=400, weight='w') ### 400 for road129k, else 180
            except:
                CL_m.remove(d) 
                break
                
    print("******************************************************")   
    print("BFS prunning size:", len(CL), "weigh based prunning size:", len(CL_m))
    print("******************************************************") 
    return CL, CL_m


# In[73]:



# Create Sample Graph And Find Probability Of Existence
def Graph_and_Prob(G, edge_att):
    Gr = G.copy()
    prob = 1
    for i in range(len(G.edges)):
        test_prob = r.uniform(0,1)
        a, b = list(edge_att.keys())[i]
        if Gr[a][b]["p"] < test_prob:
            prob *= (1 - Gr[a][b]["p"])
            Gr.remove_edge(a,b)
        else:
            prob *= Gr[a][b]["p"]
    return Gr, prob

# Graph Sampling And Existence Probability
print("*****************Graph Sampling And Existence Probability*********************")
start_time = t.time() # START TIME
GraphProbList = []
for i in tqdm(range(samples)):
    Gr, prob = Graph_and_Prob(G, edge_att)
    GraphProbList.append((Gr, prob))
MyTime.append(("Sample graph generation:",t.time() - start_time)) # END TIME
print(MyTime[-1])


def plot_graph(G):
	plt.figure(figsize=(20,10))
	pos=nx.fruchterman_reingold_layout(G)
	#pos=nx.circular_layout(G)
	#nx.draw(G)
	nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='cyan')
	nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=6)
	nx.draw_networkx_labels(G, pos, font_size=30, font_family='sans-serif')
	labels = nx.get_edge_attributes(G, 'p')
	for i in labels.keys():
		labels[i] = round(float(labels[i]),2)
	print(labels)
	labels_1 = nx.get_edge_attributes(G, 'w')
	labels_2 ={}
	for i in labels_1.keys():
		labels_2[i] = (labels[i],labels_1[i])
	print(labels_2)
	#nx.draw_networkx_edge_labels(G,pos,edge_labels= labels)
	nx.draw_networkx_edge_labels(G,pos,edge_labels= labels_2, font_size=30)
	
	plt.savefig('test_data_graph.png')
	
if dataset_name == 'test_data':
	plot_graph(G)
	print("Graph probabilities:")
	print([float(GraphProbList[i][1]) for i in range(len(GraphProbList))])
	#for i in range(len(GraphProbList)):
	#	plot_graph(GraphProbList[i][0])


# In[74]:


# Finding P(s,t)(d)
def PST(v, q, samples, GraphProbList):
    pstd = defaultdict(int)
    pstinf = 0
    for i in range(samples):
        Gr = GraphProbList[i][0]
        try:
            short_dist = nx.dijkstra_path_length(Gr, source=v, target=q, weight='w')
            pstd[short_dist] = pstd[short_dist] + GraphProbList[i][1]
        except:
            pstinf = pstinf + GraphProbList[i][1]
    return pstd, pstinf


# In[75]:


# Major Distance Table 
def MD(G, CL, Q, CL_names, Q_names, samples, GraphProbList):
    print("Major Distance Table")
    MajorDistDF = pd.DataFrame(9999, index = CL_names, columns = Q_names) 
    for i, v in tqdm(enumerate(CL)):    
        for j, q in enumerate(Q):
            pstd, pstinf = PST(v, q, samples, GraphProbList)
            if len(pstd.keys()) != 0:
                max_dist_prob_v = max(pstd.values()) ### 
                majority_dist = [d_value for d_value in pstd.keys() if pstd[d_value] == max_dist_prob_v]
                MajorDistDF.iloc[i,j] = min(majority_dist)
            
    return MajorDistDF


# In[76]:





def ExpDist(G,v,q):
    MyList = []
    MyPath = list(nx.all_simple_paths(G, source = v, target = q, cutoff = 4))
    for i in range(len(MyPath)):
        #print(MyPath[i])
        p = 1
        w = 0
        for j in range(len(MyPath[i])-1):
            a = MyPath[i][j]
            b = MyPath[i][j+1]
            
            p *= G[a][b]["p"]
            w += G[a][b]["w"]
            #print("p", p, "edge prob:", a, b, G[a][b]["p"])
            #print("w", w, "edge weight:", a, b, G[a][b]["w"])
        MyList.append((p,w))
    dist = 0
    p_sum = 0
    for p, w in MyList:
        dist += p*w
        p_sum += p
    #print(dist, p_sum)
    dist = dist/p_sum
    #print(int(dist))
    return dist

# Expected Distance Table 
def ED(G, CL, Q, CL_names, Q_names):
    print("Expected Distance Table")
    ExpDistDF = pd.DataFrame(9999, index = CL_names, columns = Q_names)
    distances = {}
    for v in tqdm(CL):
        temp = []
        for q in Q:
            try:
                dist = ExpDist(G, v, q)
            except ZeroDivisionError:
                dist = 9999               
            temp.append(dist)
            ExpDistDF.loc['D'+str(v)]['Q'+str(q)] = dist
        distances[v] = temp
    return ExpDistDF


# In[78]:



# Finding Skylines
def FindSkyline(dataset):
    print("Finding Skylines")
    pref = p.low(dataset.columns[0])
    for i in list(dataset.columns)[1:]:
        pref *= p.low(i)
    return pref.psel(dataset)

# Writing To File
def WriteMyFile(MyFile, row):
    with open (MyFile, "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()


# In[79]:



def CallMe(name, G, BFS_CL, CL, Q, CL_names, Q_names, samples, GraphProbList):
    if name == "MD":
        start_time = t.time() # START TIME
        data = MD(G, CL, Q, CL_names, Q_names, samples, GraphProbList)
        MyTime.append(("MD Distance Calculation:",t.time() - start_time)) # END TIME
        print(MyTime[-1])
    elif name == "ERD":
        start_time = t.time() # START TIME
        data = ERD(G, CL, Q, CL_names, Q_names, samples, GraphProbList)
        MyTime.append(("ERD Distance Calculation:",t.time() - start_time)) # END TIME
        print(MyTime[-1])
    elif name == "ED":
        start_time = t.time() # START TIME
        data = ED(G, CL, Q, CL_names, Q_names)
        MyTime.append(("ED Distance Calculation:",t.time() - start_time)) # END TIME
        print(MyTime[-1])
    
    start_time = t.time() # START TIME
    sky = FindSkyline(data)
    MyTime.append(("Skyline Computation:",t.time() - start_time)) # END TIME
    print(MyTime[-1])
    print("*******************************************************************")
    print(["query size:", str(len(Q)),            "skyline size:", str(len(sky.index)),            "Query:", Q,            "Skyline:", list(sky.index)] )
    print("******************************************************************")
    
    file1 = dataset_name+"_"+ name+"_"+query_sel_strategy+"_query_and_skyline_data_" + str(len(Q)) + ".csv"
    file2 = dataset_name+"_"+ name+"_"+query_sel_strategy+"_query_candidate_skyline_size_data_" + str(len(Q)) + ".csv"
    
    WriteMyFile(result_path + dataset_name + '/' + file1, ["query size:", str(len(Q)),                                                            "skyline size:", str(len(sky.index)),                                                            "Query:", Q,                                                            "Skyline:", list(sky.index)])
    WriteMyFile(result_path + dataset_name + '/' + file2, ["query size:", str(len(Q)) ,                                                            "BFS candidate size:" ,str(len(BFS_CL)),                                                            "candidate size:" ,str(len(CL)),                                                            "skyline size", str(len(sky.index))])
    
    return data


# In[80]:


##### Cross fold run  #########################
itr = 0
while itr < run:
    print("############ Cross-fold  ########### : ", i, "###")
    # Query Vertex Generation
    print("Query Vertex Generation")
    start_time = t.time() # START TIME
    if dataset_name == 'test_data':
        Q = [2,4]
    else:
        Q = get_query(G, HopDic_G, num_query_nodes, query_sel_strategy)
    MyTime.append(t.time() - start_time) # END TIME
    print(Q)
    Q_names = []
    for i in Q:
        Q_names.append("Q" + str(i))
        
    # Candidate Generation
    print("Candidate Generation")
    start_time = t.time() # START TIME
    BFS_CL, CL_m = candidate_selection(G, Q)
    MyTime.append(t.time() - start_time) # END TIME
    if len(BFS_CL) == 0:
        continue
    if len(CL_m) == 0:
        #CL = BFS_CL.copy()
        continue
    else:
        CL = CL_m.copy()
        
    CL_names = []
    for i in CL:
        CL_names.append("D" + str(i))
    print("Candidate vertices -- length: ", len(CL))
    
    if distance_type == "MD":
        data_MD = CallMe("MD", G, BFS_CL, CL, Q, CL_names, Q_names, samples, GraphProbList)
    elif distance_type == "ED":
        data_ED = CallMe("ED", G, BFS_CL, CL, Q, CL_names, Q_names, samples, GraphProbList)
    elif distance_type == "ALL":
        data_MD = CallMe("MD", G, BFS_CL, CL, Q, CL_names, Q_names, samples, GraphProbList)
        data_ED = CallMe("ED", G, BFS_CL, CL, Q, CL_names, Q_names, samples, GraphProbList)
    else:
        print("Invalid Distance Measure...")
    itr = itr + 1
    file3 = dataset_name + "_execution_times_" + str(num_query_nodes) +"_" + query_sel_strategy+ ".csv"
    WriteMyFile(result_path + dataset_name + '/' + file3, MyTime)

if dataset_name == 'test_data':
    print("Majority Distance Table")
    print(data_MD)
    print("Expected Distance Table")
    print(data_ED)
    ### static graph shortest path  ###
    Static_Dist_DF = pd.DataFrame(9999, index = CL_names, columns = Q_names)
    print(CL, CL_names)
    for v in CL:
        for q in Q:
            short_dist = nx.dijkstra_path_length(G, source=v, target=q, weight='w')
            #print(v, q, short_dist)             
            Static_Dist_DF.loc['D'+str(v)]['Q'+str(q)] = short_dist
    sky = FindSkyline(Static_Dist_DF)
    print("*******************************************************************")
    print(["query size:", str(len(Q)), "skyline size:", str(len(sky.index)), "Query:", Q, "Skyline:", list(sky.index)] )
    print("******************************************************************")
    
    print("Static graph shortest disteance")
    print(Static_Dist_DF)

# In[ ]:




