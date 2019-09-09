# Skyline_Uncertain_Graph

DySky: Dynamic Skyline Queries on Uncertain Graphs

Given a graph, and a set of query vertices (subset of the vertices), the dynamic skyline query problem returns a subset of data vertices
(other than query vertices) which are not dominated by other data vertices based on certain distance measure. In this paper, we study the dynamic skyline query problem on uncertain graphs (DySky).
The input to this problem is an uncertain graph, a subset of its nodes as query vertices, and the goal here is to return all the data
vertices which are not dominated by others. We employ two distance measures in uncertain graphs, namely, Majority Distance, and Expected Distance. Our approach is broadly divided into three steps:
Pruning, Distance Computation, and Skyline Vertex Set Generation.

Here, we have used 2 distance measures Expected Distance and Majority Distance.
The query is generated from size 2, 3, 5, 8, 10, with different query selection straties. 

(i) Random Query selection (RAND)

(ii) High degree node as query (HDEG)

(iii) High clustering coefficient (CLUST)

Execution command is as follows:

python skyline_uncertain_graph.py 'dataset-name' 'Distance-type' 'query-size' 'No of runs' 'Query-selection-strategy' 


Query-selection-strategy = ['MD', 'ED', 'ALL']

e.g. 
python skyline_uncertain_graph.py test_data ED 5 1 CLUST 

Note:
datasets should be kept in the folder ./datasets/
results should be kept in the folder ./results/<dataset-name>/

Demo execution and its result is shown for test-data.csv

Dependecy package for BNL (Block nested loop) is pypref https://github.com/patrickroocks/pypref

Please do cite the following paper incase you are using this code.

Suman Banerjee, Bithika Pal, Akshit Bhalla, and Mamata Jenamani. 
DySky: Dynamic Skyline Queries on Uncertain Graphs. ACM Joint International Conference, 9 pages. https://doi.org/xxxx
