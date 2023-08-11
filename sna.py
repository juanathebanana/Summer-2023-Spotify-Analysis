import itertools
import networkx as nx
import matplotlib.pyplot as plt
from sentiment_analysis import df1
import ast
import numpy as np
import matplotlib.cm as cm
import pandas as pd
df1['genres'] = df1['genres'].apply(ast.literal_eval)


G = nx.Graph()

for i, row in df1.iterrows():
    genres = row['genres']
    # Create edges for all combinations of genres (without repetitions)
    G.add_edges_from(itertools.combinations(genres, 2))

genres = list(set([genre for sublist in df1['genres'] for genre in sublist]))

colors = cm.rainbow(np.linspace(0, 1, len(genres)))
color_map = dict(zip(genres, colors))

# Assign colors to nodes in the graph based on their labels
node_colors = [color_map[node] for node in G.nodes]

# Compute node sizes based on degree centrality
node_sizes = [30* degree for node, degree in G.degree()]

# Draw the graph
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=node_sizes, font_size=5)
plt.show()


adjacency_matrix = nx.adjacency_matrix(G)
adjacency_df = pd.DataFrame(adjacency_matrix.toarray(), index=G.nodes(), columns=G.nodes())

# Display the DataFrame
print(adjacency_df)



degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
clustering_coefficient = nx.clustering(G)
# Create a DataFrame
metrics_df = pd.DataFrame({
    'Degree Centrality': degree_centrality,
    'Betweenness Centrality': betweenness_centrality,
})

print(metrics_df)
#create a dataframe with specif metrics
sorted_metrics_df = metrics_df.sort_values(by='Degree Centrality', ascending=False)
print(sorted_metrics_df)
metrics_df.plot(kind='bar', figsize=(10,10))
plt.title('Graph Metrics for Each Genre')
plt.ylabel('Metric Value')
plt.xlabel('Genre')
plt.show()

