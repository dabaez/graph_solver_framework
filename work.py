import statistics

from framework.experiment.utils import load_dataset

dataset = load_dataset("full")
graph_sizes = []
for graph in dataset:
    with graph as g:
        graph_sizes.append(g.number_of_nodes())

# graph sizes statistics
print(f"Number of graphs: {len(graph_sizes)}")
print(f"Min graph size: {min(graph_sizes)}")
print(f"Max graph size: {max(graph_sizes)}")
print(f"Mean graph size: {statistics.mean(graph_sizes)}")
print(f"Median graph size: {statistics.median(graph_sizes)}")
print(f"Standard deviation of graph sizes: {statistics.stdev(graph_sizes)}")
