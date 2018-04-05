
def get_second_degree(gene, g):
    l = list(g.neighbors(gene))
    neighbors = l[:]
    for index, gene in enumerate(l):
        neighbors.extend(list(g.neighbors(gene)))
    return neighbors
