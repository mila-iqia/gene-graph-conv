import sys
import pandas as pd
import networkx as nx


def infer_gene(method, genes, gene_to_infer, g, train_size, test_size, trials, penalty=False):
    mean = genes[gene_to_infer].mean()
    labels = [1 if x > mean else 0 for x in genes[gene_to_infer]]
    temp_genes = genes.drop(gene_to_infer, axis=1)

    first_degree_neighbors = temp_genes.loc[:, list(g.neighbors(gene_to_infer))].dropna(axis=1)
    second_degree_neighbors = temp_genes.loc[:, list(set(nx.ego_graph(g, gene_to_infer, radius=2)))].dropna(axis=1)

    full_results = method(temp_genes, labels, trials, train_size, test_size, penalty=penalty)

    if len(first_degree_neighbors.columns) == 0:
        first_degree_results = (.5, .0)
    else:
        first_degree_results = method(first_degree_neighbors, labels, trials, train_size, test_size, penalty=penalty)

    if len(second_degree_neighbors.columns) == 0:
        second_degree_results = (.5, .0)
    else:
        second_degree_results = method(second_degree_neighbors, labels, trials, train_size, test_size, penalty=penalty)

    first_degree_diff = full_results[0] - first_degree_results[0]
    second_degree_diff = full_results[0] - second_degree_results[0]

    data = {"gene_name": gene_to_infer,
            "auc": full_results[0],
            "std": full_results[1],
            "first_degree_auc": first_degree_results[0],
            "first_degree_std": first_degree_results[1],
            "first_degree_diff": first_degree_diff,
            "second_degree_diff": second_degree_diff,
            "second_degree_auc": second_degree_results[0],
            "second_degree_std": second_degree_results[1],
            }
    return pd.DataFrame(data, [0])


def infer_all_genes(method, genes, g, train_size, test_size, trials, penalty):
    results = pd.DataFrame(columns=["gene_name",
                                    "auc",
                                    "std",
                                    "first_degree_auc",
                                    "first_degree_std",
                                    "second_degree_auc",
                                    "second_degree_std",
                                    "first_degree_diff",
                                    "second_degree_diff"])
    print "Genes to do:" + str(len(g.nodes))
    sys.stdout.write("Gene number:")
    for index, gene in enumerate(genes):
        # This throws errors sometimes when we give really unbalanced training sets. Those genes aren't particularly interesting for this usecase though, so we can ignore them,
        try:
            data = infer_gene(method, genes, gene, g, train_size, test_size, trials, penalty)
            results = results.append(pd.DataFrame(data, index=range(0, len(data))))
        except:
            continue
        sys.stdout.write(str(index) + ", ")
    results.to_csv('results.csv')
    return results


def sample_neighbors(g, gene, num_neighbors):
    results = set([gene])
    all_nodes = set(g.nodes)
    first_degree = set(g.neighbors(gene))
    second_degree = set()
    for x in g.neighbors(gene):
        second_degree = second_degree.union(set(g.neighbors(x)))
    while len(results) < num_neighbors:
        if len(first_degree - results) > 0:
            unique = first_degree - results
            results.add(unique.pop())
        elif len(second_degree - results) > 0:
            unique = second_degree - results
            results.add(unique.pop())
        else:
            unique = all_nodes - results
            results.add(unique.pop())
    return results
