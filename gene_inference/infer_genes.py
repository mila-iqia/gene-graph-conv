import sys
import pandas as pd
import numpy as np

def infer_gene(method, dataset, gene_to_infer, train_size, test_size, trials, model=None, penalty=False):
    mean = dataset.df[gene_to_infer].mean()
    dataset.labels = [1 if x > mean else 0 for x in dataset.df[gene_to_infer]]
    temp_df = dataset.df.copy()
    dataset.df = dataset.df.drop(gene_to_infer, axis=1)
    dataset.df = dataset.df - dataset.df.mean(axis=0)
    try:
        results = method(dataset, trials, train_size, test_size, penalty=penalty)
    except Exception as e:
        dataset.df = temp_df
        raise e

    dataset.df = temp_df
    data = {"gene_name": gene_to_infer,
            "auc": results[0],
            "std": results[1]
            }
    return pd.DataFrame(data, [0])


def infer_all_genes(method, genes, train_size, test_size, trials, penalty):
    results = pd.DataFrame(columns=["gene_name",
                                    "auc",
                                    "std"])
    sys.stdout.write("num genes: " + str(len(genes.df.columns)))

    for index, gene in enumerate(genes.df):
        sys.stdout.write(str(index) + ", ")
        # This throws errors with really unbalanced training sets.
        try:
            data = infer_gene(method, genes, gene, train_size, test_size, trials, penalty)
            results = results.append(pd.DataFrame(data, index=range(0, len(data))))
        except Exception as e:
            print e
    results.to_csv('results.csv')
    return results

def compute_diff(results_1, results_2):
    union_full = results_1.loc[results_1['gene_name'].isin(results_2['gene_name'])]
    diff = results_1.loc[results_1['gene_name'].isin(results_2['gene_name'])]

    all_diff = []
    all_gene = []
    for gene in results_2['gene_name']:

        if gene not in list(results_1['gene_name']):
            continue

        # print gene
        diff = np.array(results_1.loc[results_1['gene_name'] == gene]['auc'])[0] - \
               np.array(results_2.loc[results_2['gene_name'] == gene]['auc'])[0]
        # print diff
        all_diff.append(diff)
        all_gene.append(gene)

    return all_diff, all_gene

def infer_all_genes_selective(dataset, graph, method, train_size, test_size, trials):

    # Do the same, but for first order neighbours
    results = pd.DataFrame([])
    real_df = dataset.df.copy()

    for i, gene in enumerate(dataset.node_names):
        # print gene
        if i % 100 == 0:
            print i
        try:
            first_degree = set(graph.neighbors(gene))
            first_degree.add(gene)

            dataset.df = dataset.df.loc[:, first_degree]
            results = results.append(
                infer_gene(method, dataset, gene, train_size, test_size=test_size, trials=trials, penalty=True)).reset_index(drop=True)
        except Exception as e:
            print gene, e, " Failed!"
        dataset.df = real_df

    return results

def sample_neighbors(g, gene, num_neighbors, include_self=True):
    results = set([])
    if include_self:
        results = set([gene])
    all_nodes = set(g.nodes)
    first_degree = set(g.neighbors(gene))
    second_degree = set()
    for x in g.neighbors(gene):
        second_degree = second_degree.union(set(g.neighbors(x)))
    while len(results) < num_neighbors:
        if len(first_degree) - len(results) > 0:
            unique = sorted(first_degree - results)
            results.add(unique.pop())
        elif len(second_degree) - len(results) > 0:
            unique = sorted(second_degree - results)
            results.add(unique.pop())
        else:
            unique = sorted(all_nodes - results)
            results.add(unique.pop())
    return results
