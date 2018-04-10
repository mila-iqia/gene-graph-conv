import sys
import pandas as pd


def infer_gene(method, genes, gene_to_infer, train_size, test_size, trials, penalty=False):
    mean = genes[gene_to_infer].mean()
    labels = [1 if x > mean else 0 for x in genes[gene_to_infer]]
    temp_genes = genes.drop(gene_to_infer, axis=1)

    results = method(temp_genes, labels, trials, train_size, test_size, penalty=penalty)

    data = {"gene_name": gene_to_infer,
            "auc": results[0],
            "std": results[1]
            }
    return pd.DataFrame(data, [0])


def infer_all_genes(method, genes, train_size, test_size, trials, penalty):
    results = pd.DataFrame(columns=["gene_name",
                                    "auc",
                                    "std"])
    sys.stdout.write("num genes: " + str(len(genes.columns)))

    for index, gene in enumerate(genes):
        sys.stdout.write(str(index) + ", ")
        # This throws errors sometimes when we give really unbalanced training sets. Those genes aren't particularly interesting for this usecase though, so we can ignore them,
        try:
            data = infer_gene(method, genes, gene, train_size, test_size, trials, penalty)
            results = results.append(pd.DataFrame(data, index=range(0, len(data))))
        except:
            continue
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
