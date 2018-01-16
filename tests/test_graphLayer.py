import unittest
import graphLayer
import numpy as np
import torch
from torch.autograd import Variable

class TestMaxPooling(unittest.TestCase):
    def setUp(self):
        self.sqrt_nodes = 3
        self.nb_nodes = self.sqrt_nodes*self.sqrt_nodes
        self.adj = np.identity(self.nb_nodes)
        self.adj_p1 = np.identity(self.nb_nodes)
        self.adj_p2 = np.identity(self.nb_nodes)

        def addIfOk(i, j):
            if j >=0 and j < self.nb_nodes:
                self.adj[i, j] = 1.
                self.adj[j, i] = 1.

        # It's a grid like graph. (i.e. like CNNs)
        for i in range(self.nb_nodes):
            addIfOk(i, i+self.sqrt_nodes)
            addIfOk(i, i-self.sqrt_nodes)

            if (i+1) % self.sqrt_nodes != 0.:
                addIfOk(i, i+1)
            if i % self.sqrt_nodes != 0.:
                addIfOk(i, i-1)

        # add the neighbours of neighbours.
        self.adj_p1 = self.adj.copy()
        for n1 in range(self.nb_nodes):
            for n2 in range(self.nb_nodes):
                if self.adj[n1, n2]:
                    self.adj_p1[n1] += self.adj[n2]

        self.adj_p1 = (self.adj_p1 > 0.).astype(float)

        # redo it another time.
        self.adj_p2 = self.adj_p2.copy()
        for n1 in range(self.nb_nodes):
            for n2 in range(self.nb_nodes):
                if self.adj_p1[n1, n2]:
                    self.adj_p2[n1] += self.adj_p1[n2]

        self.adj_p2 = (self.adj_p2 > 0.).astype(float)

    def testConnectivity(self):
        # Test if we are able to augment the conenctivity of the nodes
        gc = graphLayer.AugmentGraphConnectivity(kernel_size=1)

        adj_p1 = gc(self.adj)
        np.testing.assert_almost_equal(adj_p1, self.adj_p1)

        # Bigger kernel size.
        gc = graphLayer.AugmentGraphConnectivity(kernel_size=2)
        adj_p2 = gc(self.adj)
        np.testing.assert_almost_equal(adj_p2, self.adj_p2)


    def testIdToKeep(self):
        # Given a graph and a list of index to keep, can we do the max pooling?
        x = np.arange(self.nb_nodes * 2) .reshape((1, 2, self.nb_nodes))# Our value
        to_keep = np.arange(0, self.nb_nodes) % 2  # We keep one node on two.
        solution = np.zeros((9,2))

        for i in range(self.nb_nodes):
            if to_keep[i]:
                #import ipdb; ipdb.set_trace()
                for c in range(2):
                    solution[i, c] = np.max(x[:, c] * self.adj_p1[i]) # It's the max of the neighbours

        agr = graphLayer.AgregateGraph(torch.FloatTensor(self.adj_p1), to_keep)

        # import ipdb; ipdb.set_trace()
        #print solution, x.transpose((0, 2, 1))

        proposed = agr(Variable(torch.FloatTensor(x.transpose((0, 2, 1))), requires_grad=False)).data

        np.testing.assert_almost_equal(proposed.numpy().reshape(-1), solution.reshape(-1))
