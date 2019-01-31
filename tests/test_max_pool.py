import os
import unittest
from models import utils
import numpy as np
import torch
from torch.autograd import Variable


class MaxPoolTestSuite(unittest.TestCase):
    """Test cases on the academictorrents.py file."""

    def test_identity_max_pool_torch_scatter(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 1]))
        expected_result = torch.tensor([[1., 2., 3.], [5., 6., 7.]])
        x = utils.max_pool_torch_scatter(x, centroids)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_torch_scatter(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[5., 6., 7.]])
        x = utils.max_pool_torch_scatter(x, centroids)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_torch_scatter_batches(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]], [[13, 2, 34], [5, 65, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[[5., 6., 7.]], [[13., 65., 34.]]])
        x = utils.max_pool_torch_scatter(x, centroids)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_big(self):
        x = torch.FloatTensor(np.load("tests/x.npy"))
        centroids = torch.LongTensor(np.load("tests/centroids.npy"))
        adj = torch.FloatTensor(np.load("tests/adj.npy"))
        res = np.load("tests/res.npy")
        x = utils.max_pool_torch_scatter(x, centroids, adj)
        self.assertTrue((x.numpy() == res).all())

    def test_identity_max_pool_dense(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 1]))
        expected_result = torch.tensor([[1., 2., 3.], [5., 6., 7.]])
        adj = torch.FloatTensor(np.array([[1, 0], [0, 1]]))

        x = utils.max_pool_dense(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[5., 6., 7.]])
        adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
        x = utils.max_pool_dense(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_batches(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]], [[13, 2, 34], [5, 65, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()
        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[[5., 6., 7.]], [[13., 65., 34.]]])
        adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
        x = utils.max_pool_dense(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_big(self):
        x = torch.FloatTensor(np.load("tests/x.npy"))
        centroids = torch.LongTensor(np.load("tests/centroids.npy"))
        adj = torch.FloatTensor(np.load("tests/adj.npy"))
        res = np.load("tests/res.npy")
        x = utils.max_pool_dense(x, centroids, adj)
        self.assertTrue((x.numpy() == res).all())

    def test_identity_max_pool_dense_iter(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()

        centroids = torch.LongTensor(np.array([0, 1]))
        expected_result = torch.tensor([[1., 2., 3.], [5., 6., 7.]])
        adj = torch.FloatTensor(np.array([[1, 0], [0, 1]]))
        x = utils.max_pool_dense_iter(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_iter(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()

        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[5., 6., 7.]])
        adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
        x = utils.max_pool_dense_iter(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_iter_batches(self):
        # 1 batches of 2 genes with 3 channels each (ex, node, channel)
        x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]], [[13, 2, 34], [5, 65, 7]]])), requires_grad=False).float()
        x = x.permute(0, 2, 1).contiguous()

        centroids = torch.LongTensor(np.array([0, 0]))
        expected_result = torch.tensor([[[5., 6., 7.]], [[13., 65., 34.]]])
        adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
        x = utils.max_pool_dense_iter(x, centroids, adj)
        self.assertTrue((x.numpy() == expected_result.numpy()).all())

    def test_max_pool_dense_iter_big(self):
        x = torch.FloatTensor(np.load("tests/x.npy"))
        centroids = torch.LongTensor(np.load("tests/centroids.npy"))
        adj = torch.FloatTensor(np.load("tests/adj.npy"))
        res = np.load("tests/res.npy")
        x = utils.max_pool_dense_iter(x, centroids, adj)
        self.assertTrue((x.numpy() == res).all())

    # def test_identity_sparse_max_pool(self):
    #     # 1 batches of 2 genes with 3 channels each (ex, node, channel)
    #     x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
    #     x = x.permute(0, 2, 1).contiguous()
    #
    #     centroids = torch.LongTensor(np.array([0, 1]))
    #     expected_result = torch.tensor([[1., 2., 3.], [5., 6., 7.]])
    #     adj = torch.FloatTensor(np.array([[1, 0], [0, 1]]))
    #
    #     x = utils.sparse_max_pool(x, centroids, adj)
    #     self.assertTrue((x.numpy() == expected_result.numpy()).all())
    #
    # def test_sparse_max_pool(self):
    #     # 1 batches of 2 genes with 3 channels each (ex, node, channel)
    #     x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]]])), requires_grad=False).float()
    #     x = x.permute(0, 2, 1).contiguous()
    #     centroids = torch.LongTensor(np.array([0, 0]))
    #     expected_result = torch.tensor([[5., 6., 7.]])
    #     adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
    #     x = utils.sparse_max_pool(x, centroids, adj)
    #     self.assertTrue((x.numpy() == expected_result.numpy()).all())
    #
    # def test_sparse_max_pool_batches(self):
    #     # 2 batches of 2 genes with 3 channels each (ex, node, channel)
    #     x = Variable(torch.FloatTensor(np.array([[[1, 2, 3], [5, 6, 7]], [[13, 2, 34], [5, 65, 7]]])), requires_grad=False).float()
    #     x = x.permute(0, 2, 1).contiguous()
    #     centroids = torch.LongTensor(np.array([0, 0]))
    #     expected_result = torch.tensor([[[5., 6., 7.]], [[13., 65., 34.]]])
    #     adj = torch.FloatTensor(np.array([[1, 1], [1, 1]]))
    #     x = utils.sparse_max_pool(x, centroids, adj)
    #     self.assertTrue((x.numpy() == expected_result.numpy()).all())
    #
    # def test_sparse_max_pool_big(self):
    #     x = torch.FloatTensor(np.load("tests/x.npy"))
    #     centroids = torch.LongTensor(np.load("tests/centroids.npy"))
    #     adj = torch.FloatTensor(np.load("tests/adj.npy"))
    #     res = np.load("tests/res.npy")
    #     import pdb; pdb.set_trace()
    #
    #     x = utils.sparse_max_pool(x, centroids, adj)
    #     self.assertTrue((x.numpy() == res).all())
    #

if __name__ == '__main__':
    unittest.main()
