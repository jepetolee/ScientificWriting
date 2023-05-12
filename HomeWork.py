import torch
import PIL

from torchvision import transforms
import time
import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
class WisePooling(Module):

    def __init__(self):
        super(WisePooling, self).__init__()

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor_list = list()
            for j in range(graph[i].shape[0]):
                shot_boundary = graph[i][j]
                tensor_list.append(
                    torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                              shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class WiseConvolution(Module):

    def __init__(self, input_size, output_size):
        super(WiseConvolution, self).__init__()
        self.WiseConv = nn.Linear(input_size, output_size)

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor_list = list()
            for j in range(graph[i].shape[0]):
                shot_boundary = graph[i][j]

                tensor_list.append(torch.sum(self.WiseConv(input[i][shot_boundary[0]:shot_boundary[1] + 1]), dim=0).requires_grad_(True))
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class NodeConvolution(Module):

    def __init__(self, kernel, input_size, pooling_size=2):
        super(NodeConvolution, self).__init__()
        self.pooling_size = pooling_size
        self.weight1 = Parameter(torch.FloatTensor(kernel, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_list = list()
        for i in range(input.shape[0]):
            tensor = input[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    tensor_list.append(torch.sum(tensor_ * self.weight1, dim=0).requires_grad_(True))
                else:
                    tensor_list.append(
                        torch.sum(
                            tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size] * self.weight1,
                            dim=0).requires_grad_(True))
            batch_list.append(torch.stack(tensor_list, dim=0))

        return torch.stack(batch_list, dim=0).requires_grad_(True)


class GraphPooling(Module):

    def __init__(self, pooling_size=3):
        super(GraphPooling, self).__init__()
        self.pooling_size = pooling_size

    def forward(self, input, graph):
        batch_list = []
        for i in range(input.shape[0]):
            tensor = input[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True
            tensor_list = list()
            for j in range(steps):
                shot_boundary = graph[i][j]
                tensor_list.append(
                    torch.div(torch.sum(input[shot_boundary[0]:shot_boundary[1] + 1], dim=0).requires_grad_(True),
                              shot_boundary[1] - shot_boundary[0] + 1).requires_grad_(True) + 6e-3)
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]

                    tensor_list.append(torch.div(torch.sum(tensor_, dim=0).requires_grad_(True),self.pooling_size))
                else:
                    tensor_list.append(
                        torch.div(torch.sum(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size], dim=0).requires_grad_(True),self.pooling_size))
            batch_list.append(torch.stack(tensor_list, dim=0).requires_grad_(True))

        return torch.stack(batch_list, dim=0).requires_grad_(True)

class GraphAttentionPooling(Module):

    def __init__(self, in_features, pooling_size=3):
        super(GraphAttentionPooling, self).__init__()
        self.in_features = in_features
        self.W = nn.Linear(in_features, 1, bias=True)
        self.pooling_size = pooling_size

    def forward(self, batch_tensor):
        batch_list = list()
        for i in range(batch_tensor.shape[0]):
            tensor = batch_tensor[i]
            batch_size = tensor.shape[0]
            steps = batch_size // self.pooling_size
            left_tensor = False
            if batch_size % self.pooling_size != 0:
                steps += 1
                left_tensor = True

            tensor_list = list()
            for j in range(steps):
                if left_tensor is True and j == steps - 1:
                    tensor_ = torch.zeros(self.pooling_size, tensor.shape[1], requires_grad=True).cuda()
                    for i in range(batch_size % self.pooling_size):
                        tensor_[i] = tensor[self.pooling_size * j + i]
                    att_w = F.softmax(self.W(tensor_), dim=0).requires_grad_(True)
                    tensor_list.append(tensor_.T @ att_w)
                else:
                    att_w = F.softmax(
                        self.W(tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size]),
                        dim=0).requires_grad_(True)
                    tensor_list.append(
                        tensor[self.pooling_size * j + 0:self.pooling_size * j + self.pooling_size].T @ att_w)
            batch_list.append(torch.stack(tensor_list, dim=0))
        return torch.stack(batch_list, dim=0).requires_grad_(True)

class DCGNPropagate(nn.Module):
    def __init__(self, input, output):
        super(DCGNPropagate, self).__init__()
        self.Weight = Parameter(torch.FloatTensor(input, output))
        self.Bias = Parameter(torch.FloatTensor(output))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Weight.size(1))
        self.Weight.data.uniform_(-stdv, stdv)

    def forward(self, adj, x):
        graphed_net = torch.einsum('xyz,abc->xbc',adj,x)
        return torch.einsum('abc,cd->abd',graphed_net,self.Weight)+self.Bias


class DCGN(nn.Module):
    def __init__(self, input, nclass, pooling_size=3):
        super(DCGN, self).__init__()

        self.nodewiseconvolution = NodeConvolution(pooling_size, input, pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input, pooling_size=pooling_size)
        self.Propagate1 = DCGNPropagate(input, 784)

        self.NodeConvolution2 = NodeConvolution(pooling_size, 784, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(784, pooling_size=pooling_size)
        self.Propagate2 = DCGNPropagate(784, 28)

        self.classifier = nn.Sequential( nn.Linear(84,32),
                                         nn.GELU(),
                                         nn.Linear(32,nclass))

    def forward(self, x, device):

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,2
        x = self.Propagate1(adj, x)
        x = F.gelu(x)

        adj = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,32.
        x = self.Propagate2(adj, x)
        x = F.gelu(x)

        x = x.view(-1, 3*28)
        x = self.classifier(x)
        return x

    def cosine_similarity_adjacent(self, matrix1, matrix2):
        squaresum1 = torch.sum(torch.square(matrix1), dim=1)  # 1024 to 1

        squaresum2 = torch.sum(torch.square(matrix2), dim=1)  # 1024 to 1

        multiplesum = torch.sum(torch.multiply(matrix1, matrix2), dim=1)

        Matrix1DotProduct = torch.sqrt(squaresum1)
        Matrix2DotProduct = torch.sqrt(squaresum2)
        cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
        return cosine_similarity

    def get_adjacent(self, matrix):
        batch_list = list()
        for i in range(matrix.shape[0]):
            tensor = matrix[i]
            matrix_frame = tensor.shape[0]  # 4,2,1024
            AdjacentMatrix = torch.zeros(matrix_frame, matrix_frame)  # 2 X 2

            chunks = torch.chunk(tensor, matrix_frame, dim=0)
            for i in range(matrix_frame):
                for j in range(matrix_frame - i):
                    AdjacentMatrix[j][i] = self.cosine_similarity_adjacent(chunks[i], chunks[j])
                    if not i == j:
                        AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
            I = torch.eye(AdjacentMatrix.shape[0], requires_grad=True)

            AdjacentMatrix += I
            AdjacentMatrix = AdjacentMatrix.requires_grad_(True)
            D_hat = torch.sum(AdjacentMatrix, dim=0)
            D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))
            batch_list.append(D_hat @ AdjacentMatrix @ D_hat)
        return torch.stack(batch_list, dim=0).requires_grad_(True)



class DCGN2(nn.Module):
    def __init__(self, input, nclass, pooling_size=3):
        super(DCGN2, self).__init__()

        self.nodewiseconvolution = NodeConvolution(pooling_size, input, pooling_size=pooling_size)
        self.WisePooling = GraphAttentionPooling(input, pooling_size=pooling_size)
        self.Propagate1 = DCGNPropagate(input, 784)

        self.NodeConvolution2 = NodeConvolution(pooling_size, 784, pooling_size=pooling_size)
        self.AttentionPooling2 = GraphAttentionPooling(784, pooling_size=pooling_size)
        self.Propagate2 = DCGNPropagate(784, 28)

        self.classifier = nn.Sequential( nn.Linear(28,nclass))

    def forward(self, x, device):

        adj = self.WisePooling(x)
        x = self.nodewiseconvolution(x)  # 2,256
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,2
        x = self.Propagate1(adj, x)
        x = F.gelu(x)

        adj = self.AttentionPooling2(x)  # 2,64
        x = self.NodeConvolution2(x)  # 2,64
        adj = self.get_adjacent(adj).to(device).requires_grad_(True)  # 2,32.
        x = self.Propagate2(adj, x)
        x = F.gelu(x)

        x = x.view(-1, 28)
        x = self.classifier(x)
        return x

    def cosine_similarity_adjacent(self, matrix1, matrix2):
        squaresum1 = torch.sum(torch.square(matrix1), dim=1)  # 1024 to 1

        squaresum2 = torch.sum(torch.square(matrix2), dim=1)  # 1024 to 1

        multiplesum = torch.sum(torch.multiply(matrix1, matrix2), dim=1)

        Matrix1DotProduct = torch.sqrt(squaresum1)
        Matrix2DotProduct = torch.sqrt(squaresum2)
        cosine_similarity = torch.div(multiplesum, torch.multiply(Matrix1DotProduct, Matrix2DotProduct))
        return cosine_similarity

    def get_adjacent(self, matrix):
        batch_list = list()
        for i in range(matrix.shape[0]):
            tensor = matrix[i]
            matrix_frame = tensor.shape[0]  # 4,2,1024
            AdjacentMatrix = torch.zeros(matrix_frame, matrix_frame)  # 2 X 2

            chunks = torch.chunk(tensor, matrix_frame, dim=0)
            for i in range(matrix_frame):
                for j in range(matrix_frame - i):
                    AdjacentMatrix[j][i] = self.cosine_similarity_adjacent(chunks[i], chunks[j])
                    if not i == j:
                        AdjacentMatrix[j][i] = AdjacentMatrix[i][j]
            I = torch.eye(AdjacentMatrix.shape[0], requires_grad=True)

            AdjacentMatrix += I
            AdjacentMatrix = AdjacentMatrix.requires_grad_(True)
            D_hat = torch.sum(AdjacentMatrix, dim=0)
            D_hat = torch.linalg.inv(torch.sqrt(torch.diag(D_hat)))
            batch_list.append(D_hat @ AdjacentMatrix @ D_hat)
        return torch.stack(batch_list, dim=0).requires_grad_(True)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, base_width=64,
                 downsample=None, groups=1, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        width = output_channel * (base_width // 64) * groups
        self.conv1 = nn.Conv2d(input_channel, width, kernel_size=1, stride=1, bias=False)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
                               dilation=dilation)

        self.conv3 = nn.Conv2d(width, output_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample
        self.stride = stride
    def forward(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x= self.gelu(x)

        x = self.conv2(x)
        x= self.gelu(x)

        x = self.conv3(x)
        x= self.gelu(x)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        x += identity
        out =self.gelu(x)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, input_channel, output_channel, base_width=64,
                 downsample=None, groups=1, stride=1, dilation=1):
        super(Bottleneck3D, self).__init__()
        width = output_channel * (base_width // 64) * groups
        self.conv1 = nn.Conv3d(input_channel, width, kernel_size=1, stride=1, bias=False)
        self.gelu = nn.GELU()

        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False,
                               dilation=dilation)

        self.conv3 = nn.Conv3d(width, output_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample
        self.stride = stride
    def forward(self, inputs):
        identity = inputs
        x = self.conv1(inputs)
        x= self.gelu(x)

        x = self.conv2(x)
        x= self.gelu(x)

        x = self.conv3(x)
        x= self.gelu(x)

        if self.downsample is not None:
            identity = self.downsample(inputs)
        x += identity
        out =self.gelu(x)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=3, padding=3,
                               bias=False)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=5, stride=3, padding=3,
                               bias=False)
        self.gelu = nn.GELU()

        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 3, layers[1], stride=3,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4, layers[2], stride=3,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 5, layers[3], stride=3,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1,bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,downsample= downsample, groups= self.groups,
                            base_width=self.base_width, dilation=previous_dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet3D(nn.Module):

    def __init__(self, block, layers,zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=(1,7,7), stride=3, padding=3,
                               bias=False)
        self.conv2 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=(1,7,7), stride=3, padding=3,
                               bias=False)
        self.conv3 = nn.Conv3d(self.inplanes, self.inplanes, kernel_size=(1,7,7), stride=3, padding=3,
                               bias=False)
        self.gelu = nn.GELU()

        self.maxpool = nn.MaxPool3d(kernel_size=(1,4,4),stride=2)

        self.layer1 = self._make_layer(block, 2, layers[0])
        self.layer2 = self._make_layer(block, 3, layers[1], stride=1,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 5, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[1])

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):

        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, stride=stride, kernel_size=1,bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride,downsample= downsample, groups= self.groups,
                            base_width=self.base_width, dilation=previous_dilation))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.gelu(x)
        x = self.conv3(x)
        x = self.gelu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class TestingModel1(nn.Module):
    def __init__(self):
        super(TestingModel1, self).__init__()
        self.feature_extract = ResNet(Bottleneck, [2, 3, 4, 5])
        self.DCGN =DCGN(2600,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        feature_list = list()
        for i in range(input.shape[0]):
          feature_list.append( self.feature_extract(input[i].reshape(-1,3,3000,2400)))


        x = torch.stack(feature_list,dim=1).reshape(-1,20,2600)
        x = self.DCGN(x,'cuda')
        return self.softmax(x)


class TestingModel2(nn.Module):
    def __init__(self):
        super(TestingModel2, self).__init__()
        self.feature_extract = ResNet3D(Bottleneck3D, [3, 4, 6, 8])
        self.DCGN =DCGN2(11340,2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.feature_extract(input)
        x = self.DCGN(x.reshape(-1,2,11340),'cuda')
        return self.softmax(x)

def Test(crypto_name, load_model=False, adder=0, device='cuda'):

    trans = transforms.Compose([transforms.ToTensor()])
    feature_list =list()

    for number in range(20):

        decided_number = number + adder
        url = 'G:/ImgDataStorage/' + crypto_name + '/COMBINED/' + str(decided_number + 1) + '.jpg'
        crypto_chart = PIL.Image.open(url)
        crypto_chart = trans(crypto_chart).float().to(device)
        feature_list.append(crypto_chart)

    model1 = TestingModel1().to(device)

    video = torch.stack(feature_list,dim=0).to(device).reshape(20,3,3000,2400)
    start1 = time.time()
    action = model1(video)
    end1 = time.time()
    del model1
    print(end1-start1)
    model2 = TestingModel2().to(device)

    start2 = time.time()
    action = model2(video.reshape(-1,3,20,3000,2400))
    end2 = time.time()
    print(end2-start2)

if __name__ == '__main__':

    Test('BTCUSDT', load_model=False, adder=0)
