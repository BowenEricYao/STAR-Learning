from torch_scatter import scatter
from torch.nn import Sequential as Seq, Linear,Conv2d,CELU,Flatten,MaxPool2d, ReLU,Softmax,GELU
from torch_geometric.nn import MessagePassing
from torch import Tensor
import torch
import torch.nn as nn
from typing import Callable, List, Optional
import math
import torch.nn.functional as F
import snntorch as snn
class MEConv(MessagePassing):
    def __init__(self, in_channels, out_channels,features,out_features):
        super(MEConv,self).__init__(aggr='mean') #  "Max" aggregation.

        self.mlp = Seq(Conv2d(in_channels, 32,3,1,padding=1,bias=False),
                       nn.BatchNorm2d(32),
                       ReLU(),
                       MaxPool2d(2,2),
                       Conv2d(32, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Flatten())

        self.mlp2=Seq(Linear(4096, 4096,bias=False))
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)




    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.mlp(x)




        return self.propagate(edge_index, x=x)

    def message(self,x_j,x_i):
        Cos=self.cos(x_j,x_i)

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x_j,Cos

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """


        return scatter(inputs[0], index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr),scatter(inputs[1], index, dim=0, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:
        return inputs[0],x,inputs[1]
class MEConv2(MessagePassing):
    def __init__(self, in_channels, out_channels,features,out_features):
        super(MEConv2,self).__init__(aggr='mean') #  "Max" aggregation.

        self.mlp = Seq(Conv2d(in_channels, 32,3,1,padding=1),
                       nn.InstanceNorm2d(32),
                       ReLU(),
                       MaxPool2d(2,2),
                       Conv2d(32, out_channels, 3, 1, padding=1),
                       nn.InstanceNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       nn.InstanceNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       nn.InstanceNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       nn.InstanceNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       nn.InstanceNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Flatten(),
                       Linear(features,out_features,bias=False),ReLU())

        self.mlp2=Seq(Linear(4096, 4096,bias=False))
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.mlp(x)




        return self.propagate(edge_index, x=x)

    def message(self,x_j,x_i):
        Cos=self.cos(x_j,x_i)

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x_j,Cos

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """


        return scatter(inputs[0], index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr),scatter(inputs[1], index, dim=0, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:
        return inputs[0],x,inputs[1]
class MEConvnocos(MessagePassing):
    def __init__(self, in_channels, out_channels,features,out_features):
        super(MEConvnocos,self).__init__(aggr='mean') #  "Max" aggregation.

        self.mlp = Seq(Conv2d(in_channels, out_channels,3,1,padding=1),
                       ReLU(),
                       MaxPool2d(2,2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Flatten())

        self.mlp2=Seq(Linear(4096, 1024,bias=False))
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.mlp(x)




        return self.propagate(edge_index, x=x)

    def message(self,x_j,x_i):

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """


        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:
        return inputs*x,x
class Block(nn.Module):
    def __init__(self, embed_dim,num_patches, mlp_ratio=4.,  drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.conv1 = MEConv2(dim=embed_dim, num_heads=3, features=1024, qk_scale=None, num_patches=num_patches + 1,
                             embed_dim=embed_dim, mlp_ratio=4.)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x,egde, return_attention=False):
        y=self.conv1(x,egde)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
class MEConv2(MessagePassing):
    def __init__(self, dim, num_heads,features,qk_scale,num_patches,embed_dim,mlp_ratio):
        super(MEConv2,self).__init__(aggr='mean') #  "Max" aggregation.
        self.num_heads=num_heads
        self.norm1 = nn.LayerNorm(dim)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.node_dim=0
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.norm1(x)
        B, N, C = x.shape
        x=self.qkv(x)
        qkv = x.reshape(B, N, 3, self.num_heads, C// self.num_heads)

        return self.propagate(edge_index, x=qkv)

    def message(self,x_j,x_i):
        B, N, _,_,_=x_j.shape


        qkv_i = x_i.permute(2, 0, 3, 1, 4)
        qkv_j = x_j.permute(2, 0, 3, 1, 4)
        k_j= qkv_j[1]
        q_i, v_i = qkv_i[0], qkv_i[2]
        attn = (q_i @ k_j.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v_i).transpose(1, 2).reshape(B, N, self.embed_dim)
        x = self.proj(x)

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.

        """

        return scatter(inputs, index, dim=0, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:


        return inputs
class MEinConv(MessagePassing):
    def __init__(self,features):
        super(MEinConv,self).__init__(aggr='mean') #  "Max" aggregation.



        self.mlp2=Seq(Linear(1024, features,bias=True))


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.mlp2(x)

        return self.propagate(edge_index, x=x)

    def message(self,x_j):

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x_j

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """


        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:
        return inputs*x
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MEConv(3, 64,1024,1024)
        self.conv2= MEinConv( 1024)
        self.conv3 = MEinConv(1024)
        self.mlp=Seq(Linear(2048,1024))
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, support=None,query=None,center=None,cross_edge_index=None,test=False):
        if test is False:
            support_x, support_edge_index = support.x, support.edge_index
            query_x, query_edge_index = query.x, query.edge_index
            tm=torch.nn.functional.one_hot(query.y,num_classes=23)+0.5



            s_x,s_x1,s_cos = self.conv1(support_x, support_edge_index)
            s_x = self.conv2(s_x, support_edge_index)

            s_x = torch.cat([s_x1,s_x],dim=-1)
            print(s_x.size())
            s_x=self.mlp(s_x)
            q_x,q_x1,q_cos  = self.conv1(query_x, query_edge_index)
            q_x = self.conv2(q_x, query_edge_index)
            q_x = torch.cat([q_x1, q_x], dim=-1)
            q_x = self.mlp(q_x)

            pdis = []

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2)))

                a = scatter(a, support.batch, dim=0, reduce='mean')
                a = self.softmax(a)

                pdis.append(a)
            pd = torch.stack(pdis, dim=0)
            s_cos=scatter(s_cos, support.batch, dim=0, reduce='mean')
            s_x = scatter(s_x, support.batch, dim=0, reduce='mean')

            q_x = scatter(q_x, query.batch, dim=0, reduce='mean')
            pd = scatter(pd, query.batch, dim=0, reduce='mean')
            cos_loss=torch.pow(s_cos - 1, 2)
            cos_loss=torch.mean(cos_loss)
            dis = []
            s_c = 0.25 * torch.repeat_interleave(center, 5, dim=0) + s_x * 0.5


            for i in range(q_x.size(0)):
                a = (torch.exp(-(torch.pow((s_c - q_x[i, :]), 2))) + pd[i, :]).sum(1)
                a = self.softmax(a)
                # a = center@q_x[i,:]


                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)
            center = scatter(s_x, support.y, dim=0, reduce='mean') * 0.5 + 0.25 * center
            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)
            y = torch.nn.functional.one_hot(query.y, num_classes=23)
            loss = F.cross_entropy(output, y.float(), reduction='sum')+cos_loss
            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc, center
        else:
            support_x, support_edge_index = support.x, support.edge_index
            query_x, query_edge_index = query.x, query.edge_index
            s_x, s_x1,s_cos = self.conv1(support_x, support_edge_index)
            s_x = self.conv2(s_x, support_edge_index)
            s_x = torch.cat([s_x1, s_x], dim=-1)
            s_x = self.mlp(s_x)
            q_x, q_x1,q_cos = self.conv1(query_x, query_edge_index)
            q_x = self.conv2(q_x, query_edge_index)
            q_x = torch.cat([q_x1, q_x], dim=-1)
            q_x = self.mlp(q_x)

            pdis = []

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2)))

                a = scatter(a, support.batch, dim=0, reduce='mean')
                a = self.softmax(a)

                pdis.append(a)
            pd = torch.stack(pdis, dim=0)
            s_x = scatter(s_x, support.batch, dim=0, reduce='mean')
            cosd_s_x = self.conv3(s_x, cross_edge_index)
            cosd_s_x = scatter(cosd_s_x, support.y, dim=0, reduce='mean')
            q_x = scatter(q_x, query.batch, dim=0, reduce='mean')
            pd = scatter(pd, query.batch, dim=0, reduce='mean')
            dis = []
            s_c = 0.25 * torch.repeat_interleave(center, 5, dim=0)+ s_x * 0.5
            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = (torch.exp(-(torch.pow((s_c - q_x[i, :]), 2))) + pd[i, :]).sum(1)
                a = self.softmax(a)

                a = scatter(a, support.y, dim=0, reduce='mean')

                dis.append(a)

            # x = self.classifier(torch.stack(dis, dim=0))

            return torch.stack(dis, dim=0).squeeze()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def attention_loss_pre(attention_class):

    attention_bool=torch.zeros(24,dtype=torch.bool)
    attention_bool[attention_class]=True

    return attention_bool,~attention_bool
def attention_loss_pre_sub(attention_class):

    attention_bool=torch.zeros(6,dtype=torch.bool)
    attention_bool[attention_class]=True

    return attention_bool,~attention_bool
class GCN3(torch.nn.Module):
    def __init__(self,attention_class):
        super().__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()
        self.attention_class=attention_class
        self.attention_class_index, self.rest_class_index=attention_loss_pre(self.attention_class)
        self.attention_class_index, self.rest_class_index=self.attention_class_index.to(device), self.rest_class_index.to(device)
    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None,temprature_P=None,temprature_N=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []


            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))




                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=24)
            atten_loss= F.cross_entropy(output[self.attention_class_index,:], label[self.attention_class_index,:].float(), reduction='mean')
            rest_loss =F.cross_entropy(output[self.rest_class_index,:], label[self.rest_class_index,:].float(), reduction='mean')
            loss = atten_loss+rest_loss
            loss = loss + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss,atten_loss,rest_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')




            return a


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat
class GCN3_sub(torch.nn.Module):
    def __init__(self,attention_class):
        super().__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()
        self.attention_class=attention_class
        self.attention_class_index, self.rest_class_index=attention_loss_pre(self.attention_class)
        self.attention_class_index, self.rest_class_index=self.attention_class_index.to(device), self.rest_class_index.to(device)
    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None,temprature_P=None,temprature_N=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []


            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))




                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=6)
            atten_loss= F.cross_entropy(output[self.attention_class_index,:], label[self.attention_class_index,:].float(), reduction='mean')
            rest_loss =F.cross_entropy(output[self.rest_class_index,:], label[self.rest_class_index,:].float(), reduction='mean')
            loss = atten_loss+rest_loss
            loss = loss + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss,atten_loss,rest_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')




            return a


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat
class GCN3_server(torch.nn.Module):
    def __init__(self):
        super(GCN3_server,self).__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()


    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None,temprature_P=None,temprature_N=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []
            temparature_all = torch.ones(23,23) * temprature_N
            # temparature_all[(target-i).nonzero()] = temprature_N
            for i in range(23):
                temparature_all[i,i]=temprature_P

            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))




                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=24)
            atten_loss= F.cross_entropy(output[self.attention_class_index,:], label[self.attention_class_index,:].float(), reduction='mean')
            rest_loss =F.cross_entropy(output[self.rest_class_index,:], label[self.rest_class_index,:].float(), reduction='mean')
            loss = atten_loss+rest_loss
            loss = loss + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss,atten_loss,rest_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')




            return a


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat
class GCN3_server_sub(torch.nn.Module):
    def __init__(self):
        super(GCN3_server,self).__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()


    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None,temprature_P=None,temprature_N=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []
            temparature_all = torch.ones(23,23) * temprature_N
            # temparature_all[(target-i).nonzero()] = temprature_N
            for i in range(23):
                temparature_all[i,i]=temprature_P

            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))




                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=6)
            atten_loss= F.cross_entropy(output[self.attention_class_index,:], label[self.attention_class_index,:].float(), reduction='mean')
            rest_loss =F.cross_entropy(output[self.rest_class_index,:], label[self.rest_class_index,:].float(), reduction='mean')
            loss = atten_loss+rest_loss
            loss = loss + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss,atten_loss,rest_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')




            return a


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat
class GCN3_metrics(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()
        self.attention_class=22
        self.attention_class_index, self.rest_class_index=attention_loss_pre(self.attention_class)
        self.attention_class_index, self.rest_class_index=self.attention_class_index.to(device), self.rest_class_index.to(device)
    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None,temprature_P=None,temprature_N=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []
            temparature_all = torch.ones(23,23) * temprature_N
            # temparature_all[(target-i).nonzero()] = temprature_N
            for i in range(23):
                temparature_all[i,i]=temprature_P

            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))




                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=23)



            loss = F.cross_entropy(output, label.float(), reduction='mean') + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')





            return self.softmax(a)


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat



class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MEConvnocos(3, 64,1024,1024)
        self.conv2= MEinConv( 1024)
        self.mlp=Seq(Linear(2048,1024))
        self.classifier = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Dropout(),
            nn.Linear(1024, 1),
        )
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, support=None,query=None,center=None,test=False):
        if test is False:
            support_x, support_edge_index = support.x, support.edge_index
            query_x, query_edge_index = query.x, query.edge_index
            tm=torch.nn.functional.one_hot(query.y,num_classes=23)+0.5



            s_x,s_x1 = self.conv1(support_x, support_edge_index)
            s_x = self.conv2(s_x, support_edge_index)

            s_x = torch.cat([s_x1,s_x],dim=-1)
            s_x=self.mlp(s_x)
            q_x,q_x1  = self.conv1(query_x, query_edge_index)
            q_x = self.conv2(q_x, query_edge_index)
            q_x = torch.cat([q_x1, q_x], dim=-1)
            q_x = self.mlp(q_x)

            pdis = []

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2)))

                a = scatter(a, support.batch, dim=0, reduce='mean')
                a = self.softmax(a)

                pdis.append(a)
            pd = torch.stack(pdis, dim=0)

            s_x = scatter(s_x, support.batch, dim=0, reduce='mean')
            q_x = scatter(q_x, query.batch, dim=0, reduce='mean')
            pd = scatter(pd, query.batch, dim=0, reduce='mean')

            dis = []
            s_c = 0.25 * torch.repeat_interleave(center, 5, dim=0) + s_x * 0.5


            for i in range(q_x.size(0)):
                a = (torch.exp(-(torch.pow((s_c - q_x[i, :]), 2))) + pd[i, :]).sum(1)
                a = self.softmax(a)
                # a = center@q_x[i,:]

                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)
            center = scatter(s_x, support.y, dim=0, reduce='mean') * 0.5 + 0.25 * center
            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)
            y = torch.nn.functional.one_hot(query.y, num_classes=23)
            loss = F.cross_entropy(output, y.float(), reduction='sum')
            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc, center
        else:
            support_x, support_edge_index = support.x, support.edge_index
            query_x, query_edge_index = query.x, query.edge_index
            s_x, s_x1 = self.conv1(support_x, support_edge_index)
            s_x = self.conv2(s_x, support_edge_index)
            s_x = torch.cat([s_x1, s_x], dim=-1)
            s_x = self.mlp(s_x)
            q_x, q_x1 = self.conv1(query_x, query_edge_index)
            q_x = self.conv2(q_x, query_edge_index)
            q_x = torch.cat([q_x1, q_x], dim=-1)
            q_x = self.mlp(q_x)

            pdis = []

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2)))

                a = scatter(a, support.batch, dim=0, reduce='mean')
                a = self.softmax(a)

                pdis.append(a)
            pd = torch.stack(pdis, dim=0)
            s_x = scatter(s_x, support.batch, dim=0, reduce='mean')
            q_x = scatter(q_x, query.batch, dim=0, reduce='mean')
            pd = scatter(pd, query.batch, dim=0, reduce='mean')
            dis = []
            s_c = 0.25 * torch.repeat_interleave(center, 5, dim=0)+ s_x * 0.5
            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = (torch.exp(-(torch.pow((s_c - q_x[i, :]), 2))) + pd[i, :]).sum(1)
                a = self.softmax(a)

                a = scatter(a, support.y, dim=0, reduce='mean')

                dis.append(a)

            # x = self.classifier(torch.stack(dis, dim=0))

            return torch.stack(dis, dim=0).squeeze()
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block1(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
class GCN1(torch.nn.Module):
    def __init__(self,embed_dim=384,drop_rate=0,depth=6):
        super().__init__()

        self.conv2 = MEinConv(1024)
        self.mlp = Seq(Linear(192, 2048),nn.GELU(),nn.Linear(2048,2048),nn.GELU(),nn.Linear(2048,256))
        self.classifier = nn.Linear(256, 65336)
        self.norm = nn.LayerNorm(embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.patch_embed=PatchEmbed(img_size=256,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.block1=Block(embed_dim=embed_dim,num_patches=num_patches + 1)
        self.blocks = nn.ModuleList([
            Block(embed_dim=embed_dim,num_patches=num_patches + 1)
            for i in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, support=None, query=None, center=None,center1=None,center2=None, test=False,feature=False,s_index=None,sp_index=None):
        if test is False and feature is False:
            support_x, support_edge_index = support.x, support.edge_index
            query_x, query_edge_index = query.x, query.edge_index
            tm = torch.nn.functional.one_hot(query.y, num_classes=23) + 0.5
            s_x = self.prepare_tokens(support_x)

            q_x = self.prepare_tokens(query_x)
            for blk in self.blocks:
                s_x = blk(s_x,support_edge_index)
                q_x = blk(q_x, query_edge_index)

            s_x = self.norm(s_x)
            q_x = self.norm(q_x)
            s_x = s_x[:,0]
            q_x = q_x[:, 0]

            # s_x =self.mlp(s_x)
            # q_x = self.mlp(q_x)
            #
            # s_x = self.classifier(s_x)
            # q_x = self.classifier(q_x)
            pdis = []

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2))).sum(1)

                a = scatter(a, support.batch, dim=0, reduce='max')



                pdis.append(a)
            pd =torch.stack(pdis,dim=0)



            # s_x = scatter(s_x, support.batch, dim=0, reduce='mean')
            # q_x = scatter(q_x, query.batch, dim=0, reduce='mean')
            pd = scatter(pd, query.batch, dim=0, reduce='max')



            dis = []

            # s_c=torch.repeat_interleave(center,5, dim=0)


            for i in range(pd.size(0)):

                # a = center@q_x[i,:]

                a = scatter(pd[i, :], support.y, dim=0, reduce='max')

                dis.append(a)

            # s_dis=F.mse_loss(s_c, s_x)

            # x=self.classifier(torch.stack(dis,dim=0))
            output = torch.stack(dis, dim=0)
            y = torch.nn.functional.one_hot(query.y, num_classes=23)
            # loss = F.cross_entropy(output, y.float(), reduction='sum')
            loss=F.cross_entropy(output, y.float(), reduction='mean')
            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss, acc, center
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            tm = torch.nn.functional.one_hot(query.y, num_classes=23) + 0.5
            q_x = self.prepare_tokens(query_x)
            for blk in self.blocks:
                q_x = blk(q_x, query_edge_index)
            q_x = self.norm(q_x)
            q_x = q_x[:, 0]
            pdis = []
            s_x=center1.to(device)
            pdis = []
            center = center.to(device)

            for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_x - q_x[i, :]), 2))).sum(1)

                a = scatter(a, sp_index, dim=0, reduce='max')

                pdis.append(a)
            pd = torch.max(torch.stack(pdis, dim=0),dim=0)[0]

            dis = []


            # for i in range(q_x.size(0)):
                # a = center @ q_x[i, :]
            a = pd.squeeze(0)


            a = scatter(a, s_index.squeeze(), dim=0, reduce='max')

            dis.append(a)

            # x = self.classifier(torch.stack(dis, dim=0))

            return a
        elif feature is True:
            support_x, support_edge_index = support.x, support.edge_index
            s_x = self.prepare_tokens(support_x)
            for blk in self.blocks:
                s_x = blk(s_x, support_edge_index)
            s_x = self.norm(s_x)
            s_x = s_x[:, 0]
            # s_cat=s_mss_feature
            # s_mss_feature2 = self.conv2(s_mss_feature, support_edge_index)
            # s_cat = torch.cat([s_cat, s_mss_feature2], dim=-1)
            # s_mss_feature3 = self.conv2(s_mss_feature2, support_edge_index)
            # s_cat = torch.cat([s_cat, s_mss_feature3], dim=-1)
            return s_x


class CovaBlock2(nn.Module):
    def __init__(self):
        super(CovaBlock2, self).__init__()

    # calculate the covariance matrix
    def cal_covariance(self, input):

        CovaMatrix_list = []
        for i in range(len(input)):
            support_set_sam = input[i]
            B, C, h, w = support_set_sam.size()

            support_set_sam = support_set_sam.permute(1, 0, 2, 3)
            support_set_sam = support_set_sam.contiguous().view(C, -1)
            mean_support = torch.mean(support_set_sam, 1, True)
            support_set_sam = support_set_sam - mean_support

            covariance_matrix = support_set_sam @ torch.transpose(support_set_sam, 0, 1)
            covariance_matrix = torch.div(covariance_matrix, h * w * B - 1)
            CovaMatrix_list.append(covariance_matrix)

        return CovaMatrix_list

        # calculate the mahalanobis distance

    def cal_mahalanobis(self, input, CovaMatrix_list):

        B, C, h, w = input.size()
        Maha_list = []

        for i in range(B):
            query_sam = input[i]
            query_sam = query_sam.view(C, -1)
            mean_query = torch.mean(query_sam, 1, True)
            query_sam = query_sam - mean_query

            if torch.cuda.is_available():
                maha_dis = torch.zeros(1, len(CovaMatrix_list) * h * w).cuda()

            for j in range(len(CovaMatrix_list)):
                temp_dis = torch.transpose(query_sam, 0, 1) @ CovaMatrix_list[j] @ query_sam
                maha_dis[0, j * h * w:(j + 1) * h * w] = temp_dis.diag()

            Maha_list.append(maha_dis.unsqueeze(0))

        Maha_Dis = torch.cat(Maha_list, 0)  # get Batch*1*(h*w*num_classes)
        return Maha_Dis

    def forward(self, x1, x2):

        CovaMatrix_list = self.cal_covariance(x2)
        Maha_Dis = self.cal_mahalanobis(x1, CovaMatrix_list)

        return Maha_Dis