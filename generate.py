import argparse
import os
import numpy as np

from torch.autograd import Variable
import torch

from models.generator import Generator
from utils import general
from collections import Counter
import pickle

# 此函数用于对演化向量进行截断操作（参数 Z 的截断技巧）
def trunc(latent, mean_size, truncation):  # Truncation trick on Z
    t = Variable(FloatTensor(np.random.normal(0, 1, (mean_size, *latent.shape[1:]))))
    m = t.mean(0, keepdim=True)

    # 对所有演化向量进行操作
    for i,_ in enumerate(latent):
        latent[i] = m + truncation * (latent[i] - m)

    return latent

# 查询最近的运行记录，并返回运行输出路径
out = general.check_runs('uav-human-gan', id=-1)
actions_out = os.path.join(out, 'actions')
if not os.path.exists(actions_out): 
    os.makedirs(actions_out)

# 构造命令行参数解析器
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=10, help="每个类别生成的样本数量（每次迭代）")
parser.add_argument("--latent_dim", type=int, default=512, help="演化空间的维度")
parser.add_argument("--mlp_dim", type=int, default=4, help="映射网络深度")
parser.add_argument("--n_classes", type=int, default=155, help="数据集的类别数")  # 修改为 UAV-Human 的类别数
parser.add_argument("--label", type=int, default=-1, help="指定要生成的类别，-1 代表所有类别")
parser.add_argument("--t_size", type=int, default=300, help="每个时间维度的大小")
parser.add_argument("--v_size", type=int, default=17, help="每个空间维度的大小（顶点）")  # 修改为 UAV-Human 的节点数
parser.add_argument("--channels", type=int, default=3, help="通道数（坐标维度）")
parser.add_argument("--dataset", type=str, default="uav-human", help="数据集")
parser.add_argument("--model", type=str, default="runs/uav-human-gan/exp1/models/generator_uav_human.pth", help="生成器模型的路径")
parser.add_argument("--stochastic", action='store_true', help="生成一个样本，并验证随机性")
parser.add_argument("--stochastic_file", type=str, default="-", help="读取一个样本，并验证随机性")
parser.add_argument("--stochastic_index", type=int, default=0, help="定位要获取的演化向量的索引")
parser.add_argument("--gen_qtd", type=int, default=1000, help="每个类别生成的样本数")
parser.add_argument("--trunc", type=float, default=0.95, help="截断模式的正态标准底数")
parser.add_argument("--trunc_mode", type=str, default='w', choices=['z', 'w', '-'], help="截断模式（查阅文章详情）")
parser.add_argument("--mean_size", type=int, default=1000, help="用于估计平均值的样本数")
opt = parser.parse_args()
print(opt)

# 生成配置文件并保存
config_file = open(os.path.join(out,"gen_config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

# 检查是否有 CUDA，并初始化 CUDA 变量
cuda = True if torch.cuda.is_available() else False
print(cuda)

# 初始化生成器
generator = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.t_size, mlp_dim=opt.mlp_dim, dataset=opt.dataset)

if cuda:
    generator.cuda()

# 使用 CUDA 或 CPU 初始化数据类型
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# 加载生成器模型
generator.load_state_dict(torch.load(opt.model), strict=False)
generator.eval()

new_imgs = []
new_labels = []
z_s = []

# 生成要生成的类别数据
classes = np.arange(opt.n_classes) if opt.label == -1 else [opt.label]
qtd = opt.batch_size

# 如果指定了 stochastic_file，读取文件中的样本
if opt.stochastic_file != '-':
    stoch = np.load(opt.stochastic_file)
    stoch = np.expand_dims(stoch[opt.stochastic_index], 0)
    print(stoch.shape)

# 如果使用 stochastic 模式，生成一个演化向量
if opt.stochastic:
    z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim)) if opt.stochastic_file == '-' else stoch))
    z = z.repeat(qtd * len(classes), 1)

# 循环生成每个类别的样本
while len(classes) > 0:

    # 如果不是 stochastic 模式，生成演化向量
    if not opt.stochastic:
        z = Variable(FloatTensor(np.random.normal(0, 1, (qtd * len(classes), opt.latent_dim))))

    # 应用截断技巧
    z = trunc(z, opt.mean_size, opt.trunc) if opt.trunc_mode == 'z' else z
    # 生成标签
    labels_np = np.array([num for _ in range(qtd) for num in classes])
    labels = Variable(LongTensor(labels_np))
    # 使用生成器生成新的样本
    gen_imgs = generator(z, labels, opt.trunc) if opt.trunc_mode == 'w' else generator(z, labels)

    # 将生成的新图像数据应用 CPU 输出
    new_imgs = gen_imgs.data.cpu().numpy() if len(new_imgs) == 0 else np.concatenate((new_imgs, gen_imgs.data.cpu().numpy()), axis=0)
    new_labels = labels_np if len(new_labels) == 0 else np.concatenate((new_labels, labels_np), axis=0)
    z_s = z.cpu().numpy() if len(z_s) == 0 else np.concatenate((z_s, z.cpu().numpy()), axis=0)

    # 统计新标签的数量，确定不足 opt.gen_qtd 的类别
    tmp = Counter(new_labels)
    classes = [i for i in classes if tmp[i] < opt.gen_qtd]

    print('---------------------------------------------------')
    print(tmp)
    print(len(new_labels), classes)

# 如果数据集是 UAV-Human，将生成的图像数据增少一个维度
if opt.dataset == 'uav-human':
    new_imgs = np.expand_dims(new_imgs, axis=-1)

# 保存生成的数据
with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label) + '_' + str(opt.gen_qtd) + ('_trunc' + str(opt.trunc) if opt.trunc_mode != '-' else '') + ('_stochastic' if opt.stochastic else '') + '_gen_data.npy'), 'wb') as npf:
    np.save(npf, new_imgs)

# 保存生成的演化向量
with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label) + '_' + str(opt.gen_qtd) + ('_trunc' + str(opt.trunc) if opt.trunc_mode != '-' else '') + ('_stochastic' if opt.stochastic else '') + '_gen_z.npy'), 'wb') as npf:
    np.save(npf, z_s)

# 保存生成的标签数据
with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label) + '_' + str(opt.gen_qtd) + ('_trunc' + str(opt.trunc) if opt.trunc_mode != '-' else '') + ('_stochastic' if opt.stochastic else '') + '_gen_labels.npy'), 'wb') as npf:
    np.save(npf, new_labels)
