import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from shutil import copyfile
import multiprocessing

from models.generator import Generator
from models.discriminator import Discriminator
from feeder.feeder import Feeder
from utils import general

def main():
    # 查询最近的运行记录，并返回运行输出路径
    out = general.check_runs('uav-human-gan')
    models_out = os.path.join(out, 'models')
    actions_out = os.path.join(out, 'actions')
    if not os.path.exists(models_out): os.makedirs(models_out)
    if not os.path.exists(actions_out): os.makedirs(actions_out)

    # 构造命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1200, help="训练的轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="每次迭代的批量大小")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: 学习率")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: 一阶动量的衰减")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: 二阶动量的衰减")
    parser.add_argument("--n_cpu", type=int, default=0, help="批量生成过程中使用的 CPU 线程数，设置为 0 禁用多进程")
    parser.add_argument("--latent_dim", type=int, default=512, help="演化空间的维度")
    parser.add_argument("--mlp_dim", type=int, default=4, help="映射网络的深度")
    parser.add_argument("--n_classes", type=int, default=155, help="数据集的类别数")
    parser.add_argument("--t_size", type=int, default=64, help="每个时间维度的大小")
    parser.add_argument("--v_size", type=int, default=17, help="每个空间维度的大小（顶点）")
    parser.add_argument("--channels", type=int, default=3, help="通道数（坐标维度）")
    parser.add_argument("--n_critic", type=int, default=5, help="每迭代生成器训练次数前鉴别器的训练次数")
    parser.add_argument("--lambda_gp", type=int, default=10, help="WGAN-GP 中的梯度惩罚权重")
    parser.add_argument("--sample_interval", type=int, default=5000, help="样本生成的间隔间隔时间")
    parser.add_argument("--checkpoint_interval", type=int, default=10000, help="模型保存的间隔间隔时间")
    parser.add_argument("--dataset", type=str, default="uav-human", help="数据集")
    parser.add_argument("--data_dir", type=str, default="/path/to/data_c", help="数据目录路径")
    parser.add_argument("--data_split", type=str, default="train", help="数据集划分")
    opt = parser.parse_args()
    print(opt)

    # 保存配置文件，为了重现此训练过程
    config_file = open(os.path.join(out, "config.txt"), "w")
    config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
    config_file.close()

    # 复制生成器和鉴别器模型文件
    copyfile(os.path.basename(__file__), os.path.join(out, os.path.basename(__file__)))
    copyfile('models/generator.py', os.path.join(out, 'generator.py'))
    copyfile('models/discriminator.py', os.path.join(out, 'discriminator.py'))

    # 检查 CUDA 是否可用
    cuda = True if torch.cuda.is_available() else False
    print('CUDA', cuda)

    # 初始化生成器和鉴别器
    generator = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.t_size, opt.mlp_dim, dataset=opt.dataset)
    discriminator = Discriminator(opt.channels, opt.n_classes, opt.t_size, opt.latent_dim, dataset=opt.dataset)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # 配置数据输入
    # 使用修改后的 Feeder 类来加载 UAV-Human 数据集
    train_data_path = os.path.join(opt.data_dir, 'train_joint.npy')
    train_label_path = os.path.join(opt.data_dir, 'train_label.npy')
    dataloader = torch.utils.data.DataLoader(
        dataset=Feeder(
            data_path=train_data_path,
            label_path=train_label_path,
            window_size=opt.t_size,
            bone=True,   # 使用骨骼特征增强
            vel=True     # 使用速度特征增强
        ),
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=opt.n_cpu
    )

    # 初始化优化器
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

    # 样本生成函数
    # 每次训练时进行样本生成，并将生成的图像保存
    def sample_action(n_row, batches_done):
        z = Variable(Tensor(np.random.normal(0, 1, (10 * n_row, opt.latent_dim))))
        # 获取从 0 到 n_classes 的标签
        labels = np.array([num for _ in range(10) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = generator(z, labels)
        with open(os.path.join(actions_out, str(batches_done) + '.npy'), 'wb') as npf:
            np.save(npf, gen_imgs.data.cpu())

    # 计算梯度惩罚
    def compute_gradient_penalty(D, real_samples, fake_samples, labels):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        labels = LongTensor(labels)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates, labels)
        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # ----------
    #  训练
    # ----------

    loss_d, loss_g = [], []
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i

            #print("imgs.shape:", imgs.shape)
            imgs = imgs[:,:,:opt.t_size,:]
            #print("imgs.shape:", imgs.shape)
            # 配置输入
            #imgs = imgs.permute(0, 3, 1, 2)  # 将数据形状调整为 (B, C, T, V)
            real_imgs = Variable(imgs.type(Tensor))
            labels = Variable(labels.type(LongTensor))

            # ---------------------
            #  训练鉴别器
            # ---------------------

            optimizer_D.zero_grad()

            # 生成器输入的随机噪声
            z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

            # 生成一批的动作
            fake_imgs = generator(z, labels)

            # 真动作的鉴别
            real_validity = discriminator(real_imgs, labels)
            # 假动作的鉴别
            fake_validity = discriminator(fake_imgs, labels)
            # 梯度惩罚
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
            # 判别器的损失
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # 每 n_critic 次训练鉴别器后训练一次生成器
            if i % opt.n_critic == 0:

                # -----------------
                #  训练生成器
                # -----------------

                # 生成一批的动作
                fake_imgs = generator(z, labels)
                # 生成器的损失
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            loss_d.append(d_loss.data.cpu())
            loss_g.append(g_loss.data.cpu())

            # 通过样本间隔生成动作
            if batches_done % opt.sample_interval == 0:
                sample_action(n_row=opt.n_classes, batches_done=batches_done)
                general.save('uav-human-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')

            # 保存模型检查点
            if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
                torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
                torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

    loss_d = np.array(loss_d)
    loss_g = np.array(loss_g)

    # 保存损失值图像
    general.save('uav-human-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')

if __name__ == "__main__":
    multiprocessing.freeze_support()  # 适用于 Windows 系统
    multiprocessing.set_start_method('spawn')  # 设置启动方法
    main()
