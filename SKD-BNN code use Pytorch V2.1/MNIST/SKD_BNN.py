import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.patheffects as pe



def setup_my_seed(seed):
	"""
	操作:   固定网络的所有随机数，使模型结果可复现
	"""
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
	np.random.seed(seed)
	torch.manual_seed(seed)  # CPU随机种子确定

	torch.cuda.manual_seed(seed)  # GPU随机种子确定
	torch.cuda.manual_seed_all(seed)  # 所有的GPU设置种子
	torch.backends.cudnn.deterministic = True  # 确定为默认卷积算法
	torch.backends.cudnn.benchmark = False  # 模型卷积层算法预先优化关闭

	print("\n************ The Model has been Seeded ************\n")
	
	
# 交叉熵损失函数 PS-KD
class CrossEntropy_SoftLabelloss(nn.Module):
	"""
	输入:   网络输出(不需要提前softmax)，one-hot标签(和必须为1)
	输出：  loss
	"""
	def __init__(self):
		super(CrossEntropy_SoftLabelloss, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, outputs, targets, temperature):
		log_probs = self.logsoftmax(outputs / temperature)
		Soft_targets = self.softmax(targets / temperature)
		loss = (-Soft_targets * log_probs).mean(0).sum() * (temperature**2)
		return loss


# KL 散度损失函数
class KL_SoftLabelloss(nn.Module):
	"""
	输入:   网络输出(不需要提前softmax)，one-hot标签(和必须为1)
	操作：  torch.softmax(true_outputs, dim=1)
	输出：  loss
	"""
	def __init__(self):
		super(KL_SoftLabelloss, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
		self.softmax = nn.Softmax(dim=1).cuda()
		self.kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

	def forward(self, outputs, targets, temperature):
		log_probs = self.logsoftmax(outputs / temperature)
		Soft_targets = self.softmax(targets / temperature)
		loss = self.kl_criterion(log_probs, Soft_targets.detach()) * (temperature**2)
		return loss


# 特征图匹配 空间通道损失函数
class Spatial_Channel_loss(nn.Module):
	"""
	输入:   需要进行匹配的特征图，匹配目标特征图 (二者需具有相同大小 (B, C , W, H) )
	操作：  compute distillation loss by referring to both the spatial and channel differences : Mean
	SVec： 空间向量             (B, 1, C)
	CVec： 维度矩阵(B, W, H) -> (B, 1, W*H)
	输出：  loss
	"""
	def __init__(self):
		super(Spatial_Channel_loss, self).__init__()
		self.SmoothL1 = nn.SmoothL1Loss(reduction='sum').cuda()

	def forward(self, outputs, targets):
		assert outputs.size() == targets.size()

		# SVec_out = torch.sum(torch.sum(outputs, dim=3), dim=2)
		# SVec_tar = torch.sum(torch.sum(targets, dim=3), dim=2)
		# SVec_loss = torch.norm(SVec_out / torch.norm(SVec_out) - SVec_tar / torch.norm(SVec_tar))
		#
		# CVec_out = torch.sum(torch.sum(outputs, dim=1), dim=0)
		# CVec_tar = torch.sum(torch.sum(targets, dim=1), dim=0)
		# CVec_loss = torch.norm(CVec_out / torch.norm(CVec_out) - CVec_tar / torch.norm(CVec_tar))

		batch_size = targets.size()[0]
		channel_size = targets.size()[1]
		# feature_size = targets.size()[2] ** 2
		feature_size = targets.size()[1]

		# SVec_out = torch.unsqueeze(torch.mean(torch.mean(outputs, dim=3), dim=2), dim=1)
		# SVec_tar = torch.unsqueeze(torch.mean(torch.mean(targets, dim=3), dim=2), dim=1)
		# SVec_out = SVec_out / torch.unsqueeze(torch.norm(SVec_out, dim=-1), dim=1).detach()
		# SVec_tar = SVec_tar / torch.unsqueeze(torch.norm(SVec_tar, dim=-1), dim=1)
		#
		# CVec_out = torch.unsqueeze(torch.mean(outputs, dim=1).reshape(batch_size, -1), dim=1)
		# CVec_tar = torch.unsqueeze(torch.mean(targets, dim=1).reshape(batch_size, -1), dim=1)
		# CVec_out = CVec_out / torch.unsqueeze(torch.norm(CVec_out, dim=-1), dim=1).detach()
		# CVec_tar = CVec_tar / torch.unsqueeze(torch.norm(CVec_tar, dim=-1), dim=1)
		SVec_out = outputs
		SVec_tar = targets
		CVec_out = outputs
		CVec_tar = targets

		# #  空间通道注意力loss
		SVec_loss = torch.norm(SVec_out - SVec_tar) / channel_size
		CVec_loss = torch.norm(CVec_out - CVec_tar) / feature_size
		loss = (SVec_loss + CVec_loss) / batch_size

		# SVec_loss = self.SmoothL1(SVec_out, SVec_tar) / channel_size
		# CVec_loss = self.SmoothL1(CVec_out, CVec_tar) / feature_size
		# loss = (SVec_loss + CVec_loss) / batch_size

		# # 余弦相似度  空间通道注意力loss
		# SVec_CosSim_loss = (SVec_out * SVec_tar).sum(dim=-1)
		# SVec_CosSim_loss = ((torch.ones_like(SVec_CosSim_loss) - SVec_CosSim_loss) * 0.5).sum() / channel_size
		# CVec_CosSim_loss = (CVec_out * CVec_tar).sum(dim=-1)
		# CVec_CosSim_loss = ((torch.ones_like(CVec_CosSim_loss) - CVec_CosSim_loss) * 0.5).sum() / feature_size
		# loss = (SVec_CosSim_loss + CVec_CosSim_loss) / batch_size

		return loss


# 特征图匹配 行列余弦相似度损失函数
class Line_Column_CosSimloss(nn.Module):
	"""
	输入:   需要进行匹配的特征图，匹配目标特征图 (二者需具有相同大小 (B, C , W, H) )
	操作：  特征的 行 列 上计算余弦相似度
	输出：  loss
	"""
	def __init__(self):
		super(Line_Column_CosSimloss, self).__init__()

	def forward(self, outputs, targets):
		assert outputs.size() == targets.size()
		Num = targets.size()[0] * targets.size()[1] * targets.size()[2] * targets.size()[3]

		Line_CosSimloss = torch.cosine_similarity(outputs, targets, dim=-1)
		Column_CosSimloss = torch.cosine_similarity(outputs, targets, dim=-2)
		Line_CosSimloss = ((torch.ones_like(Line_CosSimloss) - Line_CosSimloss) * 0.5).sum() / Num
		Column_CosSimloss = ((torch.ones_like(Column_CosSimloss) - Column_CosSimloss) * 0.5).sum() / Num
		loss = Line_CosSimloss + Column_CosSimloss

		return loss


# 倒数对二层匹配 行方向余弦相似度损失函数
class Line_CosSimloss(nn.Module):
	"""
	输入:   需要进行匹配的特征图，匹配目标特征图 (二者需具有相同大小 (B, C) )
	操作：  特征的 行 上计算余弦相似度
	输出：  loss
	"""
	def __init__(self):
		super(Line_CosSimloss, self).__init__()

	def forward(self, outputs, targets):
		assert outputs.size() == targets.size()
		Num = targets.size()[0]

		loss = torch.cosine_similarity(outputs, targets, dim=-1)
		loss = ((torch.ones_like(loss) - loss) * 0.5).sum() / Num

		return loss


def BN_Without_gammabeta(x, eps=1e-5):
	"""
	输入:   特征图(B, C , W, H)
	操作:   进行 train阶段 的 BN 操作，但没有gamma和beta可学习参数
	"""
	assert len(x.shape) in (2, 4)
	# 全连接层
	if len(x.shape) == 2:
		mean = x.mean(dim=0)
		var = ((x - mean)**2).mean(dim=0)
	# 卷积层
	else:
		mean = x.mean(dim=(0, 2, 3), keepdim=True)
		var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
	x = (x - mean) / torch.sqrt(var + eps)
	return x


# 特征图缩放 逐channel处理 or 逐mini-batch处理
def Rough_Normalized_scale(outputs):
	# """
	#       逐channel处理
	# 输入:   特征图(B, C , W, H)
	# 输出：  粗糙归一化 scale = 1/逐 channel 最大值
	# 使用：  outputs * Rough_Normalized_scale(outputs)
	# """
	# indices = outputs.size()
	# scale = 1. / torch.squeeze(torch.max(torch.max(torch.abs(outputs), dim=3)[0], dim=2)[0], dim=0)
	# scale = scale.view(indices[0], indices[1], 1, 1)
	# return scale.detach()
	"""
	        逐mini-batch处理
		输入:   特征图(B, C , W, H)
		输出：  粗糙归一化 scale = 1/逐 mini-batch 最大值
		使用：  outputs * Rough_Normalized_scale(outputs)
	"""
	indices = outputs.size()
	scale = 1. / torch.squeeze(torch.max(torch.max(torch.max(torch.abs(outputs), dim=3)[0], dim=2)[0], dim=1)[0], dim=0)
	scale = scale.view(indices[0], 1, 1, 1)
	return scale.detach()


# 符号不一致程度计算 %
def Sign_difference(outputs, targets):
	return torch.sum(torch.abs(outputs - targets) / 2.) / outputs.numel() * 100


# 特征图 直方图显示
def Hist_Show(x, name):
	"""
	输入:   网络某一层的输出特征(B, C , W, H)
	输出:   (0, 0 , W, H)对应的直方图
	"""
	x = x.detach()
	histNum = torch.reshape(input=x[0][0], shape=(1, -1)).cpu()
	# plt.hist(histNum, bins=range(-50, 50, 1))  # BN前
	plt.hist(histNum, bins=np.arange(-2, 2, 0.04))  # BN后
	plt.savefig(name)
	plt.close('all')


class tSNE_data_Maker(nn.Module):
	"""
	输入:
		data: 倒数第二层特征图(即进入最后分类层的数据:(B, C)）
		label: Batch对应的类别标签(B, )
		step: 当前迭代次数,参与每次迭代的数据为batch_size个
		batch_size
		ceiling: 后续参与tSNE的数据个数(为batch_size的整数倍)，即散点图的打点个数
	操作:累计输入的 mini-batch 数据
	输出：达到 ceiling 上限的累计数据
	"""
	def __init__(self):
		super(tSNE_data_Maker, self).__init__()
		self.tSNE_data = np.empty(0)
		self.tSNE_label = np.empty(0)

	def forward(self, data, label, step, batch_size, ceiling):
		if step == 0:
			self.tSNE_data = np.empty(0)
			self.tSNE_label = np.empty(0)
		if (step+1)*batch_size <= ceiling:
			data = data.detach().cpu().numpy()
			label = label.detach().cpu().numpy()  # -->CPU-->numpy
			self.tSNE_data = np.append(self.tSNE_data, data.ravel())
			self.tSNE_label = np.append(self.tSNE_label, label)
			self.tSNE_data = self.tSNE_data.reshape(-1, data.shape[1])
			return self.tSNE_data, self.tSNE_label
		else:
			return self.tSNE_data, self.tSNE_label


# 特征图 tSNE显示
def tSNE_Show(data, label, name):
	"""
	输入:   倒数第二层特征图(即进入最后分类层的数据:(B, C)）, Batch对应的类别标签
	操作:   random_state=2023
	输出:   此 mini-Batch 对应的tSNE
	"""
	assert data.dtype == 'float64'
	assert label.dtype == 'float64'
	print("------------> Start t-SNE  |  Waiting time ------------>")
	data = np.vstack([data[label == i] for i in range(10)])
	label = np.hstack([label[label == i]for i in range(10)])  # 顺序，其实没有也无所谓
	label = np.trunc(label).astype(int)  # 转化为int类型，便于画图显示
	tSNE_feature = TSNE(n_components=2, init='random', learning_rate='auto', random_state=20221020, perplexity=64).fit_transform(data)
	tSNE_data_draw = {'x': tSNE_feature[:, 0], 'y': tSNE_feature[:, 1], 'Label': label}  # 打包数据
	tSNE_data_draw = pd.DataFrame(tSNE_data_draw)
	sns.set_context("notebook", font_scale=1.6)  # 设置画布长宽比
	sns.set_style("ticks")  # 图片风格，共有5种选择
	sns.lmplot(x='x', y='y', hue='Label', data=tSNE_data_draw, fit_reg=False, height=12, scatter_kws={"s": 50, "alpha": 0.9})
	plt.title(name).set_fontsize('19')
	plt.xlabel('x').set_fontsize('18')
	plt.ylabel('y').set_fontsize('18')
	plt.savefig('./{}.png'.format(name), dpi=120)
	# plt.show()
	plt.close('all')

	# 以下为第二种画图风格，不推荐使用
	# x = tSNE_feature
	# colors = label
	# palette = np.array(sns.color_palette("pastel", 10))
	# f = plt.figure(figsize=(8, 8))
	# ax = plt.subplot(aspect='equal')
	# sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int8)])
	# txts = []
	# for i in range(10):
	# 	# Position of each label.
	# 	xtext, ytext = np.median(x[colors == i, :], axis=0)
	# 	txt = ax.text(xtext, ytext, str(i), fontsize=24)
	# 	txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
	# 	txts.append(txt)
	# plt.savefig('./digits_tsne-pastel.png', dpi=120)


def Save_Label_bank(epoch, save_every, data, save):
	"""
	Save the training KD_Label_bank
	"""
	filename = os.path.join(save, str(epoch + 1) + '-CIFAR10-Label.txt')
	if (epoch + 1) % save_every == 0:
		data = data.cpu().numpy()
		np.savetxt(filename, data, fmt='%.6f')


# 产生子标签库，以更新标签库
def Up_Label_bank(outputs, target, num_classes, mode):
	"""
	输入:   网络直接输出，对应的真实标签，数据集类别个数
	操作:
	输出：  对应类别，子标签库
	"""
	assert outputs.size()[0] == target.size()[0]
	assert mode == 'all' or mode == 'top1'
	# 初始化
	Son_Label = torch.zeros(num_classes, num_classes, dtype=torch.float32).cuda()
	True_Num = torch.zeros(num_classes, dtype=torch.int64).cuda()
	with torch.no_grad():
		if mode == 'all':
			label_cate = target
		elif mode == 'top1':
			# 1，得到top1正确的输出
			_, outtop1_cate = outputs.topk(1, 1, True, True)  # 概率最大数值 概率最大位置索引
			outtop1_cate = torch.squeeze(outtop1_cate.t(), dim=0)  # 输出所属类别，不一定正确
			label_cate = torch.where(outtop1_cate == target, outtop1_cate, torch.tensor([-1]).cuda())  # 类别正确判断：正确为对应类别 错误为-1\

		# # 标签归一化：减去最小值，除以最大最小值的差值
		# scale_max, _ = torch.max(outputs, dim=1)
		# scale_min, _ = torch.min(outputs, dim=1)
		# scale_max = torch.unsqueeze(scale_max, dim=1)
		# scale_min = torch.unsqueeze(scale_min, dim=1)
		# outputs = num_classes * (outputs-scale_min)/(scale_max-scale_min)

		# # 标签minsum：减去最小值，按和为10进行缩放
		# scale_min, _ = torch.min(outputs, dim=1)
		# scale_min = torch.unsqueeze(scale_min, dim=1) 
		# outputs = outputs - scale_min
		# scale = num_classes / torch.sum(outputs, dim=1)
		# scale = torch.unsqueeze(scale, dim=1)
		# outputs = scale * outputs

		# # # 标签abssum：按绝对值和为10进行缩放 ···！
		# scale = num_classes / torch.sum(torch.abs(outputs), dim=1)
		# scale = torch.unsqueeze(scale, dim=1)
		# outputs = scale * outputs

		# # 按每个类别的最大值进行缩放，缩放后最大值为1
		# scale_max, _ = torch.max(outputs, dim=1)
		# scale_max = torch.unsqueeze(scale_max, dim=1)
		# outputs = 3.0 * outputs / scale_max

		# # 按每个类别的最大值进行缩放，缩放后最大值为1，接着进行abssum
		# scale_max, _ = torch.max(outputs, dim=1)
		# scale_max = torch.unsqueeze(scale_max, dim=1)
		# outputs = outputs / scale_max
		# scale = num_classes / torch.sum(torch.abs(outputs), dim=1)
		# scale = torch.unsqueeze(scale, dim=1)
		# outputs = scale * outputs

		# # 将标签汇聚到(-1，1)之间
		# scale_max, _ = torch.max(outputs, dim=1)
		# scale_min, _ = torch.min(outputs, dim=1)
		# scale_max = torch.unsqueeze(scale_max, dim=1)
		# scale_min = torch.unsqueeze(scale_min, dim=1)
		# outputs = (outputs-scale_min)/(scale_max-scale_min) * 0.5

		if label_cate.sum() == -1 * label_cate.numel():  # 全部错误 置为0 不对后续的标签库产生影响
			Son_Label = Son_Label
			True_Num = True_Num
		else:
			indices = torch.nonzero(label_cate + torch.ones_like(label_cate)).squeeze()  # 类别正确 的位置索引
			true_outputs = torch.index_select(outputs, 0, indices)  # 取出类别预测正确 的 输出正确标签
			true_cate = label_cate[indices]  # 正确标签对应的类别
			# 2，按照类别进行排序
			sorted_true_cate, _ = torch.sort(true_cate, descending=False, dim=-1)  # 对应的类别
			true_outputs = true_outputs[_]  # 进一步按照排序顺序重组true_outputs
			if sorted_true_cate.numel() == 1:  # 只有一个正确
				Son_Label[sorted_true_cate] = true_outputs
				True_Num[sorted_true_cate] = 1
			else:
				counter = Counter(sorted_true_cate.cpu().numpy())
				true_cate = list(counter.keys())  # 合并后 类标签
				indices = list(counter.values())  # 用于下面类切分的索引(重复个数)
				# 3，对块进行求平均处理 得到 Outputs_Label_bank
				true_outputs = torch.split(true_outputs, indices, dim=0)  # 按照indices对true_outputs划分 输出为typle类型
				for i in true_cate:
					# Son_Label[i] = true_outputs[true_cate.index(i)].mean(0)    # 对batch内的同一类标签进行平均处理
					Son_Label[i] = true_outputs[true_cate.index(i)].sum(0)  # 对batch内的同一类标签进行求和处理
				True_Num[true_cate] = torch.tensor(indices).cuda()
				# del true_outputs  # 删除元组 消除显存
	return Son_Label, True_Num


def Clustering(penul, label, num_classes):
	Clustered = torch.zeros(num_classes, penul.size()[1], dtype=torch.float32).cuda()  # 类别数的聚类存储
	Clustered_outputs = torch.zeros(penul.size()[0], penul.size()[1], dtype=torch.float32).cuda()  # 同形状的聚类输出，方便loss

	sorted, _ = torch.sort(label, descending=False, dim=-1)
	outputs = penul[_]  # 按照类别排序重组

	counter = Counter(sorted.cpu().numpy())
	true_cate = list(counter.keys())
	indices = list(counter.values())
	outputs = torch.split(outputs, indices, dim=0)  # 划分为元组

	for i in true_cate:
		Clustered[i] = outputs[true_cate.index(i)].mean(0)    # 对特征内部，同一类别的，进行求平均处理
	Clustered_outputs = Clustered[label]  # 抽取所需聚类，方便loss

	return Clustered_outputs