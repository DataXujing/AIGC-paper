## Clip

------

<!-- 中文CLIP的一个文章，MedClip -->

2021年[CLIP](https://arxiv.org/pdf/2103.00020.pdf)（Contrastive Language-Image Pre-training）即图文对比预训练，是这推动这两年多模态领域大火的奠基之作。相信大家都已经比较熟悉了。关于Clip的介绍可以参考上一章中大模型中关于Clip的介绍。

<div align=center>
    <img src="zh-cn/img/ch4/1-1/p1.png" /> 
</div>

在4亿个网络图片和对应的标题的加持下，CLIP使用简单的Info NCE loss大力出奇迹，在多个图像识别任务上的零样本预测能力吊打了很多的监督学习模型。这个特性再次强化了我们对于深度学习more data, more intelligence的印象。紧接着，就出现了非常多在CLIP预训练模型的基础上整一些花活的文章，比如在视频文本上预训练，在音频文本上预训练，等等。

CLIP在
医疗相关也有可用的图像和文本的配对数据集。最知名的应该是一些列胸片和临床报告的配对数据，比如CheXpert, MIMIC-CXR等。早在CLIP之前，ConVIRT就已经展示了Info NCE式对比学习在医疗图文数据上的能力。但是不幸的是，由于数据不够大，表现不够excited，这篇文章没有搞出一个大新闻，最终才在今年的MLHC会议上发表。当时还让作为作者之一的Christopher Manning大佬发推吐槽了一番。

简单来说，ConVIRT的思路和CLIP是一致的。在我们有胸片和对应的临床报告文本时，我们可以把每张胸片和对应的报告中的句子作为正样本对，而跟其他的报告中的句子作为负样本对。这样就可以在一个图片编码器（ResNet）和文本编码器（BERT）的加持下，愉快地做对比学习了。那么和CLIP不同的是，ConVIRT并没有考虑零样本预测的情况，而只是利用预训练的图片编码器加上一个全连接层做分类器，然后还是加标签数据做微调那一套。

在ConVIRT之后，Stanford又出了一篇GLoRIA，在之前工作的基础上加了很多注意力（attention）机制。即考虑了图像编码器中间层里的特征图和文本中每个词之前的attention，得到一个经过加权的局部特征（local representation）。相对应的就是原本的图像和文本特征，在文中叫做全局特征（global representation）。也是在这篇文章中，第一次实现了医疗图片在图文预训练之后的基于prompt的零样本预测。


### 1.MedClip

<!-- https://zhuanlan.zhihu.com/p/574963133 -->
写到这里，终于该轮到我们的工作MedCLIP出场了。MedCLIP要解决的问题，我想可以用一张图说明。

<div align=center>
    <img src="zh-cn/img/ch4/1-1/p2.png" /> 
</div><p align=center>MedCLIP要解决的问题：（1）如何解决只能利用配对图文训练的限制；（2）如何解决由于只使用配对图文作为正样本带来的假阴性样本问题。</p>

首先，跟CLIP相比，医疗领域的图像文本配对总量要小的多。CLIP可以在4亿数据上充分训练，但是，X-ray和配对的报告的公开数据集最大的也只有数十万这个级别，分别是CheXpert（20万）和MIMIC-CXR（37万）。这就严重限制了模型的能力。同时，我们其实还有大量的纯医疗图像或者纯文本数据。由于使用CLIP的对比学习方法，模型只能利用天生配对的图片+报告来训练。这就导致了医疗图文训练的天生在数据量方面的跛腿，从而很难达到CLIP那样的高度。

另外，由于假定只有配对的图片和文本是正样本，其它的都被当作负样本，很多的潜在正样本都被当作了负样本，即False Negatives。同CLIP使用的日常图文不同，X-rays之间的差别其实很小。在没有经过专业训练的普通人眼里几乎分辨不出来任何差别。并且，很多报告可能都描述了类似的症状和病情，但都被一律当作了负样本处理。这就导致了模型在训练过程中感觉到了疑惑：图片1和文本B明明匹配，却要求把它们的特征分开。这大大影响了学习到的表征的质量。

**MedCLIP怎么做**

针对上面的这两个问题，我们希望能够解耦（decouple）图片和文本的配对关系，转而用一个人工构建的弱标签系统作为匹配图片和文本的工具。见下图。

<div align=center>
    <img src="zh-cn/img/ch4/1-1/p3.png" /> 
</div><p align=center>MedCLIP的基本架构</p>

咱们这里主要关注最左边这一块。对于每条文本，我们都可以抽取它之中存在的一些关键实体，作为这条文本的弱标签。对于图片，我们有它们的标签，因为它们可能来源于已经标注好的纯图像数据集，或者有对应的报告，那么就用报告的标签作为它的标签。

在获得这两个标签后，对比学习的目标就不再是一个对角的identity矩阵，而是两个标签向量的内积，作为图片和文本之间的一个相似度。

**实验结果**

实验比较多，这里只放一个我认为最重要的。见下图。

<div align=center>
    <img src="zh-cn/img/ch4/1-1/p4.png" /> 
</div><p align=center>MedCLIP方法展现了惊人的数据效率（data efficiency）：利用1/10的数据就可以达到之前sota方法的表现。</p>

可以看到，相比GLoRIA，我们的方法在只用20K数据的时候就已经达到了更强的零样本预测能力。随着样本量的增大，MedCLIP的表现也在逐渐scale。可以期待如果更多的数据可用，它的表现还可以更好。值得注意的是，在我们的实验里，CheXpert+MIMIC-CXR作为预训练数据集。但是，因为MedCLIP的特性，我们还可以考虑加入更多的图片数据进来。

**总结**

这篇文章的方法和思想非常简单，就是一个利用外部知识来构建文本和图像的弱标签，从而能够解耦图片和文本对，做到指数级扩大可用的正负样本。同时，利用弱标签，我们能够甄别出很多的False Negative样本，从而提高模型的表征学习能力。

后续的工作可以考虑如何进一步提高弱标签的质量，以及在有噪弱标签的情况下进一步提高预训练的鲁棒性。或者，在模型架构主要是图片编码器一侧提升设计，让模型更多的抓住医疗图片的重要区域，从而提升表征的判别能力。

!> 个人观点

我个人认为MedClip不是真正意义上的多模态的模型，只是文本和文本之间相似度的匹配的学习！


### 2.中文Clip

<!-- https://github.com/OFA-Sys/Chinese-CLIP -->
<!-- https://blog.csdn.net/qq_27590277/article/details/128213439 -->

Github上有开发者基于Vit-Bert的CLIP,其中图像编码基于Vit,文本编码基于Bert,开发者称为BertCLIP,用140万的中文数据，基于Lit-tuning的方式，训练了BertCLIP模型。BertCLIP模型在中文图文相似度，文本相似度，图片相似度等任务上都有着不错的表现。详细的参考：

!> github: https://github.com/yangjianxin1/CLIP-Chinese

!> 微信文章： https://mp.weixin.qq.com/s/6gQX91M-Lt7eiMimhYRJEw

这不是我们本节内容的重点，我们本节着重介绍阿里开源的中文clip!

!> github: https://github.com/OFA-Sys/Chinese-CLIP

<!-- https://zhuanlan.zhihu.com/p/580546929 -->
<!-- https://zhuanlan.zhihu.com/p/594354204 -->

#### 1. 简介

之前的文章已经介绍过做中文CLIP的一些想法，更详细的可以看上一篇文章的一些showcase，可以看到多语言的mclip在中文图搜表现是很难让人满意的，比如搜索“对联”返回的却是圣诞相关的物品。在我们后来的实验里，在中文原生的数据集上，我们对比了OpenAI CLIP和我们中文CLIP的表现，会发现后者的优势是巨大的。当然，之前也有一些相关工作，但大多增加了算法的复杂性，并且还存在开源做得不足的问题等等。因此，我们就是想打造一个靠谱的Baseline，CLIP如果换成大规模的中文数据应当取得比较突出的效果。因此，我们收集了接近2亿的图文对数据（绝大部分都是公开的数据），在上面用CLIP的方法做预训练（当然有一些训练方法上的改动，后续会介绍我们使用的两阶段预训练的方法，如下图所示），训练出了5个规模的中文CLIP模型，包括ResNet、ViT-B/16、ViT-L/14、ViT-L/14@336px和ViT-H/14。实验做了3个公开数据集的图文检索，还做了一系列经典图像分类数据集的零样本分类，都取得不错的效果，其中检索基本达到SOTA，零样本分类也比较有竞争力。我们补充做了一些分析，证明了两阶段预训练的有效性，同时还发现了CLIP本身一些比较明显的缺陷，比如对prompt很敏感以及难以理解否定。之后，我们还会做一些更强的模型，同时还会特别关注轻量化这块，让小模型的效果能提上去。

<div align=center>
    <img src="zh-cn/img/ch4/2-1/p1.png" /> 
</div>

#### 2. 数据

我们希望我们的工作即使预训练也是可以复现的，因此为了减少壁垒，我们尽可能使用公开数据集。其中包括LAION-5B中标注了“zh”的部分，华为悟空数据集，但因为失效链接等问题，1.1亿的LAION和7千万的悟空的图才被下下来。我们还增加了一些英文的翻译数据集，比如VG和COCO。此外因为一些历史原因，我们的预训练数据集还是混入了少量私有数据集，不过我们认为应该不会对最终效果造成太大影响。

#### 3. 模型训练

先简单回顾CLIP，它的核心就是图文双塔模型，用对比学习让正例更近，让负例更远，CLIP原文的图非常易懂。可以看到负采样就是在batch内采样，和SimCLR一个道理，我们的实现上就是使用all gather让分布式增大batch size的同时也能增大负例size。

下面说下我们核心的训练方法。最朴素的方案就是用我们的数据从头开始训练，但是成本太高了，当时就设计了两阶段的方法，后续我们也通过ablation证明它是相对来说最好的方案了。两阶段训练法的核心思路就是把OpenAI那个极强的CLIP给用上，现在有的工作做得比较狠，直接把文本塔对齐CLIP的图像塔。固然在Flickr30K-CN和COCO-CN这类翻译数据集上应该能取得比较不错的效果，但是没有学过中文世界的图像的模型真的能很好地应用到中文场景吗？当然也可以直接finetune，我们也把它作为一个对比选项。

我们的核心方法在于把训练分为两阶段（如上图所示），第一阶段和LiT是一致的，冻结图像塔，让文本塔表示接近图像塔表示。当训练继续但下游精度不能再产生显著提升，即下游零样本检索的精度，我们就把训练切换到第二阶段，即解除图像塔的参数冻结，继续用contrastive tuning预训练，同样直到下游精度没有显著提升。后者的意义在于让图像塔能拟合中文世界的图像数据的分布，学习中文世界的知识。更多实验参数欢迎查看论文的附录部分。

最终我们训练了5个规模的模型，小到ResNet50，大到ViT-H，如下表所示：

<div align=center>
    <img src="zh-cn/img/ch4/2-1/p2.png" /> 
</div>

ChineseCLIP 公开了 5 个多种尺寸的中文 CLIP 模型，参数范围从 77 到 9.58 亿。训练方法上，ChinseCLIP 选择直接用 CLIP 的 ViT 和 RoBERTa 做双塔的初始化，让模型可以从一个好的起点开始训练。此外，它还提出了一种两阶段预训练方法，首先在冻结图像编码器的情况下训练模型，然后在优化所有参数的情况下进行训练，以实现增强的模型性能。

#### 3. 实验

目前我们做的实验集中在图文检索的零样本和finetuningh场景，以及最具代表性的CLIP的零样本图像分类。检索方案，我们在MUGE（https://tianchi.aliyun.com/muge ）、Flickr30K-CN、COCO-CN上基本都做到零样本和finetuning最优。具体的实验结果请参考原文。

#### 4. 总结

很简单的一项工作，我们这篇技术报告之后还会更新加入更多细节，还有一些关于轻量化的内容补充，以及短期内我们还会探索更多的玩法，当然也包括探索它更多的不足，让大家对CLIP多一分了解。

这里补充一点，这个工作还在进展中。主要还是中文CLIP除了检索外，作为中文cv backbone的作用，依然会有很多人认为LiT训练足矣。最近在筹备一个中文的benchmark，快的话下个月应该能出来一批数据可以做测试。

当然了，还是非常非常需要大家的帮助和反馈，我们才知道模型是不是真的在各个场景都好，还是说只是做好了一部分。最后也还是希望朋友们能在Github给我们加星（private了之后星全没了好伤）。也欢迎大家关注达摩院最新推出的魔搭社区，新社区刚推出肯定还有不少问题，欢迎反馈！


### 3. WuKong：100 Million Large-scale Chinese Cross-modal Pre-training Dataset and A Foundation Framework

<!-- https://zhuanlan.zhihu.com/p/473794131 -->

!> Noah-Wukong Dataset: https://wukong-dataset.github.io/wukong-dataset/

#### 0. 摘要

华为诺亚开放的一个100M（1亿）的中文多模态图文对数据集，并开放了一组预训练的大模型。预训练模型使用了先进的图像编码器: ResNet/ViT/SwinT，不同的预训练方法：CLIP/FILIP/LiT。

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p1.png" /> 
</div><p align=center>当前VLP模型预训练数据集</p>

模型：双编码器，对比学习目标函数

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p2.png" /> 
</div><p align=center>悟空数据集下预训练的中文多模态模型，双流编码</p>

基准测试：zero-shot 图像分类、图文检索、文图检索。

发现：1）将英文预训练的图像编码器迁移到中文任务时，仍然可以很好的与中文文本进行跨模态预训练，并且取得了中文下游任务的良好效果。2)来自 filip 的跨模态令牌相似性补充了各种基于patch的视觉编码器(比如 SwinT),对更好的视觉和文本表示有帮助。

#### 1. 方法

文图对齐：对比损失

跨模态相似度：1）全局特征的点积相似度（from CLIP），2）学习word-patch alignment的token-wise相似度（from FILIP）

##### 1.1 模型架构

视觉编码器：1）ResNet; 2) Vision Transformer(ViT); 3) Swin Transformer(SwinT)

文本编码器：标准Transformer，Chinese tokenizer(from Chinese BERT-Base model), WordPiece, 词汇表大小为21 128。12层，12个注意力Head，768隐藏维度。

编码器的线性映射：视觉token序列和文本token序列，线性映射到共同的多模态空间，分别接一个L2正则化。

使用全局序列表示（`[CLS]` for ViT, avg pooled patch tokens for SwinT, textual `[CLS]`）计算全局相似度 + 细粒度 
token-wise相似度（from FILIP）。

##### 1.2 训练目标

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p3.jpg" /> 
</div><p align=center>图文/文图对比损失</p>

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p4.jpg" /> 
</div><p align=center>总损失</p>

使用了两种衡量图文相似度的方法：

（1）全局相似度：(in CLIP and ALIGN )

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p5.jpg" /> 
</div><p align=center>全局相似度</p>

（2）Token-wise相似度：(in FILIP)

<div align=center>
    <img src="zh-cn/img/ch4/3-1/p6.png" /> 
</div><p align=center>token-wise相似度</p>

##### 1.3 锁定图像的文本调优(Locked-image Text Tuning, LiT-tuning)

from LiT-tuning。LiT-tuning表明固定预训练图像编码器的权重，解锁文本编码器的权重，效果最好。因此，我们在对本学习设置中也遵循这个方法。具体而言，LiT-tuning方法旨在让一个中文文本编码器从英文数据集预训练的图像编码器中去读取一个合适的表示。我们对每个编码器还额外增加了一层可学习的线性层，将两个模态的表示映射到相同维度。

### 4. FILIP: FINE-GRAINED INTERACTIVE LANGUAGE-IMAGE PRE-TRAINING
<!-- https://zhuanlan.zhihu.com/p/473794131 -->
<!-- FILIP： 细粒度交互语言图像预训练 -->

#### 0. 摘要

现有方法通常通过每个模态的全局特征的相似性来计算模态交互，这缺乏足够的信息；或者在视觉/文本token上使用交叉注意力/自注意力计算更细粒度的交互。

但是交叉/自注意力在训练和推理时并不高效。

本文引入一个大规模细粒度交互语言图像预训练方法（FILIP），来完成更细粒度的跨模态迟交互，通过使用token-wise的视觉token和文本token的最大相似度作为对比学习的目标来实现。

通过只修改对比损失，FILIP成功地利用了图像patches和文本words之间更细粒度的表达能力，同时获得了在推理时预先计算图像和文本表示的能力，保持了大规模训练和推理的有效性。

此外构建了一个新的大规模图文对数据集FILIP300M。

单词分块对齐（word-patch alignment）的可视化表明FILIP可以学习有意义的细粒度特征，并且具有一定的定位能力。

#### 1. 引言

CLIP和ALIGN的核心技巧是双流模型的全局对比对齐。这种模型架构对于像检索这样的下游任务是推理高效的，因为两种模态的编码器可以解耦，图像或文本表示可以离线预先计算。

但是CLIP和ALIGN仅通过全局特征的点积相似度来建模跨模态交互。这是粗粒度的跨模态交互。

而对于细粒度跨模态交互，之前的工作主要有两种方法：

（1）使用目标检测器抽取图像的RoI特征，然后使用VLP模型将其和配对的文本融合。

缺点是：一、需要预先计算大量RoI特征，使预训练复杂化。二、这些方法的zero-shot能力受预定义类别数目的限制，性能也受目标检测器质量的限制。

（2）将来自两个模态的token-wise或patch-wise的表示调制到一个表示空间，然后使用交叉注意力或自注意力建模细粒度交互。

缺点是：交叉、自注意力本身在训练和推理时都并不高效，且推理时不能离线预先计算特征，对于图文检索和图像分类下游任务不高效。

因此，本文提出了FILIP来解决以上限制。

我们使用对比损失，通过一种新的跨模态迟交互机制来建模细粒度语义对齐。

具体来说，使用一个视觉token和文本token之间的token-wise的最大相似度来指导对比目标。

不同于Colbert的summation，这里使用average of token-wise maximum相似度，增强了跨模态表示的学习，稳定了训练。

我们还建立了一个新的多模态数据集FILIP300M。

单词分块对齐（word-patch alignment）的可视化表明FILIP可以学习有意义的细粒度特征，并且具有一定的定位能力。

#### 2. 相关工作

VLP Models:

datasets：YFCC100M, CC12M, (dataset in CLIP, ALIGN);

pre-training tasks: 图文跨模态学习，基于语言建模的任务

多模态交互机制：单流：VisualBERT，ViLT；双流：ViLBERT，CLIP

#### 3. 方法

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p1.png" /> 
</div>

##### 3.1 细粒度对比学习

匹配的文本表示是正样本，批次内其他文本作为负样本。

图文对比损失：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p2.png" /> 
</div><p align=center>图文对比损失</p>

文图对比损失：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p3.png" /> 
</div><p align=center>文图对比损失</p>

一个mini-batch的总损失：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p4.png" /> 
</div><p align=center>总损失</p>

##### 3.2 跨模态迟交互

此前CLIP和ALIGN采用的是全局特征点积相似度：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p5.png" /> 
</div><p align=center>全局特征相似度</p>

我们提出的细粒度交互，对于第k个视觉token，计算它和所有文本token的相似度，使用最大的那个作为它的token-wise maximum similarity：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p6.png" /> 
</div><p align=center>最大相似度</p>

然后使用图像/文本中所有non-padded tokens的average token-wise maximum similarity 作为图到文或文到图的相似度。

因此，第i张图片和第j个文本的相似度为：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p7.png" /> 
</div><p align=center>细粒度图文相似度</p>

第j个文本和第i张图片的相似度为：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p8.png" /> 
</div><p align=center>细粒度文图相似度</p>

##### 3.3 prompt集成和模板

根据 radford 等人(2021)的研究，由于一词多义以及与预训练过程不一致的问题，我们还使用提示模板来增加一些下游任务的原始标签。

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p9.png" /> 
</div><p align=center>prompt模板</p>

+ prefix: 上下文描述如“a photo of a”
+ label: 数据集的类别标签
+ category description：对细粒图像分类由于的类别，如“a type of pet”
+ suffix: 后缀，比如“I like it”对零次分类性能有帮助。

##### 3.4 图像和文本增强

+ 图像增强：AutoAugment
+ 文本增强：使用back-translation重写原始文本。

##### 3.5 预训练数据集

新构建的FILIP300M + CC3M + CC12M + YFCC100M（最终约340M图文对）

#### 4. 实验

模型架构：follow CLIP(ViT-B/32, ViT-L/14)

实验结果集中在zero-shot图像分类和图文检索，具体的结果请参考原始paper.

细粒度对比的可视化：

<div align=center>
    <img src="zh-cn/img/ch4/4-1/p10.png" /> 
</div>

### 5. LiT : Zero-Shot Transfer with Locked-image Text Tuning
<!-- LiT: 使用锁图文本调优进行零次迁移 -->
<!-- https://zhuanlan.zhihu.com/p/473794131 -->

#### 0. 摘要

本文提出对比调优，一种简单的采用对比训练来对齐图文模型的方法，同时仍然利用它们的预训练。经验上我们发现锁定预训练的图像模型而不锁定文本模型效果最好。我们称这种对比调优技术为锁定图像的文本调优(LiT-tuning)，它可以教文本模型为新任务从预训练的图像模型中读取更好的表示。LiT-Tuned的模型在新的视觉任务如图像分类或检索上获得了零次迁移能力的提升。LiT-tuning方法具有广泛的适用性，它可以可靠地运用多种预训练方法(监督和非监督) ，并且使用三种不同的图像文本数据集，跨越不同的架构(ResNet、 Vision Transformers 和 MLP-Mixer)。采用基于Transformer预训练的 ViT-g/14模型，LiT-tuned模型在ImageNet测试集上实现了84.5% 的zero-shot迁移准确率，在具有挑战性的分布外的ObjectNet测试集上实现了81.1% 的准确率。

#### 1. 引言

本文我们采用对比学习框架，提出一种数据和计算高效的策略，称为对比调优(contrastive-tuning)。关键思想是使用图像-文本数据调整文本模型，同时使用预训练的强图像模型作为图像模型。

对比调优有三种调优设计，其中L表示locked，U表示unlocked且使用预训练模型初始化，u表示unlocked且随机初始化。

我们发现锁定图像效果最好。我们称之为“Locked-image Text tuning”(LiT-tuning)，它教会文本模型从预训练的图像模型中读取合适的表示。

<div align=center>
    <img src="zh-cn/img/ch4/5-1/p1.png" /> 
</div>

LiT-tuning和从头训练的CLIP和ALIGN相比，达到了更好的结果。使用预训练的ViT-g/14, LiT-tuning取得了ImageNet上84.5%的零次迁移准确率。

我们认为LiT-tuning工作良好的原因在于它将数据源和用于学习图像描述符和视觉语言对齐的技术解耦。图像-文本数据对于学习自然语言和视觉世界之间的对应关系非常有用，但同时，它可能不够精确和清晰，不足以产生最好的图像描述符。本文仔细研究这一假设，并用经验证据支持它。

#### 2. 方法

对比预训练的作用：（1）学习图像embedding；（2）学习文本embedding以和image embedding space对齐。尽管对比预训练能同时解决以上两个任务，但它可能不是最优方案。

如果不使用对比预训练，一种标注的方法是使用大规模高质量图像数据集学习图像embedding，但是这种方法的缺点是受限于预定义的类别。而图文对数据因为文本的自由形式，可以跨越真实生活中的广泛概念。但是图文对的缺点是对于学习图像embedding来说，质量低于那些人工制作的图像数据集。

因此，我们提出对比微调同时结合了上述两种方案的优点。一种具体方式是用更干净的(半)手动标记数据预训练的图像模型来初始化对比预训练。这样，独立于图像嵌入来学习图像-文本对齐，从而能够从两个数据源中获益。

除了使用监督的预训练图像模型之外，所提出的对比调整也足够灵活，可以集成任何可以产生有意义的表示的模型。我们在实验中使用自监督的预训练图像模型验证了这一点。类似的推理也可以应用于文本塔，因为有许多强大的预训练模型使用特定于文本的数据源和学习技术。

##### 2.1 设计方案和LiT-tuning

对比调优有三种调优设计，其中L表示locked且使用预训练模型初始化，U表示unlocked且使用预训练模型初始化，u表示unlocked且随机初始化。

<div align=center>
    <img src="zh-cn/img/ch4/5-1/p2.png" /> 
</div>

我们发现锁定图像效果最好。我们称之为“Locked-image Text tuning”(LiT-tuning)，它教会文本模型从预训练的图像模型中读取合适的表示。

#### 3. 实验

数据集：CC12M，YFCC100m，Our dataset(follow ALIGN收集的数据集)

<div align=center>
    <img src="zh-cn/img/ch4/5-1/p3.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch4/5-1/p4.png" /> 
</div>


### 6.ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
<!-- https://blog.csdn.net/moxibingdao/article/details/120320356 -->
<!-- https://zhuanlan.zhihu.com/p/410525192 -->

学习良好的视觉和视觉语言表征对于解决计算机视觉问题(图像检索、图像分类、视频理解)是至关重要的，目前，预训练的特征在许多NLP任务中已经展现了非常大的潜力。虽然NLP中的表示学习已经可以用没有人工注释的原始文本训练，但视觉和视觉语言表示仍然严重依赖于昂贵或需要专家知识的训练数据集。对于视觉任务，特征表示的学习主要依赖具有显式的class标签的数据集，如ImageNet或OpenImages。

对于视觉语言任务，一些使用广泛的数据集像Conceptual Captions、MS COCO以及CLIP都涉及到了数据收集和清洗的过程。这类数据预处理的工作严重阻碍了获得更大规模的数据集。在本文中，作者利用了超过10亿的图像文本对的噪声数据集，没有进行数据过滤或后处理步骤 。基于对比学习损失，使用一个简单的双编码器结构来学习对齐图像和文本对的视觉和语言表示 。

作者证明了，语料库规模的巨大提升可以弥补数据内部存在的噪声，因此即使使用简单的学习方式，模型也能达到SOTA的特征表示。当本文模型的视觉表示转移到ImageNet和VTAB等分类任务时，也能取得很强的性能。对齐的视觉和语言表示支持zero-shot的图像分类，并在Flickr30K和MSCOCO图像-文本检索基准数据集上达到了SOTA的结果。

遗憾的是代码并未开源。

#### 1.Motivation

在现有工作中，视觉和视觉语言表示学习大多是分别使用不同的训练数据源进行研究的。在视觉领域，对大规模监督数据（如ImageNet、OpenImages和JFT-300M）进行预训练对提高下游任务的性能是至关重要的。获得这种预训练的数据集需要在数据收集、采样和人工标注方面进行大量的工作，数据获取成本非常大，因此难以扩展。

预训练也是视觉语言建模的方法。然而，视觉语言的预训练数据集，如Conceptual Captions、Visual Genome Dense Captions和 ImageBERT，需要在人类标注、语义解析、清理和平衡方面进行更重的工作。因此，这些数据集的规模仅在10M个样本左右。这至少比视觉领域的数据集小一个数量级，而且比预训练的NLP数据集也小得多。

在这项工作中，作者利用了超过10亿个有噪声的图像文本对的数据集来扩展视觉和视觉语言表示学习。作者采用了Conceptual Captions的方式来获取一个大的噪声数据集。与其不同的是，作者没有用复杂的数据滤波和后处理步骤来清理数据集，而是只应用简单的基于数据频率的过滤。虽然得到的数据集有噪声，但比Conceptual Captions数据集大两个数量级。作者发现，在这样的大规模噪声数据集上预训练的视觉和视觉语言表示在广泛的任务上取得了非常强的性能。

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p1.png" /> 
</div>

作者基于在一个共享的embedding空间中对齐视觉和语言表示的训练目标，使用一个简单的双编码器体系结构来训练模型。作者将这个模型命名为**ALIGN（A Large-scale ImaGe and Noisy-text embedding）**，图像和文本编码器是通过对比损失函数学习的，将匹配的图像文本对的embedding推在一起，同时将不匹配的图像文本对的embedding分开。

这也是自监督和监督表示学习的最有效的损失函数之一。考虑到ALIGN用文本作为图像的细粒度标签，因此图像对文本的对比损失类似于传统的基于标签的分类目标；关键的区别在于这里的label是由文本编码器生成“标签”权重，而不是像ImageNet那样离散的标签。（ALIGN的模型结构如上图所示）

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p2.png" /> 
</div>

对齐的图像和文本表示自然适用于跨模态匹配/检索任务，并在相应的基准数据集测试中实现了SOTA结果。此外，这种跨模态匹配也适用于zero-shot图像分类，在不使用任何训练样本的情况下，在ImageNet中获得了76.4%的Top-1准确率 。此外，图像表示在各种下游视觉任务中也取得了不错的性能。例如，ALIGN在ImageNet中达到了88.64%的Top-1准确率 。（上图展示了跨模态检索的示例）

#### 2. 方法

##### 2.1. A Large-Scale Noisy Image-Text Dataset

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p3.png" /> 
</div>

本文的重点是扩大视觉和语言表示学习的规模。为此，作者创建了一个比现有数据集大得多的数据集。具体来说，作者遵循构建Conceptual Captions数据集的方法，以获得更大规模的图像-文本数据集。

但是，Conceptual Captions数据集还进行了大量的数据过滤和后处理工作，为了获取更大规模的数据，作者通过减轻Conceptual Captions工作中的大部分数据清洗工作来减少数据处理的工作量（作者仅根据数据的频率做了非常简单的数据过滤）。因此，作者获得了一个更大规模的数据集（18亿的图像文本对）。上图展示了数据集中的一些随机采样的例子。

##### 2.2. 预训练与任务迁移

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p4.png" /> 
</div>

ALIGN的大致框架如上图所示。

###### 2.2.1 预训练

作者使用双编码器结构用于训练对齐特征，该模型由一对图像编码器和文本编码器组成。作者使用具有全局池化的EfficientNet作为图像编码器，使用带有`[CLS]`` token embedding的BERT作为文本编码器。在BERT编码器的顶部，作者添加了一个带激活函数的全连接层，以匹配图像的维度。

图像和文本编码器都是通过normalized softmax损失函数进行优化。在训练中，将匹配的图像-文本对视为正样本，并将当前训练batch中的其他随机图像-文本对视为负样本。在训练过程中，优化以下两个损失函数：

image-to-text的对比损失：

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p5.png" /> 
</div>

text-to-image的对比损失：

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p6.png" /> 
</div>

其中，$x_i$和$y_j$分别是第$i$个图像和第$j$个文本的 normalized embedding。N是batch size，σ是temperature系数。在这里，temperature系数是非常重要的，因为图像和文本的embedding都经过了L2-normalized。在本文中，公式中的temperature系数是通过训练来获得，而不是一个超参数。

###### 2.2.2 任务迁移之Image-Text Matching & Retrieval

作者评估了ALIGN在图像对文本和文本对图像的检索任务上的性能（有finetuning和无finetuning）。测试的数据集包括Flickr30K和MSCOCO。此外，作者也在Crisscrossed Captions (CxC)数据集上测试ALIGN的性能（Crisscrossed Captions是MSCOCO的一个扩展数据集，它对caption-caption、 image-image和image-caption对进行了额外的语义相似性判断）。

通过这些扩展的标注，CxC可以实现四个模态内和模式内的检索任务，包括图像到文本检索、文本到图像检索、文本到文本检索和图像到图像的检索任务，以及三个语义文本相似性任务，包括语义文本相似性(STS)、语义图像相似性(SIS)和语义图像-文本相似度(SITS)。

###### 2.2.3. 任务迁移之 Visual Classification

作者首先将ALIGN基于zero-shot方式应用到视觉分类任务上，数据集包括ImageNet ILSVRC-2012 benchmark、ImageNet-R、ImageNet-A、ImageNet-V2。这些ImageNet数据集变种都是ImageNet的一个子集，ImageNet-R和 ImageNet-A是根据不同的分布对ImageNet采样得到的。

作者还将图像编码器迁移到了下游的视觉分类任务中，为此，作者使用了ImageNet以及一些较小的细粒度分类数据集Oxford Flowers-102、 Oxford-IIIT Pets、Stanford Cars、 Food101。

对于ImageNet，作者展示了来自两个设置的结果：只训练顶级分类层（使用冻结的对齐图像编码器）和完全微调（不冻结的对齐图像编码器）。对于细粒度的分类基准数据集测试，作者只展示了后一种设置的结果。此外，作者还在Visual Task Adaptation Benchmark数据集（由19个不同的视觉分类任务组成，每个任务有1000个训练样本）上测试了模型的鲁棒性。

#### 3.实验
##### 3.1. Image-Text Matching & Retrieval

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p7.png" /> 
</div>

上表展示了ALIGN在Flickr30K和MSCOCO数据集上基于Zero-shot和fine-tuned设置下和其他SOTA方法的对比。可以看出在Zero-shot的设置下，ALIGN在图像检索任务上比CLIP获得了7%以上的性能改进。通过微调，ALIGN的性能大大优于所有现有方法。

##### 3.2. Zero-shot Visual Classification

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p8.png" /> 
</div>

如果直接将类名的文本输入文本编码器，ALIGN就可以通过图像-文本检索任务对图像进行分类。上表展示了ALIGN和CLIP在不同分类数据集上Zero-Shot的结果，可以看出，相比于CLIP，ALIGN在大多数数据集具备性能上的明显优势。

##### 3.3. Visual Classification w/ Image Encoder Only

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p9.png" /> 
</div>

上表展示了ALIGN和其他方法在ImageNet数据集上的比较结果。通过冻结参数，ALIGN的性能略优于CLIP，并达到85.5%的SOTA准确率。微调后，ALIGN比BiT和ViT模型获得更高的精度。

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p10.png" /> 
</div>

上表展示了在VTAB（19个任务）上，ALIGN和BiT-L之间的结果比较。结果表明，采用类似的超参数选择方法，ALIGN的性能优于BiT-L。

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p11.png" /> 
</div>

上表展示了不同模型在细粒度分类任务上的迁移学习结果。

##### 3.4. Ablation Study
###### 3.4.1. Model Architectures

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p12.png" /> 
</div>

上图显示了不同图像和文本Backbone组合下的MSCOCO zero-shot检索和ImageNet KNN结果。

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p14.png" /> 
</div>

上表展示了一些ALIGN模型变体与baseline模型（第一行）的比较。第2-4行显示，embedding维度越高，模型性能越高。第5行和第6行显示，在softmax损失中使用更少的in-batch negatives（50%和25%）会降低性能。第7-9行研究了temperature参数对softmax损失的影响。

###### 3.4.2. Pre-training Datasets

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p13.png" /> 
</div>

上表的结果表明一个大规模的训练集对于扩展ALIGN模型和实现更好的性能是至关重要的。

##### 3.5. Analysis of Learned Embeddings

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p15.png" /> 
</div>

作者建立了一个简单的图像检索系统，来研究通过ALIGN训练的embedding行为。上图显示了用不存在于训练集中 text queries进行text-to-image检索的top-1结果。

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p16.png" /> 
</div>

上图显示了用“图像±文本查询”进行图像检索的结果。

##### 3.6. Multilingual ALIGN Model

<div align=center>
    <img src="zh-cn/img/ch4/6-1/p17.png" /> 
</div>

ALIGN的一个优点是，该模型是在有噪声的网络图像文本数据上进行非常简单的过滤之后训练得到的，并且没有对特定语言进行过滤。因此该模型不受语言的约束。上表显示了不同语言下zero-shot和fine-tuning的结果。

#### 4. 总结

在本文中，作者提出了一种简单的方法（ALIGN），利用大规模噪声图像-文本数据来扩大视觉和视觉语言的表示学习。作者避免了对数据预处理和标注的工作量，只需要基于数据频率的简单过滤。

在这个数据集上，作者基于对比学习损失函数训练一个非常简单的双编码器模型ALIGN。ALIGN能够进行跨模态检索，并显著优于SOTA的VSE和基于cross-attention的视觉语言模型。在视觉的下游任务中，ALIGN也可以达到与用大规模标注数据训练的SOTA模型相似的性能，甚至优于SOTA模型。