## Diffusion Model

!> 李宏毅课程： https://www.bilibili.com/video/BV1ss4y1B7v2?p=1&vd_source=def8c63d9c5f9bf987870bf827bfcb3d

!> https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php

<!-- https://www.bilibili.com/video/BV1fk4y1J753/?spm_id_from=333.788.recommend_more_video.3&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

------

<!-- DALLE, Stable diffusion -->

<!-- https://zhuanlan.zhihu.com/p/622238031 -->

<!-- https://mp.weixin.qq.com/s/5HnOAmUKDnOtf2xDX2R9Xg -->

<!-- https://zhuanlan.zhihu.com/p/612854566/?utm_id=0 -->

<!-- https://mp.weixin.qq.com/s/U13Si5nM12_b9WqGaXOMWg -->

<!-- https://mp.weixin.qq.com/s/m9ja0pVpOY8JnehLrzw5_g -->

<!-- https://mp.weixin.qq.com/s/d2y6kL5M6PgsrvDTl8xwQg -->

<!-- https://mp.weixin.qq.com/s/lTcvS9xpeKJV-rXcE3F-VQ -->

<!-- https://zhuanlan.zhihu.com/p/587727367 -->

<!-- https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php -->

<!-- https://www.bilibili.com/video/BV18a4y1T75X?p=1&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

<!-- https://www.bilibili.com/video/BV16N411K7aT?p=17&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->
<!-- https://www.bilibili.com/video/BV1Xs4y1q7uX/?spm_id_from=333.337.search-card.all.click&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

### 1. VAE

<!-- https://zhuanlan.zhihu.com/p/34998569 -->

过去虽然没有细看，但印象里一直觉得变分自编码器（Variational Auto-Encoder，VAE）是个好东西。趁着最近看概率图模型的三分钟热度，我决定也争取把 VAE 搞懂。

于是乎照样翻了网上很多资料，无一例外发现都很含糊，主要的感觉是公式写了一大通，还是迷迷糊糊的，最后好不容易觉得看懂了，再去看看实现的代码，又感觉实现代码跟理论完全不是一回事啊。终于，东拼西凑再加上我这段时间对概率模型的一些积累，并反复对比原论文 Auto-Encoding Variational Bayes，最后我觉得我应该是想明白了。其实真正的 VAE，跟很多教程说的的还真不大一样，很多教程写了一大通，都没有把模型的要点写出来。于是写了这篇东西，希望通过下面的文字，能把 VAE 初步讲清楚。

**1.分布变换**

通常我们会拿 VAE 跟 GAN 比较，的确，它们两个的目标基本是一致的——希望构建一个从隐变量 $Z$ 生成目标数据 $X$ 的模型，但是实现上有所不同。
更准确地讲，它们是假设了服从某些常见的分布（比如正态分布或均匀分布），然后希望训练一个模型 $X=g(Z)$，这个模型能够将原来的概率分布映射到训练集的概率分布，也就是说，它们的目的都是进行分布之间的变换。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p1.jpg" /> 
</div>

生成模型的难题就是判断生成分布与真实分布的相似度，因为我们只知道两者的采样结果，不知道它们的分布表达式。

那现在假设服从标准的正态分布，那么我就可以从中采样得到若干个 $Z_1,Z_2,…,Z_n$，然后对它做变换得到 $\hat{X}_1=g(Z_1),\hat{X}_2=g(Z_2),…,\hat{X}_n=g(Z_n)$，我们怎么判断这个通过 $f$ 构造出来的数据集，它的分布跟我们目标的数据集分布是不是一样的呢？

有读者说不是有 KL 散度吗？当然不行，因为 KL 散度是根据两个概率分布的表达式来算它们的相似度的，然而目前我们并不知道它们的概率分布的表达式。

我们只有一批从构造的分布采样而来的数据 $[\hat{X}_1,\hat{X}_2,...,\hat{X}_n]$，还有一批从真实的分布采样而来的数据 $[X_1,X_2,…,X_n]$（也就是我们希望生成的训练集）。我们只有样本本身，没有分布表达式，当然也就没有方法算 KL 散度。(个人认为这肯定是错的，其实我们可以拟合经验分布的！)

虽然遇到困难，但还是要想办法解决的。GAN 的思路很直接粗犷：既然没有合适的度量，那我干脆把这个度量也用神经网络训练出来吧。就这样，WGAN 就诞生了，详细过程请参考[互怼的艺术：从零直达 WGAN-GP](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247484880&idx=1&sn=4b2e976cc715c9fe2d022ff6923879a8&chksm=96e9da50a19e5346307b54f5ce172e355ccaba890aa157ce50fda68eeaccba6ea05425f6ad76&scene=21#wechat_redirect)。而 VAE 则使用了一个精致迂回的技巧。

**2.VAE慢谈**

这一部分我们先回顾一般教程是怎么介绍 VAE 的，然后再探究有什么问题，接着就自然地发现了 VAE 真正的面目。

**经典回顾**

首先我们有一批数据样本 `{X1,…,Xn}`，其整体用 $X$ 来描述，我们本想根据 `{X1,…,Xn}`得到 $X$ 的分布 $p(X)$，如果能得到的话，那我直接根据 $p(X)$ 来采样，就可以得到所有可能的 $X$ 了（包括 `{X1,…,Xn}`以外的），这是一个终极理想的生成模型了。当然，这个理想很难实现，于是我们将分布改一改：

$$p(X)=\sum_{Z}p(X|Z)p(Z)$$

这里我们就不区分`求和`还是`求积分`了，意思对了就行。此时 $p(X|Z)$ 就描述了一个由 $Z$ 来生成 $X$的模型，而我们假设 $Z$ 服从标准正态分布，也就是 $p(Z)=N(0,I)$。如果这个理想能实现，那么我们就可以先从标准正态分布中采样一个 $Z$，然后根据 $Z$ 来算一个 $X$，也是一个很棒的生成模型。

接下来就是结合自编码器来实现重构，保证有效信息没有丢失，再加上一系列的推导，最后把模型实现。框架的示意图如下：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p2.jpg" /> 
</div><p align=center>▲ VAE的传统理解</p>

看出了什么问题了吗？如果像这个图的话，我们其实完全不清楚：究竟经过重新采样出来的$Z_k$，是不是还对应着原来的 $X_k$，所以我们如果直接最小化 $D(\hat{X}_k,X_k)^2$（这里 $D$ 代表某种距离函数）是很不科学的，而事实上你看代码也会发现根本不是这样实现的。

也就是说，很多教程说了一大通头头是道的话，然后写代码时却不是按照所写的文字来写，可是他们也不觉得这样会有矛盾。

**VAE初现**

其实，**在整个 VAE 模型中，我们并没有去使用 $p(Z)$（先验分布）是正态分布的假设，我们用的是假设 $p(Z|X)$（后验分布,个人认为这也不叫后验分布这叫似然函数）是正态分布**

具体来说，给定一个真实样本 $X_k$，我们假设存在一个专属于 $X_k$ 的分布 $p(Z|X_k)$（学名叫后验分布），并进一步假设这个分布是（独立的、多元的）正态分布。

为什么要强调“专属”呢？因为我们后面要训练一个生成器 $X=g(Z)$，希望能够把从分布 $p(Z|X_k)$ 采样出来的一个 $Z_k$ 还原为 $X_k$。

如果假设 $p(Z)$ 是正态分布，然后从 $p(Z)$ 中采样一个 $Z$，那么我们怎么知道这个 $Z$ 对应于哪个真实的 $X$ 呢？现在 $p(Z|X_k)$ 专属于 $X_k$，我们有理由说从这个分布采样出来的 $Z$ 应该要还原到$X_k$ 中去。
事实上，在论文 Auto-Encoding Variational Bayes 的应用部分，也特别强调了这一点：
>In this case, we can let the variational approximate posterior be a multivariate Gaussian with a diagonal covariance structure:
><div align=center>
    <img src="zh-cn/img/ch5/1-1/p3.jpg" /> 
</div>
论文中的式 (9) 是实现整个模型的关键，不知道为什么很多教程在介绍 VAE 时都没有把它凸显出来。尽管论文也提到 $p(Z)$ 是标准正态分布，然而那其实并不是本质重要的。

再次强调，这时候每一个 $X_k$ 都配上了一个专属的正态分布，才方便后面的生成器做还原。但这样有多少个 $X$ 就有多少个正态分布了。我们知道正态分布有两组参数：均值 $μ$ 和方差 $σ^2$（多元的话，它们都是向量）。

那我怎么找出专属于 $X_k$ 的正态分布 $p(Z|X_k)$ 的均值和方差呢？好像并没有什么直接的思路。

那好吧，我就用神经网络来拟合出来。这就是神经网络时代的哲学：难算的我们都用神经网络来拟合，在 WGAN 那里我们已经体验过一次了，现在再次体验到了。

于是我们构建两个神经网络 $μ_k=f_1(X_k)，logσ^2=f_2(X_k)$ 来算它们了。我们选择拟合 $logσ^2$ 而不是直接拟合 $σ^2$，是因为 $σ^2$ 总是非负的，需要加激活函数处理，而拟合 $logσ^2$ 不需要加激活函数，因为它可正可负。

到这里，我能知道专属于 $X_k$ 的均值和方差了，也就知道它的正态分布长什么样了，然后从这个专属分布中采样一个 $Z_k$ 出来，然后经过一个生成器得到 $\hat{X}_k=g(Z_k)$。

现在我们可以放心地最小化 $D(\hat{X}_k,X_k)^2$，因为 $Z_k$ 是从专属 $X_k$ 的分布中采样出来的，这个生成器应该要把开始的 $X_k$ 还原回来。于是可以画出 VAE 的示意图：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p4.png" /> 
</div>
事实上，VAE 是为每个样本构造专属的正态分布，然后采样来重构。

**3.分布标准化**

让我们来思考一下，根据上图的训练过程，最终会得到什么结果。首先，我们希望重构$X$，也就是最小化$D(\hat{X}_k,X_k)^2$，但是这个重构过程受到噪声的影响，因为$Z_k$ 是通过重新采样过的，不是直接由 encoder 算出来的。显然噪声会增加重构的难度，不过好在这个噪声强度（也就是方差）通过一个神经网络算出来的，所以最终模型为了重构得更好，肯定会想尽办法让方差为0。

而方差为 0 的话，也就没有随机性了，所以不管怎么采样其实都只是得到确定的结果（也就是均值），只拟合一个当然比拟合多个要容易，而均值是通过另外一个神经网络算出来的。说白了，模型会慢慢退化成普通的 **AutoEncoder，噪声不再起作用**。

这样不就白费力气了吗？说好的生成模型呢？

别急别急，其实 VAE 还让所有的$p(Z|X)$都向标准正态分布看齐，这样就防止了噪声为零，同时保证了模型具有生成能力。

怎么理解“保证了生成能力”呢？

如果所有的$p(Z|X)$都很接近标准正态分布$N(0,I)$，那么根据定义：
$$p(Z)=\sum_{X}p(Z|X)p(X)=\sum_{X}N(0,I)p(X)=N(0,I)\sum_{X}p(X)=N(0,I)$$
这样我们就能达到我们的先验假设：$p(Z)$ 是标准正态分布。然后我们就可以放心地从 $N(0,I)$ 中采样来生成图像了。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p5.jpg" /> 
</div>

为了使模型具有生成能力，VAE 要求每个 $p(Z|X)$ 都向正态分布看齐。

那怎么让所有的 $p(Z|X)$ 都向 $N(0,I)$ 看齐呢？如果没有外部知识的话，其实最直接的方法应该是在重构误差的基础上中加入额外的 loss：
$$L_{\mu}=||f_1(X_k)||^2,L_{\sigma^2}=||f_2(X_k)||^2$$

因为它们分别代表了均值 $μ_k$ 和方差的对数 $logσ^2$，达到 $N(0,I)$ 就是希望二者尽量接近于 0 了。不过，这又会面临着这两个损失的比例要怎么选取的问题，选取得不好，生成的图像会比较模糊。

所以，原论文直接算了一般（各分量独立的）正态分布与标准正态分布的 KL 散度$KL(N(μ,σ^2)‖N(0,I))$作为这个额外的 loss，计算结果为：


<div align=center>
    <img src="zh-cn/img/ch5/1-1/p6.jpg" /> 
</div>

这里的 $d$ 是隐变量 $Z$ 的维度，而 $\mu_{(i)}$ 和 $σ_{(i)}^{2}$ 分别代表一般正态分布的均值向量和方差向量的第 $i$ 个分量。直接用这个式子做补充 loss，就不用考虑均值损失和方差损失的相对比例问题了。

显然，这个 loss 也可以分两部分理解：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p7.jpg" /> 
</div>

**4.推导**

由于我们考虑的是各分量独立的多元正态分布，因此只需要推导一元正态分布的情形即可，根据定义我们可以写出：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p8.jpg" /> 
</div>

整个结果分为三项积分，第一项实际上就是 $−logσ^2$ 乘以概率密度的积分（也就是 1），所以结果是 $−logσ^2$；第二项实际是正态分布的二阶矩，熟悉正态分布的朋友应该都清楚正态分布的二阶矩为 $μ^2+σ^2$；而根据定义，第三项实际上就是“-方差除以方差=-1”。所以总结果就是：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p9.jpg" /> 
</div>

**5.重参数技巧**

最后是实现模型的一个技巧，英文名是 Reparameterization Trick，我这里叫它做重参数吧。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p10.jpg" /> 
</div><p align=center>▲ 重参数技巧</p>

其实很简单，就是我们要从 $p(Z|X_k)$ 中采样一个 $Z_k$ 出来，尽管我们知道了 $p(Z|X_k)$ 是正态分布，但是均值方差都是靠模型算出来的，我们要靠这个过程反过来优化均值方差的模型，但是“采样”这个操作是不可导的，而采样的结果是可导的，于是我们利用了一个事实：

> 从$N(\mu,\sigma^2)$中采样一个$Z$,相当于从$N(0,1)$中采样一个$\epsilon$，然后让$Z=\mu+\sigma\times \epsilon$

所以，我们将从 $N(μ,σ^2)$ 采样变成了从 $N(0,1)$ 中采样，然后通过参数变换得到从$N(μ,σ^2)$中采样的结果。这样一来，“采样”这个操作就不用参与梯度下降了，改为采样的结果参与，使得整个模型可训练了。具体怎么实现，大家把上述文字对照着代码看一下，一下子就明白了。

**6.后续分析**

即便把上面的所有内容都搞清楚了，面对 VAE，我们可能还存有很多疑问。

**本质是什么**

VAE 的本质是什么？VAE 虽然也称是 AE（AutoEncoder）的一种，但它的做法（或者说它对网络的诠释）是别具一格的。在 VAE 中，它的 Encoder 有两个，一个用来计算均值，一个用来计算方差，这已经让人意外了：Encoder 不是用来 Encode 的，是用来算均值和方差的，这真是大新闻了，还有均值和方差不都是统计量吗，怎么是用神经网络来算的？

事实上，我觉得 VAE 从让普通人望而生畏的变分和贝叶斯理论出发，最后落地到一个具体的模型中，虽然走了比较长的一段路，但最终的模型其实是很接地气的。

**它本质上就是在我们常规的自编码器的基础上，对 encoder 的结果（在VAE中对应着计算均值的网络）加上了“高斯噪声”，使得结果 decoder 能够对噪声有鲁棒性；而那个额外的 KL loss（目的是让均值为 0，方差为 1），事实上就是相当于对 encoder 的一个正则项，希望 encoder 出来的东西均有零均值。**

那另外一个 encoder（对应着计算方差的网络）的作用呢？它是用来动态调节噪声的强度的。

直觉上来想，当 decoder 还没有训练好时（重构误差远大于 KL loss），就会适当降低噪声（KL loss 增加），使得拟合起来容易一些（重构误差开始下降）。反之，如果 decoder 训练得还不错时（重构误差小于 KL loss），这时候噪声就会增加（KL loss 减少），使得拟合更加困难了（重构误差又开始增加），这时候 decoder 就要想办法提高它的生成能力了。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p11.jpg" /> 
</div><p align=center>▲ VAE的本质结构</p>

说白了，重构的过程是希望没噪声的，而 KL loss 则希望有高斯噪声的，两者是对立的。所以，VAE 跟 GAN 一样，内部其实是包含了一个对抗的过程，只不过它们两者是混合起来，共同进化的。

从这个角度看，VAE 的思想似乎还高明一些，因为在 GAN 中，造假者在进化时，鉴别者是安然不动的，反之亦然。当然，这只是一个侧面，不能说明 VAE 就比 GAN 好。

GAN 真正高明的地方是：它连度量都直接训练出来了，而且这个度量往往比我们人工想的要好（然而 GAN 本身也有各种问题，这就不展开了）。

**正态分布？**

对于 $p(Z|X)$ 的分布，读者可能会有疑惑：是不是必须选择正态分布？可以选择均匀分布吗？

首先，这个本身是一个实验问题，两种分布都试一下就知道了。但是从直觉上来讲，正态分布要比均匀分布更加合理，因为正态分布有两组独立的参数：均值和方差，而均匀分布只有一组。前面我们说，在 VAE 中，重构跟噪声是相互对抗的，重构误差跟噪声强度是两个相互对抗的指标，而在改变噪声强度时原则上需要有保持均值不变的能力，不然我们很难确定重构误差增大了，究竟是均值变化了（encoder的锅）还是方差变大了（噪声的锅）。而均匀分布不能做到保持均值不变的情况下改变方差，所以正态分布应该更加合理。

**变分在哪里?**

还有一个有意思（但不大重要）的问题是：VAE 叫做“变分自编码器”，它跟变分法有什么联系？在VAE 的论文和相关解读中，好像也没看到变分法的存在？

其实如果读者已经承认了 KL 散度的话，那 VAE 好像真的跟变分没多大关系了，因为 KL 散度的定义是：

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p12.jpg" /> 
</div>

如果是离散概率分布就要写成求和，我们要证明：已概率分布 $p(x)$（或固定$q(x)$）的情况下，对于任意的概率分布 $q(x)$（或 $p(x)$），都有 $KLp(x)‖q(x))≥0$，而且只有当$p(x)=q(x)$时才等于零。

因为 $KL(p(x)‖q(x))$实际上是一个泛函，要对泛函求极值就要用到变分法，当然，这里的变分法只是普通微积分的平行推广，还没涉及到真正复杂的变分法。而 VAE 的变分下界，是直接基于 KL 散度就得到的。所以直接承认了 KL 散度的话，就没有变分的什么事了。一句话，VAE 的名字中“变分”，是因为它的推导过程用到了 KL 散度及其性质。

**条件VAE: CVAE**

最后，因为目前的 VAE 是无监督训练的，因此很自然想到：如果有标签数据，那么能不能把标签信息加进去辅助生成样本呢？

这个问题的意图，往往是希望能够实现控制某个变量来实现生成某一类图像。当然，这是肯定可以的，我们把这种情况叫做 Conditional VAE，或者叫 CVAE（相应地，在 GAN 中我们也有个 CGAN）。

但是，CVAE 不是一个特定的模型，而是一类模型，总之就是把标签信息融入到 VAE 中的方式有很多，目的也不一样。这里基于前面的讨论，给出一种非常简单的 VAE。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p13.jpg" /> 
</div><p align=center>▲ 一个简单的CVAE结构</p>

在前面的讨论中，我们希望 $X$ 经过编码后，$Z$ 的分布都具有零均值和单位方差，这个“希望”是通过加入了 KL loss 来实现的。

如果现在多了类别信息 $Y$，我们可以希望同一个类的样本都有一个专属的均值 $μ^Y$（方差不变，还是单位方差），这个 $μ^Y$让模型自己训练出来。这样的话，有多少个类就有多少个正态分布，而在生成的时候，我们就可以通过控制均值来控制生成图像的类别。事实上，这样可能也是在 VAE 的基础上加入最少的代码来实现 CVAE 的方案了，因为这个“新希望”也只需通过修改 KL loss 实现：
<div align=center>
    <img src="zh-cn/img/ch5/1-1/p14.jpg" /> 
</div>
下图显示这个简单的 CVAE 是有一定的效果的，不过因为 encoder 和 decoder 都比较简单（纯 MLP），所以控制生成的效果不尽完美。

<div align=center>
    <img src="zh-cn/img/ch5/1-1/p15.jpg" /> 
</div>

用这个 CVAE 控制生成数字 9，可以发现生成了多种样式的 9，并且慢慢向 7 过渡，所以初步观察这种 CVAE 是有效的。更完备的 CVAE 请读者自行学习了，最近还出来了 CVAE 与 GAN 结合的工作 CVAE-GAN: Fine-Grained Image Generation through Asymmetric Training，模型套路千变万化。


**7.代码**

我把 Keras 官方的 VAE 代码复制了一份，然后微调并根据前文内容添加了中文注释，也把最后说到的简单的 CVAE 实现了一下，供读者参考。

!> github:https://github.com/bojone/vae

总的来说，VAE 的思路还是很漂亮的。倒不是说它提供了一个多么好的生成模型（因为事实上它生成的图像并不算好，偏模糊），而是它提供了一个将概率图跟深度学习结合起来的一个非常棒的案例，这个案例有诸多值得思考回味的地方。

------

### 2.Stable Diffusion的原理介绍

<!--开源项目webUI： https://zhuanlan.zhihu.com/p/622238031 -->

<!--1 https://mp.weixin.qq.com/s/5HnOAmUKDnOtf2xDX2R9Xg -->

<!--2 https://zhuanlan.zhihu.com/p/612854566/?utm_id=0 -->

<!-- https://mp.weixin.qq.com/s/U13Si5nM12_b9WqGaXOMWg -->

<!--3 主要参考 https://mp.weixin.qq.com/s/m9ja0pVpOY8JnehLrzw5_g -->

<!--4. https://mp.weixin.qq.com/s/d2y6kL5M6PgsrvDTl8xwQg -->

<!--3. https://mp.weixin.qq.com/s/lTcvS9xpeKJV-rXcE3F-VQ -->

<!--读了14篇论文，终于会拿捏Diffusion了： https://zhuanlan.zhihu.com/p/587727367 -->

2022年，Stable Diffusion横空出世，成为AI行业从传统深度学习时代过渡至AIGC时代的标志模型，并为工业界和投资界注入了新的活力，让AI再次性感。


#### 2.1 Stable Diffusion原理详解

<!-- https://zhuanlan.zhihu.com/p/612854566/?utm_id=0 -->

Stable Diffusion是stability.ai开源的图像生成模型，可以说Stable Diffusion的发布将AI图像生成提高到了全新高度，其效果和影响不亚于Open AI发布ChatGPT。今天我们就一起学习一下Stable Diffusion的原理。

**图像生成的发展**

在Stable Diffusion诞生之前，计算机视觉和机器学习方面最重要的突破是 GAN（Generative Adversarial Networks 生成对抗网络）。GAN让超越训练数据已有内容成为可能，从而打开了一个全新领域——现在称之为生成建模。

然而，在经历了一段蓬勃发展后，GAN开始暴露出一些瓶颈和弊病，大家倾注了很多心血努力解决对抗性方法所面临的一些瓶颈，但是鲜有突破，GAN由此进入平台期。GAN的主要问题在于：

+ 图像生成缺乏多样性
+ 模式崩溃
+ 多模态分布学习困难
+ 训练时间长
+ 由于问题表述的对抗性，不容易训练

另外，还有一条基于似然（例如，马尔可夫随机场）的技术路线，尽管已经存在很久，但由于对每个问题的实施和制定都很复杂，因此未能产生重大影响。

近几年，随着算力的增长，一些过去算力无法满足的复杂算法得以实现，其中有一种方法叫“扩散模型”——一种从气体扩散的物理过程中汲取灵感并试图在多个科学领域模拟相同现象的方法。该模型在图像生成领域展现了巨大的潜力，成为今天Stable Diffusion的基础。

**扩散模型**

扩散模型是一种生成模型，用于生成与训练数据相似的数据。简单的说，扩散模型的工作方式是通过迭代添加高斯噪声来“破坏”训练数据，然后学习如何消除噪声来恢复数据。一个标准扩散模型有两个主要过程：正向扩散和反向扩散。

在正向扩散阶段，通过逐渐引入噪声来破坏图像，直到图像变成完全随机的噪声。在反向扩散阶段，使用一系列马尔可夫链逐步去除预测噪声，从高斯噪声中恢复数据。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p1.jpg" /> 
</div><p align=center>通过缓慢添加（去除）噪声来生成样本的正向（反向）扩散过程的马尔可夫链(图片来源: Jonathan Ho, Ajay Jain, Pieter Abbeel. 2020)</p>

对于噪声的估计和去除，最常使用的是 U-Net。该神经网络的架构看起来像字母 U，由此得名。U-Net 是一个全连接卷积神经网络，这使得它对图像处理非常有用。U-Net的特点在于它能够将图像作为入口，并通过减少采样来找到该图像的低维表示，这使得它更适合处理和查找重要属性，然后通过增加采样将图像恢复回来。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p2.jpg" /> 
</div><p align=center>一个典型的U-Net架构实例</p>

具体的说，所谓去除噪声就是从时间帧$t$向时间帧$t-1$ 的变换，其中$t$是$t_0$(没有噪声）到$t_{max}$ (完全噪声)之间的任意时间帧。变换规则为：

1. 输入时间帧$t$的图像，并且在该时间帧上图像存在特定噪声；
2. 使用 U-Net 预测总噪声量；
3. 然后在时间帧$t$的图像中去除总噪声的“一部分”，得到噪声较少的时间帧$t-1$的图像。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p3.jpg" /> 
</div><p align=center>向图片逐步增加/删除噪声</p>

从数学上讲，执行此上述方法$T$次比尝试消除整个噪声更有意义。通过重复这个过程，噪声会逐渐被去除，我们会得到一个更“干净”的图像。比如对于带有噪声的图，我们通过在初始图像上添加完全噪声，然后再迭代地去除它来生成没有噪声的图像，效果比直接在原图上去除噪声要好。

近几年，扩散模型在图像生成任务中表现出突出的性能，并在图像合成等多个任务中取代了GAN。由于扩散模型能够保持数据的语义结构，因此不会受到模式崩溃的影响。

然而，实现扩散模型存在一些困难。因为所有马尔可夫状态都需要一直在内存中进行预测，这意味着内存中要一直保存多个大型深度网络的实例，从而导致扩散模型非常吃内存。此外，扩散模型可能会陷入图像数据中难以察觉的细粒度复杂性中，导致训练时间变得太长（几天到几个月）。矛盾的是，细粒度图像生成是扩散模型的主要优势之一，我们无法避免这个“甜蜜的烦恼”。由于扩散模型对计算要求非常高，训练需要非常大的内存和电量，这使得早前大多数研究人员无法在现实中实现该模型。

**Transformer**

Transformer是来自 NLP 领域的非常著名的模型方法。Transformer在语言建模和构建对话式 AI 工具方面取得了巨大成功。 在视觉应用中，Transformer 表现出了泛化和自适应的优势，这使得它们非常适合通用学习。 它们比其他技术能够更好地捕捉文本甚至图像中的语义结构。 然而，Transformers 需要大量数据，并且与其他方法相比，在许多视觉领域的性能方面也面临着平台期。

Transformer可以与扩散模型结合，通过Transformer的“词嵌入”可以将文本插入到模型中。这意味着将词Token化后，然后将这种文本表示添加到U-Net的输入（图像）中，经过每一层U-Net神经网络与图像一起进行变换。从第一次迭代开始到之后的每一次迭代都加入相同的文本，从而让文本“作为指南”生成图像，从有完整噪声的第一次迭代开始，然后进一步向下应用到整个迭代。

**Stable Diffusion**

扩散模型最大的问题是它的时间成本和经济成本都极其“昂贵”。Stable Diffusion的出现就是为了解决上述问题。如果我们想要生成一张$1024\times 1024$
 尺寸的图像，U-Net 会使用 $1024 \times 1024$
 尺寸的噪声，然后从中生成图像。这里做一步扩散的计算量就很大，更别说要循环迭代多次直到100%。一个解决方法是将大图片拆分为若干小分辨率的图片进行训练，然后再使用一个额外的神经网络来产生更大分辨率的图像（超分辨率扩散）。

2021年发布的[Latent Diffusion](https://arxiv.org/pdf/2112.10752.pdf)模型给出了不一样的方法。 Latent Diffusion模型不直接在操作图像，而是在潜在空间中进行操作。通过将原始数据编码到更小的空间中，让U-Net可以在低维表示上添加和删除噪声。

**潜在空间(Lantent Space)**

潜在空间简单的说是对压缩数据的表示。所谓压缩指的是用比原始表示更小的数位来编码信息的过程。比如我们用一个颜色通道（黑白灰）来表示原来由RGB三原色构成的图片，此时每个像素点的颜色向量由3维变成了1维度。维度降低会丢失一部分信息，然而在某些情况下，降维不是件坏事。通过降维我们可以过滤掉一些不太重要的信息，只保留最重要的信息。

假设我们像通过全连接的卷积神经网络训练一个图像分类模型。当我们说模型在学习时，我们的意思是它在学习神经网络每一层的特定属性，比如边缘、角度、形状等……每当模型使用数据（已经存在的图像）学习时，都会将图像的尺寸先减小再恢复到原始尺寸。最后，模型使用解码器从压缩数据中重建图像，同时学习之前的所有相关信息。因此，空间变小，以便提取和保留最重要的属性。这就是潜在空间适用于扩散模型的原因。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p4.jpg" /> 
</div><p align=center>利用卷积神经网络提取最重要的属性</p>

**Latent Diffusion**

“潜在扩散模型”（Latent Diffusion Model）将GAN的感知能力、扩散模型的细节保存能力和Transformer的语义能力三者结合，创造出比上述所有模型更稳健和高效的生成模型。与其他方法相比，Latent Diffusion不仅节省了内存，而且生成的图像保持了多样性和高细节度，同时图像还保留了数据的语义结构。

任何生成性学习方法都有两个主要阶段：**感知压缩**和**语义压缩**。

1. 感知压缩

在感知压缩学习阶段，学习方法必须去除高频细节将数据封装到抽象表示中。此步骤对构建一个稳定、鲁棒的环境表示是必要的。GAN 擅长感知压缩，通过将高维冗余数据从像素空间投影到潜在空间的超空间来实现这一点。潜在空间中的潜在向量是原始像素图像的压缩形式，可以有效地代替原始图像。

更具体地说，用自动编码器 (Auto Encoder) 结构捕获感知压缩。 自动编码器中的编码器将高维数据投影到潜在空间，解码器从潜在空间恢复图像。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p5.jpg" /> 
</div><p align=center>自动编码器和解码器构成感知压缩</p>

2. 语义压缩

在学习的第二阶段，图像生成方法必须能够捕获数据中存在的语义结构。 这种概念和语义结构提供了图像中各种对象的上下文和相互关系的保存。 Transformer擅长捕捉文本和图像中的语义结构。 Transformer的泛化能力和扩散模型的细节保存能力相结合，提供了两全其美的方法，并提供了一种生成细粒度的高度细节图像的方法，同时保留图像中的语义结构。

3. 感知损失

潜在扩散模型中的自动编码器通过将数据投影到潜在空间来捕获数据的感知结构。论文作者使用一种特殊的损失函数来训练这种称为“感知损失”的自动编码器。该损失函数确保重建限制在图像流形内，并减少使用像素空间损失（例如 L1/L2 损失）时出现的模糊。

4. 扩散损失

扩散模型通过从正态分布变量中逐步去除噪声来学习数据分布。换句话说，扩散模型使用长度为$T$的反向马尔可夫链。这也意味着扩散模型可以建模为时间步长为 
$t=1,...,T$的一系列“T”去噪自动编码器。由下方公式中的$\epsilon_{\theta}$
表示：

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p6.png" /> 
</div>

公式(1)给出了扩散模型的损失函数。在潜在扩散模型中，损失函数取决于潜在向量而不是像素空间。我们将像素空间元素$x$
替换成潜在向量$\epsilon(x)$
，将$t$时间的状态$x_t$
替换为去噪U-Net在时间$t$的潜在状态$z_t$
，即可得到潜在扩散模型的损失函数，见公式(2)：

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p7.png" /> 
</div>

将公式(2)写成条件损失函数，得到公式(3)：

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p8.png" /> 
</div>

其中$\tau_{\theta}(y)$是条件$y$下的领域专用编码器(比如Transformer)。

5. 条件扩散

扩散模型是依赖于先验的条件模型。在图像生成任务中，先验通常是文本、图像或语义图。为了获得先验的潜在表示，需要使用转换器（例如 CLIP）将文本/图像嵌入到潜在向量$\tau$
中。因此，最终的损失函数不仅取决于原始图像的潜在空间，还取决于条件的潜在嵌入。

**文本-图像合成**

在 Python 实现中，我们可以使用使用 LDM v4 的最新官方实现来生成图像。 在文本到图像的合成中，潜在扩散模型使用预训练的 CLIP 模型，该模型为文本和图像等多种模态提供基于Transformer的通用嵌入。 然后将Transformer模型的输出输入到称为“diffusers”的潜在扩散模型Python API，同时还可以设置一些参数（例如，扩散步数、随机数种子、图像大小等）

**图像-图像合成**

相同的方法同样适用于图像到图像的合成，不同的是需要输入样本图像作为参考图像。生成的图像在语义和视觉上与作为参考给出的图像相似。这个过程在概念上类似于基于样式的 GAN 模型，但它在保留图像的语义结构方面做得更好。

**整体架构**

上面介绍了潜在扩散模型的各个主要技术部分，下面我们将它们合成一个整理，看一下潜在扩散模型的完整工作流程。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p9.png" /> 
</div><p align =center> 潜在扩散模型的架构（图片来源：Rombach & Blattmann, et al. 2022）</p>

上图中$x$表示输入图像，$\tilde{x}$表示生成的图像；$\epsilon$是编码器，$ \mathcal{D}$是解码器，二者共同构成了感知压缩；$z$是潜在向量；
$z_{T}$是增加噪声后的潜在向量；$\tau_{\theta}$是文本/图像的编码器（比如Transformer或CLIP），实现了语义压缩。

**总结**

本文向大家介绍了图像生成领域最前沿的Stable Diffusion模型。本质上Stable Diffusion属于潜在扩散模型(Latent Diffusion Model)。潜在扩散模型在生成细节丰富的不同背景的高分辨率图像方面非常稳健，同时还保留了图像的语义结构。 因此，潜在扩散模型是图像生成即深度学习领域的一项重大进步。 Stable Diffusion只是将潜在扩散模型应用于高分辨率图像，同时使用 CLIP 作为文本编码器。

------

#### 2.2 Stable Diffusion完整核心基础讲解

<!-- https://mp.weixin.qq.com/s/5HnOAmUKDnOtf2xDX2R9Xg -->

2022年，Stable Diffusion横空出世，成为AI行业从传统深度学习时代过渡至AIGC时代的标志模型，并为工业界和投资界注入了新的活力，让AI再次性感。

本文中，将深入浅出的讲解Stable Diffusion的核心知识，例举最有价值的应用场景，对Stable Diffusion的训练过程进行通俗易懂的分析，并尝试对其性能进行优化，希望我们能更好的入门Stable Diffusion及其背后的AIGC领域。

**1.Stable Diffusion模型原理**

Stable Diffusion（SD）模型和GAN模型一样，是生成式模型，了解GAN模型的朋友都知道，生成式模型能够生成和训练集分布相似的输出结果（拟合数据分布），在计算机视觉领域是图片，在自然语言处理领域是文字。下面是主流生成式模型各自的生成逻辑：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p1.png" /> 
</div><p align =center> 生成式模型的主流架构</p>

在这里拿GAN详细展开讲讲，由于篇幅原因，VAE和Flow-based models这里就不过多介绍。

GAN由生成器G和判别器D组成。其中，生成器主要负责生成相应的样本数据，输入一般是由高斯分布随机采样得到的噪声$Z$。而判别器的主要职责是区分生成器生成的样本与GT(Ground Truth)
样本，输入一般是GT样本与相应的生成样本，我们想要的是对GT样本输出的置信度越接近1越好，而对生成样本输出的置信度越接近0越好。与一般神经网络不同的是，**GAN在训练时要同时训练生成器与判别器，所以其训练难度是比较大的**。

我们可以将GAN中的生成器比喻为印假钞票的犯罪分子，判别器则被当作警察。犯罪分子努力让印出的假钞看起来逼真，警察则不断提升对于假钞的辨识能力。二者互相博弈，随着时间的进行，都会越来越强。在图像生成任务中也是如此，生成器不断生成尽可能逼真的假图像。判别器则判断图像是GT图像，还是生成的图像。二者不断博弈优化，最终生成器生成的图像使得判别器完全无法判别真假。好的，讲完GAN，让我们回到SD模型。SD模型有两个最基本也是最经典的应用，分别是文生图（txt2pic）和图生图（pic2pic）。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p2.png" /> 
</div><p align =center> SD文生图过程</p>

文生图是指将一段文字输入到SD模型中，经过一定的迭代次数，SD模型输出一张符合输入文字描述的图片。比如上图中输入了“天堂，巨大的，海滩”，于是生成了一个美丽沙滩的图片。


<div align=center>
    <img src="zh-cn/img/ch5/3-1/p3.png" /> 
</div><p align =center> SD图生图过程</p>

而图生图在输入文字的基础上，再输入一张图片，SD模型将根据文字的提示，将图片进行丰富，比如上图中，SD模型将“海盗船”添加在之前那个美丽的沙滩上。

在这里也分享一些朋友生成的图片，整体上都非常逼真：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p4.png" /> 
</div>

感受了SD模型强大的生成能力，大家可能会想到生成式领域上一个霸主模型GAN，与GAN模型不同的是，SD模型是属于扩散模型，并且是基于Latent的扩散模型。

!> 什么是扩散模型呢？用通俗的话来讲，就是将图像的生成过程分多步进行，逐步完善图像内容，经过20-50次的“扩散”循环，最终输出精致的图像。

下面是一个直观的例子，将随机高斯噪声矩阵通过SD模型的Inference过程，逐步去燥，最后生成一个小别墅的图片。
<div align=center>
    <img src="zh-cn/img/ch5/3-1/p5.png" /> 
</div><p align =center> SD模型的Inference过程</p>

那latent又是什么呢？基于latent的扩散模型可以在低维度的隐空间上进行“扩散”过程而不是在实际pixel空间，这样一来大大降低了内存占用和计算复杂性，这是常规扩散模型和latent扩散模型之间的主要区别。同样的，在训练中，latent扩散模型也将训练过程聚焦在latent空间中。

举个例子，如果SD模型中将数据矩阵缩小的倍数设为8，那么原本尺寸为`[3,512,512]`的数据矩阵就会进入`[3,64,64]`的latent隐空间中，内存和计算量直接缩小64倍，整体效率大大提升。

到这里，大家应该对SD模型的基本概念有一个清晰的认识了，再帮大家总结一下：
+ SD模型是生成式模型，与GAN模型有很多相似的地方，输入可以是图片，也可以是文字，输出是图片。
+ SD模型属于扩散模型，扩散模型的整理逻辑是生成过程分步化与迭代化，这给整个生成过程引入更多约束与优化提供了可能。
+ SD模型是基于latent的，将生成空间压缩到latent空间中，比起常规扩散模型，大大提高计算效率的同时，降低了内存占用，成为了SD模型爆发的关键一招。

SD模型的整体流程是一个优化噪声的艺术。

**2.Stable Diffusion模型的核心组件**

SD模型主要由自动编码器（VAE），U-Net以及文本编码器三个核心组件构成。
<div align=center>
    <img src="zh-cn/img/ch5/3-1/p6.png" /> 
</div><p align =center> SD模型文生图流程</p>

+ 自动编码器（VAE）：VAE的编码器能够将输入图像转换为低维特征(Latent)，作为U-Net的输入。VAE的解码器将隐特征升维解码成完整图像。不同的VAE结构能够为生成图片带来不同的细节与整体颜色。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p7.png" /> 
</div><p align =center>VAE的主要工作</p>

为什么VAE可以将图像压缩到一个非常小的潜空间后能再次对图像还原呢？虽然整个过程可以看作是一个有损压缩，但自然图像并不是随机的，它们具有很高的规律性：比如说一张脸上的眼睛、鼻子、脸颊和嘴巴之间遵循特定的空间关系，又比如说一只猫有四条腿，并且是一个特定的生物结构。所以如果我们生成的图像尺寸在$512\times 512$之上时，其实特征损失带来的影响非常小。

+ U-Net：预测噪声残差，结合**调度算法（PNDM，DDIM，K-LMS等）进行噪声重构**，逐步将随机高斯噪声转化成图片的隐特征。U-Net整体结构一般由ResNet模块构成，并在ResNet模块之间添加CrossAttention模块用于接收文本信息。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p8.png" /> 
</div><p align =center>常规U-Net结构
</p>


+ 文本编码器：将输入prompt进行编码，输出token embeddings向量（语意信息），通过CrossAttention方式送入扩散模型的U-Net中作为condition，对生成图像内容进行一定程度上的控制，目前SD默认的是CLIP text encoder。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p9.png" /> 
</div><p align =center>ResNet模块与CrossAttention模块的结合
</p>


**3.Stable Diffusion推理流程**

想要运行Stable Diffusion（SD），我们可以直接使用diffusers的完整pipeline流程。

```
#首先，安装相关依赖
pip install diffusers transformers scipy ftfy accelerate

#读取diffuers库
from diffusers import StableDiffusionPipeline

#初始化SD模型，加载预训练权重
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

#使用GPU加速
pipe.to("cuda")

#如GPU的内存少于10GB，可以加载float16精度的SD模型
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", torch_dtype=torch.float16)

#接下来，我们就可以运行pipeline了
prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]

# 由于没有固定seed，每次运行代码，我们都会得到一个不同的图片。
```

我们打开下载的预训练文件夹，可以看到预训练模型主要由以下几个部分组成：

+ text_encoder和tokenizer，
+ scheduler，
+ unet，
+ vae。

其中text_encoder，scheduler，unet，vae分别代表了上面讲到过的SD模型的核心结构。
同时我们还可以看到Tokenizer文件夹。Tokenizer首先将Prompt中的每个词转换为一个称为标记（token）的数字，符号化（Tokenization）是计算机理解单词的方式。然后，通过text_encoder将每个标记都转换为一个768值的向量，称为嵌入（embedding），用于U-Net的condition。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p10.png" /> 
</div><p align =center>Tokenizer的作用</p>

有时候我们运行完pipeline之后，会出现纯黑色图片，这表示我们本次生成的图片触发了NSFW机制，出现了一些违规的图片，我们可以修改seed重新进行生成。
我们可以自己设置seed，来达到对图片生成的控制

```python
import torch

#manual_seed(1024)：每次使用具有相同种子的生成器时，都会获得相同的图像输出。
generator = torch.Generator("cuda").manual_seed(1024)

image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
```

将pipeline的完整结构梳理好之后，我们再对一些核心参数进行讲解：

+ num_inference_steps。num_inference_steps表示我们对图片进行噪声优化的次数。一般来说，我们可以选择num_inference_steps = 20/25/50，数值越大，图片生成效果越好，但同时生成所需的时间就越长。
+ guidance_scale，代表无分类指引（Classifier-free guidance，guidance_scale，CFG）是一个控制文本提示对扩散过程的影响程度的值。简单来说就是在加噪阶段将条件控制下预测的噪音和无条件下的预测噪音组合在一起来确定最终的噪声。通常guidance_scale可以选7-8.5之间，如果使用非常大的值，图像可能看起来不错，但多样性会降低。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p11.png" /> 
</div>

其中$w$代表CFG，当$w$越大时，condition起的作用越大，即生成的图像更和输入文本一致，当$w$被设置为0时，图像生成是无条件的，文本提示会被忽略。
+ 输出尺寸:SD在默认情况下会输出$512\times 512$尺寸的图片。我们也可以自定义设置图片尺寸，个人建议如下：
    - 建议height和width都是8的倍数。
    - 低于512可能会导致图像质量较低。
    - 创建非正方形图像的推荐方法是在一维中使用512，另一个使用更大的值。

```python
prompt = "a photograph of an astronaut riding a horse"

# Number of denoising steps
steps = 25         

# Scale for classifier-free guidance
CFG = 7.5

image = pipe(prompt, guidance_scale=CFG, height=512, width=768, num_inference_steps=steps).images[0]
```

除了将预训练模型整体加载，我们还可以将SD模型的不同组件单独加载：

```python
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler

# 单独加载VAE模型 
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

# 单独家在CLIP模型和tokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 单独加载U-Net模型
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

# 单独加载调度算法
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
```

**4.Stable Diffusion经典应用场景**

**文本生成图像**

+ 输入：prompt, 输出:图像

**图片生成图片**

+ 输入：图像+prompt,输出：图像

与文字生成图片的过程相比，图片生成图片的预处理阶段，先把噪声添加到隐空间特征中。我们设置一个去噪强度（Denoising strength）控制加入多少噪音。如果它是0，就不添加噪音。如果它是1，则添加最大数量的噪声，使潜像成为一个完整的随机张量，如果将去噪强度设置为1，就完全相当于文本转图像，因为初始潜像完全是随机的噪声。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p13.png" /> 
</div><p align=center>去噪强度（Denoising strength）控制噪音的加入量</p>

**图片inpainting**

+ 输入：图像 + mask + prompt, 输出：图像

下面就是如何进行图像inpainting的直观过程：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p14.png" /> 
</div>

**使用controlnet辅助生成图片**

+ 输入：素描图 + prompt,输出：图像

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p15.png" /> 
</div><p align=center>使用ControlNet辅助生成图片</p>

**超分辨率重建**

+ 输入：prompt/（图像 + prompt）,输入：图像


**5.Stable Diffusion训练过程**

Stable Diffusion的整个训练过程在最高维度上可以看成是如何加噪声和如何去噪声的过程，并在针对噪声的“对抗与攻防”中学习到生成图片的能力。

具体地，在训练过程中，我们首先对干净样本进行加噪处理，采用多次逐步增加噪声的方式，直至干净样本转变成为纯噪声。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p16.png" /> 
</div><p align=center>SD训练时的加噪过程</p>

接着，让SD模型学习去噪过程，最后抽象出一个高维函数，这个函数能在纯噪声中“优化”噪声，得到一个干净样本。
其中，将去噪过程具像化，就得到使用U-Net预测噪声，并结合调度算法逐步去噪的过程。
<div align=center>
    <img src="zh-cn/img/ch5/3-1/p17.png" /> 
</div><p align=center>SD训练时的去噪过程</p>

我们可以看到，加噪和去噪过程都是逐步进行的，我们假设进行K步，那么每一步，SD都要去预测噪声，从而形成“小步快跑的稳定去噪”，类似于移动互联网时代的产品逻辑，这是足够伟大的关键一招。
与此同时，在加噪过程中，每次增加的噪声量级可以不同，假设有5种噪声量级，那么每次都可以取一种量级的噪声，增加噪声的多样性。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p18.png" /> 
</div><p align=center>多量级噪声</p>

那么怎么让网络知道目前处于K的哪一步呢？这时就需要Positional embeddings了，通过位置编码，将步数也传入网络中，从而让网络知道现在处于哪一步，和Transformer中的操作类似：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p19.png" /> 
</div><p align=center>使用Positional embeddings对训练迭代的步数进行编码</p>


**6.Stable Diffusion性能优化**

**使用TF32精度**

```python
import torch

torch.backends.cuda.matmul.allow_tf32 = True
```
TF32在性能和精度上实现了平衡。下面是TF32精度的一些作用和优势：
+ 加速训练速度：使用TF32精度可以在保持相对较高的模型精度的同时，加快模型训练的速度。
+ 减少内存需求：TF32精度相对于传统的浮点数计算（如FP32）需要更少的内存存储。这对于训练大规模的深度学习模型尤为重要，可以减少内存的占用。
+ 可接受的模型精度损失：使用TF32精度会导致一定程度的模型精度损失，因为低精度计算可能无法精确表示一些小的数值变化。然而，对于大多数深度学习应用，TF32精度仍然可以提供足够的模型精度。

**使用FP16半精度**

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
```

使用FP16半精度训练的优势：
减少了一半的内存占用，我们可以进一步将batch大小翻倍，并将训练时间减半。一些GPU如V100, 2080Ti等针对16位计算进行了优化，能自动加速3-8倍。

**对注意力模块进行切片**

当模型中的注意力模块存在多个注意力头时，可以使用切片注意力操作，使得每个注意力头依次计算注意力矩阵，从而大幅减少内存占用，但随之而来的是推理时间增加约10%。

```python
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

# 切片注意力
pipe.enable_attention_slicing()
```

**对VAE进行切片**

和注意力模块切片一样，我们也可以对VAE进行切片，让VAE每次处理Batch（32）中的一张图片，从而大幅减少内存占用。

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
#切片VAE
pipe.enable_vae_slicing()
images = pipe([prompt] * 32).images
```

**大图像切块**

当想要生成4k或者更大的图像，并且内存不充裕时，可以使用图像切块的操作，让VAE的编码器与解码器对切块后的图像逐一处理，最后从容拼接生成大图。

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
prompt = "a beautiful landscape photograph"
# 大图像切块
pipe.enable_vae_tiling()

image = pipe([prompt], width=3840, height=2224, num_inference_steps=20).images[0]
```

**CPU <-> GPU切换**

可以将整个SD模型或者SD模型的部分模块权重加载到CPU中，只有等推理时再将需要的权重加载到GPU。

```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)

#子模块CPU <-> GPU切换
pipe.enable_sequential_cpu_offload()

#整个SD模型CPU <-> GPU切换
pipe.enable_model_cpu_offload()
```

**变换Memory Format**

在CV领域，两种比较常见的memory format是channels first(NCHW)和channels last(NHWC)。将channels first转变成为channels last可能会提升推理速度，不过这也需要依AI框架和硬件而定。

在Channels Last内存格式中，张量的维度顺序为：(batch_size, height, width, channels)。其中，batch_size表示批处理大小，height和width表示图像或特征图的高度和宽度，channels表示通道数。

相比而言，Channels First是另一种内存布局，其中通道维度被放置在张量的第二个维度上。在Channels First内存格式中，张量的维度顺序为：(batch_size, channels, height, width)。

选择Channels Last或Channels First内存格式通常取决于硬件和软件平台以及所使用的深度学习框架。不同的平台和框架可能对内存格式有不同的偏好和支持程度。

在一些情况下，Channels Last内存格式可能具有以下优势：
+ 内存访问效率：在一些硬件架构中，如CPU和GPU，Channels Last内存格式能够更好地利用内存的连续性，从而提高数据访问的效率。
+ 硬件加速器支持：一些硬件加速器（如NVIDIA的Tensor Cores）对于Channels Last内存格式具有特定的优化支持，可以提高计算性能。
+ 跨平台兼容性：某些深度学习框架和工具更倾向于支持Channels Last内存格式，使得在不同的平台和框架之间迁移模型更加容易。

需要注意的是，选择内存格式需要根据具体的硬件、软件和深度学习框架来进行评估。某些特定的操作、模型结构或框架要求可能会对内存格式有特定的要求或限制。因此，建议在特定环境和需求下进行测试和选择，以获得最佳的性能和兼容性。

```python
print(pipe.unet.conv_out.state_dict()["weight"].stride())  
# 变换Memory Format
pipe.unet.to(memory_format=torch.channels_last)  
print(pipe.unet.conv_out.state_dict()["weight"].stride())
```

**使用xFormers**

使用xFormers插件能够让注意力模块优化运算，提升20%左右的运算速度。

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
).to("cuda")

# 使用xFormers
pipe.enable_xformers_memory_efficient_attention()
```

**7.Stable Diffusion的常用采样方法(调度算法)**

<!-- PNDM，DDIM，K-LMS -->
<!-- https://zhuanlan.zhihu.com/p/612572004 -->

<!-- https://post.smzdm.com/p/aev65x7z/
https://zhuanlan.zhihu.com/p/612572004
https://zhuanlan.zhihu.com/p/621083328

https://www.bilibili.com/video/BV1iW4y1D7RW/?spm_id_from=333.337.search-card.all.click&vd_source=def8c63d9c5f9bf987870bf827bfcb3d

百度：链接：https://pan.baidu.com/s/1raib0i7D2JrpFC_wJoD8yg?pwd=Et51 
提取码：Et51 -->

!> 什么是Stable diffusion

Stable diffusion是一中潜在扩散模型（latent diffusion model），这里面除了模型这两个字，潜在和扩散听上去就不太像人话了，这里我们先弄明白什么是扩散模型，它是在训练图像上逐渐添加噪声，最后变成完全随机噪声图。这个过程就像是一滴墨水滴在一杯清水里，会慢慢扩散最终均匀分布在清水里一样，扩散这个名字就是那么来的，参考[下图](https://stable-diffusion-art.com/how-stable-diffusion-work/)：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p16.png" /> 
</div>

这是一个前向扩散的过程，因为让一张图片越来越模糊没什么技术含量。

而扩散模型就是通过训练让上述过程获得逆向从随机噪声图生成清晰图像的过程。实现下图的过程，而训练的重点就是下图中的噪声预测器（noise predictor），它可以通过训练得出每次需要减掉的噪声，每次需要减多少噪声是预测出来的，这就是一个叫U-net的模型（先请记住它，下面还会反复提它），从而实现还原清晰图片的目的。

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p17.png" /> 
</div>

但是，这个过程无论是模型训练，还是推理（你生成图片的过程）都是非常贵的，需要海量的算力支持和内存需求，而stable diffusion就是让它不那么贵的解决方案，核心思想也很简单：压缩图像，它通过一个叫变分自编码器（VAE）的模型，把图像压缩到它亲妈都不认识到的程度，甚至把此类压缩方式称作降维，这种降维级别的压缩且不丢失重要信息与一套理论相关，即流形假说（Manifold hypothesis），认为高维空间的数据集可以对应到低维的潜在流形（latent manifolds ）上，训练需要找到就是一些共同的高维特征。所以经过此压缩后，这个时候图像被称作低维潜在（latent）的"图像"，作为U-net的输入，去了潜空间（latent space），请参考下图标注latent space的绿色区域，也就是U-net现在工作的地方。在低维的潜空间里一步一步降噪后，完成反向扩散的低维“图片”还得通过VAE的解码器，把图像从潜空间转换回像素空间（pixel space）。

<div align=center>
    <img src="zh-cn/img/ch5/2-1/p9.png" /> 
</div>

但是这张图片中好像除了刚刚提到潜空间和像素空间，其余的内容是什么鬼我们目前还不知道，目前的重点是压缩，因为这是让反向扩散推理以生成图片的基石，它让Stable Diffusion在消费级GPU上运行成为了可能。下面聊Stable diffusion的具体工作流程。图片来源[1]

!> 三大核心组成部分

+ VAE：包括Encoder编码器和Decoder解码器，用于图像从像素空间到潜空间的转换，或者叫降维或升维，由于用于降维的VAE Encoder 只在训练模型的阶段使用，推理过程（图像生成）只需要VAE Decoder解码器就ok了，而网上常见的VAE文件，就是对这个VAE Decoder解码器的微调改进版本，用于解决角色面部眼睛等细节方面的问题。
+ U-net：输出预测的噪声残差，用于每次迭代过程的降噪，提供了交叉注意力层，并且通过交叉注意力机制来消耗CLIP提供的条件文本嵌入U-net，输出预测的噪声残差，实现有条件（被文本指挥）的迭代降噪。
+ CLIP：将Prompt文本转化成能够让U-net使用的嵌入（embedding），达到实现文本作为条件生成图片的过程，这里反复提到嵌入这个概念是因为，可以通过微调的方式，发出带有关键词的embedding，用于调整图片的样式。你能下载到的textual inversion就是通过这种方式实现调整生成图像效果的，优点是文件的体积非常小。

!> 整体工作流程

见下图[2]：

<div align=center>
    <img src="zh-cn/img/ch5/3-1/p6.png" /> 
</div>

SD在潜空间生成随机的张量，你也可以设置种子（seed）来控制这个初始值，特定的初始值可以起到固定的作用。然后这个在潜在种子生成64 x 64的潜空间图像，Prompt通过CLIP转化77 x 768条件文本嵌入（embedding）；

U-net以这个嵌入为条件，对潜空间图像迭代降噪。U-net输出噪声的残差，通过调度器（scheduler）进行降噪计算并返回本轮的去噪样本，（而这个调度器也叫采样器或者求解器，对应了不同的算法，有各种优缺点）；

在迭代50次后（取决于你选择的采样方法，有些20次迭代就可以达到高质量的结果），最后VAE的解码器把最终的潜空间图像转化输入到像素空间，即我们能看到的像素图片，整个工作流到此结束，你得到了结果图片。

这就stable diffusion的文生图大致工作原理。

!> 采样方法（SD-Webui）

Stable diffusion webui是Stable diffusion的GUI是将stable diffusion实现可视化的图像用户操作界面，它本身还集成了很多其它有用的扩展脚本。

webui中集成了很多不同的采样方法，（上述提到的调度算法）这块也是目前AI艺术家们乐忠对比的环节，这里结合设置中提供的选项，简单粗略的介绍下它们的各自区别。

这些大部分的采样器是由Katherine Crowson根据论文实现的功能，stable diffusion官方的blog也提到过她，她在github中有一个名为[K-diffusion](https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py)的项目[3]，理论基础主要基于Jiaming Song等人的论文[4]、Karras等人的论文[5]以及前不久基于Cheng Lu等人的论文[6]

按照SD webui的顺序

+ **Euler**

基于Karras论文，在K-diffusion实现，20-30steps就能生成效果不错的图片，采样器设置页面中的 sigma noise，sigma tmin和sigma churn这三个属性会影响到它（后面会提这三个参数的作用）；

+ **Euler a**

使用了祖先采样（Ancestral sampling）的Euler方法，受采样器设置中的eta参数影响（后面详细介绍eta）；

+ **LMS**

线性多步调度器（Linear multistep scheduler）源于K-diffusion的项目实现；

+ **heun**

基于Karras论文，在K-diffusion实现，受采样器设置页面中的 sigma参数影响；

+ **DPM2**

这个是Katherine Crowson在K-diffusion项目中自创的，灵感来源Karras论文中的DPM-Solver-2和算法2，受采样器设置页面中的 sigma参数影响；

+ **DPM2 a**

使用了祖先采样（Ancestral sampling）的DPM2方法，受采样器设置中的ETA参数影响；

+ **DPM++ 2S a**

基于Cheng Lu等人的论文（改进后，后面又发表了一篇），在K-diffusion实现的2阶单步并使用了祖先采样（Ancestral sampling）的方法，受采样器设置中的eta参数影响；Cheng Lu的github中也提供已经实现的代码，并且可以自定义，1、2、3阶，和单步多步的选择，webui使用的是K-diffusion中已经固定好的版本。对细节感兴趣的小伙伴可以参考Cheng Lu的github和原论文。

+ **DPM++ 2M**

基于Cheng Lu等人的论文（改进后的版本），在K-diffusion实现的2阶多步采样方法，在Hagging face中Diffusers中被称作已知最强调度器，在速度和质量的平衡最好。这个代表M的多步比上面的S单步在采样时会参考更多步，而非当前步，所以能提供更好的质量。但也更复杂。

+ **DPM++ SDE**

基于Cheng Lu等人的论文的，DPM++的SDE版本，即随机微分方程（stochastic differential equations），而DPM++原本是ODE的求解器即常微分方程（ordinary differential equations），在K-diffusion实现的版本，代码中调用了祖先采样（Ancestral sampling）方法，所以受采样器设置中的ETA参数影响；

+ **DPM fast**

基于Cheng Lu等人的论文，在K-diffusion实现的固定步长采样方法，用于steps小于20的情况，受采样器设置中的ETA参数影响；

+ **DPM adaptive**

基于Cheng Lu等人的论文，在K-diffusion实现的自适应步长采样方法，DPM-Solver-12 和 23，受采样器设置中的ETA参数影响；

+ **Karras后缀**

LMS Karras 基于Karras论文，运用了相关Karras的noise schedule的方法，可以算作是LMS使用Karras noise schedule的版本；

DPM2 Karras，DPM2 a Karras，DPM++ 2S a Karras，DPM++ 2M Karras，DPM++ SDE Karras这些含有Karras名字的采样方法和上面LMS Karras意思相同，都是相当于使用Karras noise schedule的版本；

+ **DDIM**

“官方采样器”随latent diffusion的最初repository一起出现， 基于Jiaming Song等人的论文，也是目前最容易被当作对比对象的采样方法，它在采样器设置界面有自己的ETA；

+ **PLMS**

同样是元老，随latent diffusion的最初repository一起出现；

+ **UniPC**

最新被添加到webui中的采样器，基于Wenliang Zhao等人的论文[7]，应该是目前最快最新的采样方法，10步就可以生成高质量结果；在采样器设置界面可以自定义的参数目前也比较多

+ **UniPC variant**

bh1和bh2和vary_coeff是三种变体

hugging face的团队在diffuser中给出了他们的建议：bh1适合在无条件（没指挥，无引导）且步数小于10情况下使用，其余情况全部使用bh2。

至于vary_coeff这个，作者在论文中实验对比了在“无条件”的和bh1和bh2的区别，即bh1在5，6步表现最好，vary_coeff在7，8或9表现最好，10步以上还是bh2。

由于我们在webui的使用场景使用提示词就是“有条件”了，所以看上去bh2更合适，除非你热衷于10步以内生成图片，但webui的github上有人反应vary_coeff生成的图片背景细节更加丰富[8]。我个人使用下来看，区别不算大，可以自己对比。

一句话总结：懒得对比的话就默认bh2。

+ **UniPC skip type**

如果你生成的图片是512 x 512或者更大的话，选uniform。它更适合高分辨率图（512目前相对算高分别率）logSNR适合低分别率

logSNR在512 x 512下会出现一些奇怪的细节（模型SD1.5），quadratic稍微好一些,结论：512 x 512细节更合理， 推荐uniform

+ 采样方法小结

1.建议根据自己使用的checkpoint使用脚本跑网格图（用自己关心的参数）然后选择自己想要的结果。

2.懒得对比：请使用DPM++ 2M或DPM++ 2M Karras或UniPC，想要点惊喜和变化，Euler a、DPM++ SDE、DPM++ SDE Karras、DPM2 a Karras（注意调正对应eta值）

3.eta和sigma都是多样性相关的，但是它们的多样性来自步数的变化，追求更大多样性的话应该关注seed的变化，这两项参数应该是在图片框架被选定后，再在此基础上做微调时使用的参数。

!> 参考文献

1. High-Resolution Image Synthesis with Latent Diffusion Models https://arxiv.org/abs/2112.10752
2. how-stable-diffusion-work https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work
3. k_diffusion/sampling https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
4. Denoising Diffusion Implicit Models https://arxiv.org/abs/2010.02502
5. Elucidating the Design Space of Diffusion-Based Generative Models https://arxiv.org/abs/2206.00364
6. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps https://arxiv.org/abs/2206.00927
7. UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models https://arxiv.org/abs/2302.04867
8. UniPC Sampler support https://github.com/easydiffusion/sdkit/pull/12
9. CodeFormer https://github.com/sczhou/CodeFormer
10. GFPGAN https://github.com/TencentARC/GFPGAN
11. Face restoration in webui https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#face-restoration
12. ai-upscaler https://stable-diffusion-art.com/ai-upscaler/
13. ESRGAN https://github.com/xinntao/Real-ESRGAN
14. SwinIR https://github.com/JingyunLiang/SwinIR

<div align=center>
    <img src="zh-cn/img/ch5/3-1/Snipaste_2023-06-27_19-00-13.jpg" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch5/3-1/Snipaste_2023-06-27_19-00-32.jpg" /> 
</div>

**8.Classifier-free guidance（CFG）和Classifier-Guidance**

<!-- CFG的论文Classifier-Free Diffusion Guidance -->

<!-- https://zhuanlan.zhihu.com/p/623837604

https://zhuanlan.zhihu.com/p/582880086
https://zhuanlan.zhihu.com/p/607225186
https://zhuanlan.zhihu.com/p/640631667 -->

<!-- https://www.bilibili.com/video/BV1ch4y1W7JB/?spm_id_from=333.337.search-card.all.click&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

<!-- https://www.bilibili.com/video/BV1m84y1e7hP/?spm_id_from=333.337.search-card.all.click&vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

!> https://arxiv.org/pdf/2105.05233.pdf

!> https://arxiv.org/pdf/2207.12598.pdf

<!-- https://zhuanlan.zhihu.com/p/623837604 -->

<object data="zh-cn/img/ch5/3-1/Conditional Control（Classifier-Guidance and Classifier-Free） .pdf" type="application/pdf" width="100%" height="650px">
<!--     <embed src="http://www.africau.edu/images/default/sample.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://www.africau.edu/images/default/sample.pdf">Download PDF</a>.</p>
    </embed> -->
</object>

------

#### 2.3 硬核解读Stable Diffusion（完整版）

<!-- https://mp.weixin.qq.com/s/m9ja0pVpOY8JnehLrzw5_g -->

2022年可谓是AIGC（AI Generated Content）元年，上半年有文生图大模型DALL-E2和Stable Diffusion，下半年有OpenAI的文本对话大模型ChatGPT问世，这让冷却的AI又沸腾起来了，因为AIGC能让更多的人真真切切感受到AI的力量。这篇文章将介绍比较火的文生图模型Stable Diffusion（简称SD），Stable Diffusion不仅是一个完全开源的模型（代码，数据，模型全部开源），而且是它的参数量只有1B左右，大部分人可以在普通的显卡上进行推理甚至精调模型。毫不夸张的说，Stable Diffusion的出现和开源对AIGC的火热和发展是有巨大推动作用的，因为它让更多的人能快地上手AI作画。这里将基于Hugging Face的diffusers库深入讲解SD的技术原理以及部分的实现细节，然后也会介绍SD的常用功能，注意本文主要以SD V1.5版本为例，在最后也会简单介绍 SD 2.0版本以及基于SD的扩展应用。

**SD模型原理**

SD是CompVis、Stability.AI和LAION等公司研发的一个文生图模型，它的模型和代码是开源的，而且训练数据LAION-5B也是开源的。SD在开源90天github仓库就收获了33K的stars，可见这个模型是多受欢迎。

!> github: https://github.com/CompVis/stable-diffusion

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p1.png" /> 
</div>

SD是一个基于latent的扩散模型，它在UNet中引入text condition来实现基于文本生成图像。SD的核心来源于Latent Diffusion这个工作，常规的扩散模型是基于pixel的生成模型，而Latent Diffusion是基于latent的生成模型，它先采用一个autoencoder将图像压缩到latent空间，然后用扩散模型来生成图像的latents，最后送入autoencoder的decoder模块就可以得到生成的图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p2.png" /> 
</div>

基于latent的扩散模型的优势在于计算效率更高效，因为图像的latent空间要比图像pixel空间要小，这也是SD的核心优势。文生图模型往往参数量比较大，基于pixel的方法往往限于算力只生成64x64大小的图像，比如OpenAI的DALL-E2和谷歌的Imagen，然后再通过超分辨模型将图像分辨率提升至256x256和1024x1024；而基于latent的SD是在latent空间操作的，它可以直接生成256x256和512x512甚至更高分辨率的图像。

SD模型的主体结构如下图所示，主要包括三个模型：

+ **autoencoder(AE)**：encoder将图像压缩到latent空间，而decoder将latent解码为图像；
+ **CLIP text encoder**：提取输入text的text embeddings，通过cross attention方式送入扩散模型的UNet中作为condition；
+ **UNet**：扩散模型的主体，用来实现文本引导下的latent生成。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p3.png" /> 
</div>

对于SD模型，其autoencoder模型参数大小为84M，CLIP text encoder模型大小为123M，而UNet参数大小为860M，所以SD模型的总参数量约为1B。

!> autoencoder

autoencoder是一个基于encoder-decoder架构的图像压缩模型，对于一个大小为$H\times H\times 3$的输入图像，encoder模块将其编码为一个大小为$h\times w \times 3$的latent，其中$f=H/h=W/w$为下采样率（downsampling factor）。在训练autoencoder过程中，除了采用**L1重建损失**外，还增加了**感知损失（perceptual loss，即LPIPS，具体见论文The Unreasonable Effectiveness of Deep Features as a Perceptual Metric）**以及基于**patch的对抗训练**。辅助loss主要是为了确保重建的图像局部真实性以及避免模糊，具体损失函数见latent diffusion的loss部分。同时为了防止得到的latent的标准差过大，采用了两种正则化方法：第一种是**KL-reg**，类似VAE增加一个latent和标准正态分布的KL loss，不过这里为了保证重建效果，采用比较小的权重（～10e-6）；第二种是**VQ-reg**，引入一个VQ （vector quantization）layer，此时的模型可以看成是一个VQ-GAN，不过VQ层是在decoder模块中，这里VQ的codebook采样较高的维度（8192）来降低正则化对重建效果的影响。 latent diffusion论文中实验了不同参数下的autoencoder模型，如下表所示，可以看到当$f$较小和$c$较大时，重建效果越好（PSNR越大），这也比较符合预期，毕竟此时压缩率小。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p4.png" /> 
</div>

论文进一步将不同$f$的autoencoder在扩散模型上进行实验，在ImageNet数据集上训练同样的步数（2M steps），其训练过程的生成质量如下所示，可以看到过小$f$的（比如1和2）下收敛速度慢，此时图像的感知压缩率较小，扩散模型需要较长的学习；而过大的$f$其生成质量较差，此时压缩损失过大。


<div align=center>
    <img src="zh-cn/img/ch5/4-1/p5.png" /> 
</div>

当$f$在4～16时，可以取得相对好的效果。SD采用基于KL-reg的autoencoder，其中下采样率$f=8$，特征维度为$c=4$，当输入图像为`512x512`大小时将得到`64x64x4`大小的latent。 autoencoder模型时在OpenImages数据集上基于`256x256`大小训练的，但是由于autoencoder的模型是全卷积结构的（基于ResnetBlock），所以它可以扩展应用在尺寸>256的图像上。下面我们给出使用diffusers库来加载autoencoder模型，并使用autoencoder来实现图像的压缩和重建，代码如下所示：

```python
import torch
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image

#加载模型: autoencoder可以通过SD权重指定subfolder来单独加载
autoencoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
autoencoder.to("cuda", dtype=torch.float16)

# 读取图像并预处理
raw_image = Image.open("boy.png").convert("RGB").resize((256, 256))
image = np.array(raw_image).astype(np.float32) / 127.5 - 1.0
image = image[None].transpose(0, 3, 1, 2)
image = torch.from_numpy(image)

# 压缩图像为latent并重建
with torch.inference_mode():
    latent = autoencoder.encode(image.to("cuda", dtype=torch.float16)).latent_dist.sample()
    rec_image = autoencoder.decode(latent).sample
    rec_image = (rec_image / 2 + 0.5).clamp(0, 1)
    rec_image = rec_image.cpu().permute(0, 2, 3, 1).numpy()
    rec_image = (rec_image * 255).round().astype("uint8")
    rec_image = Image.fromarray(rec_image[0])
rec_image
```

这里我们给出了两张图片在256x256和512x512下的重建效果对比，如下所示，第一列为原始图片，第二列为512x512尺寸下的重建图，第三列为256x256尺寸下的重建图。对比可以看出，autoencoder将图片压缩到latent后再重建其实是有损的，比如会出现文字和人脸的畸变，在256x256分辨率下是比较明显的，512x512下效果会好很多。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p7.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/ch5/4-1/p6.png" /> 
</div>

这种有损压缩肯定是对SD的生成图像质量是有一定影响的，不过好在SD模型基本上是在512x512以上分辨率下使用的。为了改善这种畸变，stability.ai在发布SD 2.0时同时发布了两个在LAION子数据集上精调的autoencoder，注意这里只精调autoencoder的decoder部分，SD的UNet在训练过程只需要encoder部分，所以这样精调后的autoencoder可以直接用在先前训练好的UNet上（这种技巧还是比较通用的，比如谷歌的Parti也是在训练好后自回归生成模型后，扩大并精调ViT-VQGAN的decoder模块来提升生成质量）。我们也可以直接在diffusers中使用这些autoencoder，比如mse版本（采用mse损失来finetune的模型）：

```
autoencoder = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse/")
```

对于同样的两张图，这个mse版本的重建效果如下所示，可以看到相比原始版本的autoencoder，畸变是有一定改善的。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p8.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/ch5/4-1/p9.png" /> 
</div>

由于SD采用的autoencoder是基于KL-reg的，所以这个autoencoder在编码图像时其实得到的是一个高斯分布DiagonalGaussianDistribution（分布的均值和标准差），然后通过调用sample方法来采样一个具体的latent（调用mode方法可以得到均值）。由于KL-reg的权重系数非常小，实际得到latent的标准差还是比较大的，latent diffusion论文中提出了一种rescaling方法：首先计算出第一个batch数据中的latent的标准差$\hat{\sigma}$，然后采用$\frac{1}{\hat{\sigma}}$的系数来rescale latent，这样就尽量保证latent的标准差接近1（防止扩散过程的SNR较高，影响生成效果，具体见latent diffusion论文的D1部分讨论），然后扩散模型也是应用在rescaling的latent上，在解码时只需要将生成的latent除以$\frac{1}{\hat{\sigma}}$，然后再送入autoencoder的decoder即可。对于SD所使用的autoencoder，这个rescaling系数为0.18215。

!> CLIP text encoder

SD采用CLIP text encoder来对输入text提取text embeddings，具体的是采用目前OpenAI所开源的最大CLIP模型：clip-vit-large-patch14，这个CLIP的text encoder是一个transformer模型（只有encoder模块）：层数为12，特征维度为768，模型参数大小是123M。对于输入text，送入CLIP text encoder后得到最后的hidden states（即最后一个transformer block得到的特征），其特征维度大小为`77x768`（77是token的数量），这个细粒度的text embeddings将以cross attention的方式送入UNet中。在transofmers库中，可以如下使用CLIP text encoder：

```python
from transformers import CLIPTextModel, CLIPTokenizer

text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder").to("cuda")
# text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# 对输入的text进行tokenize，得到对应的token ids
prompt = "a photograph of an astronaut riding a horse"
text_input_ids = text_tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).input_ids

# 将token ids送入text model得到77x768的特征
text_embeddings = text_encoder(text_input_ids.to("cuda"))[0]

```

值得注意的是，这里的tokenizer最大长度为77（CLIP训练时所采用的设置），当输入text的tokens数量超过77后，将进行截断，如果不足则进行paddings，这样将保证无论输入任何长度的文本（甚至是空文本）都得到77x768大小的特征。 在训练SD的过程中，CLIP text encoder模型是冻结的。在早期的工作中，比如OpenAI的GLIDE和latent diffusion中的LDM均采用一个随机初始化的tranformer模型来提取text的特征，但是最新的工作都是采用预训练好的text model。比如谷歌的Imagen采用纯文本模型T5 encoder来提出文本特征，而SD则采用CLIP text encoder，预训练好的模型往往已经在大规模数据集上进行了训练，它们要比直接采用一个从零训练好的模型要好。

!> UNet

SD的扩散模型是一个860M的UNet，其主要结构如下图所示（这里以输入的latent为64x64x4维度为例），其中encoder部分包括3个CrossAttnDownBlock2D模块和1个DownBlock2D模块，而decoder部分包括1个UpBlock2D模块和3个CrossAttnUpBlock2D模块，中间还有一个UNetMidBlock2DCrossAttn模块。encoder和decoder两个部分是完全对应的，中间存在skip connection。注意3个CrossAttnDownBlock2D模块最后均有一个2x的downsample操作，而DownBlock2D模块是不包含下采样的。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p10.png" /> 
</div>

其中CrossAttnDownBlock2D模块的主要结构如下图所示，text condition将通过CrossAttention模块嵌入进来，此时Attention的query是UNet的中间特征，而key和value则是text embeddings。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p11.png" /> 
</div>

SD和DDPM一样采用预测noise的方法来训练UNet，其训练损失也和DDPM一样：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p12.png" /> 
</div>

这里的$c$为text embeddings，此时的模型是一个条件扩散模型。基于diffusers库，我们可以很快实现SD的训练，其核心代码如下所示（这里参考diffusers库下examples中的finetune代码）：

```python
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

# 加载autoencoder
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
# 加载text encoder
text_encoder = CLIPTextModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="tokenizer")
# 初始化UNet
unet = UNet2DConditionModel(**model_config) # model_config为模型参数配置
# 定义scheduler
noise_scheduler = DDPMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)

# 冻结vae和text_encoder
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)

for step, batch in enumerate(train_dataloader):
    with torch.no_grad():
        # 将image转到latent空间
        latents = vae.encode(batch["image"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor # rescaling latents
        # 提取text embeddings
        text_input_ids = text_tokenizer(
            batch["text"],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
  ).input_ids
  text_embeddings = text_encoder(text_input_ids)[0]
    
    # 随机采样噪音
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # 随机采样timestep
    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # 将noise添加到latent上，即扩散过程
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # 预测noise并计算loss
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

 opt.step()
    opt.zero_grad()
```

注意的是SD的noise scheduler虽然也是采用一个1000步长的scheduler，但是不是linear的，而是scaled linear，具体的计算如下所示：

```python
betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
```

在训练条件扩散模型时，往往会采用Classifier-Free Guidance（这里简称为CFG）(我们在上一节给了详细的介绍），所谓的CFG简单来说就是在训练条件扩散模型的同时也训练一个无条件的扩散模型，同时在采样阶段将条件控制下预测的噪音和无条件下的预测噪音组合在一起来确定最终的噪音，具体的计算公式如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p13.png" /> 
</div>

这里的$w$为guidance scale，当$w$越大时，condition起的作用越大，即生成的图像其更和输入文本一致。CFG的具体实现非常简单，在训练过程中，我们只需要以一定的概率（比如10%）随机drop掉text即可，这里我们可以将text置为空字符串（前面说过此时依然能够提取text embeddings）。这里并没有介绍CLF背后的技术原理，感兴趣的可以阅读CFG的论文Classifier-Free Diffusion Guidance以及guided diffusion的论文Diffusion Models Beat GANs on Image Synthesis。CFG对于提升条件扩散模型的图像生成效果是至关重要的。


**训练细节**

前面我们介绍了SD的模型结构，这里我们也简单介绍一下SD的训练细节，主要包括训练数据和训练资源，这方面也是在SD的Model Card上有说明。 首先是训练数据，SD在laion2B-en数据集上训练的，它是laion-5b数据集的一个子集，更具体的说它是laion-5b中的英文（文本为英文）数据集。laion-5b数据集是从网页数据Common Crawl中筛选出来的图像-文本对数据集，它包含5.85B的图像-文本对，其中文本为英文的数据量为2.32B，这就是laion2B-en数据集。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p14.png" /> 
</div>

下面是laion2B-en数据集的元信息（图片width和height，以及文本长度）统计分析：其中图片的width和height均在256以上的样本量为1324M，在512以上的样本量为488M，而在1024以上的样本为76M；文本的平均长度为67。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p15.png" /> 
</div>


laion数据集中除了图片（下载URL，图像width和height）和文本（描述文本）的元信息外，还包含以下信息：
+ msimilarity：使用CLIP ViT-B/32计算出来的图像和文本余弦相似度；
+ pwatermark：使用一个图片水印检测器检测的概率值，表示图片含有水印的概率；
+ punsafe：图片是否安全，或者图片是不是NSFW，使用基于CLIP的检测器来估计；
+ AESTHETIC_SCORE：图片的美学评分（1-10），这个是后来追加的，首先选择一小部分图片数据集让人对图片的美学打分，然后基于这个标注数据集来训练一个打分模型，并对所有样本计算估计的美学评分。

上面是laion数据集的情况，下面我们来介绍SD训练数据集的具体情况，SD的训练是多阶段的（先在256x256尺寸上预训练，然后在512x512尺寸上精调），不同的阶段产生了不同的版本：

+ SD v1.1：在laion2B-en数据集上以256x256大小训练237,000步，上面我们已经说了，laion2B-en数据集中256以上的样本量共1324M；然后在laion5B的高分辨率数据集以512x512尺寸训练194,000步，这里的高分辨率数据集是图像尺寸在1024x1024以上，共170M样本。
+ SD v1.2：以SD v1.1为初始权重，在improved_aesthetics_5plus数据集上以512x512尺寸训练515,000步数，这个improved_aesthetics_5plus数据集上laion2B-en数据集中美学评分在5分以上的子集（共约600M样本），注意这里过滤了含有水印的图片（pwatermark>0.5)以及图片尺寸在512x512以下的样本。
+ SD v1.3：以SD v1.2为初始权重，在improved_aesthetics_5plus数据集上继续以512x512尺寸训练195,000步数，不过这里采用了CFG（以10%的概率随机drop掉text）。
+ SD v1.4：以SD v1.2为初始权重，在improved_aesthetics_5plus数据集上采用CFG以512x512尺寸训练225,000步数。
+ SD v1.5：以SD v1.2为初始权重，在improved_aesthetics_5plus数据集上采用CFG以512x512尺寸训练595,000步数

其实可以看到SD v1.3、SD v1.4和SD v1.5其实是以SD v1.2为起点在improved_aesthetics_5plus数据集上采用CFG训练过程中的不同checkpoints，目前最常用的版本是SD v1.4和SD v1.5。 SD的训练是采用了32台8卡的A100机器（32 x 8 x A100_40GB GPUs），所需要的训练硬件还是比较多的，但是相比语言大模型还好。单卡的训练batch size为2，并采用gradient accumulation，其中gradient accumulation steps=2，那么训练的总batch size就是32x8x2x2=2048。训练优化器采用AdamW，训练采用warmup，在初始10,000步后学习速率升到0.0001，后面保持不变。至于训练时间，文档上只说了用了150,000小时，这个应该是A100卡时，如果按照256卡A100来算的话，那么大约需要训练25天左右。


**模型评测**

上面介绍了模型训练细节，那么最后的问题就是模型评测了。对于文生图模型，目前常采用的定量指标是FID（Fréchet inception distance）和CLIP score，其中FID可以衡量生成图像的逼真度（image fidelity），而CLIP score评测的是生成的图像与输入文本的一致性，其中FID越低越好，而CLIP score是越大越好。当CFG的gudiance scale参数设置不同时，FID和CLIP score会发生变化，下图为不同的gudiance scale参数下，SD模型在COCO2017验证集上的评测结果，注意这里是zero-shot评测，即SD模型并没有在COCO训练数据集上精调。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p16.png" /> 
</div>

可以看到当gudiance scale=3时，FID最低；而当gudiance scale越大时，CLIP score越大，但是FID同时也变大。在实际应用时，往往会采用较大的gudiance scale，比如SD模型默认采用7.5，此时生成的图像和文本有较好的一致性。从不同版本的对比曲线上看，SD的采用CFG训练后三个版本其实差别并没有那么大，其中SD v1.5相对好一点，但是明显要未采用CFG训练的版本要好的多，这说明CFG训练是比较关键的。 目前在模型对比上，大家往往是比较不同模型在COCO验证集上的zero-shot FID-30K（选择30K的样本），大家往往就选择模型所能得到的最小FID来比较，下面为eDiff和GigaGAN两篇论文所报道的不同文生图模型的FID对比（由于SD并没有给出FID-30K，所以大家应该都是自己用开源SD的模型计算的，由于选择样本不同，可能结果存在差异）：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p17.png" /> 
</div>

可以看到SD虽然FID不是最好的，但是也能达到比较低的FID（大约在8～9之间）。不过虽然学术界常采用FID来定量比较模型，但是FID有很大的局限性，它并不能很好地衡量生成图像的质量，也是因为这个原因，谷歌的Imagen引入了人工评价，先建立一个评测数据集DrawBench（包含200个不同类型的text），然后用不同的模型来生成图像，让人去评价同一个text下不同模型生成的图像，这种评测方式比较直接，但是可能也受一些主观因素的影响。总而言之，目前的评价方式都有一定的局限性，最好还是直接上手使用来比较不同的模型。

**SD的主要应用**

下面来介绍SD的主要应用，这包括**文生图，图生图以及图像inpainting**。其中文生图是SD的基础功能：根据输入文本生成相应的图像，而图生图和图像inpainting是在文生图的基础上延伸出来的两个功能。

!> 文生图

根据文本生成图像这是文生图的最核心的功能，下图为SD的文生图的推理流程图：首先根据输入text用text encoder提取text embeddings，同时初始化一个随机噪音noise（latent上的，512x512图像对应的noise维度为64x64x4），然后将text embeddings和noise送入扩散模型UNet中生成去噪后的latent，最后送入autoencoder的decoder模块得到生成的图像

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p18.png" /> 
</div>

使用diffusers库，我们可以直接调用StableDiffusionPipeline来实现文生图，具体代码如下所示：

```python
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# 组合图像，生成grid
def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

# 加载文生图pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", # 或者使用 SD v1.4: "CompVis/stable-diffusion-v1-4"
    torch_dtype=torch.float16
).to("cuda")

# 输入text，这里text又称为prompt
prompts = [
    "a photograph of an astronaut riding a horse",
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    "An avocado armchair",
    "A white dog wearing sunglasses"
]

generator = torch.Generator("cuda").manual_seed(42) # 定义随机seed，保证可重复性

# 执行推理
images = pipe(
    prompts,
    height=512,
    width=512,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    generator=generator
).images

grid = image_grid(images, rows=1, cols=4)
grid

```

生成的图像效果如下所示:

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p19.png" /> 
</div>

这里可以通过指定width和height来决定生成图像的大小，前面说过SD最后是在512x512尺度上训练的，所以生成512x512尺寸效果是最好的，但是实际上SD可以生成任意尺寸的图片：一方面autoencoder支持任意尺寸的图片的编码和解码，另外一方面扩散模型UNet也是支持任意尺寸的latents生成的（UNet是卷积+attention的混合结构）。然而，生成512x512以外的图片会存在一些问题，比如生成低分辨率图像时，图像的质量大幅度下降，下图为同样的文本在

256x256尺寸下的生成效果：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p20.png" /> 
</div>

如果是生成512x512以上分辨率的图像，图像质量虽然没问题，但是可能会出现重复物体以及物体被拉长的情况，下图为分别为768x512和512x768尺寸下的生成效果，可以看到部分图像存在一定的问题

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p21.png" /> 
</div>

所以虽然SD的架构上支持任意尺寸的图像生成，但训练是在固定尺寸上（512x512），生成其它尺寸图像还是会存在一定的问题。解决这个问题的办法就相对比较简单，就是采用多尺度策略训练，比如NovelAI提出采用Aspect Ratio Bucketing策略来在二次元数据集上精调模型，这样得到的模型就很大程度上避免SD的这个问题，目前大部分开源的基于SD的精调模型往往都采用类似的多尺度策略来精调。比如我们采用开源的dreamlike-diffusion-1.0模型（基于SD v1.5精调的），其生成的图像效果在变尺寸上就好很多：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p22.png" /> 
</div>


另外一个参数是num_inference_steps，它是指推理过程中的去噪步数或者采样步数。SD在训练过程采用的是步数为1000的noise scheduler，但是在推理时往往采用速度更快的scheduler：只需要少量的采样步数就能生成不错的图像，比如SD默认采用PNDM scheduler，它只需要采样50步就可以出图。当然我们也可以换用其它类型的scheduler，比如DDIM scheduler和DPM-Solver scheduler。我们可以在diffusers中直接替换scheduler，比如我们想使用DDIM：

```
from diffusers import DDIMScheduler

# 注意这里的clip_sample要关闭，否则生成图像存在问题，因为不能对latent进行clip
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
```
换成DDIM后，同样的采样步数生成的图像如下所示，在部分细节上和PNDM有差异：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p23.png" /> 
</div>

当然采样步数越大，生成的图像质量越好，但是相应的推理时间也更久。这里我们可以试验一下不同采样步数下的生成效果，以宇航员骑马为例，下图展示了采样步数为10，20，30，50，70和100时的生成图像，可以看到采样步数增加后，图像生成质量是有一定的提升的，当采样步数为30时就能生成相对稳定的图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p24.png" /> 
</div>

我们要讨论的第三个参数是guidance_scale，前面说过当CFG的guidance_scale越大时，生成的图像应该会和输入文本更一致，这里我们同样以宇航员骑马为例来测试不同guidance_scale下的图像生成效果。下图为guidance_scale为1，3，5，7，9和11下生成的图像对比，可以看到当guidance_scale较低时生成的图像效果是比较差的，当guidance_scale在7～9时，生成的图像效果是可以的，当采用更大的guidance_scale比如11，图像的色彩过饱和而看起来不自然，所以SD默认采用的guidance_scale为7.5。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p25.png" /> 
</div>

过大的guidance_scale之所以出现问题，主要是由于训练和测试的不一致，过大的guidance_scale会导致生成的样本超出范围。谷歌的Imagen论文提出一种dynamic thresholding策略来解决这个问题，所谓的dynamic thresholding是相对于原来的static thresholding，static thresholding策略是直接将生成的样本clip到[-1, 1]范围内（Imagen是基于pixel的扩散模型，这里是将图像像素值归一化到-1到1之间），但是会在过大的guidance_scale时产生很多的饱含像素点。而dynamic thresholding策略是先计算样本在某个百分位下（比如99%）的像素绝对值，然后如果它超过1时就采用来进行clip，这样就可以大大减少过饱和的像素。两种策略的具体实现代码如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p26.png" /> 
</div>

dynamic thresholding策略对于Imagen是比较关键的，它使得Imagen可以采用较大的guidance_scale来生成更自然的图像。下图为两种thresholding策略下生成图像的对比：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p27.png" /> 
</div>

虽然SD是基于latent的扩散模型，但依然可以采用类似的dynamic thresholding策略，感兴趣的可以参考目前的一个开源实现：sd-dynamic-thresholding，使用dynamic thresholding策略后，SD可以在较大的guidance_scale下生成相对自然的图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p28.png" /> 
</div>

另外一个比较容易忽略的参数是negative_prompt，这个参数和CFG有关，前面说过，SD采用了CFG来提升生成图像的质量。使用CFG，去噪过程的噪音预测不仅仅依赖条件扩散模型，也依赖无条件扩散模型：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p29.png" /> 
</div>

这里的negative_prompt便是无条件扩散模型的text输入，前面说过训练过程中我们将text置为空字符串来实现无条件扩散模型，所以这里：`negative_prompt = None = ""`。但是有时候我们可以使用不为空的negative_prompt来避免模型生成的图像包含不想要的东西，因为从上述公式可以看到这里的无条件扩散模型是我们想远离的部分。下面我们来举几个具体的例子，首先来看生成人物图像的一个例子，这里的输入文本为"a portrait of a beautiful blonde woman"，其生成的图像如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p30.png" /> 
</div>

可以看到生成的图像效果并不好，比如出现一些脸部的畸变，但是我们可以设置negative_prompt来提升生成效果，这里我们将negative_prompt设置为"cropped, lowres, poorly drawn face, out of frame, poorly drawn hands, blurry"，这些描述都是负面的。改变negative_prompt后，生成的图像效果有一个明显的提升：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p31.png" /> 
</div>

第二个例子是一个建筑物，这里的输入文本为"A Hyperrealistic photograph of German architectural modern home"，默认图像生成效果如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p32.png" /> 
</div>

虽然生成的图像效果不错，但是如果只想要一个干净的建筑物，而不想背景中含有树木和草地等，此时我们可以通过设置negative prompt来达到这种效果。这里将negative prompt设为"trees, bushes, leaves, greenery"，其生成的建筑物就干净了很多：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p33.png" /> 
</div>

**可以看到合理使用negative prompt能够帮助我们去除不想要的东西来提升图像生成效果。**一般情况下，输入的text或者prompt我们称之为“正向提示词”，而negative prompt称之为“反向提示词”，想要生成的好的图像，不仅要选择好的正向提示词，也需要好的反向提示词，这和文本生成模型也比较类似：都需要好的prompt。这里也举一个对正向prompt优化的例子（这个例子来源于微软的工作Optimizing Prompts for Text-to-Image Generation），这里的原始prompt为"A rabbit is wearing a space suit"，可以看到直接生成的效果其实是不尽人意的：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p34.png" /> 
</div>

但是如果我们将prompt改为"A rabbit is wearing a space suit, digital Art, Greg rutkowski, Trending cinematographic artstation"，其生成的效果就大大提升：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p35.png" /> 
</div>

这里我们其实只是在原有的prompt基础加上了一些描述词，有时候我们称之为“魔咒”，不同的模型可能会有不同的魔咒。 上述我们讨论了SD的文生图的主要参数，这里简单总结一下：

+ SD默认生成512x512大小的图像，但实际上可以生成其它分辨率的图像，但是可能会出现不协调，如果采用多尺度策略训练，会改善这种情况；
+ 采用快速的noise scheduler，SD在去噪步数为30～50步时就能生成稳定的图像；
+ SD的guidance_scale设置为7～9是比较稳定的，过小和过大都会出现图像质量下降，实际使用中可以根据具体情况灵活调节；
+ 可以使用negative prompt来去除不想要的东西来改善图像生成效果；
+ 好的prompt对图像生成效果是至关重要的。

上边我们介绍了如何使用SD进行文生图以及一些主要参数，在最后我们也给出文生图这个pipeline的内部流程代码，如下所示

```python

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


model_id = "runwayml/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

# 定义参数
prompt = [
    "A dragon fruit wearing karate belt in the snow",
    "A small cactus wearing a straw hat and neon sunglasses in the Sahara desert",
    "A photo of a raccoon wearing an astronaut helmet, looking out of the window at night",
    "A cute otter in a rainbow whirlpool holding shells, watercolor"
]
height = 512
width = 512
num_inference_steps = 50
guidance_scale = 7.5
negative_prompt = ""
batch_size = len(prompt)
# 随机种子
generator = torch.Generator(device).manual_seed(2023)


with torch.no_grad():
 # 获取text_embeddings
 text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
 # 获取unconditional text embeddings
 max_length = text_input.input_ids.shape[-1]
 uncond_input = tokenizer(
     [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
 )
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
 # 拼接为batch，方便并行计算
 text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

 # 生成latents的初始噪音
 latents = torch.randn(
     (batch_size, unet.in_channels, height // 8, width // 8),
     generator=generator, device=device
 )
 latents = latents.to(device, dtype=torch.float16)

 # 设置采样步数
 noise_scheduler.set_timesteps(num_inference_steps, device=device)

 # scale the initial noise by the standard deviation required by the scheduler
 latents = latents * noise_scheduler.init_noise_sigma # for DDIM, init_noise_sigma = 1.0

 timesteps_tensor = noise_scheduler.timesteps

 # Do denoise steps
 for t in tqdm(timesteps_tensor):
     # 这里latens扩展2份，是为了同时计算unconditional prediction
     latent_model_input = torch.cat([latents] * 2)
     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

     # 使用UNet预测噪音
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

     # 执行CFG
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

     # 计算上一步的noisy latents：x_t -> x_t-1
     latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
 # 注意要对latents进行scale
 latents = 1 / 0.18215 * latents
 # 使用vae解码得到图像
    image = vae.decode(latents).sample
```


!> 图生图

图生图（image2image）是对文生图功能的一个扩展，这个功能来源于SDEdit这个工作，其核心思路也非常简单：给定一个笔画的色块图像，可以先给它加一定的高斯噪音（执行扩散过程）得到噪音图像，然后基于扩散模型对这个噪音图像进行去噪，就可以生成新的图像，但是这个图像在结构和布局和输入图像基本一致。


<div align=center>
    <img src="zh-cn/img/ch5/4-1/p36.png" /> 
</div>

对于SD来说，图生图的流程图如下所示，相比文生图流程来说，这里的初始latent不再是一个随机噪音，而是由初始图像经过autoencoder编码之后的latent加高斯噪音得到，这里的加噪过程就是扩散过程。要注意的是，去噪过程的步数要和加噪过程的步数一致，就是说你加了多少噪音，就应该去掉多少噪音，这样才能生成想要的无噪音图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p37.png" /> 
</div>

在diffusers中，我们可以使用`StableDiffusionImg2ImgPipeline`来实现文生图，具体代码如下所示：

```python
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# 加载图生图pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 读取初始图片
init_image = Image.open("init_image.png").convert("RGB")

# 推理
prompt = "A fantasy landscape, trending on artstation"
generator = torch.Generator(device="cuda").manual_seed(2023)

image = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.8, #表示对输入图片加噪音的程度，这个值越大加的噪音越多
    guidance_scale=7.5,
    generator=generator
).images[0]

image
```

相比文生图的pipeline，图生图的pipeline还多了一个参数strength，这个参数介于0-1之间，表示对输入图片加噪音的程度，这个值越大加的噪音越多，对原始图片的破坏也就越大，当strength=1时，其实就变成了一个随机噪音，此时就相当于纯粹的文生图pipeline了。下面展示了一个具体的实例，这里的第一张图为输入的初始图片，它是一个笔画的色块，我们可以通过图生图将它生成一幅具体的图像，其中第2张图和第3张图的strength分别是0.5和0.8，可以看到当strength=0.5时，生成的图像和原图比较一致，但是就比较简单了，当strength=0.8时，生成的图像偏离原图更多，但是图像的质感有一个明显的提升

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p38.png" /> 
</div>

图生图这个功能一个更广泛的应用是在风格转换上，比如给定一张人像，想生成动漫风格的图像。这里我们可以使用动漫风格的开源模型anything-v4.0，它是基于SD v1.5在动漫风格数据集上finetune的，使用它可以更好地利用图生图将人物动漫化。下面的第1张为输入人物图像，采用的prompt为"masterpiece, best quality, 1girl, red hair, medium hair, green eyes"，后面的图像是strength分别为0.3-0.9下生成的图像。可以看到在不同的strength下图像有不同的生成效果，其中strength=0.6时我觉得效果是最好的。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p39.png" /> 
</div>

总结来看，图生图其实核心也是依赖了文生图的能力，其中strength这个参数需要灵活调节来得到满意的图像。在最后，我们也给出图生图pipeline的内部主要代码，如下所示：

```python
import PIL
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


model_id = "runwayml/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

# 预处理init_image
def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

# 参数设置
prompt = ["A fantasy landscape, trending on artstation"]
num_inference_steps = 50
guidance_scale = 7.5
strength = 0.8
batch_size = 1
negative_prompt = ""
generator = torch.Generator(device).manual_seed(2023)

init_image = PIL.Image.open("init_image.png").convert("RGB")

with torch.no_grad():
 # 获取prompt的text_embeddings
 text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
 # 获取unconditional text embeddings
 max_length = text_input.input_ids.shape[-1]
 uncond_input = tokenizer(
     [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
 )
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
 # 拼接batch
 text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

 # 设置采样步数
 noise_scheduler.set_timesteps(num_inference_steps, device=device)
 # 根据strength计算timesteps
 init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
 t_start = max(num_inference_steps - init_timestep, 0)
 timesteps = noise_scheduler.timesteps[t_start:]


 # 预处理init_image
 init_input = preprocess(init_image)
    init_latents = vae.encode(init_input.to(device, dtype=torch.float16)).latent_dist.sample(generator)
    init_latents = 0.18215 * init_latents

 # 给init_latents加噪音
 noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
 init_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
 latents = init_latents # 作为初始latents


 # Do denoise steps
 for t in tqdm(timesteps):
     # 这里latens扩展2份，是为了同时计算unconditional prediction
     latent_model_input = torch.cat([latents] * 2)
     latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

     # 预测噪音
     noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

     # CFG
     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

     # 计算上一步的noisy latents：x_t -> x_t-1
     latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
    
 # 注意要对latents进行scale
 latents = 1 / 0.18215 * latents
 # 解码
 image = vae.decode(latents).sample
```

!> 图像inpainting

最后我们要介绍的一项功能是图像inpainting，它和图生图一样也是文生图功能的一个扩展。SD的图像inpainting不是用在图像修复上，而是主要用在图像编辑上：给定一个输入图像和想要编辑的区域mask，我们想通过文生图来编辑mask区域的内容。SD的图像inpainting原理可以参考论文Blended Latent Diffusion，其主要原理图如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p40.png" /> 
</div>

它和图生图一样：首先将输入图像通过autoencoder编码为latent，然后加入一定的高斯噪音生成noisy latent，再进行去噪生成图像，但是这里为了保证mask以外的区域不发生变化，在去噪过程的每一步，都将扩散模型预测的noisy latent用真实图像同level的nosiy latent替换。 在diffusers中，使用`StableDiffusionInpaintPipelineLegacy`可以实现文本引导下的图像inpainting，具体代码如下所示：

```python
import torch
from diffusers import StableDiffusionInpaintPipelineLegacy
from PIL import Image

# 加载inpainting pipeline
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# 读取输入图像和输入mask
input_image = Image.open("overture-creations-5sI6fQgYIuo.png").resize((512, 512))
input_mask = Image.open("overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

# 执行推理
prompt = ["a mecha robot sitting on a bench", "a cat sitting on a bench"]
generator = torch.Generator("cuda").manual_seed(0)

with torch.autocast("cuda"):
    images = pipe(
        prompt=prompt,
        image=input_image,
        mask_image=input_mask,
        num_inference_steps=50,
        strength=0.75,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        generator=generator,
    ).images
```

下面是一个具体的生成效果，这里我们将输入图像的dog换成了mecha robot或者cat，从而实现了图像编辑。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p41.png" /> 
</div>

要注意的是这里的参数guidance_scale也和图生图一样比较重要，要生成好的图像，需要选择合适的guidance_scale。如果guidance_scale=0.5时，生成的图像由于过于受到原图干扰而产生一些不协调，如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p42.png" /> 
</div>

合适的prompt也比较重要，比如如果我们去掉prompt中的"sitting on a bench"，那么编辑的图像效果也会出现不协调：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p43.png" /> 
</div>

无论是上面的图生图还是这里的图像inpainting，我们其实并没有去finetune SD模型，只是扩展了它的能力，但是这两样功能就需要精确调整参数才能得到满意的生成效果。 这里，我们也给出`StableDiffusionInpaintPipelineLegacy`这个pipeline内部的核心代码：

```python

import PIL
import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0

model_id = "runwayml/stable-diffusion-v1-5"
# 1. 加载autoencoder
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
# 2. 加载tokenizer和text encoder 
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
# 3. 加载扩散模型UNet
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
# 4. 定义noise scheduler
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False, # don't clip sample, the x0 in stable diffusion not in range [-1, 1]
    set_alpha_to_one=False,
)

# 将模型复制到GPU上
device = "cuda"
vae.to(device, dtype=torch.float16)
text_encoder.to(device, dtype=torch.float16)
unet = unet.to(device, dtype=torch.float16)

prompt = "a mecha robot sitting on a bench"
strength = 0.75
guidance_scale = 7.5
batch_size = 1
num_inference_steps = 50
negative_prompt = ""
generator = torch.Generator(device).manual_seed(0)

with torch.no_grad():
    # 获取prompt的text_embeddings
    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    # 获取unconditional text embeddings
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    # 拼接batch
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # 设置采样步数
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    # 根据strength计算timesteps
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = noise_scheduler.timesteps[t_start:]


    # 预处理init_image
    init_input = preprocess(input_image)
    init_latents = vae.encode(init_input.to(device, dtype=torch.float16)).latent_dist.sample(generator)
    init_latents = 0.18215 * init_latents
    init_latents = torch.cat([init_latents] * batch_size, dim=0)
    init_latents_orig = init_latents
    # 处理mask
    mask_image = preprocess_mask(input_mask)
    mask_image = mask_image.to(device=device, dtype=init_latents.dtype)
    mask = torch.cat([mask_image] * batch_size)
    
    # 给init_latents加噪音
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
    init_latents = noise_scheduler.add_noise(init_latents, noise, timesteps[:1])
    latents = init_latents # 作为初始latents


    # Do denoise steps
    for t in tqdm(timesteps):
        # 这里latens扩展2份，是为了同时计算unconditional prediction
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t) # for DDIM, do nothing

        # 预测噪音
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 计算上一步的noisy latents：x_t -> x_t-1
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        # 将unmask区域替换原始图像的nosiy latents
        init_latents_proper = noise_scheduler.add_noise(init_latents_orig, noise, torch.tensor([t]))
        latents = (init_latents_proper * mask) + (latents * (1 - mask))

    # 注意要对latents进行scale
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
```

另外，runwayml在发布SD 1.5版本的同时还发布了一个inpainting模型：runwayml/stable-diffusion-inpainting，与前面所讲不同的是，这是一个在SD 1.2上finetune的模型。原来SD的UNet的输入是64x64x4，为了实现inpainting，现在给UNet的第一个卷机层增加5个channels，分别为masked图像的latents（经过autoencoder编码，64x64x4）和mask图像（直接下采样8x，64x64x1），增加的权重填零初始化。在diffusers中，可以使用StableDiffusionInpaintPipeline来调用这个模型，具体代码如下：

```python
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm.auto import tqdm
import PIL

# Load pipeline
model_id = "runwayml/stable-diffusion-inpainting/"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = ["a mecha robot sitting on a bench", "a dog sitting on a bench", "a bench"]

generator = torch.Generator("cuda").manual_seed(2023)

input_image = Image.open("overture-creations-5sI6fQgYIuo.png").resize((512, 512))
input_mask = Image.open("overture-creations-5sI6fQgYIuo_mask.png").resize((512, 512))

images = pipe(
    prompt=prompt,
    image=input_image,
    mask_image=input_mask,
    num_inference_steps=50,
    generator=generator,
    ).images

```

其生成的效果图如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p44.png" /> 
</div>

经过finetune的inpainting在生成细节上可能会更好，但是有可能会丧失部分文生图的能力，而且也比较难迁移其它finetune的SD模型。

------
**SD 2.0**

Stability.AI公司在2022年11月（stable-diffusion-v2-release）放出了SD 2.0版本，这里我们也简单介绍一下相比SD 1.x版本SD 2.0的具体改进点。SD 2.0相比SD 1.x版本的主要变动在于模型结构和训练数据两个部分。


<div align=center>
    <img src="zh-cn/img/ch5/4-1/p45.png" /> 
</div>

首先是模型结构方面，SD 1.x版本的text encoder采用的是OpenAI的CLIP ViT-L/14模型，其模型参数量为123.65M；而SD 2.0采用了更大的text encoder：基于OpenCLIP在laion-2b数据集上训练的CLIP ViT-H/14模型，其参数量为354.03M，相比原来的text encoder模型大了约3倍。两个CLIP模型的对比如下所示：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p46.png" /> 
</div>

可以看到CLIP ViT-H/14模型相比原来的OpenAI的L/14模型，在imagenet1K上分类准确率和mscoco多模态检索任务上均有明显的提升，这也意味着对应的text encoder更强，能够抓住更准确的文本语义信息。另外是一个小细节是SD 2.0提取的是text encoder倒数第二层的特征，而SD 1.x提取的是倒数第一层的特征。由于倒数第一层的特征之后就是CLIP的对比学习任务，所以倒数第一层的特征可能部分丢失细粒度语义信息，Imagen论文（见论文D.1部分）和novelai（见novelai blog）均采用了倒数第二层特征。对于UNet模型，SD 2.0相比SD 1.x几乎没有改变，唯一的一个小的变动是：SD 2.0不同stage的attention模块是固定attention head dim为64，而SD 1.0则是不同stage的attention模块采用固定attention head数量，明显SD 2.0的这种设定更常用，但是这个变动不会影响模型参数。 然后是训练数据，前面说过SD 1.x版本其实最后主要采用laion-2B中美学评分为5以上的子集来训练，而SD 2.0版本采用评分在4.5以上的子集，相当于扩大了训练数据集，具体的训练细节见model card。 另外SD 2.0除了512x512版本的模型，还包括768x768版本的模型（https://huggingface.co/stabilityai/stable-diffusion-2） ，所谓的768x768模型是在512x512模型基础上用图像分辨率大于768x768的子集继续训练的，不过优化目标不再是noise_prediction，而是采用Progressive Distillation for Fast Sampling of Diffusion Models论文中所提出的 v-objective。 下图为SD 2.0和SD 1.x版本在COCO2017验证集上评测的对比，可以看到2.0相比1.5，CLIP score有一个明显的提升，同时FID也有一定的提升。但是正如前面所讨论的，FID和CLIP score这两个指标均有一定的局限性，所以具体效果还是上手使用来对比。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p47.png" /> 
</div>

Stability.AI在发布SD 2.0的同时，还发布了另外3个模型：stable-diffusion-x4-upscaler，stable-diffusion-2-inpainting和stable-diffusion-2-depth。stable-diffusion-x4-upscaler是一个基于扩散模型的4x超分模型，它也是基于latent diffusion，不过这里采用的autoencoder是基于VQ-reg的，下采样率为$f=4$。在实现上，它是将低分辨率图像直接和noisy latent拼接在一起送入UNet，因为autoencoder将高分辨率图像压缩为原来的1/4，而低分辨率图像也为高分辨率图像的1/4，所以低分辨率图像的空间维度和latent是一致的。另外，这个超分模型也采用了Cascaded Diffusion Models for High Fidelity Image Generation所提出的noise conditioning augmentation，简单来说就是在训练过程中给低分辨率图像加上高斯噪音，可以通过扩散过程来实现，注意这里的扩散过程的scheduler与主扩散模型的scheduler可以不一样，同时也将对应的noise_level（对应扩散模型的time step）通过class labels的方式送入UNet，让UNet知道加入噪音的程度。stable-diffusion-x4-upscaler是使用LAION中>2048x2048大小的子集（10M）训练的，训练过程中采用512x512的crops来训练（降低显存消耗）。SD模型可以用来生成512x512图像，加上这个超分模型，就可以得到2048x2048大小的图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p48.png" /> 
</div>

在diffusers库中，可以如下使用这个超分模型（这里的noise level是指推理时对低分辨率图像加入噪音的程度）：

```python
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# let's download an  image
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
response = requests.get(url)
low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
low_res_img = low_res_img.resize((128, 128))

prompt = "a white cat"

upscaled_image = pipeline(prompt=prompt, image=low_res_img, noise_level=20).images[0]
upscaled_image.save("upsampled_cat.png")
```

stable-diffusion-2-inpainting是图像inpainting模型，和前面所说的runwayml/stable-diffusion-inpainting基本一样，不过它是在SD 2.0的512x512版本上finetune的。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p49.png" /> 
</div>

stable-diffusion-2-depth是也是在SD 2.0的512x512版本上finetune的模型，它是额外增加了图像的深度图作为condition，这里是直接将深度图下采样8x，然后和nosiy latent拼接在一起送入UNet模型中。深度图可以作为一种结构控制，下图展示了加入深度图后生成的图像效果：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p50.png" /> 
</div>

你可以调用diffusers库中的`StableDiffusionDepth2ImgPipeline`来实现基于深度图控制的文生图：

```python
import torch
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16,
).to("cuda")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)

prompt = "two tigers"
n_propmt = "bad, deformed, ugly, bad anotomy"
image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]
```

除此之外，Stability AI公司还开源了两个加强版的autoencoder：ft-EMA和ft-MSE（前者使用L1 loss后者使用MSE loss），前面已经说过，它们是在LAION数据集继续finetune decoder来增强重建效果。

------

**SD 2.1**

在SD 2.0版本发布几周后，Stability AI又发布了SD 2.1。SD 2.0在训练过程中采用NSFW检测器过滤掉了可能包含色情的图像（punsafe=0.1），但是也同时过滤了很多人像图片，这导致SD 2.0在人像生成上效果可能较差，所以SD 2.1是在SD 2.0的基础上放开了限制（punsafe=0.98）继续finetune，所以增强了人像的生成效果。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p51.png" /> 
</div>

和SD 2.0一样，SD 2.1也包含两个版本：512x512版本和768x768版本。

**SD unclip**

Stability AI在2023年3月份，又放出了基于SD的另外一个模型：stable-diffusion-reimagine，它可以实现单个图像的变换，即image variations，目前该模型已经在在huggingface上开源：stable-diffusion-2-1-unclip。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p52.png" /> 
</div>

这个模型是借鉴了OpenAI的DALLE2（又称unCLIP)，unCLIP是基于CLIP的image encoder提取的image embeddings作为condition来实现图像的生成。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p53.png" /> 
</div>

SD unCLIP是在原来的SD模型的基础上增加了CLIP的image encoder的nosiy image embeddings作为condition。具体来说，它在训练过程中是对提取的image embeddings施加一定的高斯噪音（也是通过扩散过程），然后将noise level对应的time embeddings和image embeddings拼接在一起，最后再以class labels的方式送入UNet。在diffusers中，你可以调用`StableUnCLIPImg2ImgPipeline`来实现图像的变换：

```python
import requests
import torch
from PIL import Image
from io import BytesIO

from diffusers import StableUnCLIPImg2ImgPipeline

#Start the StableUnCLIP Image variations pipeline
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
)
pipe = pipe.to("cuda")

#Get image from URL
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/stable_unclip/tarsila_do_amaral.png"
response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")

#Pipe to make the variation
images = pipe(init_image).images
images[0].save("tarsila_variation.png")

```

其实在SD unCLIP之前，已经有Lambda Labs开源的sd-image-variations-diffusers，它是在SD 1.4的基础上finetune的模型，不过实现方式是直接将text embeddings替换为image embeddings，这样也同样可以实现图像的变换。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p54.png" /> 
</div>

这里SD unCLIP有两个版本：sd21-unclip-l和sd21-unclip-h，两者分别是采用OpenAI CLIP-L和OpenCLIP-H模型的image embeddings作为condition。如果要实现文生图，还需要像DALLE2那样训练一个prior模型，它可以实现基于文本来预测对应的image embeddings，我们将prior模型和SD unCLIP接在一起就可以实现文生图了。KakaoBrain这个公司已经开源了一个DALLE2的复现版本：Karlo，它是基于OpenAI CLIP-L来实现的，你可以基于这个模型中prior模块加上sd21-unclip-l来实现文本到图像的生成，目前这个已经集成了在`StableUnCLIPPipeline`中，或者基于`stablediffusion`官方仓库来实现。
<div align=center>
    <img src="zh-cn/img/ch5/4-1/p55.png" /> 
</div>

------

**SD的其它特色应用**

在SD模型开源之后，社区和研究机构也基于SD实现了形式多样的特色应用，这里我们也选择一些比较火的应用来介绍一下。

!> 个性化生成

个性化生成是指的生成特定的角色或者风格，比如给定自己几张肖像来利用SD来生成个性化头像。在个性化生成方面，比较重要的两个工作是英伟达的Textual Inversion和谷歌的DreamBooth。Textual Inversion这个工作的核心思路是基于用户提供的3～5张特定概念（物体或者风格）的图像来学习一个特定的text embeddings，实际上只用一个word embedding就足够了。Textual Inversion不需要finetune UNet，而且由于text embeddings较小，存储成本很低。目前diffusers库已经支持textual_inversion的训练。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p56.png" /> 
</div>

DreamBooth原本是谷歌提出的应用在Imagen上的个性化生成，但是它实际上也可以扩展到SD上（更新版论文已经增加了SD）。DreamBooth首先为特定的概念寻找一个特定的描述词[V]，这个特定的描述词只要是稀有的就可以，然后与Textual Inversion不同的是DreamBooth需要finetune UNet，这里为了防止过拟合，增加了一个class-specific prior preservation loss（基于SD生成同class图像加入batch里面训练）来进行正则化。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p57.png" /> 
</div>

由于finetune了UNet，DreamBooth往往比Textual Inversion要表现的要好，但是DreamBooth的存储成本较高。目前diffusers库已经支持dreambooth训练，你也可以在sd-dreambooth-library中找到其他人上传的模型。 DreamBooth和Textual Inversion是最常用的个性化生成方法，但其实除了这两种，还有很多其它的研究工作，比如Adobe提出的Custom Diffusion，相比DreamBooth，它只finetune了UNet的attention模块的KV权重矩阵，同时优化一个新概念的token。


<div align=center>
    <img src="zh-cn/img/ch5/4-1/p58.png" /> 
</div>

!> 风格化finetune模型

SD的另外一大应用是采用特定风格的数据集进行finetune，这使得模型“过拟合”在特定的风格上。之前比较火的novelai就是基于二次元数据在SD上finetune的模型，虽然它失去了生成其它风格图像的能力，但是它在二次元图像的生成效果上比原来的SD要好很多。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p59.png" /> 
</div>

目前已经有很多风格化的模型在huggingface上开源，这里也列出一些：

+ andite/anything-v4.0：二次元或者动漫风格图像

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p60.png" /> 
</div>

+ dreamlike-art/dreamlike-diffusion-1.0：艺术风格图像

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p61.png" /> 
</div>

+ prompthero/openjourney：mdjrny-v4风格图像

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p62.png" /> 
</div>

更多的模型可以直接在huggingface text-to-image模型库上找到。此外，很多基于SD进行finetune的模型开源在civitai上，你也可以在这个网站上找到更多风格的模型。 值得说明的一点是，目前finetune SD模型的方法主要有两种：一种是直接finetune了UNet，但是容易过拟合，而且存储成本大；另外一种低成本的方法是基于微软的LoRA，LoRA本来是用于finetune语言模型的，但是现在已经可以用来finetune SD模型了，具体可以见博客Using LoRA for Efficient Stable Diffusion Fine-Tuning。

!> 图像编辑

图像编辑也是SD比较火的应用方向，这里所说的图像编辑是指的是使用SD来实现对图片的局部编辑。这里列举两个比较好的工作：谷歌的prompt-to-prompt和加州伯克利的instruct-pix2pix。 谷歌的prompt-to-prompt的核心是基于UNet的cross attention maps来实现对图像的编辑，它的好处是不需要finetune模型，但是主要用在编辑用SD生成的图像。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p63.png" /> 
</div>


谷歌后面的工作Null-text Inversion有进一步实现了对真实图片的编辑：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p64.png" /> 
</div>

instruct-pix2pix这个工作基于GPT-3和prompt-to-prompt构建了pair的数据集，然后在SD上进行finetune，它可以输入text instruct对图像进行编辑：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p65.png" /> 
</div>

!> 可控生成

可控生成是SD最近比较火的应用，这主要归功于ControlNet，基于ControlNet可以实现对很多种类的可控生成，比如边缘，人体关键点，草图和深度图等等。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p66.png" /> 
</div>

其实在ControlNet之前，也有一些可控生成的工作，比如stable-diffusion-2-depth也属于可控生成，但是都没有太火。我觉得ControlNet之所以火，是因为这个工作直接实现了各种各种的可控生成，而且训练的ControlNet可以迁移到其它基于SD finetune的模型上（见Transfer Control to Other SD1.X Models）：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p67.png" /> 
</div>

与ControlNet同期的工作还有腾讯的T2I-Adapter以及阿里的composer-page：

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p68.png" /> 
</div>

!> stable-diffusion-webui

最后要介绍的一个比较火的应用stable-diffusion-webui其实是用来支持SD出图的一个web工具，它算是基于gradio框架实现了SD的快速部署，不仅支持SD的最基础的文生图、图生图以及图像inpainting功能，还支持SD的其它拓展功能，很多基于SD的拓展应用可以用插件的方式安装在webui上。

<div align=center>
    <img src="zh-cn/img/ch5/4-1/p69.png" /> 
</div>


**后话**

在OpenAI最早放出DALLE2的时候，我曾被它生成的图像所惊艳到，但是我从来没有想到图像生成的AIGC会如此火爆，技术的发展太快了，这得益于互联网独有的开源精神。我想，没有SD的开源，估计这个方向可能还会沉寂一段时间。

**参考文献**

+ High-Resolution Image Synthesis with Latent Diffusion Models
+ https://huggingface.co/CompVis/stable-diffusion-v1-4
+ https://huggingface.co/runwayml/stable-diffusion-v1-5
+ https://github.com/huggingface/diffusers
+ https://huggingface.co/blog/stable_diffusion
+ https://github.com/CompVis/latent-diffusion
+ https://laion.ai/blog/laion-5b/
+ https://arxiv.org/abs/2303.05511
+ https://arxiv.org/abs/2211.01324
+ https://arxiv.org/abs/2205.11487
+ https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/
+ https://stability.ai/blog/stablediffusion2-1-release7-dec-2022


------
------

### 3.项目实战：文本生成实战

这部分介绍stable diffusion常用的微调方法，并基于太乙stable diffusion实现文本生成图片的实战，其实可以非常容易的迁移到图片生成图片，图片inpainting等任务。

下面我们将详细介绍stable diffusion中的微调方法：DreamBooth,LoRA,textual inversion,hypernetwork和LyCORIS。

<!-- https://zhuanlan.zhihu.com/p/619348969 -->

<!-- https://zhuanlan.zhihu.com/p/615739257 -->

<!-- https://zhuanlan.zhihu.com/p/631370055 -->

<!-- 太乙：https://zhuanlan.zhihu.com/p/580103864 -->

<!-- https://zhuanlan.zhihu.com/p/580966411 -->

<!-- 
### 4.项目实战：图片生成图片
### 5.项目实战：图片inpainting
### 6.项目实战：使用controlnet辅助生成图片
 -->


#### 3.1 DreamBooth:Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
 
 <!-- https://zhuanlan.zhihu.com/p/615739257 -->
 <!-- https://zhuanlan.zhihu.com/p/620577688 -->
 <!-- https://mp.weixin.qq.com/s/NE3Gkr64G3XADVdujtzRXw -->

 <!-- https://zhuanlan.zhihu.com/p/615739257 -->
 <!-- https://zhuanlan.zhihu.com/p/630754310 -->

!> arxiv: https://arxiv.org/abs/2208.12242

!> github: https://github.com/XavierXiao/Dreambooth-Stable-Diffusion

**1. 简述**

现在的文图生成模型参数规模已经达到了数十亿甚至数百亿的量级，通过输入文本Prompt描述，可合成质量高，多样化的图片。但是这种图片缺乏对特定物体的精确在现能力，特别是对个人定制的物体，很难实现个人物体在风格、造型等方面的实现能力。

对此，Google团队提出一种文图模型生成内容的"定制化"能力，通过一种基于new autogenous class-specific prior preservation loss的微调策略，训练少量几张需要定制化物体的图片，实现该生成物体在风格上、背景等多样化的再创造，在保持物体特征不变的前提下，光照、姿势、视角以及背景等内容可以自定义。如下图所示，输入4张柯基的照片，可是保持柯基的外貌特征不变的前提下，将其背景改为Acropolis，或者狗窝、水里等。画面看上去充满了创意。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p1.png" /> 
</div><p align=center>Figure 1: With just a few images (typically 3-5) of a subject (left), DreamBooth—our AI-powered photo booth—can generate a myriad of images of the subject in different contexts (right)</p>

Dreambooth将希望个性化目标内容（例如：图中的柯基狗）绑定到一个新词`[V]`上，实现精准的生成目标内容。通俗理解为：通过prompt无法精准的定位视觉生成空间中的目标柯基，只能在dog这个类(CLASS)里面，带有随机性的生成各种各样的狗，即使prompt描述的非常精细，也无法与视觉空间形成精准的对应，如果在目标柯基的视觉位置上放置一个精准定位词`[V]`，就像是在目标位置放置一个GPS，这样就可以从茫茫的狗群里面精准定位到柯基。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p2.png" /> 
</div><p align=center> Dreambooth通俗理解示意图</p>

 
该方案主要贡献有两点：

+ 通过少量参考图，可以精确的控制个性化内容生成，并且在保留个性化内容的关键特征的同时，对内容的风格创意和形式合成方面的具备较好的泛化性。
+ 利用few-shot的方式对原模型进行微调时，提出一种正则损失来保留模型的先验知识，也能精准的生成目标内容。

**2. 具体方案**

本文选用的Cascaded文图生成模型Imagen（Cascaded Text-to-Image Diffusion Models，Google王者归来Imagen）采用的是典型的两段式结构模型，即：

1. 先根据描述文本Prompt生成low-resolution的图片，
2. 在根据low-resolution图生成high-resolution图（即Imagen中的Super-resolution模型）。

训练的过程也是分为两个夹阶段：

**第一阶段：**先根据训练图片集（一般为3~5张）和描述文本（带有`[V]`标识，例如：A `[V]` dog），去训练low-resolution的生成模型。由于训练数据少，模型参数量大，极容易产生overfitting，另外模型对标识符`[V]`是没有先验知识的，所以也容易产生language drift，为此提出一种正则损失函数：autogenous, class-specific prior preservation loss，这个损失函数既可以保留模型对目标class的先验知识，也可以将目标准确的圈定在class内的确定视觉空间中，如Figure4中的第一阶段。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p3.png" /> 
</div><p align=center>autogenous, class-specific prior preservation loss</p>

**第二阶段：**将训练集中的图片组成`<low-resolution, high resolution>`样本对，去finetune Imagen中的SR部分模型，使得最终能够生成目标内容的更多细节，如Figure4中的第二阶段。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p4.png" /> 
</div><p align=center>Figure 4: Fine-tuning.</p>

Dreambooth的主要思路就是通过以上两个阶段的训练来实现。其中有几个细节点需要展开：


!> 2.1 如何构建[V]，为什么需要[V]？

`[V]`是一个GPS定位器（文中比喻为indentifier），为了保证不与先验知识中其他的词在语义上冲突，构建时需要尽可能唯一性，建议用生僻词（不建议随便通过字母和数字构建，如"xxy5syt00"，因为在通过tokenizer分词之后，可能会出现具有较强先验的词，可以使用`T5-XXL`中tokenizer词表`5000~10000`范围内的`1~3`个token进行组合）。`[V]`告诉生成模型，在class的范围内，目标内容的具体"位置信息"，使得生成模型可以精准的生成目标内容。

!> 2.2 如何解决Overfitting和Language drift的问题？

用少量的训练样本，采用denoising loss对大模型进行微调的方式，不可避免出现Overfitting和Language-drift的问题。对于Overfitting的问题，常用的方式是添加正则、固定部分参数、dropout等，但是实际上无法知晓模型各部分参数具体在生成过程中所起到的作用。对与Language drift问题，由于大模型是用非常大的各类数据集进行训练的，微调时却使用非常小的数据，很容易造成`知识的遗忘`。如果在训练集的文本中制定微调的范围，即CLASS，则可以缓解这种遗忘。

对应以上问题，Dreambooth提出一种Prior-Preservation Loss：
<div align=center>
    <img src="zh-cn/img/ch5/5-1/p5.png" /> 
</div>

其中$\tilde{x}_ {\theta}(\alpha_t x+\sigma_t \epsilon,c)-x$,
部分是训练模型学习输入的少量样本。$\tilde{x}_ {\theta}(\alpha_{t^{'}} x_{pr}+\sigma_{t^{'}} \epsilon^{'},c_{pr})-x_{pr}$
部分则是训练原模型自己生成的样本，防止模型的知识遗忘。通过上述Preservation Loss，Google通过实验验证了其能较好的解决Overfitting和Language-drift问题。

**3. 实验结果**

在开头就已经强调了本文的两个重要贡献，其中"对内容的风格创意和形式合成方面的具备较好的泛化性"，看起来有点不太清晰，具体体现在哪些方面效果上呢？结合文章的实验部分具体从一下几方面进行效果展示：

!> 3.1 Recontextualization个性化内容的重构效果

个性化内容重构能力是微调的最基本要求了。通过输入带有`[V]`和`CLASS`标识的Prompt，输入到微调后的模型中，可以生成与训练集中内容相同的目标，例如下图中训练集中红色款式的书包，可以直接进行生成，生成的书包基本还原了训练集的效果，包括细小的小装饰物。另外还可以向Prompt中添加背景修饰。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p6.png" /> 
</div><p align=center>Figure 5: Recontextualization of a backpack, vase, and teapot subject instances.</p>

!> 3.2 Art Renditions 艺术渲染效果

进一步验证微调后的效果，还验证了对目标图中的内容进行各种艺术效果的改变，例如`"a [V] [class noun] in the style of [famous sculptor]"`中将famous sculptor分别改为"Vincent Van Gogh"等，最终图中内容呈现了对用的艺术形式。并且风格与内容看起来和谐，同时又保留了内容的主要特征。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p7.png" /> 
</div><p align=center>Figure 6: Artistic renderings of a dog instance in the style of famous painters.</p>

!> 3.3 Expression Manipulation 表述的可控效果

为了验证模型确实是理解了图中的内容，还验证了训练集中不存在的知识表达，例如下图中的dog分别进行不同的表情效果，这类表情在原图中是不具备的，模型的生成结果是建立在准确捕捉到训练集中的内容以及控制生成文本的表述意图。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p8.png" /> 
</div><p align=center>Figure 7: Expression manipulation of a dog instance.</p>

!> 3.4 Novel View Synthesis 多视角合成效果

不仅对局部内容进行理解和生成，也对整体形态进行了理解和生成。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p9.png" /> 
</div><p align=center>Figure 8: Text-guided view synthesis.</p>

!> 3.5 Accessorization 外观装饰效果

另外还验证了与大模型知识中学习到的内容进行融合交互，展示了模型的创新能力。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p10.png" /> 
</div><p align=center>Figure 9: Outfitting a dog with accessories.</p>

!> 3.6 Property Modification 属性编辑效果

最后验证了模型对图中内容的某类属性的理解和创新生成。

<div align=center>
    <img src="zh-cn/img/ch5/5-1/p11.png" /> 
</div><p align=center>Figure 10: Modification of subject properties while preserving their key features.</p>

#### 3.2 LoRA:LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
<!-- https://zhuanlan.zhihu.com/p/627133491 -->
!> arxiv:https://arxiv.org/abs/2106.09685v2 

!> github:https://github.com/microsoft/LoRA

**1. 简述**

这两年大火的AIGC领域中的LLM和AI绘画都是基于超大参数量的模型和超大训练数据。各领域为了跟上技术的发展，也积极投身于领域模型定制化训练，但是训练大模型需要巨大的投入，一般中小企业很难承担的起。例如175Billion参数量的GPT-3，据非权威信息GPT3的一次训练成本为140万美元，更何况这种超大模型的训练难度加上试错成本就决定了这只能是头部大公司玩得起的。

不过，微软提出了一种低秩自适应训练方式Low-Rank Adaptation, Lora，该方式固定预训练模型，往模型中添加少量可训练的tensor矩阵，通过对该部分可训练的参数，实现对整个模型输出效果的定制化。通过这种方式需要训练的参数量只有原模型的1/1000甚至1/10000，并且也大大缩小了训练时长，从根本上解决了中小企业训练大模型难的问题。

举个例子，如下图：
<div align=center>
    <img src="zh-cn/img/ch5/6-1/p1.png" /> 
</div><p align=center>Figure 1: Our reparametrization. We only train A and B.</p>

原模型的训练需要训练$W$ ，Lora的方式则是保持$W$不变，只训练$A$和$B$,由于$A*B=W^{'}$ 得到的 
$W^{'}$维度大小与$W$相同，训练时输入分别与$W^{'}$和$W$进行计算，然后将输出相加，反向获取的梯度只用于更新$A$和$B$。推理阶段前则直接将$W^{'}$和$W$相加，然后替换$W$，这样整个推理流程和参数计算量没有发生变化。

总的来说Lora的主要优点有：

+ 训练前后预训练模型保持不变，Lora可以看作为"插件"，可以添加、删除，以及替换，操作容易，资源占用少。
+ 训练效率高，只需要训练Lora部分，参数量小，且硬件要求不高。
+ 训练后将Lora添加到原模型中，推理速度不受影响，保持不变。
+ 可多个Lora进行叠加，具备融合效果。

**2. 具体方案**

!> 2.1 用低秩矩阵代替全秩矩阵

LoRA的方案可以应用在多种结构的神经网络模型中，用低秩的$A$和$B$点乘替代原模型dense layer结构中的full-ranked weight matrices，减少模型训练的参数量，进而可用于低成本微调大模型。文章中有一段话奠定了该方案的理论基础：

> "Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace."

理论上解释了使用低秩的参数可以使得大模型学习到，对任意smaller subspace的投影。即让大模型通过少量的样本，学习到样本的特征子空间。

!> 2.2 对原模型训练的影响

由于LoRA在微调完成之后，是直接与原模型中的权重矩阵相加，并未带来额外的参数量，所以推理的速度不受影响。训练阶段，只需要训练LoRA的参数，原模型参数不变，并且在计算梯度的时候无需对保存原模型参数的梯度变量，所以训练所需要的显存也小于全参数微调，另外LoRA中控制秩的参数$r$越小，占用的资源和训练收敛的越快。相反,$r$如果与全秩相同，那其实就接近原模型的微调了。

!> 2.3 向Transformer模型中添加LoRA

Transformer的模型结构主要包括了Attention、MLP和Norm几部分，其中Attention的计算公式中有3个矩阵参数WQ/WK/WV，将Lora应用在该部分如下图：

<div align=center>
    <img src="zh-cn/img/ch5/6-1/p2.png" /> 
</div><p align=center>APPLYING LORA TO TRANSFORMER</p>

替换Attention中WQ/WK/WV矩阵，由此实现对Q/K/V生成的训练，最后完成原模型的效果微调。同时，微调过程中大概能减少2/3的显存（使用Adam优化器，大概是因为Adam在训练中包括了gradients和Adam states）。另外训练完成之后LoRA部分的模型大小也只有几十MB，训练的效率也有25%的提升。

至于为什么将LoRA应用在Transformer中该部分，其实文章也是经过消融对比实验得出的最佳方式：

<div align=center>
    <img src="zh-cn/img/ch5/6-1/p3.png" /> 
</div><p align=center>Table 5: Validation accuracy on WikiSQL and MultiNLI after applying LoRA to different types of attention weights in GPT-3, given the same number of trainable parameters.</p>

在相同LoRA参数量的前提下，针对self-attention中的WQ/WK/WV矩阵（W0是multi-head中的矩阵，这里不展开了）进行LoRA替换是最佳的方案。

**3. 实验结果**

本文分别在RoBERTa、DeBERTa和GPT2模型上验证LoRA微调方案在多种NLP任务上的效果。可以发现，LoRA在对比有着相同训练参数量时，在各类任务上都要比RoBERTa、DeBERTa的Adaption的方式要好，相比全参数微调FT的效果，LoRA也能够咋部分任务上表现略高，整体上与FT相当

<div align=center>
    <img src="zh-cn/img/ch5/6-1/p4.png" /> 
</div><p align=center>Table 2: RoBERTabase, RoBERTalarge, and DeBERTaXXL with different adaptation methods on the GLUE benchmark.</p>

另外对比了常见大模型微调的方案，"BitFit"， "Prefix-embedding tuning"，"Prefix-layer tuning"和花式"Adapter tuning"。在GPT2上的个任务微调结果，具体为：

<div align=center>
    <img src="zh-cn/img/ch5/6-1/p5.png" /> 
</div><p align=center>Table 3: GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG Challenge.</p>

可以发现LoRA的微调方式不仅是参数量最小的，也是在多种任务上效果最好的。
另外对于LoRA中仅有的超参r，文章也给出了对比实验的结果，如下：

<div align=center>
    <img src="zh-cn/img/ch5/6-1/p6.png" /> 
</div><p align=center>Table 6: Validation accuracy on WikiSQL and MultiNLI with different rank r.</p>

LoRA的大模型方式，不仅在LLM的微调中也比较常见,也在AI绘画Stable Diffusino模型中应用很广，，所以熟练使用LoRA对大模型进行微调还是很有必要亲自实践下的。

#### 3.3 Textual Inversion(Embedding), HyperNetwork和Aesthetic Gradients
<!-- textual inversion: https://zhuanlan.zhihu.com/p/622280974 -->
<!-- https://zhuanlan.zhihu.com/p/620570659 -->

##### 3.3.1 Textual Inversion
<!-- https://zhuanlan.zhihu.com/p/632867668 -->

!> arxiv: https://arxiv.org/abs/2208.01618v1

!> github: https://github.com/rinongal/textual_inversion

**1. 简述**

AI绘画模型具备非常强大的生成能力，可以根据任意文本描述生成指定的内容，并在在风格、构图、场景等方面具备很强的创造性。但是如果生成具备某种唯一特征的内容，则很难通过文本描述来生成。对此Nvidia团队提出一种AI绘画模型微调方案Textual Inversion，可以实现微调带有目标内容的3~5张图片，进而使得模型能够准确学会该目标的唯一特征，并且还可以与为改内容合成新的背景、风格等。例如下图：

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p1.png" /> 
</div><p align=center>Figure 1: Examples</p>

输入几张训练样本，并且用特定的词$S_{*}$ 来表示该内容，通过训练之后，在文本中添加$S_{*}$ 及其他描述，即可生成不同风格以及不同造型的目标内容。**该训练过程中模型的参数保持不变，只微调特定词$S_{*}$ 对应的embedding向量**，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p2.png" /> 
</div><p align=center>Figure 2: Outline of the text-embedding and inversion process.</p>

训练过程中，在文本编码器部分，$S_{*}$ 分词之后的token通过embedding look-up table映射为特征向量$v_{*}$ 与文本中其他部分一同嵌入到Unet模型中进行图片引导生成，通过重构目标计算梯度回传到embedding部分来更新$v_{*}$ 对应的embeding向量。
<div align=center>
    <img src="zh-cn/img/ch5/7-1/p3.png" /> 
</div>

这个逆向的将目标的特征微调到文本embedding中也对应的该方案的名字Textual Inversion。该方案不仅能到特定的物体比如Figure1中的雕像，也可以用在人物、风格等场景下，并且多个训练后的embedding合一叠加使用，例如下图：
<div align=center>
    <img src="zh-cn/img/ch5/7-1/p4.png" /> 
</div><p align=center>Figure 7: Compositional generation using two learned pseudo-words.</p>

该图训练了一种风格$S_{style}$以及三个特定的物体$S_{clock},S_{cat},S_{crasft}$ 
 ，并且分别将三种物体与该风格结合，生成看制定风格的特定内容.

**2. 具体方案**

本方案可行的依据是微调训练的embedding可以足够表示目标特征的语义信息，原文如下：

> Prior work has shown that this embedding space is expressive enough to capture basic image semantics (Cohen et al., 2022; Tsimpoukelli et al., 2021).
> 
与LoRA、Dreambooth相比，该方案需要微调的参数量最小，训练数据最少（对于较为复杂的物体特征的拟合能力是有限的，从文章后面给出的人物的效果可以发现）。本方案主要包括三部分：

!> 2.1Text embeddings

文本编码器首先对输入描述Prompt文本进行分词，将文本序列拆分为token序列。然后，通过查询词表将将文本进行embedding化，其中特殊字符$S_{*}$ 构建时不和词表中的词相同，作为词表中新增的一个token，并且其对应的embedding为随机初始化$v_x$
 。通过将$v_x$与描述文本一同进行编码为$c_{\theta}$ 
 ，进而"引导"生成模型生成目标内容。

!> 2.2 Latent Diffusion Models

这部分是目前Diffusion模型中应用最广泛的方式，原理参考[改善Diffusion效率问题，Latent Diffusion Model(for stable diffusion)](https://zhuanlan.zhihu.com/p/556302507)

!> 2.3 Textual inversion

该方案训练目标依然是最小化重构损失：

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p5.png" /> 
</div>

训练数据只需要3~5张目标图片，其对于的prompt文本是由模板构建，例如`“A photo of S∗”, “A rendition of S∗”`。训练GPU为`2*V100`，对剑学习率0.005，训练步数约5000左右最佳。

**3. 实验结果**

本方案从以下几个方面进行了实验效果验证：

!> 3.1 目标内容复现

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p6.png" /> 
</div><p align=center> Figure 3: Object variations generated using our method, the CLIP-based reconstruction of DALLE-2 (Ramesh et al., 2022), and human captions of varying lengths.</p>

!> 3.2 文本引导效果合成

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p7.png" /> 
</div><p align=center> Figure 4: Additional text-guided personalized generation results.</p>

!> 3.3 风格迁移

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p8.png" /> 
</div><p align=center> Figure 6: The textual-embedding space can represent more abstract concepts, including styles.</p>

!> 3.4 多内容效果组合

<div align=center>
    <img src="zh-cn/img/ch5/7-1/p9.png" /> 
</div><p align=center> Figure 7: Compositional generation using two learned pseudo-words.</p>

实践中发现该方案对于特征比较精细的内容难以学习的比较准确，例如人脸、服装等。适合做一个简单风格类或者表情这方面。


##### 3.3.2 HyperNetwork
<!-- hyperNetworks； https://zhuanlan.zhihu.com/p/632909746 -->
<!-- https://zhuanlan.zhihu.com/p/632909746 -->

!> arxiv: https://arxiv.org/pdf/1609.09106v3.pdf

!> github: https://github.com/antis0007/sd-webui-multiple-hypernetworks

**1. 简述**

超网络（HyperNetworks）是Google在16年提出的一种模型微调方案，其主要通过微调小参数模型结构去替代原大模型中的权重矩阵（这个思路和后面的LoRA的思路相，详见：最流行的训练方式Lora）。Google将该方案在深度卷积模型和长序列循环网络模型中进行验证。1. 在LSTM中实现了非共享的参数layer效果，在多种生成式下游任务中取得了SOTA的效果，例如：语言模型、机器翻译等。2. 在CNN类结构的图像识别类任务中也取得了SOTA的效果。

例如下图中，原大模型中参数$W_1$和$W_2$可以通过超网络结构$g(z^j)$ 
其中$j=1,2,...$来替换。

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p1.jpg" /> 
</div><p align=center> Figure 1: A hypernetwork generates the weights for a feedforward network.</p>

**2. 具体方案**

本文分别从两类极端模型结构上进行了超网络效果的验证，即一个是深度模型、一个是循环模型。循环模型每一步特征在参数共享的模型中进行循环传递，这样容易导致梯度消失/梯度爆炸（当然后续有改进方案，例如LSTM、GRU等）。深度卷积模型纵向layer之间不进行参数共享，使得整个模型随着深度的增加变得冗余。超网络介于两者之间，可以看作为一种弹性的参数共享。超网络在两类模型上的具体应用细节如下：

!> 2.1 静态超网络：deep CNN中的权重因式分解

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p2.png" /> 
</div><p align=center> A hypernetwork generates the weights for CNN</p>

将CNN中的卷积核$K^j$采用小参数网络$g(z^j)$进行替代，如上图所示。替代的网络模型$g(z^j)$数学表达式为

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p3.png" /> 
</div>

这样其可训练参数量包括$W_i,B_i,W_{out},B_{out}$ 
以及$z^j$,其参数量大小为$N_z\*D +D\*(N_z+1)\*N_i+f_{size}\*N_{out}\*f_{size}\*(d+1)$
 ，相比原参数要小很多（注意公式中$W_{out}$
和$B_{out}$为共享参数），文中也解释了分两层结构可以减少模型参数，使模型更加紧凑。对于不同大小的卷积核，本文将整块拆分为不同数量的相同小块，将乘操作转为加操作。文章还通过对比使用展示了超网络对原大模型权重的学习能力，例如下图：

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p4.png" /> 
</div><p align=center>Figure 2: Kernels learned by a ConvNet to classify MNIST digits (left). Kernels learned by a hypernetwork generating weights for the ConvNet (right).</p>

两者可视化展示出了主要特征的相似性，说明了超网络可以用小参数来学习大模型中参数量更大的权重。

!> 2.2 动态超网络：RNN中的自适应权重生成

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p5.png" /> 
</div><p align=center>ADAPTIVE WEIGHT GENERATION FOR RECURRENT NETWORKS</p>

本文针对RNN中的网络结构提出动态超网络，该网络中参数权重随着$t$发生变化。部分参数共享，这样成为relaxed weight-sharing，CNN与RNN中参数的特点。将超参数应用在RNN中的模型成为HyperRNN。

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p6.png" /> 
</div><p align=center>RNN与HyperRNN公式对比</p>

从上面结构图和公式对比可以看出，HyperRNN自身也形成了RNN结构，这样原大模型RNN中不同层的参数是不一样的，并且可以通过调整超参数中参数的维度使得HyperRNN的参数量远低于原大模型的参数量。详细的参数计算就不写了，看论文吧，公式太多了，敲起来手疼。

**3. 实验结果**

实验部分针对静态超网络和动态超网络分别进行了多种任务的对比实验。包括针对静态超网络的图像识别，数据集为MNIST和CIFAR-10。针对动态超网络的语言模型和手写识别任务，数据集为Penn Treebank和Hutter Prize Wikipedia。

结果：hypernetwork：99.24%， baseline:99.28%，超网络参数是原模型1/3，效果却接近。

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p7.png" /> 
</div><p align=center>Table 2: CIFAR-10 Classification with hypernetwork generated weights.</p>

结果：超网络参数是原模型1/6~1/15，效果相差2%。

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p8.png" /> 
</div><p align=center>Table 3: Bits-per-character on the Penn Treebank test set.</p>

结果：超网络参数如果和原模型相近，效果略高。

动态超网络方面的结果为：

<div align=center>
    <img src="zh-cn/img/ch5/8-1/p9.png" /> 
</div><p align=center>Table 4: Bits-per-character on the enwik8 test set.</p>

结果：超网络参数如果和原模型相近，效果略高。

##### 3.3.3 Aesthetic Gradients:Personalizing Text-to-Image Generation via Aesthetic Gradients

<!-- https://zhuanlan.zhihu.com/p/629635371 -->

!> arxiv: https://arxiv.org/abs/2209.12330

!> github: https://github.com/vicgalle/stable-diffusion-aesthetic-gradients

**1. 简述**

这篇文章是对Stable Diffusion模型进行微调的一种方案，实现图片生成风格的自定义，创建独特的美学风格。

本方案核心idea理解起来比较简单，即通过微调文本编码器（clip text encoder）将文本编码输出表示由原特征空间A投影到另外的"美学"特征空间B，进而在diffusion去噪过程中，逐步生成具备该"美学"风格的图片内容。

<div align=center>
    <img src="zh-cn/img/ch5/9-1/p1.png" /> 
</div><p align=center>原特征空间A投影到另外的美学特征空间B.</p>

由于这种特征空间的转换是在微调时通过控制clip text encoder训练时的"梯度"朝着“美学”风格方向收敛，所以该方案命名为"Aesthetic Gradients，美学梯度"模型定制化方案。

下面是文中的两种微调后的美学风格生成的效果，两种美学风格分别是基于SAC8+和LAION7+数据训练的，效果如下图所示：

<div align=center>
    <img src="zh-cn/img/ch5/9-1/p2.png" /> 
</div><p align=center>Figure 1: Stable diffusion generations for the original model and personalized variants using SAC8+ and LAION7+ aesthetic embeddings.</p>

在保持主要内容和整体构图的情况下，画面的风格确实发生了明显的变化。

**2. 具体方案**

<div align=center>
    <img src="zh-cn/img/ch5/9-1/p3.png" /> 
</div><p align=center>Stable Diffusion的模型结构</p>

Stable Diffusion模型生成图片的主要流程是：通过文本编码器（Text Encoder），常用的有Bert、T5、CLIP等，这里实验CLIP模型中的文本编码器部分，将描述文本Prompt进行编码，然后作为Condition插入到Unet模型中，去引导模型一步步生成目标图片。

<div align=center>
    <img src="zh-cn/img/ch5/9-1/p4.png" /> 
</div>

以上就是本文方案的主要思路了。

**3. 实验结果**

本文构建了两类美学数据集$SAC_{8+}$和$LION_{7+}$
 ，分别训练CLIP中text encoder，并且说明该方案具备泛化性，构建了25个不同复杂长度的prompt文本。下面展示部分效果图，完整的结果就去看论文吧。

 <div align=center>
    <img src="zh-cn/img/ch5/9-1/p5.png" /> 
</div><p align=center>Figure 3: Further qualitative results using different aesthetic embeddings.（节选）</p>

**4. 写在最后**

该方案主要是进行风格训练，且需要具备相同美学特征的图构建为美学数据集，所以这个数据集中包含的图片内容需要较为全面，这样得到的e特征向量才是位于比较有明显特定美学风格特征的特征空间中心位置。该类风格效果其实比较明显，并且前后的构图基本相似，代码开源了对其细节感兴趣的就去拆代码吧。

 
#### 3.4 LyCORIS (LoCon和LoHa)
<!--  https://zhuanlan.zhihu.com/p/631370055
https://www.bilibili.com/read/cv23398173/
 -->
<!-- https://www.dongwm.com/post/stable-diffusion-models/ -->
<!-- http://www.dtmao.cc/python/104959.html -->

LyCORIS (Lora beYond Conventional methods) 是最近开始流行的一种新的模型，如其名字是一种超越传统方法的 Lora，但是要比 LoRA 能够微调的层级多，它的前身是 LoCon (LoRA for convolution layer)。

LoCon 和 LoHA (LoRA with Hadamard Product representation) 都是 LyCORIS 的模型算法，如果 C 站模型下载页面如果明确说是 LoCon 那就是 LoHA

现在 stable-diffusion-webui 还没有自带它，所以需要先安装扩展:https://github.com/KohakuBlueleaf/a1111-sd-webui-lycoris。 

**1.LoCon：Lora for convolution**

+ 将Lora扩展到卷积层。
+ 方法：
    - 在代码上只是将低了卷积的输出通道数
    - 理论：将一个卷积核（一个通道）展开，可以看成是参数矩阵的一列。将所有卷积核按列排布可以得到类似于transformer中的参数矩阵，即卷积操作也是矩阵相乘。通过降低channel数再提高channel数实现降低参数量的目的
    
 <div align=center>
    <img src="zh-cn/img/ch5/10-1/p1.png" /> 
</div>

**2.LoHa：LoRA with Hadamard Product representation**

+ 对Lora的改进，将hadamard product(矩阵对应元素相乘，区别矩阵点乘)应用到矩阵低秩分解中。

<div align=center>
    <img src="zh-cn/img/ch5/10-1/p2.png" /> 
</div>

传统的低秩分解算法，需要保证分解后的秩的维度小于2R，而通过LoHa的进一步拆解，使得矩阵的秩扩展到 $R^2$,解决了原生LoRA受到低秩的限制。

+ 总结
    - LoCon和RoHa都能实现更细粒度的微调。LoCon可以对实现更细粒度的控制，从全图的调整优化为细粒度的部件调整。RoHa更注重于低秩矩阵分解本身，引入Hadamard Product，将秩的维度 从2R扩展到 $R^2$。这两个插件都包含在LyCORIS库
    - Lora可以和其他微调方法一起使用以降低微调参数量，常用的是和DreamBooth一起降低参数量
这些方法需要的数据量都较小。一般来讲，微调的参数量越多，需要的数据量也越大（DreamBooth例外）
一般而言，数据越多，效果越好

<div align=center>
    <img src="zh-cn/img/ch5/10-1/p3.png" /> 
</div>

#### 3.5 基于太乙stable diffusion的医疗文生图微调及应用

<!-- https://www.zhihu.com/tardis/zm/art/641325338?source_id=1005 -->
<!-- https://github.com/IDEA-CCNL/Fengshenbang-LM#%E5%AE%89%E8%A3%85 -->

<!-- https://17yongai.com/1624.html -->

<!-- https://github.com/IDEA-CCNL/Fengshenbang-LM/issues/346 -->

<!-- https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/stable_diffusion_chinese/taiyi_handbook.md -->

<!-- https://docs.qq.com/doc/DWklwWkVvSFVwUE9Q -->

太乙stable diffusion model是首个中文stable diffusion模型，开发团队是：DEA 研究院封神榜团队在过去快速积累的基础上，其技术文档如下：

<object data="zh-cn/img/ch5/11-1/太乙绘画使用手册1.1.pdf" type="application/pdf" width="100%" height="650px">
<!--     <embed src="http://www.africau.edu/images/default/sample.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="http://www.africau.edu/images/default/sample.pdf">Download PDF</a>.</p>
    </embed> -->
</object>


!> 构建自己的数据集

我们想让其生成一些消化内镜的图像，使用文生图的方式实现，输入文本prompt,生成$512 \times 512$的内镜图像，其数据集的构建如下图所示：

<div align=center>
    <img src="zh-cn/img/ch5/11-1/p1.jpg" /> 
</div>

!> 环境安装

我们使用NGC上的Pytorch的镜像，镜像的版本为：`nvcr.io/nvidia/pytorch:23.03-py3`

通过如下命令进入容器：

```sh
nvidia-docker run -it -v /home/myuser/LLMs/Fengshenbang-LM-main:/workspace/diffusion -p 5055:5055 -p 5056:5056 -p 5067:5057 nvcr.io/nvidia/pytorch:23.03-py3
```

安装需要的package

```sh
torch==2.0.1+cu118
torchvision==0.15.2+cu118
xformers== 0.0.20
diffusers==0.18.1
transformers==4.30.2
```

注意: `pip uninstall transformer_engine`,否则会报错!

安装`fengshenbang`

```sh
cd diffusion
pip install --editable .
```

!> finetune

上面章节我们详细介绍了stable diffusion中的微调方法包括DreamBooth,LoRA,textual inversion,hypernetwork,Aesthetic Gradients和LyCORIS等，下面我们基于消化内镜图像进行微调太乙stable diffusion model.

下载预训练的模型：https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1

```sh
cd Fengshenbang-LM-main/fengshen/examples/finetune_taiyi_stable_diffusion
```


修改`finetune.sh`文件

```sh
#!/bin/bash
#SBATCH --job-name=finetune_taiyi # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=8 # number of tasks to run per node
#SBATCH --cpus-per-task=30 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8 # number of gpus per node
#SBATCH -o %x-%j.log # output and error log file names (%x for job id)
#SBATCH -x dgx050

# pwd=Fengshenbang-LM/fengshen/examples/pretrain_erlangshen
# ROOT_DIR=../../workspace
ROOT_DIR=./


export TORCH_EXTENSIONS_DIR=${ROOT_DIR}/torch_extendsions

MODEL_NAME=taiyi-stablediffusion-1B
MODEL_ROOT_DIR=$ROOT_DIR/${MODEL_NAME}
if [ ! -d ${MODEL_ROOT_DIR} ];then
  mkdir ${MODEL_ROOT_DIR}
fi

NNODES=1
GPUS_PER_NODE=1

MICRO_BATCH_SIZE=1

# 如果你不用Deepspeed的话 下面的一段话都可以删掉 Begin
CONFIG_JSON="$MODEL_ROOT_DIR/${MODEL_NAME}.ds_config.json"
ZERO_STAGE=1
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $CONFIG_JSON
{
    "zero_optimization": {
        "stage": ${ZERO_STAGE}
    },
    "bf16": {
        "enabled": false
    },
    "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE
}
EOT
export PL_DEEPSPEED_CONFIG_PATH=$CONFIG_JSON
### End

DATA_ARGS="\
        --dataloader_workers 1 \
        --train_batchsize $MICRO_BATCH_SIZE  \
        --val_batchsize $MICRO_BATCH_SIZE \
        --test_batchsize $MICRO_BATCH_SIZE  \
        --datasets_path ./mc_dataset \
        --datasets_type txt \
        --resolution 512 \
        "

MODEL_ARGS="\
        --model_path pretrain/Taiyi-Stable-Diffusion-1B-Chinese-v0.1 \
        --learning_rate 1e-4 \
        --weight_decay 1e-1 \
        --warmup_ratio 0.01 \
        "

MODEL_CHECKPOINT_ARGS="\
        --save_last \
        --save_ckpt_path ${MODEL_ROOT_DIR}/ckpt \
        --load_ckpt_path ${MODEL_ROOT_DIR}/ckpt/last.ckpt \
        "

TRAINER_ARGS="\
        --max_epoch 10 \
        --gpus $GPUS_PER_NODE \
        --num_nodes $NNODES \
        --strategy deepspeed_stage_${ZERO_STAGE} \
        --log_every_n_steps 100 \
        --precision 32 \
        --default_root_dir ${MODEL_ROOT_DIR} \
        --replace_sampler_ddp False \
        --num_sanity_val_steps 0 \
        --limit_val_batches 0 \
        "
# num_sanity_val_steps， limit_val_batches 通过这俩参数把validation关了

export options=" \
        $DATA_ARGS \
        $MODEL_ARGS \
        $MODEL_CHECKPOINT_ARGS \
        $TRAINER_ARGS \
        "

python3 finetune.py $options
#srun -N $NNODES --gres=gpu:$GPUS_PER_NODE --ntasks-per-node=$GPUS_PER_NODE --cpus-per-task=20 python3 pretrain_deberta.py $options

```


开始微调：

```sh
./finetune.sh
```

<div align=center>
    <img src="zh-cn/img/ch5/11-1/p1.png" /> 
</div>

显示log:

```
tensorboard --logdir=version_0/  --port=5055
```

<div align=center>
    <img src="zh-cn/img/ch5/11-1/p2.png" /> 
</div>


!> 基于gradio的demo开发测试

```python
from diffusers import DiffusionPipeline
import gradio as gr
# 可以在代码中快速关闭NSFW（Not safe for work）检测：https://borrowastep.net/p/-stablediffusion-nsfw--8avcvhpmu
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py
pipeline = DiffusionPipeline.from_pretrained("./taiyi-stablediffusion-1B/hf_out_9_14780")
pipeline = pipeline.to("cuda")

def generate(text, steps):
    image = pipeline(text,num_inference_steps=steps,guidance_scale=7.5).images[0]
    return image

if __name__ == "__main__":

    demo = gr.Interface(title="太乙中文 stable diffusion 模型 微调生成内镜图像",
        css="",
        fn=generate,
        inputs=[gr.Textbox(lines=3, placeholder="输入你想生成的图片描述", label="prompt"),gr.Slider(maximum=100, value=50, minimum=1, label='Step Time')],
        outputs=[gr.outputs.Image(label="图片",type="pil")]
        )

    demo.launch(server_name="0.0.0.0",server_port=5056)
```

<div align=center>
    <img src="zh-cn/img/ch5/11-1/p3.png" /> 
</div>

生成的效果还不错！！！

#### 3.6 ControlNet

<!-- https://zhuanlan.zhihu.com/p/609075353 -->

<!-- https://juejin.cn/post/7210369671656505399 -->

!> github: https://github.com/lllyasviel/ControlNet

!> arxiv: https://arxiv.org/abs/2302.05543v1


**1.ControlNet是干嘛的**

我们知道现在文本到图像生成很火爆，你只需要输入文字就可以获得对应的输出，这个任务的发展多亏了扩散模型的发展。

而我们今天要说的ControlNet的作用就是能够控制扩散模型，生成更接近用户需求的图。

我们用几个图示范一下：
假如我们只用文字去生成图像，我们给模型输入文本prompt：`A girl, long hair, blond hair, black eyes, black coat, winter, snow。`

那模型会出来这些图：

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p1.png" /> 
</div>

我们可以看到，我们输入的要素都符合了。但是差别又很大，大在哪里？每个妹子动作差异都好大。
那现在我们就可以掏出ControlNet了，用ControlNet来控制你生成的人物的动作。
比如下图这样，可以看到即使我换了画风（换了模型的checkpoint）生成的人物动作也是一样的， 并且满足我前边输入的条件。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p2.png" /> 
</div>

那我做了什么才让模型能稳定输出相同的人物动作呢？下面请出我们的仙女姐姐[尚尚](https://juejin.cn/user/105206371061245)：

为我们生成的妹子提供动作的是我们的尚影嫣小姐姐，我把她的照片放到ControlNet中作为控制条件，这里使用的是ControlNet的Depth功能，ControlNet会根据小姐姐提取一个深度图，然后按照人物深度图轮廓去生成我们的人物。

下图你看到的就是文本prompt控制和ControlNet图像条件控制相互作用的结果

既符合文本prompt：`a girl, long hair, blond hair, black eyes, black coat, winter, snow`，又保持了小姐姐的大致动作形态。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p3.png" /> 
</div>

当然，如果我们把文字控制删除一点，只留下`a girl, long hair, black eyes`，我们可以得到更接近小姐姐原图的结果。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p4.png" /> 
</div>

好了，美女看完了，那我们就介绍一下这个论文吧。


**2.ControlNet 网络设计**

在一个扩散模型中，如果不加ControlNet的扩散模型，其中原始的神经网络
$F$输入$x$获得$y$，参数$\theta$表示。
$$y=f(x;\theta)$$

也就是下图这个样子。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p5.png" /> 
</div>

ControlNet中，就是将模型原始的神经网络锁定，设为**locked copy**。
然后将原始网络的模型复制一份，称之为**trainable copy**，在其上进行操作施加控制条件。然后将施加控制条件之后的结果和原来模型的结果相加获得最终的输出。
经过这么一顿操作之后，施加控制条件之后，最后将原始网络的输出修改为：

$$y_c=F(x;\theta)+ZF(x+Z(c;\theta_{z1});\theta_c);\theta_{z2})$$

其中zero convolution，也就是零卷积层$Z$是初始化weight和bias为0，两层零卷积的参数为$\theta_{z1},\theta_{z2}$.

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p6.png" /> 
</div>

将控制条件通过零卷积之后，与原始输入相加，相加之后进入ControlNet的复制神经网络块中，将网络输出再做一次零卷积之后与原始网络的输出相加。

初始化之后未经训练的ControlNet参数应该是这样的：

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p7.png" /> 
</div>

也就是说ControlNet未经训练的时候，输出为0，那加到原始网络上的数字也是0。这样对原始网络是没有任何影响的，就能确保原网络的性能得以完整保存。之后ControlNet训练也只是在原始网络上进行优化，这样可以认为和微调网络是一样的。


**3.ControlNet in Stable Diffusion**

上一部分描述了ControlNet如何控制单个神经网络块，论文中作者是以Stable Diffusion为例子，讲了如何使用ControlNet对大型网络进行控制。下图可以看到控制Stable Diffusion的过程就是将Encoder复制训练，decoder部分进行skip connection。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p8.png" /> 
</div>

在这之前需要注意：

Stable Diffusion有一个预处理步骤，将512×512的图片转化为64×64的图之后再进行训练，因此为了保证将控制条件也转化到64×64的条件空间上，训练过程中添加了一个小网络$E$将图像空间条件转化为特征图条件。
$$c_f=E(c_i)$$

这个网络$E$是四层卷积神经网络，卷积核为4×4，步长为2，通道16，32，64，128，初始化为高斯权重。这个网络训练过程是和整个ControlNet进行联合训练。或者我们可以把他的图改吧改吧，画成这样：

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p9.png" /> 
</div>

**4.训练过程**

训练的目标函数为：

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p10.png" /> 
</div>

使用的就是人家Stable Diffusion原始的目标函数改了改。

先看一下原始的Stable Diffusion的目标函数：


<div align=center>
    <img src="zh-cn/img/ch5/12-1/p11.png" /> 
</div>

将采样$z_t$使用网络$ϵ_θ$去噪之后和原图经过网络$ϵ$获得的潜变量计算$L_2$ loss，看其重建的效果。

那再回到

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p12.png" /> 
</div>

将原始图像经过$\epsilon$之后获得潜变量，和经过网络$\epsilon_\theta$重建之后的图算$L_2$ loss。原来Stable Diffusion中解码器要处理的是采样$z_t$和时间步长$t$，在这里加了两个控制条件：
+ 文字prompt $c_t$
+ 任务相关的prompt $c_f$

训练过程中将50 %的文本提示$c_t$随机替换为空字符串。这样有利于ControlNet网络从控制条件中识别语义内容。这样做的目的是，当Stable Diffusion没有prompt的时候，编码器能从输入的控制条件中获得更多的语义来代替prompt。（这也就是classifier-free guidance。）

**5.效果！**

这一部分作者主要是讲了如何训练不同控制条件的ControlNet的，训练方法感兴趣的自己看，这里简单展示一下作者提供好的训练出来的模型。用《青蛇劫起》里边小青做一下示范：

> Canny Edge

使用Canny边缘检测生成边缘线稿，再将作为扩散模型输入。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p13.png" /> 
</div>

> HED

使用hed边界检测。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p14.png" /> 
</div>

> Depth

使用深度图生成。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p15.png" /> 
</div>

> Normal Maps

使用法线图生成图像。提供了Midas计算深度图并转换为法线图的扩展版本的模型。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p16.png" /> 
</div>

> Human Pose

使用姿势检测，获得人体骨骼的可视化姿势图像。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p17.png" /> 
</div>

> User Sketching

使用人类涂鸦进行生成。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p18.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p19.png" /> 
</div>

> Semantic Segmentation

使用语义分割。

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p20.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p21.png" /> 
</div>

> Hough Line

使用m-lsd直线检测算法。（论文中还提到了使用传统的霍夫变换直线检测）

<div align=center>
    <img src="zh-cn/img/ch5/12-1/p22.png" /> 
</div>

> 其他

论文中还提到了其他的，比如动漫线稿之类的，但是没有提供对应的模型，所以这里无法展示，感兴趣的可以自己去看一下论文。