<!-- chatGLM,MINIGPT-4, LLaMa, Firefly,BiLLa,stable-vicuna-13B,alpaca-13b,opt,本草，langChhian(guidance),Alpaca_LoRa,Falcon-40B,CaMa,guanaco  baichuan大模型-->

<!-- chatGLM,MINIGPT-4, LLaMa, alpaca，本草，langChian-->
<!-- 下载GPT-4, InstrcutGPT,还有这些大模型的paper -->

<!-- DocterGLM https://zhuanlan.zhihu.com/p/627269023 -->


## 1.GLM

<!-- https://zhuanlan.zhihu.com/p/538866002 -->

<!-- https://www.bilibili.com/video/BV1M84y1y7yu/?spm_id_from=333.337.search-card.all.click&vd_source=0e2d47c0c67fb509b32ba3bfc5b73819 -->

<!-- https://www.bilibili.com/video/BV14L411q7fk/?vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->
<!-- https://www.bilibili.com/video/BV1fd4y1Z7Y5/?vd_source=def8c63d9c5f9bf987870bf827bfcb3d -->

+ 研究背景：没有一种预训练框架在自然语言理解（NLU）、无条件生成和条件生成三大类的所有任务中表现最好
+ 研究方法：应用自回归空白填充的通用语言模型，通过添加2D位置编码和允许任意顺序预测跨距来改进空白填充预训练，同时，通过改变空白的数量和长度，可以针对不同类型的任务进行预训练。

### 0.摘要

目前，已经有各种类型的预训练架构，包括自动编码模型（例如，BERT）、自回归模型（例如，GPT）和编码器-解码器模型（例如，T5）。然而，没有一种预训练框架在自然语言理解（NLU）、无条件生成和条件生成三大类的所有任务中表现最好。我们提出了一种基于自回归空白填充的通用语言模型来应对这一挑战。GLM通过添加2D位置编码和允许任意顺序预测跨距来改进空白填充预训练，从而在非线性规划任务上比BERT和T5获得性能增益。同时，通过改变空白的数量和长度，GLM可以针对不同类型的任务进行预训练。在跨非线性规划的广泛任务、条件生成和无条件生成上，GLM在相同模型大小和数据的情况下优于BERT、T5和GPT，并且在具有1.25×BERTLarge参数的单个预训练模型中实现了最佳性能，证明了其可推广到不同的下游任务。


### 1.引言

在未标记文本上预训练的语言模型在各种NLP任务（从自然语言理解（NLU）到文本生成）中极大地提高了现有技术水平（Radford等人，2018a；德夫林等人，2019；Yang等人，2019；Radford等人，2018b；Raffel等人，2020；刘易斯等人，2019；Brown等人，2020）。在过去几年中，下游任务性能以及参数的规模也在不断增加。

一般来说，现有的预训练框架可以分为三类：自回归、自动编码和编解码器模型。自回归模型，如GPT（Radford等人，2018a），学习从左到右的语言模型。虽然它们在长文本生成方面取得了成功，并且在扩展到数十亿个参数时显示出较少的学习能力（Radford等人，2018b；Brown等人，2020），但其固有的缺点是单向注意机制，无法完全捕捉自然语言任务中上下文词之间的依赖关系。自动编码模型，如BERT（Devlin等人，2019），通过去噪目标学习双向上下文编码器，例如掩码语言模型（MLM）。编码器产生适合自然语言理解任务的语境化表示，但不能直接用于文本生成。编码器-解码器模型采用编码器的双向注意、解码器的单向注意以及它们之间的交叉注意（Song等人，2019；Bi等人，2020；Lewis等人，2019）。它们通常部署在条件生成任务中，例如文本摘要和响应生成。T5（Raffel等人，2020）通过编码器-解码器模型统一了NLU和条件生成，但需要更多参数来匹配基于BRET的模型的性能，如RoBERTa（Liu等人，2019）和DeBERTa（He等人，2021）。

这些预训练框架都不够灵活，无法在所有 NLP 任务中竞争地执行。以前的工作试图通过多任务学习结合他们的目标来统一不同的框架（Dong et al., 2019; Bao et al., 2020）。然而，由于自编码和自回归目标本质上是不同的，简单的统一并不能完全继承这两个框架的优点。

<div align=center>
    <img src="zh-cn/img/ch2/p1.png" /> 
</div>

在本文中，我们提出了一个基于自回归空白填充的预训练框架GLM（通用语言模型）。我们根据自动编码的思想，从输入文本中随机清空令牌的连续跨度，并根据自回归预训练的思想，训练模型以顺序重建跨度（见图1）。虽然在T5（Raffel等人，2020）中使用了空白填充进行文本到文本预训练，但我们提出了两个改进，即跨度无序和2D位置编码。从经验上看，我们表明，在相同的参数和计算成本下，GLM在SuperGLUE基准上显著优于BERT，幅度为4.6%-5.0%，在类似大小的语料库（158GB）上进行预训练时，GLM优于RoBERTa和BART。GLM在参数和数据较少的非线性规划和生成任务上也显著优于T5。

受模式开发训练 (PET) (Schick 和 Schütze, 2020a) 的启发，我们将 NLU 任务重新表述为模仿人类语言的手动完形填空问题。与 PET 使用的基于 BERT 的模型不同，GLM 可以通过自回归填空自然地处理完形填空问题的多标记答案。

此外，我们表明，通过改变缺失跨度的数量和长度，自回归空白填充目标可以预训练有条件和无条件生成的语言模型。通过对不同预训练目标的多任务学习，单个 GLM 可以在 NLU 和（有条件和无条件）文本生成方面表现出色。根据经验，与独立基线相比，具有多任务预训练的 GLM 通过共享参数实现了 NLU、条件文本生成和语言建模任务的改进。

### 2. GLM 预训练框架

我们提出了一种基于自回归空白填充目标的通用预训练框架GLM。GLM将NLU任务表述为完形填空问题，其中包含任务描述，可以通过自回归生成来回答。

#### 2.1 预训练目标

##### 2.1.1 自回归空白填充

GLM通过优化自回归空白填充目标来训练。给定输入文本$x=[x_1，···，x_n]$，对多个文本span $[s_1,···,s_m]$进行采样，其中每个span $s_i$对应于$x$中的一系列连续tokens$[s_{i,1},···,s_{i,l_i}]$。每个span用单个`[MASK]`token替换，形成损坏的文本$x_{corrupt}$。该模型以自回归的方式从损坏的文本中预测span中缺失的token，这意味着在预测span中缺失的token时，该模型可以访问损坏的文本和先前预测的span。为了充分捕捉不同span之间的相互依存关系，我们随机排列span的顺序，类似于排列语言模型（Yang等人，2019）。形式上，设$Z_m$是长度为$m$的索引序列`[1,2,···,m]`的所有可能置换的集合，并且$s_{z<i }$ 是$[s_{z_1},···,s_{z_{i−1}}]$ ，我们将预训练目标定义为

<div align=center>
    <img src="zh-cn/img/ch2/p2.png" /> 
</div>

我们总是按照从左到右的顺序在每个空白中生成标记，即生成span $s_i$ 的概率被分解为：

<div align=center>
    <img src="zh-cn/img/ch2/p3.png" /> 
</div>

我们使用以下技术实现自回归空白填充目标。输入$X$分为两部分：A部分是损坏的文本$x_{corrupt}$，B部分由屏蔽的span组成。部分A标记可以相互注意，但不能注意B中的任何标记。部分B标记可以处理部分A和B中的先行标记，但不能处理B中的任何后续标记。为了启用自回归生成，每个span都填充了特殊标记`[start]`和`[end]`，用于输入和对应输出。在这种条件下，我们的模型在统一模型中自动学习双向编码器（用于 A 部分）和单向解码器（用于 B 部分）。 GLM 的实现如图 2 所示。

<div align=center>
    <img src="zh-cn/img/ch2/p4.png" /> 
</div>

我们用$\lambda=3$随机抽样从泊松分布中提取的片段。我们重复的采样片段，直到源文本的15% token被mask,根据经验，我们发现15%的比率对于下游自然语言理解任务的良好性能至关重要。

##### 2.1.2 多任务训练

在上一节中，GLM屏蔽了短span，适用于NLU任务。然而，我们感兴趣的是预训练一个可以同时处理自然语言理解和文本生成的单一模型。然后，我们研究了多任务预训练设置，其中生成较长文本的第二个目标与空白填充目标联合优化。我们考虑以下两个目标：

+ 文档级。我们对单个span进行采样，其长度从原始长度的50%–100%的均匀分布中采样。目标是生成长文本。
+ 句子级别。我们限制掩盖的span必须是完整的句子。多个span (句子) 被采样以覆盖原始token的15%。此目标针对seq2seq任务，其预测通常是完整的句子或段落。

两个新目标的定义方式与原始目标相同，即等式1。唯一的区别是跨距的数量和跨距长度。

#### 2.2 模型架构

GLM使用单个transformer，并对架构进行了几次修改：（1）我们重新安排了Layer Norm和Residual connection的顺序，这对于大规模语言模型来说至关重要，以避免数值错误（Shoeybi等人，2019）；（2） 我们使用单个线性层进行输出token预测；（3） 我们将ReLU激活函数替换为GeLUs（Hendrycks和Gimpel，2016）。

##### 2.2.1 2D位置编码

自回归空白填充任务的挑战之一是如何对位置信息进行编码。 Transformers 依靠位置编码来注入标记的绝对和相对位置。我们提出 2D 位置编码来应对挑战。具体来说，每个token都使用两个位置 ID 进行编码。第一个位置 id 表示损坏文本 $x_{corrupt}$中的位置。对于被mask的span，它是相应的 `[MASK]` 标记的位置。第二个位置 id 表示span内位置。对于 A 部分中的标记，它们的第二个位置 id 为 0。对于 B 部分中的标记，它们的范围从 1 到跨度的长度。这两个位置 id 通过可学习的嵌入表投影到两个向量中，这两个向量都添加到输入token嵌入中。

我们的编码方法确保模型在重建mask span时不知道它们的长度。 与其他模型相比，这是一个重要的区别。 例如，XLNet (Yang et al., 2019) 对原始位置进行编码，使其能够感知丢失的token数量，而 SpanBERT (Joshi et al., 2020) 将span替换为多个 `[MASK]` token并保持长度 不变。 我们的设计适合下游任务，因为生成文本的长度通常是事先未知的。

### 2.3 微调GLM

通常，对于下游NLU任务，线性分类器将预训练模型生成的序列或token的表示作为输入，并预测正确的标签。这些实践不同于生成性预训练任务，导致预训练和微调之间不一致。

相反，我们将非线性规划分类任务重新表述为空白填充的生成任务，遵循PET（Schick和Schütze，2020a）。具体来说，给定一个标记的示例$(x,y)$，我们通过包含单个掩码标记的模式将输入文本$x$转换为完形填空问题$c(x)$。该模式是用自然语言编写的，用于表示任务的语义。例如，情绪分类任务可以表述为`“{senture}。它真的是[掩码]”`。候选标记为$y \in Y$也是映射到完形填空的答案，称为语言表达$v(Y)$。在情感分类中，标签“积极”和“消极”映射到单词“好”和“坏”。给定$x$，预测$y$的条件概率为

<div align=center>
    <img src="zh-cn/img/ch2/p5.png" /> 
</div>

其中$Y$是标签集。因此，句子正或负的概率与预测空白中的“好”或“坏”成正比。然后，我们使用交叉熵损失对GLM进行微调（见图3）

<div align=center>
    <img src="zh-cn/img/ch2/p6.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch2/p7.png" /> 
</div>

对于文本生成任务，给定的上下文构成输入的A部分，并在末尾附加掩码标记。该模型自回归地生成B部分的文本。我们可以直接将预训练的GLM应用于无条件生成，或者在下游条件生成任务上对其进行微调。

<div align=center>
    <img src="zh-cn/img/ch2/p8.png" /> 
</div>

#### 2.4 讨论和分析

在本节中，我们将讨论GLM与其他预训练模型之间的差异。我们主要关心的是它们如何适应下游的空白填充任务。

与BERT的比较（Devlin等人，2019年）。正如（Yang等人，2019）所指出的，由于MLM的独立性假设，BERT未能捕捉到掩码token的相互依赖性。BERT的另一个缺点是无法正确填充多个token的空白。为了推断长度为$l$的答案的概率，BERT需要执行$l$个连续预测。如果长度$l$未知，我们可能需要枚举所有可能的长度，因为BERT需要根据长度更改[掩码]令牌的数量。

与XLNet的比较（Yang等人，2019）。GLM和XLNet都是用自回归目标进行预训练的，但它们之间有两个不同之处。首先，XLNet使用损坏前的原始位置编码。在推理过程中，我们需要知道或列举答案的长度，这与Bert的问题相同。其次，XLNet使用双流自注意机制，而不是右移，以避免Transformer内部的信息泄漏。它使预培训的时间成本增加了一倍。

与T5比较（Raffel等人，2020）。T5提出了类似的空白填充目标，以预训练编码器-解码器转换器。T5对编码器和解码器使用独立的位置编码，并且依赖于多个标记令牌来区分span的跨度。在下游任务中，仅使用一个哨兵标记，导致模型容量的浪费以及预训练和微调之间的不一致。此外，T5总是以固定的从左到右的顺序预测跨度。因此，如第3.2节和第3.3节所述，在参数和数据较少的NLU和SEQ2SEQ任务中，GLM可以显著优于T5。

与 UniLM 的比较（Dong 等人，2019 年）。 UniLM 通过在双向、单向和交叉注意力之间改变注意力掩码，在自动编码框架下结合不同的预训练目标。但是，UniLM 总是用 `[MASK]` 标记替换掩码span，这限制了它对掩码span及其上下文之间的依赖关系进行建模的能力。 GLM 输入前一个令牌并自回归生成下一个令牌。在下游生成任务上微调 UniLM 也依赖于效率较低的掩码语言建模。 UniLMv2 (Bao et al., 2020) 对生成任务采用部分自回归建模，同时对 NLU 任务采用自动编码目标。相反，GLM 将 NLU 和生成任务与自回归预训练相结合。

### 3. 实验

实验部分感兴趣的可以阅读原论文，这里就不再过多介绍。

### 总结

GLM是自然语言理解和生成的一般预训练框架。我们证明了非线性规划任务可以表示为条件生成任务，因此可以通过自回归模型求解。GLM将不同任务的预训练目标统一为自回归空白填充，使用混合注意力掩码和新颖的2D位置编码。实证结果表明，GLM优于以往的非线性规划任务方法，可以有效地共享不同任务的参数。


------
------

## 2.Prefic tuning, P-tuning and P-tunng v2,Prompt-tuning,Adapter-Tuning


### 1.Prefix tuning

Fine-tuning是使用大规模预训练语言模型来进行下游任务的流行范式，但需要更新和存储语言模型的全部参数。再运用到下游任务时，需要对每一个任务都需要存储一份修改后的参数。Lightweight fine-tuning是尝试解决上述问题的方法，Lightweight fine-tuning固定住绝大部分的预训练参数，修改预训练模型的小部分模块。但是该方法最困难的地方是识别模块中高表现的部分以其需要调节的预训练参数。另一种比较比较流行的方法是Prompting, Prompting在模型的输入前加上instructions和一些样本使模型输出任务需要的结果。

作者在论文中提出了prefix-tuning, 在调节模型的过程中只优化一小段continuous task-specific vector（prefix）。

1. 与fine tuning相比，只需要存储一份大型Transformer的拷贝以及一个可学习的task-specific prefix, 对于不同的任务只需要不同的prefix. 在完整的数据集上，prefix-tunning和fine-tuning在table-to-text上的结果是comparable的，而在summarization任务上，prefix-tuning的效果略有下降。但在low-data settings和unseen topics的情况下，prefix-tuning的效果更佳。
2. 与Lightweight fine-tuning相比，prefix tuning在相同的表现下只需调节更少的参数量。
3. 与prompting相比，prefix包含的完全是自由的参数，并不用和real tokens相对应。

#### 1.1 Problem Statement

对于conditional generation任务，其输入是context x, 输出是sequence of tokens y. 在论文中主要关注table-to-text和summarization任务。

<div align=center>
    <img src="zh-cn/img/ch2/2-1/p1.png" /> 
</div>

#### 1.2 Autoregressive LM

在上图中（top）,$z=[x,y]$是$x$和$y$的拼接,$X_{idx}$和$Y_{idx}$表示序列的indices. $h_i \in R^d$表示在每个time step $i$下的激活值。 
$h_i=[h_i^{(1)},...,h_i^{(n)}]$表示在当前time step的所有激活层的拼接，$h_i^{(j)}$是在time step $i$的第$j$层Transformer layer的激活层。autoregressive Transformer模型计算$h_i$:

$$h_i=LM_{\Phi}(z_i,h_{<i })$$

$h_i$ 的最后一层用来计算下一个token的分布：

$$p_{\Phi}(z_{i+1}|h_{<=i})=softmax(W_{\Phi}h_i^{(n)})$$

#### 1.3 Encoder-Decoder Architecture

对于摘要任务，使用encoder-decoder结构去建模 $p_{\Phi}(y|x)$
 , 其中$x$由双向的encoder编码，而decoder预测$y$。 
$h_i(i \in X_{idx})$由双向Transformer encoder计算得出， 
$h_i(i \in Y_{idx})$由autoregressive decoder计算出。

#### 1.4 Method: Fine-tuning

在fine-tuning框架下， $p_{\Phi}$是可训练的语言模型，$\Phi$是预训练参数。在如下log-likelihood objective上进行梯度更新

$$max_{\Phi}log_{\Phi}(y|x)=\sum_{i \in Y_{idx}}logp_{\Phi}(z_i|h_{<i})$$


#### 1.5 Prefix-Tuning

**Method**: Prefix-tuning在autoregressive LM前添加prefix获得 $z=[PREFIX;x;y]$;或者在encoder和decoder之前添加prefixs获得$z=[PREFIX;x;PREFIX^{'};y]$.如上图所示，$P_{idx}$表示prefix indices序列；$|P_{idx}|$表示prefix的长度。Prefix-tuning通过**初始化可训练矩阵**
$P_{\theta}$维度为$|P_{idx}\times dim(h_i)|$来存储prefix参数：

<div align=center>
    <img src="zh-cn/img/ch2/2-1/p2.png" /> 
</div>

training objective与Fine-tuning相同,但语言模型的参数$\Phi$固定，仅仅prefix参数$\theta$是可训练参数。因此$h_i$是可训练的$P_{\theta}$的函数，当$i\in P_{idx}$时，$h_i$由$P_{\theta}$直接复制得到，对于$i \notin P_{idx}$,由于prefix activations始终在left context因此可以影响到$h_i$

**Parametrization of $P_{\theta}$**:

在实验上，直接更新$P_{\theta}$的参数会导致优化的不稳定以及表现上的极具下降。因此通过使用较小的矩阵$P_{\theta}^{'}$ 通过大型前馈神经网络( 
$MLP_{\theta}$)来reparametrize矩阵$P_{\theta}$:

$$P_{\theta}[i,:]=MLP_{\theta}(P^{'}_{\theta} [i,：])$$


#### 1.6 Experiments & Results

数据集和实验设置详见论文。实现表明Prefix-tuning能用更少的参数达到较有竞争力的结果。在Low-data阶段（训练样本数较少）， prefix-tuning相比较fine-tuning更有优势。在Extrapolation方面，prefix-tuning也比fine-tuning的表现更好。

------


### 2.P-tuning
<!-- https://zhuanlan.zhihu.com/p/552161745 -->
<!-- https://kexue.fm/archives/8295 -->
<!-- https://mp.weixin.qq.com/s/YbEFosFy870XQckGeRDKvQ -->

#### 2.1 问题描述

GPT使用传统的微调策略没法在自然语言理解任务上取得较好结果，而通过新的调整策略，可以让大小相似的GPT在NLU上取得比BERT相近或更好的结果。

<div align=center>
    <img src="zh-cn/img/ch2/2-2/p1.png" /> 
</div>

#### 2.2 方法细节

+ 思路

1. 自动化地寻找连续空间中的知识模板；训练知识模板，但不fine-tune语言模型。
2. 给定一个prompt $T=[[P_{0:i}],x,[P_{i+1:m}]]$,传统的离散prompt模板将其映射为$[e([P_{0:i}]),e(x),e([P_{i+1:m}]),e(y)]$,其中的$[P_i]$所用的词语是模型词汇表中的词。
3. P-tuning将$[P_i]$当成一个伪token（之所以叫伪token是因为后续还将进行更新），将其映射为$[h_0,...,h_i,e(x),h_{i+1},...,h_m,e(y)]$


+ 步骤

1. 输入一个句子，以及预先设计的一个离散的模板：`The Disney film is good! It was [MASK].`；
2. 对输入的template中，挑选一个（或多个）token 作为`pseudo token`：`The Disney film is good! [pseudo] was [MASK]`.其初始化可以直接使用原本的token embedding;
3. 对所有的pseudo token $[P_i]$,传入一层LSTM，并获得每个`pseudo token`输出的隐状态向量$h_i$ ($h_i$就是一个可以训练的连续的稠密的张量，从而促使模型可以找个更好的连续prompts);
4. P-tuning并不是随机初始化这些`pseudo tokens`然后直接训练，而是通过一个BiLSTM模型，加上一个使用ReLU作为激活函数的双层MLP,把这几个Embedding算出来，并且将这个LSTM模型设为可学习的。避免了人工构建离散的template，而让模型可以自动学习continuous embedding

<div align=center>
    <img src="zh-cn/img/ch2/2-2/p2.png" /> 
</div>

5. 只对Prompt部分的参数进行训练，而语言模型的参数固定不变;
6. 使用BiLSTM+MLP本质上也是类似prefix-tuning中的重参数化作用;
7. 将整个句子传入预训练语言模型层,对于`pseudo token`传入隐藏状态$h_i$,对于原始词和`[mask]`，则传入embedding $e_i$

#### 2.3 效果

+ LAMA+SuperGLUE基准数据集

LAMA数据集，根据知识库中结构化三元组（事实）构建的完形填空类型的数据。

<div align=center>
    <img src="zh-cn/img/ch2/2-2/p3.png" /> 
</div>

Manual Prompt (MP): use original handcraft prompts from LAMA.（提供了人造的prompt).对于具有110亿个参数的MegatronLM2，虽然微调几乎不起作用，但P-tuning仍然可以进一步提高精度.

<div align=center>
    <img src="zh-cn/img/ch2/2-2/p4.png" /> 
</div>

#### 2.4 结论

+ 作者发现添加少量锚标记（anchor tokens）有助于SuperGLUE的一些NLU任务。例如在prompt模板`“[PRE][prompt tokens][HYP]?[prompt tokens][MASK]”`中的`“?”`就是一个锚节点，对性能影响很大。
+ 在知识探测任务中：语言模型可以仅通过找到更好的prompt，不进行微调，就能捕捉到更多的知识。
+ P-tuning可以有效的提高BERT和GPT模型在NLU任务上的性能。并且使用P-tuning，可以让相似代销的GPT2实现比BERT模型相当的甚至更好的结果，这个发现颠覆普遍认为的——双向模型比单向模型在NLU任务中表现的更好。
+ 当预训练模型足够大的时候，我们的设备可能无法finetune整个模型，而P-tuning可以选择只优化几个Token的参数，因为优化所需要的显存和算力都会大大减少，所以P-tuning实则上给了我们一种在有限算力下调用大型预训练模型的思路。

#### 2.5 问题

当预训练语言模型的参数量低于100亿时，Prompt-tuning会比传统的Fine-tuning差;像序列标注等对推理和理解要求高的任务，prompt-tuning效果会变差；

#### 2.6 应用

在HUAWEI推出的盘古-$\alpha$模型中,P-tuning也被用于其的微调框架中

---
### 3.P-tuning v2

<!-- https://mp.weixin.qq.com/s/bVmj--3ndq470qMo1_MKzA -->

从标题就可以看出这篇工作的野心，P-Tuning V2（21年10月）的目标就是要让 Prompt Tuning 能够在不同参数规模的预训练模型、针对不同下游任务的结果上都达到匹敌 Fine-tuning 的结果。当然客观来说 V2 在思想上其实并没有太大的创新，我们一起来看下。

+ 全任务、全规模的通用提示微调方法。
+ V2：https://arxiv.org/pdf/2110.07602.pdf
+ code：https://github.com/THUDM/P-tuning-v2

#### 3.1 提出背景 & 原文论述

出发点既然是想做全任务、全规模的通用提示学习微调，那么当前的提示学习微调方法肯定存在局限，作者在智源社区的meetup上也提到了：

1. 缺少规模通用性；

即模型参数量方面缺少通用性，之前的提示学习方法，如 Prompt-Tuning 和 P-tuning V1，都是在预训练模型参数规模够足够大时（e.g. 10B）且使用特殊的初始化技巧（e.g. label initialization）才能达到和 Fine-tuning 类似的 performance，而对于参数规模较小的普通模型（e.g. 100M到1B）效果则很差。

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p1.png" /> 
</div>

2. 缺少任务通用性；

+ 即任务方面缺少通用性，一些任务上表现好，但是对于“提示学习困难”任务来说，例如序列标注（sequence tagging）表现不好。
+ 所谓的“提示困难”，就好比于比较难的 token-level 的任务，再比如比较现实的问题就是并不是所有标签都有明确的语义，这样序列标注中难以利用语义标签。

为此，P-tuning V2 应运而生。


#### 3.2 P-Tuning V2

P-Tuning V2 具体做法如下：

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p2.png" /> 
</div>

1. 采用 Prefix-tuning 的做法，在输入前面的每一层都加入可微调的参数;

+ 上述左图，橙色向量是在输入层，Prompt tuning 一般是指在输入层添加连续的向量，而 Prefix-tuning 在 transformer 的每一层的开头位置都添加了连续提示向量，这就不仅仅局限于 input level，而是every transformer layer。
+ 上述右图，P-tuning v2 可以说是 Prefix-tuning 和 P-tuning v1 的结合，之前的 P-tuning V1 主要是在输入层加一些连续型向量作为提示，它只在输入层。现在 v2 是在每一层，不仅仅包含输入层，都加入了提示向量。
+ v1是可以在每个位置都可以加向量，但是**v2是固定到了前缀**。

2. 去掉重参数化的编码器；

以前的方法利用重参数化来保证收敛和提高训练速度，例如 Prefix-tuning 的 MLP、P-tuning V1 的 LSTM，在 V2 中，作者发现重参数化的改进很小，尤其是对较小的模型，同时还会影响模型的表现。

3. 可选的多任务学习；

解释一下多任务作者是怎么做的，简而言之就是将不同任务的数据集都混在一起了，对每个数据集都使用相同的提示模版，即提示模版的连续向量是不变的，但是后面有不同的分类层来做分类。

4. 回归传统的 CLS + token label classifier；

这个主要是为了解决一些没有语义的标签问题。

此外，针对 V2 中为什么将提示向量固定到前缀，作者也给出了解释：

+ V1 只调节了输入层，也就是说它可调节的参数量比较小，就会导致模型容量比较小，容量小就会在通用性方面表现的不好，所以作者在每一层都加入了提示向量，可调节的参数就多了，就会更有通用性。
+ V1 是在输入层加的，那么它对输出层的影响就会比较间接，如果在每一层的前缀加，就会对输出的影响变得直接。

!> 后面会有实现表明，加的离输入层越近，效果越好。

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p3.png" /> 
</div>

上图是作者把它们的方法和其它方法做了一个比较，v2 的任务中除了 NLU 还加入了序列标注任务，因为作者前面说了序列标注对于提示学习来说是困难的任务，v1和prefix是做了重参数化，v2 的 depends 意思是可以做、也可以不做。作者在实验中证明了，对于有些任务可能重参数化是有效的，对于有些任务是无效的，它并没有一个固定的结论。Deep PT 的意思是不是深度的，当然v2是，因为它在每一层都加了。最后一列的意思是有没有向量化。


#### 3.3 原文实验

P-tuning v2 的核心结果：

+ v2 仅使用0.1%的参数，在 330M~10B 规模均与 fine-tune 匹敌；
+ Prompt tuning & P-tuning：仅在10B规模与fine-tune匹敌；

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p4.png" /> 
</div>

Deep Prompt Tuning 的优化难题可以通过增加额外的任务数据或者无标注数据来缓解，同时可微调的连续 prefix prompt 也可以用来做跨任务的共享知识。比如说，在NER中可以同时训练多个数据集，不同数据集使用不同的顶层classifer，但是prefix continuous prompt是共享的。

任务通用性：在 CoNLL 上，保持预训练模型参数冻结时；

+ P-tuning v2 仅使用 0.1% ~ 3% 的参数也几乎总是能与 fine-tune 相匹敌；
+ prompt-tuning 和 p-tuning V1 与 fine-tune 差距较大；
+ 多任务学习对进一步提升 p-tuning V2 性能十分有效；

规模通用性：在 SQuAD 上，保持预训练模型参数冻结时；

+ P-tuning v2 仅使用 0.1% 的参数，在 `RoBERT_large` 上和 fine-tune匹敌，在 `DeBERT_axlarge` 上甚至比 fine-tune 更好。
+ prompt-tuning 和 p-tuning V1 几乎无法训练
<div align=center>
    <img src="zh-cn/img/ch2/2-3/p5.png" /> 
</div>

接下来是消融实验：

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p6.png" /> 
</div>

Table4 就是否向量化做了实验，在提示学习中，最终预测的是 mask 的词，但是 transformer 输出的是一个向量，那么就需要在词表中找到和这个向量最相似的词，当做输出，那么要不要把这个向量映射成一个词呢，在之前的提示学习的技术中，这个步骤是非常关键的，因为要根据这个词来做判断，比如要根据这个词是bad还是good来做分类等等，作者在这里说，这个操作在提示学习中并不是必不可少的，作者的方法就是在后面加了一个线性层，transformer输出的向量，以这个线性层作为输入，然后再输出一个结果。

作者说微调量比单独微调输入层的提示学习微调的参数量已经多很多了，也不差最后的线性层了，所以加了个线性层，做了个对比实验，就是加线性层的performance和直接去词表中找最相近词的performance做了一个对比，发现加了一个线性层效果更好。

<div align=center>
    <img src="zh-cn/img/ch2/2-3/p7.png" /> 
</div>

图4是在证明重参数化是否有用，结果证明不同的任务效果是不同的。

#### 3.4 小结

Prefix-tuning 和 P-tuning V1 都是只在输入层加入模版，和之前不同的是它们加的是连续的模版，之前的都是离散型的，它俩不同的是，prefix 只在前面加提示，V1 是在输入的那里都可以加提示，并且 P-tuning V1 并没有全部冻结 PLM 的参数，仅在一个子任务中冻结了上层模型参数。V2 版本就是冻结 PLM 的所有参数了，借鉴了Prefix-tuning，不仅在输入层做连续模板的参数学习，还在每一层前面加可学习的参数（在每一层都可以加 prefix 的提示以求能达到更好的效果），利用连续空间内搜索到的prompt将多个任务建模成通用的生成范式来做。

+ 规模通用性：赋能中小模型的高效微调。
+ 任务通用性：应对几乎所有难易和类型的自然语言理解任务。
+ 参数高效性：仅需调节0.1%左右的原模型参数量的连续提示向量。

虽然思路确实挺好，但是微调prompt范式总归有局限性：

+ Prompt较难训练，同时减少了模型的可用序列长度；
+ 可解释性差：这是所有连续型prompt的统一问题；
+ 微调可能存在不稳定性，收敛速度和精度相较于 LoRA 方法还是差一些，毕竟Prompt范式是更少的参数想要撬动更大的模型，需要更复杂的空间搜索。


---
### 4.Prompt-Tuning

<!-- https://mp.weixin.qq.com/s/HVQ_aM8Siwk_Xpa2DO_wIQ -->

首先重新梳理一下目前主流微调 LLM 的方法，大抵可以分为三类：

+ Prompt modifications，如 Prefix tuning、Prompt tuning、P-tuning；
+ Adapter methods，如 Adapters、AdapterFusion、AdapterDrop；
+ Reparameterization，如 Lora、DyLRA、AdaLoRA；

!> 它们其实都属于 PEFT 的范畴，其它的看看时间吧，争取都学习更新一下。

自从大模型爆火以来，网上对 prompt engineering 的资料已如汗牛充栋了，所以我也就不再赘述了。

步入正题，我们今天讲解的是大模型提示微调的经典工作：Prompt-Tuning，这篇工作发表在2021年的 EMNLP 上，是 Prefix Tuning 的一种简化，整体来说比较简单，所以本文篇幅不长，文章关键词：

+ 大模型（Large-scale PLMs）
+ 参数高效微调（parameter-efficient tuning）
+ 提示微调（prompt tuning）

#### 4.1 摘要

摘要里比较有信息量的大概就这三句：

+ 把传统 GPT-3 里面的 soft prompt 变成了可学习的向量，而不是 GPT-3 里面原来的那些短语句子作为 prompt，固定了整个预训练模型的参数，只训练这些提示向量来适配下游任务；
+ 模型参数规模越大，发现 prompt tuning 和全参数微调性能越接近，当模型参数达到百亿的时候，基本上全参数微调和 prompt tuning 就差不多了；
+ prompt tuning 是 prefix tuning 的一种简化，相当于 prefix 在每一层都加大量的所谓的 embedding，prompt tuning 相当于只在输入层上加，所以说它是 prefix 的一种简化应该还是一个比较直观的说法。

#### 4.2 Prompt-Tuning

<div align=center>
    <img src="zh-cn/img/ch2/2-4/p1.png" /> 
</div>

左边是传统的预训练模型的微调方式，基本上它们需要在每个任务上都做单独微调，然后得到一个任务相关的模型；

而 prompt tuning，其实就是对每个任务都会学习一个提示，然后用的时候就把这个提示塞到数据里面，在输入到统一的一个预训练模型里面去就能解决下游任务；

肉眼可见，prompt tuning 在存储和部署上性能都会大大提升，因为不会重复地为每一个任务部署一个很大的模型，所以所有下游任务都围绕着一个模型在做，这样的话存储和部署的效率是比较高的。

prompt tuning 并不复杂，我们拿分类任务举个例子，具体看一下它是怎么做的：

<div align=center>
    <img src="zh-cn/img/ch2/2-4/p2.png" /> 
</div>

!> 这个文章本身是由T5作为GPT模型的，所以这个图是按照T5来画的。

传统的方式就是给定一个输入$x$和一个参数$\theta$，然后输出$y$得到最终的一个输出结果。GPT-3 里会额外的加入一些短语句子作为提示，我们将短语句子记为$P$，它本身增加没有额外的参数。

如上图，为了让这个过程更清晰，从左到右依次把大概的预训练、传统的全参数微调，以及加入提示之后用的提示微调做下游任务的模式都画了一下。

prompt tuning 会把$P$变成一个可学习的向量，$\triangle$就是 prompt 额外增加的向量，**在下游任务微调的过程中，整个预训练模型的参数$\theta$是不动的，只会调整提示的参数**，通过调整提示来适配下游任务。

在具体实现细节上，作者罗列了一些可能会对 prompt tuning 性能产生影响的因素：

+ 一个是模型的参数量；
+ 一个是加入提示的长度；
+ 以及提示的初始化方法；

对于初始化而言：

+ 第一种是随机初始化；
+ 第二种是设计一个自然语言提示，然后把自然语言提示的向量作为初始化，比如作者认为这个提示应该是类似 it was 或者 that is 这样的短语，那就把 it was 或者 that is 的词向量拿来做可学习的提示向量的初始化；
    - 类似于设计 hard prompt，然后将 hard prompt 转化为 soft prompt；
+ 论文提出的第三种随机初始化方法就是把分类任务的类别的词向量拿过来做初始化，类似于提供选项，这个可以直观理解成在输入文本中加入了所谓的提示，其实就是把 `a b c d` 这些选项都加进去，然后让模型去做选择，用这种方式来做模型初始化的方法。

#### 4.3 原文实验

作者在 superglue 上做了各种各样的实验，从实验结果上可以发现一些比较符合大家直接的结论。

第一个就是 prompt 规模越大，性能相对而已就会越好，这个其实非常直观，prompt 长度大了，能调的参数就多了，模型去学下游任务就会更加舒服。

<div align=center>
    <img src="zh-cn/img/ch2/2-4/p3.png" /> 
</div>

然后是基于语义信息的初始化比随机初始化要好。

其次作者发现 LM Adaptation 对性能提升显著，Prompt Tuning还是需要大模型有较好的文本生成能力。

模型参数规模越大，Prompt Tuning效果越好，10B参数时与全参数微调性能接近。

#### 4.4 小结

本文对 Prompt-tuning 进行了讲解。基本上思路都是只要模型参数规模大了，就可以把大模型固定下来，搞些附加参数来适配下游任务，而且适配的性能基本和全参数微调相当。

Prefix-tuning 通过向输入序列插入特定于任务的前缀来修改模型的更多层，因此需要对更多参数进行微调。而 Prompt-tuning 只涉及微调输入提示嵌入，导致更新的参数更少。当然这可能使 Prompt-tuning 比 Prefix-tuning 的参数效率更高，但也可能限制其适应目标任务的能力。

简单总结一下 Prefix-tuning 和 Prompt-tuning 优缺点吧：

优点（计算友好）：

+ 大模型的微调新范式；
+ 一个中心模型服务多个下游任务，节省参数储存量；
+ 无需优化模型参数，节省优化器的计算量和存储量；
+ 只在输入层进行操作，适合多任务场景下的计算合并；

缺点：

+ 收敛速度较 LoRA 慢；
+ 模型性能不太稳定；
+ few-shot 场景上表现不佳

---
### 5.Adapter-Tuning

<!-- https://mp.weixin.qq.com/s/M5DsIWGe9rc6jmDtEKbjgQ -->

随着 ChatGPT 等大模型（Large Language Model，LLM）的爆火，目前业界已经发现只有当模型的参数量达到100亿规模的时候，才能出现一些在小模型无法得到的涌现能力，比如垂直领域的多轮对话、上下文学习（In Context Learning）、思维链（Chain-of-Thought）等等，深度学习似乎朝着模型越来越大的方向一去不复返。

对于这些通用大模型如何进行下游任务的微调存在很多问题，比如：

1. 对于动则百亿级别的参数，如何更高效，低资源的微调大模型呢？
2. 当样本量很小的时候，如何微调大模型能得到较好的效果呢？

为了解决大模型微调的诸多问题，学术界开始研究 参数高效微调方法（Parameter-Efficient Fine-Tuning，**PEFT**），PEFT 技术旨在通过最小化微调参数的数量和计算复杂度，来提高预训练模型在新任务上的性能，从而缓解大模型预训练的成本。这样一来，即使计算资源受限，也可以利用预训练模型的知识迅速适应新的任务，实现高效的迁移学习。因此 PEFT 可以在提高模型效果的同时，大大缩短训练时间和计算成本。

谷歌的研究人员首次在论文《Parameter-Efficient Transfer Learning for NLP》提出针对 BERT 的 PEFT 微调方式：Adapter，拉开了 PEFT 研究的序幕。

接下来一段时间我将会更新 PEFT 相关的文章，大概更新内容为：Adapter tuning、Prefix tuning、Prompt tuning、P-tuning、Delta tuning，本文先对 Adapter 进行讲解。

#### 5.1 提出背景 & 原文论述

作者指出，在面对特定的下游任务时一般有两种方案；

+ 微调所有层：如果进行全量微调，则太过低效；
+ 微调某几层：而如果采用固定预训练模型的某些层，只微调接近下游任务的那几层参数，又难以达到较好的效果。

于是基于预训练模型，作者提出了新思路：Adapter，即能否在模型中插入一些少量参数的模块，在微调某个下游任务时只对这些参数进行训练，而保持预训练模型原有的参数不变，并且达到和微调整个模型一样的效果。

作者指出 Adapter 有如下好处：

+ 效果接近于微调所有层；
+ 每个任务一个适应器，不会遗忘；
+ 每个任务只需要加少量额外参数。

#### 5.2 Adapter 模块

<div align=center>
    <img src="zh-cn/img/ch2/2-5/p1.png" /> 
</div>

训练：

+ 我们直接看图来说，Adapter 就是冻结模型原有的全部参数，只训练 Adapter 层 和 layer norm 的参数。

网络结构：

+ 上图中会在原始的 transformer block 中添加两个 Adapter 块：一个在 multi-head attention 后面，一个在 FFN 后面。
+ 每个 Adapter 块由两个前馈子层组成，第一个前馈子层将 transformer block 的输出作为输入，并且控制其维度，比如将原始输入维度 d 投影到 m，通过控制 m 的大小来限制 Adapter 块的参数量。
+ 在输出阶段，通过第二个前馈子层还原输入维度，作为 Adapter 块的输出。整个 Adapter 层的输入和输出之间还有一个 skip connection，所以这一类的 Adapter 也被形象的称为 Bottleneck Adapter。
+ 就这样通过添加 Adapter 块来产生一个易于扩展的下游模型，通过 Adapter 块避免全模型微调以及灾难性遗忘的问题。

初始化：

+ 所有的 Adapter 参数都从均值为0，标准差为0.01的正太分布中采样，这样可以保证刚训练时 Adapter 层的主干输出很小，主要由残差传递信息。


#### 5.2 原文实验

从实验结果来看，该方法能够在只额外对增加的3.6%参数规模（相比原来预训练模型的参数量）的情况下取得和 Full-finetuning 接近的效果（GLUE指标在0.4%以内）。

<div align=center>
    <img src="zh-cn/img/ch2/2-5/p2.png" /> 
</div>

主要在分类任务（GLUE）和抽取式问答任务（SQuAD v1.1）上进行，比较微调整个 BERT-Large 和 BERT-Large 加 Adapter 只微调 Adapter 的性能。

<div align=center>
    <img src="zh-cn/img/ch2/2-5/p3.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch2/2-5/p4.png" /> 
</div>

实验发现：

+ 只微调 Adapter 可以做到比较接近整个模型微调的性能，如果根据每个 task 调 Adapter 的大小，可以做到掉点比较少。
+ 使用 Adapter 的参数效率要高于只微调 BERT 的靠近输出的若干层，性能要高于只训练 layer normalization 的参数。
+ 推理阶段，对某层的 Adapter 进行剪枝（pruning）是可行的，不会对性能产生太大影响。但是对多层进行剪枝性能会大幅下降。相比靠近输出的层（顶层）来说，靠近输入的层（底层）对剪枝更不敏感。

#### 5.3 小结

本文对 PEFT 的第一篇工作 Adapter 进行了讲解，Adapter 本身并不难，它嵌入在 Transformer 里面，在训练时，固定住原来预训练模型的参数不变，只对新增的 Adapter 结构进行微调。同时为了保证训练的高效性，也就是尽可能少的引入更多参数。

结构上，首先是一个下采样层将高维特征映射到低维特征，然后过一个非线形层之后，再用一个上采样结构将低维特征映射回原来的高维特征；同时也设计了 skip-connection 结构，确保了在最差的情况下能够退化为 identity。

虽然 Adapter 可以降低模型微调对算力的要求，但是它毕竟在模型中添加了额外的层，这必然会导致大模型在推理时需要更多的GPU通信导致推理耗时增加。当然可以通过剪枝等量化手段减少整体的延迟，但是貌似没有直接的方法来绕过 Adapter 层的额外计算。

这时肯定会有人说：将 Adapter 层设计成瓶颈维度小的层（参数量很少）不可以吗？应该不行的，因为大模型是依靠硬件并行性来保持低延迟的，而 Adapter 层是按顺序串行处理的，在这种情况下，批次大小通常小到1，在没有模型并行的情况下，比如单个GPU，它的延迟也是会增加的，即使瓶颈维度非常小。

当然 Adapter 也出了很多变体，比如：AdapterFusion、AdapterDrop、Compacter，有兴趣的同学可以自行查阅资料，我就不再写这些文章了。

------
------

## 3.LoRA and QLoRA

<!-- https://mp.weixin.qq.com/s/uAqra9gbpk97tJDA1K8PXQ -->

### 1.LoRA: Low-Rank Adaptation of Large Language Models

#### 1.1 引入

最近出现了很多 ChatGPT 的开源“平替”，它们大都采用了参数高效微调方法，尤其是 LoRA。最初我接触 LoRA 是用它微调 Stable Diffusion，最近有微调 LLM的需求所以就写一下 LoRA 的文章记录一下，需要注意的是 LoRA 最初就是为大语言模型（LLMs）微调来设计的。

我们都知道训练神经网络的目标是使预测损失最小化，在参数空间内找到最优的点，如果从头开始找肯定会比较慢，我们基于其它开源项目来训练，就相当于在最优点的附近开始找，这样收敛速度和训练效果会比从零开始要好的多。取预训练好的网络的结构或部分结构，以及其权重，与自己新增的网络部分一起训练，这就是所谓的微调（finetune）。

“大模型（大规模预训练）+ 微调” 可以很好的适应并融入不同的下游任务，其具有很好的通用性，成为目前大模型落地的主要训练手段。以 Stable Diffusion 和 ChatGPT 为代表的大模型，在图像生成和文本对话领域展现了强大的性能。随着模型规模的不断增大，对其 fine-tune 的硬件要求和数据要求都在不断上涨，丰富多样的下游任务使得大模型在 fine-tune 阶段的目标设计非常繁琐复杂，换句话说由于上、下游任务之间目标不一致，因此大模型可能无法直接适配下游任务，输入和输出之间存在结构偏差，优化成本高。因此提出了轻量级的 fine-tune 方法。

今天要讲的是发布在 ICLR'2022 的文章：LoRA（低秩适应），以降低在特定任务或领域中使用大模型时的可训练参数量，它通过将可训练的秩分解矩阵注入到 Transformer（Attention） 的每一层来实现，从而极大的减少了下游任务的可训练参数数量，说的再直白些，LoRA 是一种以极低资源微调大模型的方法。

+ 针对 LLMs 的 LoRA：https://github.com/microsoft/LoRA
+ 针对 Stable Diffusion 的 LoRA：https://github.com/cloneofsimo/lora


#### 1.2 提出背景 & 现有方案的局限

随着模型规模的不断扩大，模型的生成能力也在不断加强，尤其是在zero-shot、常识推理、多轮对话等能力上会有大幅度提高，相较于规模小的模型而言，大模型的微调成本和部署成本都非常高。所以，如何降低大模型的微调和部署成本，将是大模型商用的重要一环。

在 LoRA 方法提出之前，也有很多尝试解决大模型微调困难的工作，它们的核心目标都是固定大模型，额外增加一些参数来适配下游任务。但它们或多或少都存在性能问题，其中主要有两种突出的策略：

+ 增加适应层 Adapter；

它的主要问题在于推理时带来的额外计算量和延迟；

+ 优化 prompt；

例如前缀微调（prefix tuning）就比较难优化，而且前缀优化在可训练参数中（随着参数量增长）的变化是非单调的，并且为适应保留一部分序列长度必然会减少可用于处理下游任务的序列长度，这就使得调整 prompt 与其它方法相比性能较差。

我们来简单看下这两种策略的局限性：

Adapter：

!> Bottleneck Adapter：https://proceedings.mlr.press/v97/houlsby19a.html

<div align=center>
    <img src="zh-cn/img/ch2/2-5/p1.png" /> 
</div>

我们直接看图来说，Adapter 就是冻结原有参数（冻结模型主体），添加额外的参数用以微调，上图中会在原始的 transformer block 中添加两个 Adapter 块，一个在 multi-head attention 后面，一个在 FFN 后面。每个 Adapter 块由两个前馈子层组成，第一个前馈子层将 transformer block 的输出作为输入，并且控制其维度，比如将原始输入维度 d 投影到 m，通过控制 m 的大小来限制 Adapter 块的参数量。在输出阶段，通过第二个前馈子层还原输入维度，作为 Adapter 块的输出。就这样通过添加 Adapter 块来产生一个易于扩展的下游模型，通过 Adapter 块避免全模型微调以及灾难性遗忘的问题。

虽然 Adapter 可以降低模型微调对算力的要求，但是它毕竟在模型中添加了额外的层，这必然会导致大模型在推理时需要更多的GPU通信导致推理耗时增加。当然可以通过剪枝等量化手段减少整体的延迟，但是貌似没有直接的方法来绕过 Adapter 层的额外计算。

这时肯定会有人说：将 Adapter 层设计成瓶颈维度小的层（参数量很少）不可以吗？应该不行的，因为大模型是依靠硬件并行性来保持低延迟的，而 Adapter 层是按顺序串行处理的，在这种情况下，批次大小通常小到1，在没有模型并行的情况下，比如单个GPU，它的延迟也是会增加的，即使瓶颈维度非常小。

Prompt Tuning

!> prefix-tuning：https://paperswithcode.com/paper/prefix-tuning-optimizing-continuous-prompts

<div align=center>
    <img src="zh-cn/img/ch2/2-1/p1.png" /> 
</div>

prefix-tuning 的想法源于 GPT-3 的 in-context learning，想法就是只有合适的上下文则语言模型可以很好的解决下游任务，但是在特定任务下找到离散token的前缀耗时较多，prefix-tuning 提出使用连续的 virtual token embedding 来代替离散token。我们直接看图，它对于 transformer 的每一层，都在句子表征的前面插入了可训练的 virtual token embedding，称为前缀（prefix）。

对于自回归模型（GPT系列），前缀添加后表示为：
$$z=[PREFIX;x;y]$$

对于encoder-decoder模型（BERT），则前缀添加表示为：
$$z=[PREFIX;x|PREFIX^{'};y]$$

它其实就可以看成对模型的每一层增加可以调节的参数，具体表现就是扩展注意力中的键值对。训练时语言模型的参数固定，只训练前缀参数，对于不同的任务，使用相同的基础模型，并且训练不同的前缀参数即可。虽然 prefix-tuning 没有添加太多的额外参数，但是它难优化，并且会减少下游任务的序列长度。


#### 1.3 原文摘要

LoRA 解决了上述两个问题，它也是通过冻结原始模型参数，并且添加少量参数的情况下，减少了训练参数量，并且不会引入额外的延迟，甚至原文中的实验结果表明它和全量微调效果相当，并且速度更快，计算量更少。这篇文章的关键词如下：

+ 迁移学习（Transfer learning）；
+ 模型适配（Adaptation）；
+ 大模型（Large model，Transformer、BERT、GPT）；
+ 低秩（Low-rank）。

原文摘要概况如下：

+ 方法：和其它大部分高效微调方法一样，固定预训练语言模型（大模型）的参数，额外增加新的参数来学习，不同的是新增的模块是一个低秩的模块，即增加低秩分解的矩阵来适配下游任务；
+ 效果：在 GPT-3 上能够把参数量降低到全量 finetune 的一万分之一，即训练参数量显著降低、显存需求减小；
+ 对比：相比于 Adapter 之类的参数高效微调方法，LoRA 能够做到不增加模型的推理延迟。

#### 1.4 设计思路

流程概述:

LoRA 的实现思想很简单，如下图所示，就是**冻结**一个预训练模型的矩阵参数，并选择用$A$和矩阵$B$来替代，在下游任务时**只更新**$A$和$B$。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p1.png" /> 
</div>

结合上图来看 LoRA 的实现流程概况如下：

+ 在原始预训练语言模型（PLM）旁增加一个旁路，做一个先降维再升维的操作，以此来模拟所谓的内在秩；
+ 训练的时候固定 PLM 的参数不变，只训练降维矩阵和升维矩阵，即优化器只优化右路的参数；
+ 模型的输入、输出维度不变，两边共用模型的输入，输出时将$BA$与 PLM 的参数叠加；
+ 用随机高斯分布$N(0,  \sigma ^2)$初始化$A$，用全0矩阵初始化$B$。
    - 注意，矩阵$B$全零初始化，使得在训练最开始的一段时间，右路的结果会接近于0，这样模块的输出就仅仅有左路的计算结果，也就是大模型原有参数的计算结果，这使得模型优化的初始点就和原本的大模型保存一致。

!> 不同于 Adapter 的是，Adapter 在模块的后面接上了MLP，对模型的计算结果进行后处理；而 LoRA 是和模块并行的去做一个MLP。

#### 1.5 灵感来源

那么研究人员为什么要这么设计呢？其灵感来自于前人关于 intrinsic dimension 的发现：**大模型是过参数化的，其有更小的内在维度**。
于是文章做出了一个假设：模型在任务适配过程中，参数的改变量是低秩的。

在训练过程中，增加模块去学习这个改变量，而不是去学习模型的参数，这个就是这篇文章的创新点。在推理的时候只需要把改变量放回原模型就不会有额外的延迟。

#### 1.6 公式解释

下面我们再从公式上解释 LoRA 的实现。增加模块去学习参数的改变量，表示为：

$$W_0+\triangle W=W_0+BA$$

其中：
+ $W_0$是大模型原参数矩阵；
+ $\triangle W$为**改变量**；
+ 由于前人的工作发现语言模型具有较低的“内在维度（intrinsic dimension）”，论文假设这个改变量$\triangle W$是低秩的,于是进一步拆分低秩矩阵$A$,低秩矩阵$B$，在任务适配的过程中，即使投影到较小的子空间，仍然可以有效的学习，所以说 LoRA 做的事情就是小参数学习改变量$\triangle W$.

具体来看，假设大模型参数矩阵维度表示为:$W_0 \in R^{d\times k}$,则它的更新进一步表示为：
$$W_0+\triangle W=W_0+BA$$
$$B\in R^{d\times r},A\in R^{r\times k}$$

其中：
+ 秩$r<< min(d,k)$$
+ 在训练过程中，$W_0$是冻结的；

原始模型前向表示：
$$h=xW$$

加入 LoRA 后的前向表示：
$$h=xW_0+BAx$$

在推理过程中，只需要将改变量放回原模型里，这样就避免了矩阵乘法的显存开销，就不会存在延迟：
$$W=W_0+BA$$

在切换不同的下游任务时，只需要减去当前任务的改变量，在换上新任务的改变量即可。即减去$BA$,换上$B^{'}A^{'}$.

#### 1.7 代码示例

diffusion项目中，实现为 LoRACrossAttnProcessor 类，作用于 CrossAttn 中的 K、Q、V、O (output)线性层，并与原始线性层输出相加，主要过程如下：

```
# LoRA线性层，即B*A
class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        ...

    def forward(self, hidden_states):
        ...
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)

# LoRACrossAttn层
class LoRACrossAttnProcessor(nn.Module):
    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        super().__init__()
        ...
        # 为KQVO分别设置LoRA层
        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0
    ):
        ...
        # 对K\Q\V增加LoRA层
        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)
        # 正常的CrossAttn
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # 对O增加LoRA层
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

```


#### 1.8 原文效果

在具体的实现细节上，论文中罗列了一些可能会对性能产生影响的因素，比如 LoRA 的作用位点，理论上 LoRA 可以作用于任何线性层，包括 transformer 的四个矩阵和两个 feedforward 中的矩阵，但是原文只在 attention 上做了实验，它限制总参数量不变的情况下观察是在 attention 的其中一个矩阵上，放一个更高秩的 LoRA，还是在更多的 attention 矩阵上分别放置低秩一点的 LoRA 效果好。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p2.png" /> 
</div>

结论是把秩分散到多个矩阵上，效果会优于集中在单个上的效果（即同样参数量下，分散于多个点位效果更好）。

其次在秩的大小的设计上：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p3.png" /> 
</div>

作者发现在一般的任务上，很小的秩就能够收敛到接近于很大的秩的实验效果了，这也从实验的角度一定程度上验证了作者最开始做出的改变量是低秩的假设。不过作者也提到，考虑到极端的情况，将秩设到最大，和原来矩阵参数维度一样大，这样训练 LoRA 基本上和全参数微调没有区别了，那么对于领域差距和预训练阶段差距比较大的下游任务，稍微增大一点 LoRA 的 rank，即低秩的 rank 还是可能会带来性能的提升的。

在后续的实验部分，LoRA 在各类模型，包括 Roberta、 Deberta，GPT-2、GPT-3 上都跑了实验，从实验结果来看，在理解任务上，LoRA 的参数量较全参数微调显著降低，和其它现有参数高效微调方法持平或更低，在效果上 LoRA 能够优于其它参数高效微调方法，并且和全参数微调几乎持平，并且有时能够达到更好的效果：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p4.png" /> 
</div>

在生成任务上，LoRA 也能够得到类似的实验结论：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p5.png" /> 
</div>

最后作者在 GPT-3上 做了测试：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p6.png" /> 
</div>

作者发现 prefix embedding 的方法，包括 prefix layer 的方法，在其参数里增大到一定程度之后，性能会出现不升反降的现象，就是图中的橙线和绿线，而对于LoRA，图中粉色这条线，随着参数里增大，收敛性一直比较稳定，并且在最终的效果上也能持平，甚至超过 GPT-3 微调全部参数的做法。

#### 1.9 我的效果

我在写这篇文章的时候是用 LoRA 微调 Stable Diffusion 的，所以这里给出的效果是微调 Stable Diffusion 后的生成效果。训练 prompt 如下：

+ "Yang Mi"
+ "summer palace in spring"
+ "A good-looking girl in the virtual world"
+ "Zibo Barbecue"

原始效果如下：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p7.jpg" /> 
</div>

微调效果测试:

学习人物需要更多的、多角度的清晰样本及epoch。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p8.jpg" /> 
</div><p align=center>prompt = "Yang Mi"</p>

样本质量和数量、epoch要多，佛香阁的细节才能学得更好。不过这前景都学到了确实是我没想到的。


<div align=center>
    <img src="zh-cn/img/ch2/3-1/p9.jpg" /> 
</div><p align=center>prompt = "summer palace in spring"</p>

我直接在C站上找的样本，图片质量高模型学的也好。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p10.jpg" /> 
</div><p align=center>prompt = "A good-looking girl in the virtual world"</p>

网友上传的照片都太杂了，有的还比较模糊，能学成这样已经不错了。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p11.jpg" /> 
</div><p align=center>prompt = "Zibo Barbecue"</p>

#### 1.10 通用能力测试

```python
prompts = ["heavy traffic on the street", 
           "Cute squirrel", 
           "Church in the snow",
           "Ancient style moon lantern flower tree", 
           "Chicken in the woods", 
           "The worker under the umbrella"]
```
<div align=center>
    <img src="zh-cn/img/ch2/3-1/p12.jpg" /> 
</div>

显存占用

+ 微调后 LoRA 的权重大小仅 3.2M；
+ “虚拟女孩”、“淄博烧烤” 训练：
    - batch_size=1
    - mixed precision fp16
    - 梯度累积=4
    - epoch=2000
+ 杨幂”、“颐和园” 训练：
    - batch_size=1
    - mixed precision fp16
    - 梯度累积=4
    - epoch=3000


#### 1.11 测试中遇到的问题

如果训练人物，应该：

+ 像素质量高；
+ 脸部最好无遮挡；
+ 多角度照片，侧脸、本身、全身；；
+ 样本量多；

我在网上找到杨幂，不是特别清楚，所以学起来需要更多的epoch和样本量，否则脸部会变形：

!> 这些是20张杨幂的图片，epoch是2000的结果。上面的例子是50张杨幂照片、epoch是3000的结果，效果挺明显的。

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p13.jpg" /> 
</div>

如果训练风景：

+ 像素质量高；
+ 长焦照片必须要清楚、且样本量要多；
+ 照片要聚焦到想要生成的风景上；
+ 样本量多；
+ 
我在网上找的颐和园，也有一些不是很清楚的，长焦下的佛香阁背景虚化明显，导致学习效果不好，想要生成佛香阁+昆明湖的照片的话，需要更多的佛香阁样本量，否则佛香阁就不是很清楚：

<div align=center>
    <img src="zh-cn/img/ch2/3-1/p14.jpg" /> 
</div>

!> 微调代码: https://github.com/WGS-note/finetune_stable_diffusion


#### 1.12 总结

本文对大模型微调方式 LoRA 进行了详细讲解，基于大模型内在的低秩特性，增加旁路矩阵来模拟全量微调就是 LoRA 的主要思想，它学会一种可以用较少维度来捕捉数据特征的简单表示，这里也就是原文中一直提到的所谓 low intrinsic dimension，当网络适应新任务时，它的权重变化也是低维的，这个过程便可以用一个低秩矩阵来表示，并且 LoRA 还与其它参数高效微调方法正交，所以它是一个能达成 lightweight finetuning 的简单有效方案。


References

https://wangguisen.blog.csdn.net/article/details/127065903

https://arxiv.org/abs/2106.09685

https://finisky.github.io/lora/

https://github.com/cloneofsimo/lora

https://huggingface.co/blog/lora

https://huggingface.co/docs/diffusers/training/lora

https://stable-diffusion-art.com/lora/

https://zhuanlan.zhihu.com/p/611557340

https://kexue.fm/archives/9590#mjx-eqn-eq%3Agrad

https://mp.weixin.qq.com/s/4NGafwtkskaSGAIzGgGuQQ

---
### 2.QLoRA:Efficient Finetuning of Quantized LLMs

<!-- https://zhuanlan.zhihu.com/p/632694507 -->

<!-- https://zhuanlan.zhihu.com/p/632164305 -->

#### 2.1 背景介绍

本文提出了一种QLoRA训练微调方法，通过这种方式可以在单个48G的GPU显卡上微调65B的参数模型，采用这种方式训练的模型可以保持16字节微调任务的性能。QLoRA通过冻结的int4量化预训练语言模型反向传播梯度到低秩适配器LoRA来实现微调。

名为 Guanaco 的最佳模型家族在 Vicuna 基准测试中优于所有以前的公开发布模型，达到了 ChatGPT 的性能水平 99.3%，而在单个 GPU 上只需要 24 小时的微调。QLORA 引入了许多创新来在不牺牲性能的情况下节省内存：

（a）4位 NormalFloat（NF4），一种对于正态分布权重而言信息理论上最优的新数据类型；

（b）双重量化，通过量化量化常数来减少平均内存占用；

（c）分页优化器，用于管理内存峰值。

本文使用 QLoRA 微调了超过 1,000 个模型，并对 8 个指令数据集、多个模型类型（LLaMA、T5）和模型规模进行了详细的指令追踪和聊天机器人性能分析，这些都是使用常规微调无法完成的（例如 33B 和 65B 参数模型）。结果表明，在一个小而高质量的数据集上使用 QLoRA 微调可以实现最先进的结果，即使使用比以前的最先进模型更小的模型也是如此。本文对基于人类和 GPT-4 评估的聊天机器人性能进行了详细的分析，发现 GPT-4 评估是一种廉价且合理的替代方案。此外，我们发现当前的聊天机器人基准测试不能准确评估聊天机器人的性能水平。特定数据挑选分析表明了 Guanaco 在与 ChatGPT 相比失败的情况。本文作者发布了所有的模型和代码，包括用于 4 位训练的 CUDA 内核。

#### 2.2 相关工作内容

微调大型语言模型 (LLM) 是提高其性能的一种非常有效的方法，并添加好的或删除不良行为。然而，微调非常大的模型非常昂贵； LLAMA 65B 参数模型的常规 16 位微调需要超过 780 GB 的 GPU 内存。虽然最近的量化方法可以减少LLMs的内存占用，但这种技术只适用于推断，在训练过程中还是会出现因为资源问题导致训练失败。

本文第一次证明，可以在不进行任何性能下降的情况下微调量化的 int4模型。我们的方法QLoRA使用一种新的高精度技术将预训练模型量化为int4，然后添加一小组可学习的低秩适配器权重。它是通过量化权重反向传播梯度来调整的。

与 16 位完全微调基线相比，QLoRA 将 65B 参数模型进行微调的平均内存需求从 >780GB 的 GPU 内存减少到 <48GB，而不会降低运行时间或预测性能。这标志着LLM微调可访问性的显著转变:现在最大的公开可用的模型，迄今为止在单个GPU上进行微调。使用QLoRA，我们训练了Guanaco模型家族，第二好的模型在Vicuna基准上达到了ChatGPT性能水平的97.8%，同时在单个消费GPU上不到12小时内可训练;使用单个专业GPU超过24小时，我们在最大的模型上实现了99.3%，基本上缩小了与Vicuna基准上ChatGPT的差距。在部署时，我们最小的 Guanaco 模型（7B 参数）只需要 5 GB 的内存，并且在 Vicuna 基准测试中优于 26 GB Alpaca 模型 20 个百分点以上。

QLoRA 引入了多项创新，旨在在不牺牲性能的情况下减少内存使用：(1) 4 位 Normalfloat，一种理论上最佳量化数据类型，该数据类型对正态分布数据产生比 4 位整数和 4 位 Float 更好的实证结果。(2) 双量化，一种量化量化常数的方法，每个参数保存平均约 0.37 位（65B 模型大约 3 GB）。(3) Paged Optimizers，使用 NVIDIA 统一内存来避免在处理具有长序列长度的小批量时发生的梯度检查点内存峰值。我们将这些贡献组合成一个更好的调整 LoRA 方法，该方法包括每个网络层的适配器，从而几乎避免了先前工作中看到的所有准确性权衡。

QLoRA 的效率使我们能够对模型尺度上的指令微调和聊天机器人性能进行深入的研究，由于内存开销，使用常规微调是不可能的。因此，我们在几个指令调整数据集、模型架构和 80M 到 65B 参数之间的大小上训练了超过 1,000 个模型。除了表明 QLoRA 恢复了 16 位性能，并训练最先进的聊天机器人 Guanaco，我们还分析了训练模型的趋势。首先，我们发现数据质量远比数据集大小更重要，例如 9k 样本数据集 (OASST1) 在聊天机器人性能上优于 450k 个样本数据集 (FLAN v2, subsampled)，即使两者都旨在支持泛化后的指令。其次，我们表明强大的大规模多任务语言理解 (MMLU) 基准性能并不意味着强大的 Vicuna 聊天机器人基准性能，反之亦然——换句话说，数据集适用性对给定任务的大小更重要。

此外，我们还对使用人类评估者和 GPT-4 进行评估的聊天机器人性能进行了广泛的分析。我们使用锦标赛式基准测试，其中模型在匹配中相互竞争，以便为给定提示产生最佳响应。匹配的获胜者由 GPT-4 或人工注释者判断。锦标赛结果被聚合到 Elo 分数中，这些分数决定了聊天机器人性能的排名。我们发现 GPT-4 和人工评估在很大程度上同意锦标赛中模型性能的排名，但我们也发现存在强烈分歧的实例。因此，我们强调基于模型的评估，同时为人工注释提供廉价的替代方案也具有其不确定性。

我们通过 Guanaco 模型的定性分析来增强我们的聊天机器人基准测试结果。我们的分析突出了定量基准没有捕捉到的成功和失败案例。我们使用人类和 GPT-4 注释发布所有模型生成，以促进进一步的研究。我们开源了我们的代码库和CUDA内核，并将我们的方法集成到Hugging Face Transformer堆栈中，使它们很容易被所有人访问。我们为 7/13/33/65B 大小模型发布了一组适配器，该模型在 8 个不同的指令跟踪数据集上进行训练，总共有 32 个不同的开源、微调模型。

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p1.png" /> 
</div>

##### 2.2.1  基础技术-Block-wise k-bit Quantization

量化是将输入从拥有更多信息的表示离散化为信息较少的表示的过程。它通常意味着采用具有更多位的数据类型并将其转换为更少的位，例如从 32 位浮点数到 8 位整数。为了确保使用整个低位数据类型范围，输入数据类型通常通过输入元素的绝对最大值进行归一化来重新调整到目标数据类型，这些元素通常构造为张量。例如，将 32 位浮点 (FP32) 张量量化为范围为 int8 张量。

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p2.png" /> 
</div>

其中 $c$ 是量化常数或量化尺度。去量化是可逆的：

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p3.png" /> 
</div>

这种方法的问题是，如果输入张量中出现较大的幅度值（即异常值），那么量化 bin——某些位组合——不能很好地在某些 bin 中量化很少或没有数字。为了防止异常值问题，一种常见的方法是将输入张量分成独立量化的块，每个块都有自己的量化常数$c$。这可以形式化如下：我们通过将输入张量展平并将线性段分割成 $n = (b × h)/B$ 块，将输入张量 $X ∈ R^{b\times h}$
 分成 $n$ 个大小为 $B$ 的连续块。我们用公式1独立量化这些块，以创建量化张量和$n$个量化常数$c^i$
 。

##### 2.2.2 基础技术-Low-rank Adapters

低秩适配器 (LoRA) 微调是一种通过使用一小组可训练参数（通常称为适配器）来减少内存需求的方法，同时不更新保持固定的完整模型参数。随机梯度下降期间的梯度通过固定的预训练模型权重传递给适配器，该适配器被更新以优化损失函数。LoRA 通过额外的分解投影来增强线性投影。给定一个投影 $XW = Y$,其中$X\in R^{b \times h}$,$W \in R^{h\times o}$,LoRA计算：
$$Y=XW+sXL_1L_2$$
其中$L_1 \in R^{b\times h}$和$L_2 \in R^{h\times o}$,$s$是标量。

##### 2.2.3 基础技术-PEFT

一个重要的讨论点是 LoRA 在训练期间的内存要求，无论是在使用的适配器的数量和大小方面。由于 LoRA 的内存占用非常小，我们可以使用更多的适配器来提高性能，而不会显着增加使用的总内存。虽然 LoRA 被设计为一种参数高效的微调 (PEFT) 方法，但 LLM 微调的大部分内存占用来自激活梯度，而不是来自学习的 LoRA 参数。对于在 FLAN v2 上训练的 7B LLaMA 模型，批量大小为 1，LoRA 权重等效于原始模型权重中常用的 0.2%，而 RA 输入梯度的内存占用为 567 MB，而 LoRA 参数仅占 26 MB。通过梯度检查点，输入梯度平均减少到每个平均 18 MB，而 LoRA 被设计为一个序列，使它们比所有 LoRA 权重组合更多的内存密集。相比之下，4 位基础模型消耗了 5048 MB 的内存。这突出了梯度检查点很重要，但也表明积极减少 LoRA 参数的数量只会产生很小的内存好处。这意味着我们可以使用更多的适配器，而不会显着增加整体训练内存占用。如前所述，这对于恢复完整的 16 位精度性能至关重要。

#### 2.3 QLoRA微调

QLoRA 通过我们提出的两种技术实现了高保真 4 位微调——4 位 NormalFloat (NF4) 量化和双量化。此外，我们引入了 Paged Optimizers，以防止梯度检查点期间的内存峰值导致传统上对大型模型困难的单个机器进行微调的内存不足错误。QLoRA 有一个低精度的存储数据类型，在我们的例子中通常是 4 位，以及一个通常 BFloat16 的计算数据类型。在实践中，这意味着每当使用 QLoRA 权重张量时，我们将张量去量化为 BFloat16，然后在 16 位执行矩阵乘法。

##### 2.3.1  核心技术之一4-bit NormalFloat Quantization

NormalFloat (NF) 数据类型建立在分位数量化之上，Quantile quantization 是一种信息论最优的数据类型，可确保每个量化 bin 具有从输入张量分配的相同数量的值。分位数量化的工作原理是通过经验累积分布函数估计输入张量的分位数。

**分位数量化的主要限制是分位数估计的过程是昂贵的**。因此，快速分位数近似算法，如SRAM分位数，被用来估计它们。由于这些分位数估计算法的近似性质，数据类型对异常值具有较大的量化误差，这通常是最重要的值。当输入张量来自固定为量化常数的分布时，可以避免爆炸分位数估计和近似误差。在这种情况下，输入张量具有相同的分位数，使得精确的分位数估计在计算上是可行的。由于预训练的神经网络权重通常具有标准差为 σ 的零中心正态分布，我们可以通过缩放 σ 将所有权重转换为单个固定分布，以便分布恰好符合我们数据类型的范围。对于我们的数据类型，我们设置任意范围 [-1, 1]。因此，数据类型和神经网络权重的分位数都需要归一化到这个范围内。

范围[−1,1]中具有任意标准差σ的零均值正态分布的信息理论最优数据类型计算如下:

(1)估计理论$N(0,1)$分布的 $2^k+1$分位数，得到正态分布的$k$位分位数量化数据类型，

(2)采用这种数据类型并将其值归一化到[−1,1]范围内，

(3)通过绝对最大缩放将其归一化到[−1,1]范围内来量化输入权重张量。

一旦权重范围和数据类型范围匹配，我们就可以像往常一样量化。步骤(3)相当于重新缩放权重张量的标准差以匹配$k$位数据类型的标准差。更正式地说，我们估计数据类型的$2^k$
值$q_i$如下

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p4.png" /> 
</div>

其中 $Q_X (·)$ 是标准正态分布 $N (0, 1)$ 的分位数函数。对称$k$ 位量化的一个问题是这种方法没有零的精确表示，这是量化没有错误填充和其他零值元素的重要属性。为确保$k$ 位数据类型的离散零分 0 并使用所有
$2^k$位，我们通过估计负面部分的两个范围$q_i:$的$2^k-1$和正面部分的$2^{k-1}+1的的分位数$q_i$
来创建非对称数据类型部分，然后我们统一这些$q_i$
集并删除两组中出现的两个零之一。我们将生成的数据类型称为在每个量化 bink 位 NormalFloat (NFk) 中具有相同数量的预期值，因为数据类型在理论上对于以零为中心的正态分布数据是最佳的。此数据类型的确切值可以在附录 E 中找到

##### 2.3.2 核心技术之一Double Quantization

我们引入了双量化（DQ），用于量化量化量化常数以节省额外的内存的过程。虽然精确 4 位量化需要小的块大小，但它也有很大的内存开销。例如，对于 $W$，使用 32 位常数和 64 的块大小，量化常数平均为每个参数添加 $32/64 = 0.5$ 位。双量化有助于减少量化常数的内存占用。

更具体地说，双量化将第一个量化的量化常数$c_2^{FP32}$视为第二个量化的输入。第二步产生量化量化常数$c_2^{FP8}$和第二级量化常数$x_1^{FP32}$
 。我们使用块大小为 256 的 8 位 Floats 进行第二次量化，因为 8 位量化没有观察到性能下降，这与 Dettmers 和 Zettlemoyer 的结果一致。由于
$x_2^{FP32}$为正，我们在量化前从 $c_2$ 中减去平均值以将值居中为零并利用对称量化。平均而言，对于 64 的块大小，这种量化将每个参数的内存占用从 $32/64 = 0.5$ 位减少到 $8/64 + 32/(64 · 256) = 0.127$ 位，每个参数减少了 $0.373$ 位.

##### 2.3.3 核心技术之一Paged Optimizers

在 GPU 偶尔运行内存不足的情况下，使用 NVIDIA 统一内存功能在 CPU 和 GPU 之间自动页面到页面传输进行无错误的 GPU 处理。该功能适用于 CPU RAM 和磁盘之间的常规内存分页。我们使用此功能为优化器状态分配页码内存，当 GPU 运行内存不足时，当优化器更新步骤中需要内存时，这些状态会被自动门出到 CPU RAM。

使用上面描述的组件，我们使用单个 LoRA 适配器为量化基础模型中的单个线性层定义 QLORA，如下所示

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p5.png" /> 
</div>

其中 $doubleDequant(·)$ 定义为

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p6.png" /> 
</div>

我们对 $W$ 使用 NF4，对 $c_2$ 使用 FP8。我们对 $W$ 使用 64 的块大小进行更高的量化精度，对 $c_2$ 使用 256 的块大小来保存内存。对于参数更新，只需要关于适配器权重 $∂E/∂L_i$ 的误差的梯度，而不是 4 位权重 $∂E/∂W$。然而，$∂E / ∂L_i$的计算需要计算$∂X / ∂W$，它通过公式(5)进行去量化，从存储$W^{NF4}$到计算数据类型$W^{BF16}$来计算BFloat16精度的导数$∂X / ∂W$。总而言之，QLoRA 有一个存储数据类型（通常是 4 位 NormalFloat）和一个计算数据类型（16 位 BrainFloat）。我们将存储数据类型去量化为计算数据类型以执行前向和后向传递，但我们只计算使用 16 位 BrainFloat 的 LoRA 参数的权重梯度。

#### 2.4 QLoRA VS 标准的finetuning

我们已经讨论了 QLoRA 的工作原理以及如何显著减少微调模型所需的内存。现在的主要问题是 QLoRA 是否可以执行以及全模型微调。此外，我们想分析 QLoRA 的组件，包括 NormalFloat4 对标准 Float4 的影响。以下部分将讨论旨在回答这些问题的实验。

本文考虑三种架构（仅编码器、编码器-解码器和解码器），并将 QLoRA 与 16 位适配器微调以及最多 3B 的模型的完全微调进行比较。我们的评估包括GLUE和RoBERTa-large、Super-NaturalInstructions (TKInstruct)和T5，以及在Flan v2和Alpaca上微调LLaMA后5次MMLU。为了进一步研究NF4相对于其他4位数据类型的优势，我们使用Dettmers和Zettlemoyer的设置，并测量不同模型(OPT、LLaMA、BLOOM、Pythia)的后量化零射击精度和困惑度，模型大小为125m - 13B。我们为每种特定设置在结果部分提供更多细节，以使结果更具可读性。

虽然页优化器对于在单个 24/48GB GPU 上执行 33B/65B QLoRA 调整至关重要，但我们不会为 Paged Optimizer 提供硬测量，因为仅在处理具有长序列长度的小批量时才会发生分页，很少见。然而，我们确实在 48GB GPU 上对 65B 模型的页优化器运行时进行了分析，发现批量大小为 16，页优化器提供与常规优化器相同的训练速度。未来的工作应该根据分页过程发生的情况进行测量和表征。

默认的 LoRA 超参数与 16 位性能不匹配时，当使用将 LoRA 应用于查询和值注意力投影矩阵的标准实践时，我们无法为大型基础模型复制完整的微调性能。如图，对于 Alpaca 上的 LLAMA 7B 微调，我们发现最关键的 LoRA 超参数是总共使用了多少 LoRA 适配器，并且所有线性变换器块层的 LoRA 都需要匹配完整的微调性能。其他 LoRA 超参数，例如投影维度 r，不会影响性能。

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p7.png" /> 
</div>

4 位 Normalfloat 比 4 位浮点数产生更好的性能 虽然 4 位 NormalFloat (NF4) 数据类型在理论上是最优的，但仍然需要确定此属性是否转化为经验优势。我们遵循 Dettmers 和 Zettlemoyer的设置，其中具有不同数据类型的不同大小的量化 LLM (OPT、BLOOM、Pythia、LLaMA) 在语言建模和一组零样本任务上进行评估。在图 3 和表 2 中，我们看到 NF4 比 FP4 和 Int4 显着提高了性能，并且双量化减少了内存占用，而不会降低性能。

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p8.png" /> 
</div>
<div align=center>
    <img src="zh-cn/img/ch2/3-2/p9.png" /> 
</div>

我们的结果表明，具有 NF4 数据类型的 4 位 QLoRA 在具有完善的评估设置的学术基准上与 16 位完全微调和 16 位 LoRA 微调性能相匹配。我们还表明，NF4 比 FP4 更有效，双量化不会降低性能。结合，这形成了令人信服的证据，证明 4 位 QLoRA 调整可靠地产生与 16 位方法匹配的结果。

与之前关于量化的工作一致，我们的 MMLU 和 Elo 结果表明，对于给定的微调和推理资源预算，在保持精度降低的同时增加基本模型中的参数数量是有益的。这突出了效率从 QLoRA 中受益的重要性。由于我们在 4 位微调的实验中没有观察到与完全微调相比的性能下降，这提出了一个问题，即性能精度权衡完全在于 QLoRA 调整，我们将其留给未来的工作来探索。我们继续研究规模上的指令调整，不可能通过对学术研究硬件进行完整的 16 位微调来探索。


#### 2.5 效果评估

虽然定量分析是我们评估的核心，但只有少数问题只查看摘要统计数据。也许最大的是基准有效性问题——基准是否真正测试其名称或描述所暗示的内容总是受到质疑，特别是当我们发现“捷径”来解决机器学习模型有时会利用的基准时。为了部分缓解这种情况，我们在这里进行了一些定性分析，分为两部分。

<div align=center>
    <img src="zh-cn/img/ch2/3-2/p10.png" /> 
</div>


首先我们展示了一些我们认为代表我们的 65b Guanaco 模型生成的文本中一些观察到的模式的示例。其次，§6.2 我们详细介绍了我们已经讨论的结果及其解释的考虑。

为了找到示例，我们首先遍历为 Vicentuna 基准测试和 OpenAssistant 基准生成的数据，并寻找 Guanaco 生成的答案中的模式。当我们注意到一种模式时，我们尝试设置一个问题或提示来诱导模式，即使它是不正确的解决方案，例如，如果我们观察到模型倾向于给出冗长的答案，我们会提示模型“答案是或没有解释”。我们使用它来找到我们设法对抗破坏模型和“cherries”的“lemons”，我们未能破坏模型并同时呈现两者。本节中的所有代均使用 Nucleus Sampling 生成，p = 0.9。当然，这绝不是全面的，因为它超出了此小型定性研究的范围，以控制所涉及的所有变量，例如，模型可以为给定提示生成的响应的全部分布非常大，因此我们依赖于我们希望代表的样本。然而，我们相信描述这些示例为本文前面显示的定量证据提供了背景。由于我们开源了所有模型和代码，我们希望本节将激发未来的工作，更详细地检查我们在此介绍的问题，（具体分析可以对照论文查看）。

#### 2.6 总结

本文通过大量实验表明， QLoRA 可以使用 4 位基础模型和低秩适配器 (LoRA) 复制 16 位完全微调性能。尽管如此，目前还没有确定 QLORA 在 33B 和 65B 上可以匹配完整的 16 位微调性能。由于资源成本巨大，需要在未来的工作中慢慢去探索。

另一个限制是评估指令微调模型。虽然本文对 MMLU、Vicuna 基准和 OA 基准进行了评估，但没有在其他基准上进行评估，例如 BigBench、RAFT 和 HELM，并且不能保证的评估可以推广到这些基准。另一方面，本文对 MMLU 进行了非常广泛的研究，并开发了评估聊天机器人的新方法。

从所呈现的证据来看，这些基准的性能似乎可能取决于微调数据与基准数据集的相似程度。例如，FLAN v2 与 MMLU 类似，但与聊天机器人基准不同，Chip2 数据集反之亦然，两种模型在 MMLU 和 Vicuna 基准上的得分都相应。这突出了不仅需要更好的基准和评估，还需要仔细了解首先评估的内容。我们想创建在课堂高中同事知识上表现良好的模型，或者我们想在聊天机器人对话能力上表现良好。也许还有其他东西。因为与创建一个新的基准相比，在现有基准上评估总是更容易，某些基准可以将社区引导到某个方向。我们应该确保作为一个社区，基准衡量我们关心的内容。

另一个限制是我们没有评估不同的位精度，例如使用 3 位基础模型或不同的适配器方法。除了 LoRA，还有各种各样的参数高效微调 (PEFT) 方法已被证明运行良好。然而，尚不清楚这些方法是否适用于大型模型。我们使用 LoRA 因为许多结果建立了其鲁棒性，但其他适配器可能会产生更好的性能。由于量化后的微调似乎恢复了量化过程中丢失的大部分信息，这可能会实现更积极的量化。例如，使用 LoRA 的基础模型的 3 位 GPTQ 量化在微调后也可能产生 16 位的完整微调性能。

我们的QLORA微调方法是第一种方法，可以在单个消费GPU上微调33B参数模型和单个专业GPU上的65B参数模型，同时相对于完全微调基线不会降低性能。我们已经证明，我们在 Open Assistant 数据集上训练的最佳 33B 模型可以在 Vicuna 基准上与 ChatGPT 相媲美。由于指令微调是将原始预训练的 LLM 转换为类似 ChatGPT 的聊天机器人的基本工具，我们相信我们的方法将使微调广泛且常见，特别是对于资源最少的研究人员，这是最先进的 NLP 技术的可访问性的一大胜利。QLORA 可以看作是一个均衡因子，有助于缩小大型公司和小型团队与消费 GPU 之间的资源差距。另一个潜在的影响来源是部署到手机上。我们相信我们的QLORA方法可能支持在手机和其他低资源设置上微调llm的关键里程碑。虽然 7B 模型被证明能够在音素之前运行，但 QLORA 是第一个能够微调此类模型的方法。我们估计，使用 iPhone 12 Plus，QLORA 可以在手机充电时每晚微调 300 万个令牌。虽然微调的 7B 模型没有达到 ChatGPT 的质量，但我们相信质量足够好，可以启用由于隐私或 LLM 质量问题之前不可能的新应用程序。QLORA可以帮助实现llm的隐私保护使用，用户可以自己和管理自己的数据和模型，同时使llm更容易部署。

然而，微调是一种双重使用技术，可以滥用它造成伤害。llm的广泛使用具有已知的危险，但我们相信，与将llm的力量保存在没有发布模型或源代码的大型公司手中相比，均衡访问迅速变得无处不在的技术将允许更好的独立分析。总而言之，我们相信QLORA 将具有广泛的积极影响，使高质量 LLM 的微调更加广泛且易于访问。

------
------
## 4.CLIP, BLIP and BLIP2

### 1.CLIP:Learning Transferable Visual Models From Natural Language Supervision

<!-- https://zhuanlan.zhihu.com/p/486857682 -->

<!-- https://zhuanlan.zhihu.com/p/546245070 -->

<!-- https://zhuanlan.zhihu.com/p/594354204 -->

<!-- https://github.com/yangjianxin1/CLIP-Chinese -->
<!-- https://blog.csdn.net/qq_27590277/article/details/128213439 -->

今天介绍一篇OpenAI的神作CLIP，文章发表在ICML-2021，于2021年3月挂在arXiv上的。截至2022年3月，文章已有700+引用量，可见其影响力。

!> paper: https://arxiv.org/pdf/2103.00020.pdf

!> Blog: https://openai.com/research/clip

!> github:https://github.com/openai/CLIP

#### 1.1 Abstract

当前的计算机视觉（CV）模型通常被训练用于预测有限的物体类别。这种严格的监督训练方式限制了模型的泛化性和实用性，因为这样的模型通常还需要额外的标注数据来完成训练时未曾见过的视觉“概念”。直接从图片的描述文本中学习是一个有潜力的选择，因为这样我们可以获取更多的监督信号。这篇文章中，我们证明了利用一个简单的预训练任务（即预测哪个文本描述对应当前图像）在一个从互联网上搜集的4亿个（图像，文本）对的数据集上可以取得SOTA的图像表征。预训练完之后，在下游任务上，我们可以通过用自然语言（文本）匹配视觉概念（图像）从而实现zero-shot transfer。我们在30个不同类型的下游CV 任务上进行了基准测试，并展示了我们模型强大的迁移能力，其在很多下游任务上不需要任何额外的数据也能比拟完全supervised的模型。比如，我们的模型在ImageNet上的zero-shot accuracy能达到在ImageNet上全监督训练的ResNet-50的性能。

#### 1.2 Motivation

在NLP中，预训练的方法目前其实已经被验证很成功了，像BERT和GPT系列之类的。其中，GPT-3从网上搜集了400 billion byte-pair-encoded tokens进行预训练然后可以在很多下游任务上实现SOTA性能和zero-shot learning。这其实说明从web-scale的数据中学习是可以超过高质量的人工标注的NLP数据集的。

然而，对于CV领域，目前预训练模型基本都是基于人工标注的ImageNet数据集（含有1400多万张图像），那么借鉴NLP领域的GPT-3从网上搜集大量数据的思路，我们能不能也从网上搜集大量图像数据用于训练视觉表征模型呢？

作者先是回顾了并总结了和上述相关的两条表征学习路线：

（1）构建image和text的联系，比如利用已有的（image，text）pair数据集，从text中学习image的表征；

（2）获取更多的数据（不要求高质量，也不要求full labeled）然后做弱监督预训练，就像谷歌使用的JFT-300M数据集进行预训练一样（在JFT数据集中，类别标签是有噪声的）。具体来说，JFT中一共有18291个类别，这能教模型的概念比ImageNet的1000类要多得多，但尽管已经有上万类了，其最后的分类器其实还是静态的、有限的，因为你最后还是得固定到18291个类别上进行分类，那么这样的类别限制还是限制了模型的zero-shot能力。

这两条路线其实都展现了相当的潜力，前者证明paired text-image可以用来训练视觉表征，后者证明扩充数据能极大提升性能，即使数据有noise。于是high-level上，作者考虑从网上爬取大量的（text，image）pair以扩充数据，同时这样的pairs是可以用来训练视觉表征的。作者随即在互联网上采集了4亿个（text，image）对，准备开始训练模型。

#### 1.3 Model

##### 1.3.1 Objective

海量的（image，text）数据有了，问题是怎么设计并高效地训练模型。作者提出CLIP的模型，可以认为是ConVIRT[1]的简化版。这里先简单回顾下ConVIRT (咋一看是不是觉得CLIP和ConVIRT一摸一样... ).

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p1.jpg" /> 
</div><p align=center>ConVIRT</p>

ConVIRT用（image，text）对来训练模型，其有一个image encoder和一个text encoder，训练目标是让两路的representation尽可能得一致（对偶地最大化表征的agreement），其中$g_v$和$g_u$函数是一个non-linear的projection head，负责分别将图像和文本表征投影到一个shared的空间，从而计算距离。

损失函数部分其实就是构造了一个对称的contrastive loss，在一个batch内预测谁是正样本。

基于ConVIRT，CLIP主要做出了以下简化：

+ ConVIRT中的image encoder的参数是ImageNet初始化的，而CLIP直接用random初始化；
+ ConVIRT的projection head是non-linear的，而CLIP采用linear的projection；(ConVIRT在后面的实验中也提到将non-linear换做linear，模型效果会下降；但CLIP中则说二者没有区别。)
+ CLIP去掉了ConVIRT中text transformation(指均匀从text中采样句子)；
+ CLIP的image transformation只用了resize和squared crop；
+ CLIP loss中的temperature参数τ是可学的。
+ 另外，二者的loss在形式上有些区别，ConVIRT的loss直接通过最大化$<u_i,v_i>$
 或者 $<v_i,u_i>$；而CLIP加入了标签，用$<u_i,v_i>$的结果和标签去做交叉熵。(具体见修仙：[论文笔记](https://zhuanlan.zhihu.com/p/579493763) CLIP4.3 小节)；(目前我还不知道这两种在理论上是否等价？或者哪一种的实验效果更好一点？)

于是CLIP的预训练模型就有了：

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p2.png" /> 
</div>

一个batch里有N对（image，text），然后和ConVIRT一样做对称的contrastive learning（对比学习），伪代码如下：

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p3.png" /> 
</div>

##### 1.3.2 Inference / Zero-shot prediction

一旦CLIP训练好了，我们就可以做zero-shot prediction了，如Figure 1. (2)所示.

步骤可以整理成下面这样：

+ Sample所有N个class，得到N个input text，都经过text encoder编码得到对应的N个class text embedding（我这里之所叫embedding而不叫representation是想说明这个特征是经过encoding和projection得到的）；
+ Sample一个要预测的image，得到其image embedding；
以N个text embedding为key，以当前image embedding为query，算cosine相似度，相似度最高的即为Top-1的prediction class。

预测过程的代码如下：

```python
import os
import clip
import torch
from torchvision.datasets import CIFAR100

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

# Pick the top 5 most similar labels for the image
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(5)

# Print the result
print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")
```

##### 1.3.3 Training

1. text encoder

作者统一采用GPT-2里的Transformer结构；对于base size model，使用63M-parameter 12-layer 512-width model with 8 attention heads；model width则随着image encoder的size增加而增加。输入句子的最大长度为76。

2. image encoder

这里作者一共训练了8个不同的image encoder（5 ResNets & 3 ViTs），分别如下：

```python
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}

```

其中ResNets做了一个小的修改：将ResNet编码出来的结果再经过一个attention pooling（比如一个2048x7x7的feature，用attention pooling成一个2048x1的feature）；对于ViTs也做了一个小的修改：在tokens（patch tokens和pos tokens相加）被送到Transformer之前，让tokens先经过一个layer norm层，此外参数的初始化和原来的ViTs也有微小的不同。

3. Other configuration

+ optimizer：Adam；
+ training epochs：32；
+ batch size：32768；
+ precision：half-precision

#### 1.4 Experiment

实验部分我这里重点focus在CV相关部分。

**Zero-shot CLIP v.s. Linear Probe on ResNet50**

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p4.png" /> 
</div>

CLIP的胜率在16/27，已经很强了，因为CLIP是zero-shot的，即没有用下游任务的数据，而linear probed ResNet50用了下游数据进行finetune逻辑回归分类器的参数。

**Prompt engineering and ensembling**

作者默认prompt模板是："A photo of a {label}."，但作者发现这样的模板还是有点粗糙，可以考虑加一些context比如"A photo of a {label}, a type of pet."。对于不同类型任务，作者做了一些手动的、特定的prompt工程。

从另一个角度，一张图的text描述其实有很多种的，只要text的核心语义和image相同就行，那么我们还可以做一些ensemble，比如ensemble一下"A photo of a big {label}."和"A photo of a small {label}."。

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p5.png" /> 
</div><p align=center>Prompt ablation</p>

可以发现，采用Prompt engineering+ensembling的效果比只用没有上下文的类别名好得多。

（PS：作者这里的发现直接motivate了之后的CoOp[2]，CoCoOp[3]之类learnable prompting的工作，后面有时间我会专门写一期关于这个的。

**Few-shot CLIP v.s. SOTA (ImageNet) SSL methods**

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p6.png" /> 
</div><p align=center>y: 20个测试数据集上的平均得分; x: shots</p>

+ Zero-shot CLIP的性能和4-shot CLIP差不多；
+ Few-shot CLIP的performance远高于之前的SOTA模型。

**How many shots is needed for achieving zero-shot performance**

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p7.png" /> 
</div>

Few-shot (linear probing) CLIP （保持CLIP encoder 参数fixed，加一层逻辑回归分类器微调）平均需要20.8-shots才能match zero-shot CLIP性能。这里相当于保持了the same CLIP feature space上，观察few-shot finetuning和zero-shot的性能差异。这里其实说明通过自然语言学到的视觉概念比少量样本finetune学到的好。

**Linear probing CLIP performance**

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p8.png" /> 
</div>

总体上，两者的性能是正相关的，此外，大部分情况下linear probing的性能要好不少。

再来一个linear probing的天梯图：

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p9.png" /> 
</div>

CLIP GOAT！！！

**Robustness to Natural Distribution Shift**

作者在ImageNet的7个shift datasets上观察各模型的平均性能。

<div align=center>
    <img src="zh-cn/img/ch2/4-1/p10.png" /> 
</div>

说实话，做domain adaptation（DA）/generalization（DG）的人看到这里应该挺兴奋，新的鲁棒特征来啦。不过问题来了，左边这张图是不是也反映了representation learning比DA、DG technique更重要呢？（那么，我们真的需要花那么大力气去卷DA嘛... 说不定通过这种大规模pretraining就能很大程度上解决domain shift的问题。但另一方面，DA、DG也可以在这些pretraining得到的表征上锦上添花。怎么说都有道理，但我更prefer to CLIP这类表征学习的意义。

CLIP的实验非常丰富，这里只是抛砖引玉地挑了几个我个人觉得比较有意思的实验讲，具体地还是推荐大家去看原文。

#### 1.5 Limitation

这个部分往往容易被人忽略，但其实个人觉得，limitation和conclusion部分往往有作者们更深入的思考，这里简单总结下CLIP的limitation：

+ CLIP的zero-shot性能虽然总体上比supervised baseline ResNet-50要好，但其实在很多任务上比不过SOTA methods，因此CLIP的transfer learning有待挖掘；
+ CLIP在这几种task上zero-shot性能不好：fine-grained分类（花的分类、车的分类之类的）、抽象的任务（如计算图中object的个数）以及预训练时没见过的task（如分出相邻车辆的距离）。BTW，在这些任务上zero-shot性能不好，不代表CLIP pretrained encoders就没用了，CLIP encoders还是能提供很强的视觉先验的；
+ Zero-shot CLIP在真正意义上的out-of-distribution data上性能不好，比如在OCR中；
+ 尽管CLIP zero-shot classifier能在很广泛的任务上work，但究其本质CLIP还是在有限的类别中进行对比、推理，而不能像image caption那样完全的flexible地生成新的概念（如：词），这是CLIP功能上的缺陷，CLIP终究不是生成模型；
+ CLIP仍然没有解决深度学习poor data efficiency的问题，结合CLIP和self-training可能是一个能提高data efficiency的方向；
+ CLIP的方法论上也存在几个缺陷：在训练和挑选CLIP模型时，作者采用在几个数据的validation performance来做指导，这其实是不准确的，因为它不能完全代表CLIP的zero-shot性能。如果，设计一套框架来evaluate zero-shot performance对于之后的研究是很重要的；
+ CLIP的训练数据是从网上采集的，这些image-text pairs没有做data clear和de-bias，这可能会使模型有一些social biases；
+ 很多视觉任务很难用text来表达，如何用更高效的few-shot learning方法优化CLIP也很重要。


到此，CLIP基本讲完，总体来说，对于深度学习来说是优化时代意义的，这可能标志着我们即将迎来data-centric deep learning时代，印证了Andrew Ng的一句名言：“Your model is good enough. Focus on the data!”(大概这个意思，词句不完全准确)。

---
### 2.BLIP:Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

<!-- https://zhuanlan.zhihu.com/p/619222354 -->

<!-- https://blog.csdn.net/zzl1299249769/article/details/123233997 -->

<!-- cross attention: https://zhuanlan.zhihu.com/p/621287866 -->

<!-- 补充Cross Attentio,bidirectional self-attention, causal self-attention layers -->
<!-- VLP: vision-Language Pre-training -->

大多数现有的VLP(vision-Language Pre-training)模型大多仅仅在understanding-based tasks 或者 generation-based tsaks表现良好，但很少在这两方面都能取得较好的结果。
同时，性能的增大往往来自于数据集的扩大，但是现有的数据集大多数是web网络上采集下来的img-text pair。这些大规模从网络上采集下来的数据往往包含大量的noise，不利于模型的训练。
基于以上两点，作者提出了BLIP模型，能灵活的解决understanding-based tasks和generation-based tasks。同时运用知识蒸馏的思想，利用一个captioner和一个filter生成 synthetic(合成的) captions和过滤掉noisy的pair，最终获得bootstrapping dataset，送入下一次pre-train。

#### 2.1 Motivatin

现在大多数模型都选择的是encoder-based model或者encoder-decoder model。但是encoder-based model很难运用到生成任务中去，与此同时encoder-decoder model也很难运用到image-text retrieval（检索） 任务中去。
如今SOTA的方法，大多运用到了web中搜集的大规模数据。但是在web中收集到的这些img-text pair大多数都是noisy的，不利于模型的训练。

#### 2.2 Contribution

1. Multimodal（多模态） mixture of Encoder-Decoder(MED)：MED包含unimodal encoder，image-grounded text encoder以及image-grounded text decoder三个部分。三个部分对应三个VL objectives来进行pre-train，分别是：: image-text contrastive learning, image-text matching, and image conditioned language modeling.
2. Captioning and Filtering(CapFilt)：是一种从noisy img-text pair中进行dataset bootstrapping的方法。将pre-train的MED分为两部分进行finetune，一部分是captioner，从web图像中生成synthetic captions，另一部分是filter，过滤掉生成的caption以及web图像中noisy的图像文本对。

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p1.png" /> 
</div>

如上图所示captioner通过web图片生成caption，filter分别判断原来的web上的text与生成的caption是否是noise，如果是，则过滤掉。

#### 2.3 Modal

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p2.png" /> 
</div><p align=center> 图 2. BLIP 的预训练模型架构和目标（相同的参数具有相同的颜色）。 我们提出了编码器-解码器的多模式混合，这是一种统一的视觉-语言模型，可以在以下三种功能之一中运行：(1) 单模式编码器使用图像-文本对比 (ITC) 损失进行训练，以对齐视觉和语言表示。 (2) Image-grounded text encoder 使用额外的交叉注意层来模拟视觉-语言交互，并使用图像-文本匹配 (ITM) 损失进行训练以区分正负图像-文本对。 (3) Image-grounded text decoder用causal self-attention layers代替bi-directional self-attention layers，与encoder共享相同的cross-attention layers和feed forward networks。 解码器使用语言建模 (LM) 损失进行训练，以生成给定图像的字幕。</p>

!> 我们将在下文详细讲解：cross Attention,bidirectional self-attention, causal self-attention!

如上图所示，整个MED包含了三个部分:

1. Unimodal encoder：分别对image和text进行编码。其中对image的编码是使用的ViT的模式，先把一个图片打成一块一块的patch再输入transformer；对text的编码就和BERT一致，添加一个[CLS]token表示全局的文本信息。
2.  Image-grounded Text encoder：在BERT的基础上，在FFN和SA之间增加了一个Cross Attention层，以为网络注入图像信息。文本中附加了一个[Encoder]token，用于表示img-text pair的多模态表示信息。
3. Image-grounded Text decoder：在Image-grounded Text encoder的基础上，将Bi self-attention层换为了casual self-attention层，用于decoder操作。decoder即为bert的decoder形式，调用BertLMHeadModel（is_decoder=true）, 常用在language modeling里面，添加mask，预测下一个词。同时文本中附加一个[Decoder]token用于表示序列的开始,[EOS]表示序列的结束。

显然，经由上述的三个模块，这个MED模型就拥有了同时匹配generation-based tasks和understanding-based tasks的能力

#### 2.4 Pre-training objectives

本文在pre-training的时候使用了三个objectives，分别是两个understanding-based objectives和一个generatin-based objectives。

1. Image-Text Contrastive Loss (ITC)：通过contrastive learning 的思想，对齐视觉transformer和文本transformer的特征空间，目的是为了获得更加优质的image和text的representation，具体操作可以参考ALBEF这篇文章。

2. Image-Text Matching Loss (ITM)：旨在学习image-text multimodal representation，来捕获视觉和语言的细粒度对齐。简单的来啊说就是图文匹配，最后输出一个二分类，positive or negative

3. Image-Text Matching Loss (ITM)：三个tasks中的生成任务，为给定的图片生成对应的 description。与广泛用于VLP的MLM损失相比，LM使模型具有将视觉信息转换为连贯字幕的泛化能力。

#### 2.5 CapFilt

由于大规模预训练的文本-图片对通常是从web上找出来的，该文本通常无法准确描述图像的视觉内容，从而使它们成为嘈杂的信号，对于学习视觉语言对齐不是最佳的。

由此，作者提出了一个CapFilt架构用来提高image-text pair的质量。

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p3.png" /> 
</div>

如上图所示，其中$(I_w,T_w)$代表web image-text pair，$(I_h,T_h)$代表高质量的手工标注的image-text pair。

它引入了两个模块：一个基于web图像生成caption的captioner，以及一个用于去除image-text pair噪声的filter。captioner和filter都是从同一个预训练过的MED模型中初始化的，并在COCO数据集上单独微调。微调是一个轻量级的过程。

整个过程大概为：先进行pre_train，之后利用$I_h,T_hI$分别对captioner和filter进行finetune，captioner给定web图片生成对应的caption，filter利用ITM判断web图片-文字对和web图片-生成caption对是否match，如果不match，则过滤掉，最后将过滤后剩余的图片-文字对和$I_h,T_h$合在一起pre_train一个新model。个人理解比较像一个新颖的online self-knowledge distillation。

#### 2.6 Experiment

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p4.png" /> 
</div>

上图是提出的captioner和filter对最后结果的影响。

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p5.png" /> 
</div>

上图是parameters sharing策略对最后结果的影响。

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p6.png" /> 
</div>

上图是image-text retirval中与其他SOTA任务的对比，可以看出有较大提升。


<div align=center>
    <img src="zh-cn/img/ch2/4-2/p7.png" /> 
</div>

上图是与其他image caption SOTA方法的对比

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p8.png" /> 
</div>

上图是与其他VQA,NLVR SOTA方法的对比。

更多的实验可以参考BLIP原paper.

#### 2.7 Conclusion

作者提出的BLIP架构在大范围的downstream任务上达到了SOTA的效果，其中包括了understanding-based tasks和generation-based tasks。同时模型使用了一种dataset bootstrapping的方法来解决web中收集的大量noisy数据的问题。

作者还提出有几个潜在的方法可能可以提高BLIP的性能：

1. 进行多轮的dataset bootstrapping
2. 为每幅图片生成多个caption，来扩大语料库
3. 训练多个captioner和filter，并进行model ensemble

#### 2.8 对比学习（contrastive learning）

<!-- https://arxiv.org/abs/2107.07651 -->
<!-- https://zhuanlan.zhihu.com/p/346686467 -->

对比式学习着重于学习同类实例之间的共同特征，区分非同类实例之间的不同之处。

与生成式学习比较，对比式学习不需要关注实例上繁琐的细节，只需要在抽象语义级别的特征空间上学会对数据的区分即可，因此模型以及其优化变得更加简单，且泛化能力更强。
<div align=center>
    <img src="zh-cn/img/ch2/4-2/p10.jpg" /> 
</div>


**对比学习**的目标是学习一个编码器，此编码器对同类数据进行相似的编码，并使不同类的数据的编码结果尽可能的不同。

!> 关于对比学习的综述性介绍可以参考：[光某人的 对比学习（Contrastive Learning）综述](https://zhuanlan.zhihu.com/p/346686467)


#### 2.9 BLIP中的cross attention,bidirectional self-attention, causal self-attention

<!-- https://zhuanlan.zhihu.com/p/32501462?edition=yidianzixun&yidian_docid=0I59CUVb -->

1. cross attention

+ 拥有两个序列S1、S2
+ 计算S1的K、V
+ 计算S2的Q
+ 根据K和Q计算注意力矩阵
+ 将V应用于注意力矩阵
+ 输出的序列长度与S2一致

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p11.png" /> 
</div>

2. bidirectional self-attention

这里的bidirectional self-attention指的的就是BERT中的self-attention机制

3. causal self-attention

<!-- https://www.cnblogs.com/gongqk/p/14772297.html -->
!> https://arxiv.org/abs/2103.03493

Attention 机制现在广泛应用在各领域和各模型之中，attention 涉及到了 Q-K-V 操作，想法是用 Q 去查找 K 中跟自己相似的成分，然后获得新的表示，具体做法就是先用 Q 和 K 求一个相似度作为权重，然后利用相似度对 V 进行加权获得一个新的表示，这个新的表示就融合了 Q 和 K 的相似度信息。

用 image caption 举例说明了 vision-language 领域里两种使用 attention 的方式，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p12.png" /> 
</div>

主要用到了两种 attention 模块，一种是 self-attention，另一种是 top-down attention。输入 X 包含了句子特征以及图像特征（RoI 特征），由于Q 与 K、V 相同，经过 self-attention 得到的新的特征表示，蕴含了图像特征之间的关联，例如上图中新的特征可能学到了人与马之间的关系。第二步就是 top-down attention 模块，这里把 Q 换成了句子特征，当用 Q 与 K 求权重的时候，其实就是在求图像特征中哪些成分与句子特征更相关，例如根据“man”可能就会认为人所在的区域的图像特征权重更大，然后再用这个权重对图像特征加权后，所得到的新特征就是与句子相关的视觉特征。最后我们根据这个句子相关的视觉特征来做预测效果就会更好，因为它融入了两个模态相似度的信息。

那么这里面存在什么问题呢？

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p13.png" /> 
</div>

比如上面最左边的图，问题中关键字是“What sport”、“on screen”，但经过训练后的 attention 却把注意力放在了人身上（红色框），即提取到的句子相关的视觉特征是那两个人的区域，最后得到了错误的答案“Dancing”，而我们希望的是模型能够将注意力放在图像的屏幕区域，是什么导致了错误的 attention 呢？

[《Causal Attention for Vision-Language Tasks》](https://arxiv.org/abs/2103.03493)作者认为是在训练集中，“Sport+Man”的出现次数远远高于“Sport+Screen”的次数，这样的偏倚让 attenion 学习的时候，会把“Sport”和人所在区域的图像特征联系起来，认为它们二者具有高相关性。但如果在测试集中，“Sport”和人所在区域的图像特征并没有这么高的相关性时（即训练集和测试集的分布不一致），那么在测试集中预测的时候带上这样的偏倚，很可能就会做出错误的预测。

**因果图**

如上面分析，数据集带来了 bias，从而产生了一些虚假的相关性（“Sport”和人图像特征），而建模和消除虚假相关性正是因果理论擅长的事，现在来看看作者是怎么对整个 vision-language 进行因果建模的，当然这是作者自己的想法，因果图并不是唯一确定的东西。

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p14.png" /> 
</div>

首先如第一张图所示，X 表示输入的数据，即原始的句子以及图像特征，Z 代表了句子相关的图像知识，X 和 Z 之间就存在着一个 `X->Z` 的因果关系，因为 Z 是 X 通过 attention 机制生成的嘛。然后利用 Z 去对最终结果 Y 做预测，显然这里也存在着 `Z->Y`，所以第一张图说明了从 `X->Z->Y` 的一条因果路径，即 X 通过 attention 机制做出的预测，这也是本文的重点研究目标。

如果只有这一条路径显然就不存在虚假的相关性，那么下一步作者就对为什么会产生虚假相关性这一点进行了建模，如下图所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p15.png" /> 
</div>

这里 C 表示常识，`C->X` 表明视觉数据或者特征本质上是由常识生成的，例如第一张图中人骑马的图可以认为是常识“人可以骑马”生成的。M 表示 `{person, horse}` 的 object 集，它也是从图像中提取出来的（例如使用 Faster R-CNN），而它本身的值域也是由常识 C 决定的，最后对词的预测是根据 object 集做出的预测，所以是 `M->Y`。

**从因果角度看 Attention**

有了因果图后，首先先从因果角度看下 attention 机制，即 `X->Z->Y` 这条因果路径，传统的模型是基于相关性

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p16.png" /> 
</div>

这里是只考虑 `X->Z->Y` 因果图下的公式，还是比较直观的，`P(Y|Z=z)`表示知识对 `Y` 的预测，`P(Z=z|X)`表示根据 `X` 来选择相关的知识，不同的知识重要程度不同。按照我的理解，z 就是 attention 机制里的 K 和 V，`P(Z=z|X)`其实就是 Q 和 K 求到的权重 α。

但公式里是根据这个 `P(Z=z|X)` 对每个 z 对 Y 的预测结果 `P(Y|Z=z)` 求期望，也就是 IS-Sampling 操作，而 attention 是先根据 α对 z 求了个期望，用这个期望的 z 再去做预测。这个细微的区别我看了几遍论文才看出来，按照作者的意思这两个是等价的，而且由于 attention 是先对输入求了期望，然后光把这个期望值丢进网络 forward 一边，肯定要比把所有输入全部 forward 然后在期望代价要小得多。

论文第 7 部分的公式 (19) 有类似的推导，即公式 (19) 的最后一行，本来按照前面的推导求期望应该停留在 g 外面，一开始不知道为啥作者的推导直接塞到函数的输入里了，后来我觉得应该是反正还不知道拟合结果怎样，那不如就先对输入求个期望，然后对期望 forward 之后的结果，让它和这两个操作反过来（先 forward 再期望）的结果一样不就行了。

总之，attention 的 Q-K-V 操作可以和这个条件概率公式对应起来了。

**消除偏倚**

正如前面构建的因果图，如果直接拟合 `P(Y|X)`会带来 bias，bias 产生的原因是 C 这个 confounder，即 `X<-C->M->Y` 这条非因果路径，由于我们又没有 C 的数据，所以 back-door 是别想了。而我们想求的是 `X->Z->Y`这条因果路径事实上也不需要 C 的数据。首先看 `X->Z`，X 和 Z 之间唯一能让信息流动的就只有这一条，别的路径统统被 `M->Y<-Z` 给对撞没了，所以 X 和 Z 之间没有混杂。

关键是 Z 和 Y 之间存在混杂，不过幸运地是这个混杂可以通过对 X 进行 adjust 给消除掉，而 X 的数据是我们有的，所以接下来就简单了，如下进行 back-door （关于 back-door 可以参考下别人的讲解的，简单来说就是分情况讨论，在不同的 X 下，`P(Y|X=x,Z)` 是该情况下 Z 对 Y 的因果效应，那么根据 X 的不同情况求个平均即可）：

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p17.png" /> 
</div>

CS-Sampling跟上面的IS-Sampling一样也是求期望的操作，区别在于前者是来自于不同的样本，后者仅来自于当前样本。同时为了和 do(X) 里的 X 区分开，这里换成 x′
。在后面会看到作者也和 attention 里的做法类似，直接把 CS-Sampling 丢给输入 Z 了。

有了 X 对 Z 因果以及 Z 到 Y 的因果，那么自然就能得到 X 到 Y 的因果（通过 Z）。结合两个公式，即把 `P(Y|X)` 展开式里的 `P(Y|Z)` 替换为 `P(Y|do(X))`，得到

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p18.png" /> 
</div>

即去偏倚后的 attention 比原来多了一个求期望的步骤。

**IS-ATT(In-Sample Attention) 和 CS-ATT(Cross-Sample Attention)**

本论文核心就是要实现上面这个`P(Y|do(X))`，首先我们先构造一个函数 `g(⋅)`来拟合 `P(Y|Z,X)`
，为了表示分布在 `g`外面套一个 softmax，如下
$$P(Y|Z,X)=Softmax[g(Z,X)]$$

最终结果 `P(Y|do(X))`就是 `P(Y|Z,X)`计算了两次期望（IS-Sampling 以及 CS-Sampling），然后如前面所说，为了减少数据 forward 次数，直接把这两个求期望塞到最原始的输入那里去做（具体推导可以见论文第 7 部分），总之这里直接放结果

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p19.png" /> 
</div>

据这个推导，作者提出了两个 attention，一个就对应了原来的 attention，即 IS-ATT，另一个就是 CS-ATT，如下图

<div align=center>
    <img src="zh-cn/img/ch2/4-2/p20.png" /> 
</div>

---
### 3.BLIP2:Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models


<!-- https://blog.csdn.net/qq_41994006/article/details/129221701 -->

<!-- https://zhuanlan.zhihu.com/p/624647342 -->
<!-- https://zhuanlan.zhihu.com/p/613247637 -->
<!-- https://zhuanlan.zhihu.com/p/606364639 -->


!> https://arxiv.org/pdf/2301.12597.pdf

**TL;DR**

2023 年 Salesforce 出的文章，提出了 BLIP2，一种通用而有效的预训练策略，它从现成的冻结参数的图像编码器和冻结参数的大型语言模型中引导视觉语言预训练，在 Image Captioning、VQA、Image-Text Retrieval（图文检索） 任务的多个数据集上取得 SOTA

**背景**

+ 由于大规模模型的端到端训练，视觉和语言预训练的成本变得越来越高
+ 为了降低计算成本并抵消灾难性遗忘的问题，希望在 Vision-language pre-training (VLP) 中固定视觉模型参数与语言模型参数。然而，由于语言模型在其单模态预训练期间没有看到图像，因此冻结它们使得视觉语言对齐尤其具有挑战性

**本文方案**

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p1.png" /> 
</div>

+ 本文提出了BLIP-2，这是一种通用而有效的预训练策略，它从现成的冻结预训练图像编码器和冻结的大型语言模型中引导视觉语言预训练
+ BLIP-2通过一个轻量级的 Querying Transformer （Q-Former是一个轻量级的 transformer，它使用一组可学习的查询向量来从冻结图像编码器中提取视觉特征，为LLM提供最有用的视觉特征，以输出所需的文本） 弥补了模态 gap，该 Transformer 分两个阶段进行预训练
    - 第一阶段从冻结图像编码器引导视觉-语言表示学习，强制 Q-Former 学习与文本最相关的视觉表示（学习表征特征）
    - 第二阶段基于冻结的语言模型引导从视觉到语言的生成学习，将Q-Former的输出连接到冻结的LLM，并对Q-Former进行训练，使其输出视觉表示能够被LLM解释（生成任务）
+ 由于Image Encoder和LLM是冻结的，因此BLIP2的训练十分高效，比Flamingo少了54倍参数，但性能还提升了8.7%。为什么说BLIP2提出了一个通用的框架，是因为frozen的Image Encoder和LLM可以任意替换，文中Image Encoder是VIT, LLM是Flan T5.

BLIP2 优势：

+ 2 stage 训练（表征学习阶段和生成学习阶段）有效利用冻结的预训练图像模型和语言模型，视觉问答、图像字幕和图像文本检索三个任务上取得了 SOTA
+ 由 LLM (如 OPT、FlanT5) 提供支持，BLIP-2 可以被提示执行遵循自然语言指令的 zero-shot 图像到文本生成，这实现了诸如视觉知识推理、视觉对话等新兴功能
由于使用了冻结的单模态模型和轻量级的Q-Former，BLIP-2比现有技术的计算效率更高。比如，BLIP-2 在 zero-shot VQAv2 上比 Flamingo 高 8.7%，可训练参数减少了 54 倍。

**模型架构**

!> 第一阶段的任务：表征特征

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p2.png" /> 
</div>

+ 提出Q-Former作为可训练模块，以弥合冻结图像编码器和冻结LLM之间的差距。它从图像编码器中提取固定数量的输出特征，与输入图像分辨率无关
+ 如图所示，Q-Former中有两个子模块，一个是Image Transformer，与Image Encoder交互用于视觉特征的抽取，第二个是Text Transformer，其可以用于encode也可以用于decode（与BLIP不同，BLIP中encode和decode的SA层参数不共享。）
+ 值得注意的是Image Transformer的Input， Learned Queries是一定长度的可训练的参数向量，论文采用了32个query embeddding且维度和text保持一致都是768(Query Embedding 就没有[CLS]token的概念)。
+ Q-Former 包含 188M 参数，导入 BERTbase pretrain 参数，cross-attention layers 随机初始化，query 维度 `32×768`（明显小于图像编码器输出特征维度 `257 × 1024`）

训练任务：

+ Image-Text Contrastive Learning（ITC)， 文本端还是采用CLS token，但值得注意的是 Query Embedding 就没有[CLS]token的概念，因此对于一张图片的每一个query，我们都计算其与文本端CLStoken的相似度，并取最大的一个;学习对齐图像表示和文本表示，以使它们的相互信息最大化。通过对比正负对的图像-文本相似性来实现这一点
为了避免信息泄漏，采用了 unimodal self-attention 掩码，不允许 query 和文本相互查看.
+ Image-grounded Text Generation(ITG), 强迫Q-Former去抽取能捕获绝大部分文本信息的视觉特征，注意生成任务的时候用的是Multi-modal Causal Attention，以防数据泄露。训练 Q-Former 生成文本，将输入图像作为条件;由于 Q-Former 的架构不允许冻结图像编码器和文本 tokens 之间的直接交互，因此生成文本所需的信息必须首先由 query 提取，然后通过 self-attention 传递给文本 tokens。因此，query 被迫提取包含文本所有信息的视觉特征;使用 multimodal causal self-attention 掩码来控制 query 与文本交互。query 可以相互关注，但不能关注文本 tokens。每个文本 tokens 都可以处理所有 query 及其以前的文本标记.
+ Image-Text Matching(ITM)，细颗粒的文本与图像的对齐任务。与ITC类似，因为有多个queries，作者会把每一个query和text的CLS放入一个二分类头中获取logit，最后取平均作为最后的output logit。这是一个二进制分类任务，要求模型预测图像文本对是正（匹配）还是负（不匹配）;使用 bi-directional self-attention 掩码，所有查询和文本都可以相互关注。因此，输出的 query embedding 捕获多模态信息


!> 第二阶段任务：文本生成任务

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p3.png" /> 
</div>

+ 在生成预训练阶段，将 QFormer（附带冻结图像编码器）连接到冻结 LLM，以获取LLM 的生成语言能力
+ 使用 FC 层将QFormer 输出的 query embedding 线性投影到与 LLM 的文本 embedding 相同的维度。将投影的 query embedding 附加到输入文本 embedding
    - 它们用作 soft visual prompts，以 Q-Former 提取的视觉表示为 LLM 提供条件
    - 由于 Q-Former 已经有预训练以提取富含语言信息性的视觉表示，因此它有效地充当了一个信息 bottleneck，将最有用的信息提供给 LLM，同时去除不相关的视觉信息，减轻了 LLM 学习视觉语言对齐的负担，从而减轻了**灾难性遗忘问题**
+ 实验两种类型的LLM：基于解码器的LLM和基于编码器-解码器的LLMs
    - 对于基于解码的LLM，对语言建模损失进行预训练，其中冻结的LLM的任务是生成基于Q-Former视觉表示的文本
    - 对于基于编码器-解码器的LLM，使用前缀语言建模损失进行预训练，将文本分成两部分。前缀文本与视觉表示连接，作为LLM编码器的输入。后缀文本用作LLM解码器编码器的生成目标

训练任务：

+ Pre-train 数据和 BLIP 一样
    - 总共 129M 图片，来源于
    - COCO
    - Visual Genome
    - CC3M
    - CC12M
    - SBU
    - LAION400M (115M图片)
    - 使用 CapFilt 方法为网络图片合成 caption，也即基于 BLIPlarge 为每张图生成 10 个 captions。基于 CLIP ViT-L/14 模型产生的图像文本相似性，将合成 caption 与原始网络 caption 一起排序。保持每个图像的前两个 caption 作为训练数据，并在每个预训练步骤中随机抽取一个 caption

+ Pre-trained image encoder and LLM
    - vision transformer：将最后一层删除，使用倒数第二层特征
    - ViT-L/14 from CLIP
    - ViT-G/14 from EVA-CLIP

+ frozen language model
    - decoder-based LLMs： unsupervised-trained OPT
    - encoder-decoder-based LLMs：instruction-trained FlanT5

+ Pre-training settings
    - 一阶段训练 250k steps，二阶段训练 80k steps
    - 第一阶段对ViT-L/ViT-G使用2320/1680的 batchsize，在第二阶段对OPT/FlanT5使用1920/1520的 batchsize
    - 在预训练期间，将冻结的ViTs和LLM参数转换为FP16，但FlanT5除外，使用BFloat16。与 32 bit 模型相比没有精度损失
    - 使用一台16-A100（40G）机器，最大型号ViT-G和FlanT5 XXL第一阶段需要不到6天，第二阶段需要不超过3天


**实验结果**

Instructed Zero-shot Image-to-Text Generation: BLIP2 支持通过指令控制图像到文本的生成，只需在视觉提示后附加文本提示作为LLM的输入。下图展示了各种各样的 zero-shot 图像到文本生成功能的示例，包括视觉知识推理、视觉注释推理、视觉对话、个性化图像到文本生成

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p4.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p5.png" /> 
</div>

Zero-shot VQA:对于 OPT 模型，prompt 为 "Question: {} Answer:"；对于 FlanT5 模型，prompt 为 "Question: {} Short answer:"
还将长度惩罚设置为-1，这鼓励更短的答案与人类注释更一致。BLIP2 在 VQAv2 和 GQA 上取得 SOTA 结果，在 OK-VQA 上差于 Flamingo，原因可能是 OK-VQA 更关注开放世界知识，而不是视觉理解，Flamingo80B 中的 70B Chinchilla 语言模型拥有比11B FlanT5XXL 更多的知识。更强的图像编码器或更强的LLM都会带来更好的性能

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p6.png" /> 
</div>

VQA finetune 实验：

<div align=center>
    <img src="zh-cn/img/ch2/4-3/p7.png" /> 
</div>

finetune 实验配置
+ 微调 Q-Former 和图像编码器的参数，同时保持LLM冻结
+ 为了提取与问题更相关的图像特征，Q-Former 以 question 为条件。具体而言，问题 token 被作为 Q-Former 的输入，并通过 self-attention 与 query 交互，这可以引导 Q-Former cross attention 层关注提供更多信息的图像区域
+ 基于 VQAv2 和 Visual Genome 的数据进行 finetune


**Thoughts**

基于 Q-former 来作为视觉和文本模态的连接模块看起来很合理，有效降低训练的资源消耗
文章的 Limitation 中提到，BLIP2 在 VQA 任务中的 in-context learning 效果一般，作者将缺乏上下文学习能力归因于文章使用的预训练数据集：每个样本只包含一个图像-文本对。导致 LLM 无法从中学习单个序列中多个图像文本对之间的相关性。
Flamingo 为了解决这个问题，使用一个 close-sourced 的交错图像和文本数据集（M3W），每个序列有多个图像-文本对

------
------
## 5. LLaMa：Open and Efficient Foundation Language Models

<!-- https://zhuanlan.zhihu.com/p/617745693 -->

论文介绍了LLaMA，它是一组基础语言模型，参数范围从7B到65B。在数万亿的tokens上训练的模型，并表明可以专门使用公开可用的数据集来训练最先进的模型，而无需求助于专有和不可访问的数据集。特别是，LLaMA-13B在大多数基准测试中都优于GPT-3（175B），并且LLaMA-65B与最好的模型Chinchilla-70B和PaLM-540B具有竞争力。

### 5.1 介绍

在大量文本语料库上训练的大型语言模型（LLM）已经显示出它们从文本指令或几个例子中执行新任务的能力（Brown et al.，2020）。当将模型缩放到足够大时，这些few-shot 特性首次出现（Kaplan等人，2020年），导致一系列工作集中于进一步缩放这些模型（Chowdhery等人，2022年；Rae等人，2021）。这些努力是基于这样一种假设，即更多的参数将带来更好的性能。然而，Hoffmann等人最近的工作（2022）表明，对于给定的计算预算，最佳性能不是通过最大的模型实现的，而是通过在更多数据上训练的较小模型实现的。

Hoffmann等人（2022）的缩放规律的目标是确定如何为特定的训练计算预算最佳地缩放数据集和模型大小。然而，这个目标忽略了推理预算，这在大规模服务于语言模型时变得至关重要。在这种情况下，给定目标性能水平，首选模型不是训练最快的，而是推理最快的。尽管训练大型模型以达到一定的性能水平可能更便宜，但训练时间更长的小型模型最终推理更便宜。例如，尽管Hoffmann等人（2022）建议在200Btoken上训练10B模型，但论文发现即使在1T tokens之后，7B模型的性能也会继续提高。

这项工作的重点是训练一系列语言模型，通过训练比通常使用的tokens更多的tokens，在不同的推理预算下实现尽可能好的性能。由此产生的模型称为LLaMA，其参数范围从7B到65B，与现有的最佳LLM相比具有竞争力。例如，LLaMA-13B在大多数基准测试中都优于GPT-3，尽管它比GPT-3小10倍。我们相信，这个模型将有助于LLM的访问和研究民主化，因为它可以在单个GPU上运行。在规模的高端，论文的65B参数模型也与最好的大型语言模型（如Chinchilla或PaLM-540B）具有竞争力。

与Chinchilla、PaLM或GPT-3不同，论文只使用公开可用的数据，使论文工作与开源兼容，而大多数现有模型依赖于未公开或未记录的数据（例如“Books–2TB”或“Social media conversations”）。存在一些例外，特别是OPT（Zhang等人，2022）、GPT-NeoX（Black等人，2022）。

在本文的其余部分中，论文概述了我们对Transformer架构所做的修改（Vaswani et al.，2017），以及训练方法。然后，报告的模型的性能，并在一组标准基准上与其他LLM进行比较。最后，使用负责任的人工智能社区的一些最新基准，揭示了模型中编码的一些偏见。

### 5.2 方法

论文的训练方法类似于之前工作中描述的方法（Brown et al.，2020；Chowdhery et al.，2022），并受到Chinchilla比例规律的启发（Hoffmann et al.，2021）。使用标准优化器在大量文本数据上训练大型Transformer。

#### 5.2.1 预训练数据集

训练数据集是表1中报告的几个来源的混合，涵盖了一组不同的领域。在大多数情况下，论文重用已被用来训练其他LLM的数据源，但限制只能使用公开可用且与开源兼容的数据。这导致了以下数据及其在训练集中所代表的百分比的混合：

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p1.png" /> 
</div><p align=center> 表1：预训练数据。用于预训练的数据混合，对于每个子集，我们列出了采样比例、在1.4T tokens上训练时在该子集上执行的epochs数以及磁盘大小。1T tokens上的预训练运行具有相同的采样比例。</p>

**英语CommonCrawl[67%]**。论文使用CCNet pipline 预处理了2017年至2020年的五个CommonCrawl 转储（Wenzek et al.，2020）。该过程在行级别消除重复数据，使用fastText线性分类器执行语言识别以删除非英语页面，并使用ngram语言模型过滤低质量内容。此外，训练了一个线性模型来对维基百科中用作参考文献的页面进行分类。随机抽样的页面，以及未被分类为参考文献的丢弃页面。

**C4 [15%]**。在探索性实验中，观察到使用不同的预处理CommonCrawl数据集可以提高性能。因此，将公开可用的C4数据集（Raffel et al.，2020）包含在数据中。C4的预处理还包括重复数据消除和语言识别步骤：与CCNet的主要区别在于质量过滤，它主要依赖于启发式方法，如标点符号的存在或网页中的单词和句子的数量。

**Github [4.5%]**。论文使用Google BigQuery上提供的公共GitHub数据集。只保留了在Apache、BSD和MIT许可证下分发的项目。此外，使用基于行长度或字母数字字符比例的启发式方法过滤低质量文件，并使用正则表达式删除样板文件，如标头。最后，在文件级别对生成的数据集进行重复数据消除，并进行精确匹配。

**维基百科[4.5%]**。添加了2022年6月至8月期间的维基百科转储，涵盖20种语言，使用拉丁语或西里尔文：bg、ca、cs、da、de、en、es、fr、hr、hu、it、nl、pl、pt、ro、ru、sl、sr、sv、uk。论文处理数据以删除超链接、注释和其他格式样板。

**Gutenberg 和 Books3 [4.5%]**。论文在训练数据集中包括两个图书语料库：Gutenberg项目，其中包含公共领域的图书，以及ThePile的Books3部分（Gao et al.，2020），这是一个用于训练大型语言模型的公开数据集。论文在书本本级别执行重复数据消除，删除内容重叠超过90%的书本。

**ArXiv[2.5%]**。我们处理arXiv Latex文件，将科学数据添加到数据集中。继Lewkowycz等人（2022）之后，删除了第一节之前的所有内容以及参考书目。还删除了.tex文件中的注释，并内联扩展了用户编写的定义和宏，以提高论文之间的一致性。

**Stack Exchange [2%]**。包括Stack Exchange，这是一个高质量问答网站，涵盖了从计算机科学到化学的一系列不同领域。保留了28个最大网站的数据，删除了文本中的HTML标签，并按分数（从高到低）对答案进行了排序。

**Tokenizer**。使用字节对编码（BPE）算法（Sennrich et al.，2015）对数据进行标记，使用PensionePiece（Kudo和Richardson，2018）的实现。值得注意的是，论文将所有数字拆分为单个数字，并回退到字节以分解未知的UTF-8字符。

总体而言，整个训练数据集在标记化后包含大约1.4T的tokens。对于大多数训练数据，每个token在训练期间只使用一次，但维基百科和图书领域除外，论文在这两个领域执行了大约两个epochs。

#### 5.2.2 架构

继最近对大型语言模型的研究之后，论文网络基于Transformer架构（Vaswani et al.，2017）。论文利用了随后提出的各种改进，并在不同的模型中使用，如PaLM。以下是与原始建筑的主要区别，以及论文在那里找到了这一变化的灵感：

**预归一化[GPT3]**。为了提高训练稳定性，对每个Transformer子层的输入进行归一化，而不是对输出进行归一化。使用了Zhang和Sennrich（2019）引入的RMSNorm规范化函数。

**SwiGLU激活功能[PaLM]**。用Shazeer（2020）引入的SwiGLU激活函数取代了ReLU非线性，以提高性能。论文使用$\frac{2}{3}4d$的尺寸，而不是PaLM中的4d。

**旋转嵌入[GPTNeo]**。删除了绝对位置嵌入，而是在网络的每一层添加了Su等人（2021）引入的旋转位置嵌入（RoPE）

#### 5.2.3 优化器

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p2.png" /> 
</div><p align=center> 表2：模型大小、体系结构和优化超参数。</p>

论文的模型使用AdamW优化器（Loshchilov和Hutter，2017）进行训练，具有以下超参数：`β1=0.9，β2=0.95`。使用余弦学习率计划，使得最终学习率等于最大学习率的`10%`。论文使用0.1的权重衰减和1.0的梯度剪裁。使用2000个预热步骤，并随着模型的大小而改变学习率和批次大小（详见表2）。

#### 5.2.4 高效实现

论文进行了一些优化，以提高模型的训练速度。首先，使用causal多头注意力的有效实现来减少内存使用和运行时间。此实现在xformers库中提供，受到Rabe和Staats（2021）的启发，并使用了Dao等人（2022年）提供的向后方法。这是通过不存储注意力权重和不计算由于语言模型任务的因果性质而被掩盖的`key/query`分数来实现的。

为了进一步提高训练效率，减少了在带有检查点的后向传球过程中重新计算的激活次数。更准确地说，保存了计算成本高昂的激活，例如线性层的输出。这是通过手动实现Transformer层的向后功能来实现的，而不是依赖PyTorch autograd。如Korthikanti等人所述，为了充分受益于这种优化，需要通过使用模型和序列并行性来减少模型的内存使用。（2022）。此外，论文还尽可能多地重叠激活的计算和GPU之间通过网络的通信（由于all_reduce操作）。

当训练65B参数模型时，在2048 A100 GPU和80GB RAM上处理大约380个tokens/秒/GPU。这意味着，在包含1.4T tokens的数据集上进行训练大约需要21天。

### 5.3 主要结果

根据之前的工作（Brown等人，2020），论文考虑了zero-shot和few-shot任务，并报告了总共20个基准的结果：

**Zero-shot**。论文提供了任务的文本描述和一个测试示例。该模型要么使用开放式生成提供答案，要么对提出的答案进行排名。

**Few-shot**。提供了一些任务示例（介于1和64之间）和一个测试示例。该模型将该文本作为输入，并生成答案或对不同的选项进行排序。

将LLaMA与其他基座模型进行了比较，即非公开可用的语言模型GPT-3（Brown等人，2020）、Gopher（Rae等人，2021）、Chinchilla（Hoffmann等人，2022）和PaLM（Chowdhery等人，2022。在第4节中，还简要比较了LLaMA与OPT-IML（Iyer et al.，2022）和Flan-PaLM（Chung et al.，2021）等指令调优模型。

论文在自由形式生成任务和多选任务上评估LLaMA。在多选任务中，目标是根据提供的上下文，在一组给定的选项中选择最合适的答案。在给定上下文的情况下，选择具有最高可能性的答案。遵循Gao等人（2021）的方法，使用由补全字符数归一化的可能性，但某些数据集（OpenBookQA、BoolQ）除外，论文遵循Brown等人（2020）的方法，给定“Answer"的completion 似然进行归一化：

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p3.png" /> 
</div>

#### 5.3.1 常识推理

考虑了八个标准常识推理基准：BoolQ（Clark等人，2019年）、PIQA（Bisk等人，2020年）、SIQA（Sap等人，201九年）、HellaSwag（Zellers等人，20119年）、WinoGrande（Sakaguchi等人，2021）、ARC easy and challenge（Clarks等人，2018年）和OpenBookQA（Mihaylov等人，2018）。这些数据集包括Cloze和Winograd风格的任务，以及多选问题回答。论文在zero-shot环境中进行评估，就像在语言模型社区中一样。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p4.png" /> 
</div><p align=center> 表3：常识推理任务的zero-shot性能。</p>

在表3中，将不同规模的现有模型和相应论文的报告编号进行了比较。首先，LLaMA-65B在除BoolQ之外的所有报告基准上都优于Chinchilla-70B。同样，除了在BoolQ和WinoGrande上，该模型在任何地方都超过了PaLM-540B。LLaMA-13B模型虽然小了10倍，但在大多数基准测试中也优于GPT-3。

#### 5.3.2 闭卷问答

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p5.png" /> 
</div><p align=center> 表4：自然问题，精确的匹配性能。</p>


<div align=center>
    <img src="zh-cn/img/ch2/4-4/p6.png" /> 
</div><p align=center> 表5:TriviaQA。zero-shot和few-shot在过滤的开发集上精确匹配性能。</p>

论文在两个闭卷问答基准上将LLaMA与现有的大型语言模型进行了比较：自然问题（Kwiatkowski et al.，2019）和TriviaQA（Joshi et al.，2017）。对于这两个基准，报告了在闭卷环境中的精确匹配性能，即模型无法访问包含回答问题的证据的文档。在表4中，报告了NaturalQuestions的性能，在表5中，报告了TriviaQA。在这两个基准上，LLaMA-65B在zero-shot和few-shot设置中实现了最先进的性能。更重要的是，LLaMA-13B在GPT-3和Chinchilla的这些基准测试中也具有竞争力，尽管它比GPT-3小5-10倍。该模型在推理过程中运行在单个V100 GPU上。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p7.png" /> 
</div><p align=center> 图3：Natural Questions（左）和TriviaQA（右）的格式化数据集示例。</p>


#### 5.3.3 阅读理解

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p8.png" /> 
</div><p align=center> 表6：阅读理解。zero-shot精度。</p>

根据RACE阅读理解基准评估模型（Lai et al.，2017）。这个数据集是从为中国中学生和高中生设计的英语阅读理解考试中收集的。遵循Brown等人（2020）的评估设置，并在表6中报告结果。在这些基准测试中，LLaMA-65B与PaLM-540B具有竞争力，并且LLaMA-13B的性能优于GPT-3几个百分点。

#### 5.3.4 数学推理

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p9.png" /> 
</div><p align=center> 表7：定量推理数据集上的模型性能。对于多数投票，使用与Minerva相同的设置，MATH使用k=256个样本，GSM8k使用k=100个样本（Minerva 540B MATH使用k=64个样本，GSM 8k使用k=40个样本）。LLaMA-65B在GSM8k上的表现优于Minerva 62B，尽管它尚未在数学数据上进行微调。</p>


根据两个数学推理基准评估模型：MATH（Hendrycks等人，2021）和GSM8k（Cobbe等人，2021）。MATH是一个用LaTeX编写的12K中学和高中数学问题的数据集。GSM8k是一组中学数学问题。在表7中，与PaLM和Minerva进行了比较（Lewkowycz等人，2022）。Minerva是一系列对从ArXiv和Math网页中提取的38.5B tokens进行微调的PaLM模型，而PaLM和LLaMA都没有对数学数据进行微调。PaLM和Minerva的数字取自Lewkowycz等人（2022），比较有和没有maj1@k。maj1@k表示为每个问题生成k个样本并进行多数投票的评估（Wang et al.，2022）。在GSM8k上，观察到LLaMA-65B的性能优于Minerva-62B，尽管它尚未在数学数据上进行微调。

#### 5.3.5 代码生成

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p10.png" /> 
</div><p align=center> 表8：代码生成的模型性能。报告了HumanEval和MBPP的pass@score。HumanEval的生成是在零样本和MBBP中进行的，带有类似于Austin等人（2021）的3次触发提示。标有*的数值取自Chowdhery等人（2022）的数字。</p>

论文评估了模型在两个基准上根据自然语言描述编写代码的能力：HumanEval（Chen等人，2021）和MBPP（Austin等人，2021）。对于这两项任务，模型都会收到用几句话描述的程序，以及一些输入输出示例。在HumanEval中，它还接收一个函数签名，并且提示被格式化为自然代码，并在文档字符串中包含文本描述和测试。该模型需要生成一个符合描述并满足测试用例的Python程序。在表8中，比较了模型与未在代码上进行微调的现有语言模型的pass@1分数，即PaLM和LaMDA（Thoppilan et al.，2022）。PaLM和LLaMA是在包含相似数量代码tokens的数据集上进行训练的。

如表8所示，对于类似数量的参数，LLaMA优于其他通用模型，如LaMDA和PaLM，这些模型没有专门针对代码进行训练或微调。具有13B参数和更多参数的LLaMA在HumanEval和MBPP上都优于LaMDA-137B。LLaMA-65B的性能也优于PaLM-62B，即使训练时间更长。这个 pass@1 该表中报告的结果是通过在0.1℃下取样获得的。这个pass@100 和 pass@80 在温度为0.8时获得度量。论文使用与Chen等人（2021）相同的方法来获得pass@k。

可以通过对特定于代码的tokens进行微调来提高代码的性能。例如，PaLM编码器（Chowdhery等人，2022）增加了pass@1 PaLM在HumanEval上的得分从PaLM的26.2%上升到36%。其他专门针对代码进行预训练的模型在这些任务上的表现也优于一般模型（Chen等人，2021；Nijkamp等人，2022年；Fried等人，2022.）。对代码tokens的微调超出了本文的范围。

#### 5.3.6 大规模多任务语言理解

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p11.png" /> 
</div><p align=center> 表9：大规模多任务语言理解（MMLU）。5-shot精度。</p>

Hendrycks等人引入的大规模多任务语言理解基准（MMLU）。（2020）由涵盖人文学科、STEM和社会科学等各个知识领域的多项选择题组成。论文使用基准提供的示例，在5-shot设置中评估模型，并在表9中报告结果。在这个基准上，观察到LLaMA-65B在大多数领域中平均落后于Chinchilla70B和PaLM-540B几个百分点。一个潜在的解释是，论文在训练前的数据中使用了有限数量的书籍和学术论文，即ArXiv、Gutenberg和Books3，总计只有177GB，而这些模型是在高达2TB的书籍上训练的。Gopher、Chinchilla和PaLM使用的大量书籍也可以解释为什么Gopher在这个基准上优于GPT-3，而在其他基准上却具有可比性。

#### 5.3.7 训练期间性能的演变

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p12.png" /> 
</div><p align=center> 图1:7B、13B、33B和65模型的训练tokens的训练损失。LLaMA-33B和LLaMA65B在1.4T tokens上进行训练。较小的模型在1.0T tokens上进行训练。所有模型都使用4M个tokens的批次大小进行训练。</p>

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p13.png" /> 
</div><p align=center> 图2：训练期间问答和常识推理表现的演变。</p>

在训练过程中，论文跟踪了模型在一些问答和常识基准上的性能，并在图2中进行了报告。在大多数基准测试中，性能稳步提高，并与模型的训练困惑相关（见图1）。SIQA和WinoGrande是例外。最值得注意的是，在SIQA上，观察到性能有很多差异，这可能表明该基准不可靠。在WinoGrande上，表现与训练困惑度并不相关：LLaMA-33B和LLaMA-65B在训练中表现相似。

### 5.4 指令微调

在本节中，展示了对指令数据的短暂微调可以快速改进MMLU。尽管LLaMA-65B的非微调版本已经能够遵循基本指令，但论文观察到，非常少量的微调可以提高MMLU的性能，并进一步提高模型遵循指令的能力。由于这不是本文的重点，论文只进行了一个实验，遵循与Chung等人相同的协议。（2022）来训练指令模型LLaMA-I。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p14.png" /> 
</div><p align=center> 表10：指令微调–MMLU（5-shot）。在MMLU上进行指令微调和不进行指令微调的中等尺寸模型的比较。</p>

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p15.png" /> 
</div><p align=center> 表16:MMLU。测试集上每个域的详细5-shot结果。</p>

在表10中，报告了MMLU上的指令模型LLaMA-I的结果，并与现有的中等规模的指令微调模型进行了比较，即OPT-IML（Iyer et al.，2022）和Flan-PaLM系列（Chung et al.，2021）。所有报告的数字都来自相应的论文。尽管这里使用的指令微调方法很简单，但论文在MMLU上达到了68.9%。LLaMA-I（65B）在MMLU上的性能优于现有的中等大小的指令微调模型，但仍远未达到最先进的水平，即MMLU上GPT代码-davinci-002的77.4（数字取自Iyer等人（2022））。MMLU在57个任务上的性能细节可以在附录的表16中找到。


### 5.5 偏见、毒性和错误信息

大型语言模型已被证明可以再现和放大训练数据中存在的偏见（Sheng等人，2019；Kurita等人，2019），并生成有毒或攻击性内容（Gehman等人，2020）。由于论文的训练数据集包含很大一部分来自Web的数据，论文认为确定论文的模型生成此类内容的潜力至关重要。为了了解LLaMA-65B的潜在危害，论文在不同的基准上进行了评估，这些基准衡量了有毒成分的产生和刻板印象的检测。虽然我们选择了语言模型社区使用的一些标准基准来表明这些模型的一些问题，但这些评估不足以充分理解与这些模型相关的风险。


#### 5.5.1 RealToxicityPrompts

语言模型可以产生有毒的语言，例如侮辱、仇恨言论或威胁。一个模型可以产生非常大范围的毒性内容，这使得彻底的评估具有挑战性。最近的几项工作（Zhang等人，2022；Hoffmann等人，2022）认为真实毒性提示基准（Gehman等人，2020）是他们模型毒性的指标。真实毒性提示由模型必须完成的大约10万个提示组成；通过向PerspectiveAPI 发出请求来自动评估毒性评分。无法控制第三方PerspectiveAPI使用的管道，因此很难与以前的模型进行比较。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p16.png" /> 
</div><p align=center> 表11：真实毒性提示。在这个基准测试的100k提示上运行贪婪解码器。“尊重”版本是以“以礼貌、尊重和公正的方式完成以下句子：”开头的提示，而“基本”则没有。分数是使用困惑度API获得的，分数越高表示毒性越大。</p>

对于每一个10万个提示，论文都贪婪地用论文的模型生成，并测量它们的毒性评分。每个提示的得分范围从0（无毒）到1（有毒）。在表11中，报告了Real ToxicityPrompts的基本提示和尊重提示类别的平均得分。这些分数与论文在文献中观察到的分数“相当”（例如，Chinchilla的分数为0.087），但这些工作与论文的方法不同（在采样策略、提示次数和API时间方面）。观察到，毒性随着模型的大小而增加，尤其是对于尊重提示。这在之前的工作中也观察到了（Zhang et al.，2022），但Hoffmann et al.（2022）除外，尽管Chinchilla和Gopher的大小不同，但他们没有看到它们之间的区别。这可以解释为，更大的模型Gopher的性能比Chinchilla差，这表明毒性和模型大小之间的关系可能只适用于模型家族。

#### 5.5.2 CrowS-Pairs

论文评估了CrowSPairs模型中的偏差（Nangia等人，2020）。该数据集可以测量9类偏见：性别、宗教、种族/肤色、性取向、年龄、国籍、残疾、外表和社会经济地位。每个例子都由一个刻板印象和一个反刻板印象组成，论文使用两个句子在zero-shot设置下的复杂度来衡量刻板印象句子的模型偏好。因此，分数越高，则表示偏见越大。论文与表12中的GPT-3和OPT-175B进行了比较。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p16.png" /> 
</div><p align=center>表12：CrowS对。比较了LLaMA-65B与OPT-175B和GPT3-175B中包含的偏差水平。得分越高，表示偏见越大。</p>

LLaMA的平均值与这两种模型相比略为有利。论文的模型在宗教类别上尤其有偏见（与OPT-175B相比增加了10%），其次是年龄和性别。尽管有多个过滤步骤，但论文预计这些偏见会来自CommonCrawl。

#### 5.5.3 WinoGender

为了进一步调查论文的模型对性别类别的偏见，论文查看了WinoGender基准（Rudinger et al.，2018），这是一个共同参考的分辨率数据集。WinoGender是由Winograd模式构成的，通过确定模型共同参考解决性能是否受到代词性别的影响来评估偏见。

更确切地说，每个句子有三个提及：“职业”、“参与者”和“代词”，其中代词共同指代职业或参与者。论文提示模型确定共指关系，并根据句子的上下文来衡量它是否正确。其目的是揭示与职业相关的社会偏见是否已被该模型所捕捉。例如，WinoGender数据集中的一句话是“护士通知患者他的轮班将在一小时后结束。论文评估了使用三个代词时的表现：“她/她/她”、“他/他/他”和“他们/他们/某人”（与代词的语法功能相对应的不同选择）。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p17.png" /> 
</div><p align=center>表13：Wino性别。LLaMA模型对不同代词（“她/她/她”和“他/他/他”）的共同参考解析准确性。观察到，论文的模型在“他们/他们/某人”代词上的表现比在“她/她”和“他/他/他”上的表现更好，这可能表明存在偏见。</p>

在表13中，报告了数据集中包含的三种不同代词的共同参考分数。观察到，与“她/她/她”和“他/他/他”代词相比，论文的模型在对“他们/他们/某人”代词执行共同参考解析方面明显更好。在之前的工作中也进行了类似的观察（Rae等人，2021；Hoffmann等人，2022年），这可能表明存在性别偏见。事实上，在“她/她/她”和“他/他/他”代词的情况下，模型可能使用职业的多数性别来执行共同参考解决，而不是使用句子的证据。

为了进一步研究这一假设，查看了WinoGender数据集中“她/她”和“他/他/他”代词的“gotcha”格集。这些情况对应于代词与职业的大多数性别不匹配的句子，而职业是正确的答案。在表13中，观察到我们的模型LLaMA-65B在gotcha例子中犯了更多的错误，清楚地表明它捕捉到了与性别和职业相关的社会偏见。“她/她/她”和“他/他/他”代词的表现有所下降，这表明无论性别如何，都存在偏见。

#### 5.5.4 TruthfulQA

TruthfulQA（Lin等人，2021）旨在衡量模型的真实性，即其识别声明真实性的能力。Lin等人（2021）考虑了“真实”的定义，即“真实世界的文字真相”，而不是仅在信仰体系或传统背景下才是真实的主张。该基准可以评估模型产生错误信息或虚假声明的风险。这些问题以不同的风格写成，涵盖38个类别，并被设计成对抗性的。

<div align=center>
    <img src="zh-cn/img/ch2/4-4/p18.png" /> 
</div><p align=center>表14：真实质量保证。报告了经过专门训练的模型通过OpenAI API评分的真实和真实*信息性答案的分数。遵循Ouyang等人使用的QA提示风格。（2022），并从同一篇论文中报告了GPT-3的性能。</p>

在表14中，报告了论文的模型在两个问题上的性能，以衡量真实模型以及真实和信息的交叉点。与GPT-3相比，论文的模型在这两个类别中的得分都更高，但正确答案的比率仍然很低，**这表明论文的模型很可能会产生错误答案的幻觉。**

### 5.6 碳足迹

论文模型的训练消耗了大量的能量，导致了二氧化碳的排放。遵循了最近关于这一主题的文献，并在表15中对总能源消耗和由此产生的碳足迹进行了细分。论文遵循Wu等人的公式。（2022）估计训练模型所需的瓦时Wh，以及碳排放吨tCO2eq。对于Wh，使用以下公式：

`Wh = GPU-h×(GPU power consumption)×PUE`

其中将功率使用效率（PUE）设置为1.1。由此产生的碳排放取决于用于训练网络的数据中心的位置。例如，BLOOM使用的网格排放0.057千克二氧化碳当量/千瓦时，导致27吨二氧化碳当量，OPT使用的网格释放0.231千克二氧化碳当量/KWh，导致82吨二氧化碳当量。在这项研究中，论文有兴趣比较在同一数据中心训练这些模型的碳排放成本。因此，论文没有考虑数据中心的位置，而是使用0.385 kg CO2eq/KWh的美国全国平均碳强度因子。这导致了以下碳排放量的公式：

`tCO2eq = MWh × 0.385`

为了进行公平的比较，将相同的公式应用于OPT和BLOOM。对于OPT，假设992 A100-80B需要34天的训练（见他们的日志4）。最后，论文估计在大约5个月的时间里，使用了2048个A100-80GB来开发我们的模型。这意味着，在论文的假设下，开发这些模型的成本约为2638兆瓦时，总排放量为1015吨二氧化碳当量。论文希望发布这些模型将有助于减少未来的碳排放，因为训练已经完成，而且其中一些模型相对较小，可以在单个GPU上运行。

!> 关于LLaMA-65B的生成的一些实例参见paper

### 5.7 高效微调技术QLoRA实战，基于LLaMA-65B微调仅需48G显存，真香

!> 感谢吃果冻不吐果冻皮 大佬

<!-- https://mp.weixin.qq.com/s/b4OixyHEvL_YfOJZukC2Ig -->

<!-- https://zhuanlan.zhihu.com/p/619426866 -->

1. 环境搭建

基础环境配置如下：

+ 操作系统: CentOS 7
+ CPUs: 单个节点具有 1TB 内存的 Intel CPU，物理CPU个数为64，每颗CPU核数为16
+ GPUs: 8 卡 A800 80GB GPUs
+ Python: 3.10 (需要先升级OpenSSL到1.1.1t版本（点击下载OpenSSL），然后再编译安装Python)，点击下载Python
+ NVIDIA驱动程序版本: 515.65.01，根据不同型号选择不同的驱动程序，点击下载。
+ CUDA工具包: 11.7，点击下载
+ NCCL: nccl_2.14.3-1+cuda11.7，点击下载
+ cuDNN: 8.8.1.3_cuda11，点击下载

上面的NVIDIA驱动、CUDA、Python等工具的安装就不一一赘述了。

创建虚拟环境并激活虚拟环境（qlora-venv-py310-cu117）：

```
cd /home/guodong.li/virtual-venv 
virtualenv -p /usr/bin/python3.10 qlora-venv-py310-cu117 
source /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/bin/activate
```

安装transformers、accelerate、peft库。

```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 8f093fb
pip install .

git clone https://github.com/huggingface/accelerate.git
cd accelerate/
git checkout 665d518
pip install .

git clone https://github.com/huggingface/peft.git
cd peft/
git checkout 189a6b8
pip install .
```

安装其他依赖库：

```
pip install -r requirements.txt
```

其中，requirements.txt内容如下：

```
bitsandbytes==0.39.0
einops==0.6.1
evaluate==0.4.0
scikit-learn==1.2.2
sentencepiece==0.1.99
tensorboardX
```

2. 数据集准备

数据集直接使用alpaca-lora项目提供的`alpaca_data.json`、`alpaca_data_cleaned_archive.json`或`alpaca_data_gpt4.json`即可。

3. 模型权重格式转换

首先，对原始的 LLaMA 30B/65B 大模型进行模型权重格式转换为Huggingface Transformers格式。模型转换的具体步骤请参考之前的文章：[从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://mp.weixin.qq.com/s/I4h3WXGwqEPVKbgy-BmpoA)。

本文会使用到 LLaMA 7B 和 65B 模型，需预先转换好。

4. 模型微调

```
git clone https://github.com/artidoro/qlora.git
cd qlora
git checkout cc48811

python qlora.py \
--dataset "/data/nfs/guodong.li/data/alpaca_data_cleaned.json" \
--model_name_or_path "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b" \
--output_dir "/home/guodong.li/output/llama-7b-qlora" \
--per_device_train_batch_size 1 \
--max_steps 1000 \
--save_total_limit 2
```

模型情况下，会将模型的不同层放置在不同层已进行模型并行。

模型训练过程：

```
python qlora.py \
> --dataset "/data/nfs/guodong.li/data/alpaca_data_cleaned.json" \
> --model_name_or_path "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b"  \
> --output_dir "/home/guodong.li/output/llama-7b-qlora"  \
> --per_device_train_batch_size 1 \
> --max_steps 1000 \
> --save_total_limit 2

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
/home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/opt/rh/devtoolset-9/root/usr/lib/dyninst'), PosixPath('/opt/rh/devtoolset-7/root/usr/lib/dyninst')}
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda-11.7/lib64/libcudart.so.11.0
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
Found a previous checkpoint at: /home/guodong.li/output/llama-7b-qlora/checkpoint-250
loading base model /data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b...
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████| 33/33 [00:17<00:00,  1.93it/s]
Loading adapters from checkpoint.
trainable params: 79953920.0 || all params: 3660320768 || trainable: 2.184341894267557
loaded model
Adding special tokens.
Found cached dataset json (/home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0)
Loading cached split indices for dataset at /home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0/cache-d071c407d9bc0de0.arrow and /home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0/cache-e736a74b2c29e789.arrow
Loading cached processed dataset at /home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0/cache-01d50099f3f094d7.arrow
torch.float32 422326272 0.11537932153507864
torch.uint8 3238002688 0.8846206784649213
{'loss': 1.4282, 'learning_rate': 0.0002, 'epoch': 0.0}
{'loss': 1.469, 'learning_rate': 0.0002, 'epoch': 0.01}
...
{'loss': 1.4002, 'learning_rate': 0.0002, 'epoch': 0.08}
{'loss': 1.4261, 'learning_rate': 0.0002, 'epoch': 0.08}
{'loss': 2.4323, 'learning_rate': 0.0002, 'epoch': 0.09}
 25%|██████████████████████▎                                                                  | 250/1000 [25:34<1:10:31,  5.64s/it]Saving PEFT checkpoint...
{'loss': 1.6007, 'learning_rate': 0.0002, 'epoch': 0.09}
{'loss': 1.6187, 'learning_rate': 0.0002, 'epoch': 0.09}
...
{'loss': 1.6242, 'learning_rate': 0.0002, 'epoch': 0.16}
{'loss': 1.6073, 'learning_rate': 0.0002, 'epoch': 0.16}
{'loss': 1.6825, 'learning_rate': 0.0002, 'epoch': 0.17}
{'loss': 2.6283, 'learning_rate': 0.0002, 'epoch': 0.17}
 50%|█████████████████████████████████████████████▌                                             | 500/1000 [50:44<49:21,  5.92s/it]Saving PEFT checkpoint...
{'loss': 1.619, 'learning_rate': 0.0002, 'epoch': 0.17}
{'loss': 1.5394, 'learning_rate': 0.0002, 'epoch': 0.18}
...
{'loss': 1.5247, 'learning_rate': 0.0002, 'epoch': 0.25}
{'loss': 1.6054, 'learning_rate': 0.0002, 'epoch': 0.25}
{'loss': 2.3289, 'learning_rate': 0.0002, 'epoch': 0.26}
 75%|██████████████████████████████████████████████████████████████████▊                      | 750/1000 [1:15:27<23:37,  5.67s/it]Saving PEFT checkpoint...
{'loss': 1.6001, 'learning_rate': 0.0002, 'epoch': 0.26}
...
{'loss': 1.6287, 'learning_rate': 0.0002, 'epoch': 0.34}
{'loss': 2.3511, 'learning_rate': 0.0002, 'epoch': 0.34}
100%|████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [1:42:08<00:00,  7.34s/it]Saving PEFT checkpoint...
{'train_runtime': 6132.3668, 'train_samples_per_second': 2.609, 'train_steps_per_second': 0.163, 'train_loss': 1.7447978076934814, 'epoch': 0.34}
100%|████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [1:42:12<00:00,  6.13s/it]
Saving PEFT checkpoint...
***** train metrics *****
  epoch                    =       0.34
  train_loss               =     1.7448
  train_runtime            = 1:42:12.36
  train_samples_per_second =      2.609
  train_steps_per_second   =      0.163
```

模型输出权重文件：

```
tree -h llama-7b-qlora
llama-7b-qlora
├── [ 167]  all_results.json
├── [ 316]  checkpoint-1000
│   ├── [ 528]  adapter_config.json
│   ├── [  75]  adapter_model
│   │   ├── [ 528]  adapter_config.json
│   │   ├── [610M]  adapter_model.bin
│   │   └── [  27]  README.md
│   ├── [610M]  adapter_model.bin
│   ├── [  21]  added_tokens.json
│   ├── [3.1G]  optimizer.pt
│   ├── [  27]  README.md
│   ├── [ 14K]  rng_state.pth
│   ├── [ 627]  scheduler.pt
│   ├── [  96]  special_tokens_map.json
│   ├── [ 742]  tokenizer_config.json
│   ├── [488K]  tokenizer.model
│   ├── [ 11K]  trainer_state.json
│   └── [5.6K]  training_args.bin
├── [ 316]  checkpoint-750
│   ├── [ 528]  adapter_config.json
│   ├── [  75]  adapter_model
│   │   ├── [ 528]  adapter_config.json
│   │   ├── [610M]  adapter_model.bin
│   │   └── [  27]  README.md
│   ├── [610M]  adapter_model.bin
│   ├── [  21]  added_tokens.json
│   ├── [3.1G]  optimizer.pt
│   ├── [  27]  README.md
│   ├── [ 14K]  rng_state.pth
│   ├── [ 627]  scheduler.pt
│   ├── [  96]  special_tokens_map.json
│   ├── [ 742]  tokenizer_config.json
│   ├── [488K]  tokenizer.model
│   ├── [8.0K]  trainer_state.json
│   └── [5.6K]  training_args.bin
├── [   0]  completed
├── [ 199]  metrics.json
├── [ 11K]  trainer_state.json
└── [ 167]  train_results.json

4 directories, 35 files
```

显存占用：

```
> nvidia-smi
Sun Jun 11 19:32:39 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:34:00.0 Off |                    0 |
| N/A   40C    P0    66W / 300W |   3539MiB / 81920MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:35:00.0 Off |                    0 |
| N/A   54C    P0    77W / 300W |   3077MiB / 81920MiB |     24%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:36:00.0 Off |                    0 |
| N/A   55C    P0    75W / 300W |   3077MiB / 81920MiB |      8%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:37:00.0 Off |                    0 |
| N/A   57C    P0    81W / 300W |   3077MiB / 81920MiB |     14%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   4  NVIDIA A800 80G...  Off  | 00000000:9B:00.0 Off |                    0 |
| N/A   60C    P0    83W / 300W |   3077MiB / 81920MiB |      8%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   5  NVIDIA A800 80G...  Off  | 00000000:9C:00.0 Off |                    0 |
| N/A   61C    P0   228W / 300W |   3077MiB / 81920MiB |     25%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   6  NVIDIA A800 80G...  Off  | 00000000:9D:00.0 Off |                    0 |
| N/A   53C    P0   265W / 300W |   3077MiB / 81920MiB |      6%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   7  NVIDIA A800 80G...  Off  | 00000000:9E:00.0 Off |                    0 |
| N/A   46C    P0    78W / 300W |   6891MiB / 81920MiB |     12%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     37939      C   python                           2513MiB |
|    1   N/A  N/A     37939      C   python                           2819MiB |
|    2   N/A  N/A     37939      C   python                           2819MiB |
|    3   N/A  N/A     37939      C   python                           2819MiB |
|    4   N/A  N/A     37939      C   python                           2819MiB |
|    5   N/A  N/A     37939      C   python                           2819MiB |
|    6   N/A  N/A     37939      C   python                           2819MiB |
|    7   N/A  N/A     37939      C   python                           3561MiB |
+-----------------------------------------------------------------------------+
```


5. 模型权重合并

新增模型权重合并文件（export_hf_checkpoint.py），将lora权重合并回原始权重。

```python
import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

BASE_MODEL = os.environ.get("BASE_MODEL", None)
LORA_MODEL = os.environ.get("LORA_MODEL", "tloen/alpaca-lora-7b")
HF_CHECKPOINT = os.environ.get("HF_CHECKPOINT", "./hf_ckpt")



assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=decapoda-research/llama-7b-hf`"  # noqa: E501

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    #load_in_8bit=False,
    torch_dtype=torch.bfloat16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    # TODO
    # "tloen/alpaca-lora-7b",
    LORA_MODEL,
    #device_map={"": "cpu"},
    #torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights
for layer in lora_model.base_model.model.model.layers:
    layer.self_attn.q_proj.merge_weights = True
    layer.self_attn.v_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
#assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, HF_CHECKPOINT , state_dict=deloreanized_sd, max_shard_size="400MB"
)

```

接下来，就可以使用合并后的权重文件进行模型推理了。

6. 模型推理

新增推理代码（`inference.py`）：

```python
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

model_id = "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b"
merge_model_id = "/home/guodong.li/output/llama-7b-merge"

#model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(merge_model_id, load_in_4bit=True, device_map="auto")

tokenizer = LlamaTokenizer.from_pretrained(model_id)

#print(model)

device = torch.device("cuda:0")

#model = model.to(device)

text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n------------------------------------------------\nInput: ")

line = input()
while line:
  inputs = tokenizer(line, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
  print("Output: ",tokenizer.decode(outputs[0], skip_special_tokens=True))
  print("\n------------------------------------------------\nInput: ")
  line = input()

```

运行过程：


```
> CUDA_VISIBLE_DEVICES=1  python inference.py

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
/home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/opt/rh/devtoolset-9/root/usr/lib/dyninst'), PosixPath('/opt/rh/devtoolset-7/root/usr/lib/dyninst')}
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda-11.7/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████| 39/39 [00:07<00:00,  5.02it/s]
Hello, my name is 23 and i have been doing this for the last 6 months. I have been a great

------------------------------------------------
Input:
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Input:\n\n\n### Response:
Output:  Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGive three tips for staying healthy.\n\n### Input:\n\n\n### Response: 1. Eat healthy food.\n2. Stay active.\n3. Eat

------------------------------------------------
Input:

```

显存占用：

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    1   N/A  N/A     21373      C   python                           5899MiB |
+-----------------------------------------------------------------------------+

```

除此之外，还可以不进行合并权重，直接进行推理，具体如下所示。

新增推理代码（inference_qlora.py）：

```python

from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel

model_id = "/data/nfs/guodong.li/pretrain/hf-llama-model/llama-7b"
lora_weights = "/home/guodong.li/output/llama-7b-qlora/checkpoint-1000/adapter_model"

#model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(
    model,
    lora_weights,
)



tokenizer = LlamaTokenizer.from_pretrained(model_id)

#print(model)

device = torch.device("cuda:0")

#model = model.to(device)

text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n------------------------------------------------\nInput: ")

line = input()
while line:
  inputs = tokenizer(line, return_tensors="pt").to(device)
  outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
  print("Output: ",tokenizer.decode(outputs[0], skip_special_tokens=True))
  print("\n------------------------------------------------\nInput: ")
  line = input()
```

显存占用：

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    1   N/A  N/A     10500      C   python                           7073MiB |
+-----------------------------------------------------------------------------+


```
可以看到，此时模型推理的显存占用会高于合并之后进行模型推理。

当然，将lora权重合并会base模型权重还可以通过merge_and_unload()方法，如下所示：

```python
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(
    model,
    lora_weights,
)
model = model.merge_and_unload()
```

前面仅对7B模型进行了尝试，而 LLaMA-65B 模型对于显存的占用效果如何呢，是否如官方所说仅需48G显存足矣了呢？带着疑问，接下来我们使用QLoRA对LLaMA-65B进行微调。


!> 微调LLaMA-65B大模型

模型训练过程：

```
CUDA_VISIBLE_DEVICES=0 python qlora.py \
>     --model_name_or_path /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b \
>     --dataset /data/nfs/guodong.li/data/alpaca_data_cleaned.json \
>     --output_dir /home/guodong.li/output/llama-65b-qlora \
>     --logging_steps 10 \
>     --save_strategy steps \
>     --data_seed 42 \
>     --save_steps 100 \
>     --save_total_limit 2 \
>     --evaluation_strategy steps \
>     --eval_dataset_size 128 \
>     --max_eval_samples 200 \
>     --per_device_eval_batch_size 1 \
>     --max_new_tokens 32 \
>     --dataloader_num_workers 3 \
>     --group_by_length \
>     --logging_strategy steps \
>     --remove_unused_columns False \
>     --do_train \
>     --do_eval \
>     --do_mmlu_eval \
>     --lora_r 64 \
>     --lora_alpha 16 \
>     --lora_modules all \
>     --double_quant \
>     --quant_type nf4 \
>     --bf16 \
>     --bits 4 \
>     --warmup_ratio 0.03 \
>     --lr_scheduler_type constant \
>     --gradient_checkpointing \
>     --source_max_len 16 \
>     --target_max_len 512 \
>     --per_device_train_batch_size 1 \
>     --gradient_accumulation_steps 16 \
>     --max_steps 200 \
>     --eval_steps 50 \
>     --learning_rate 0.0001 \
>     --adam_beta2 0.999 \
>     --max_grad_norm 0.3 \
>     --lora_dropout 0.05 \
>     --weight_decay 0.0 \
>     --seed 0 \
>     --report_to tensorboard

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please run

python -m bitsandbytes

 and submit this information together with your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
bin /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so
/home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/cuda_setup/main.py:149: UserWarning: WARNING: The following directories listed in your path were found to be non-existent: {PosixPath('/opt/rh/devtoolset-7/root/usr/lib/dyninst'), PosixPath('/opt/rh/devtoolset-9/root/usr/lib/dyninst')}
  warn(msg)
CUDA SETUP: CUDA runtime path found: /usr/local/cuda-11.7/lib64/libcudart.so
CUDA SETUP: Highest compute capability among GPUs detected: 8.0
CUDA SETUP: Detected CUDA version 117
CUDA SETUP: Loading binary /home/guodong.li/virtual-venv/qlora-venv-py310-cu117/lib/python3.10/site-packages/bitsandbytes/libbitsandbytes_cuda117.so...
loading base model /data/nfs/guodong.li/pretrain/hf-llama-model/llama-65b...
The model weights are not tied. Please use the `tie_weights` method before using the `infer_auto_device` function.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [01:33<00:00,  1.16s/it]
adding LoRA modules...
trainable params: 399769600.0 || all params: 33705172992 || trainable: 1.1860778762206212
loaded model
Adding special tokens.
Found cached dataset json (/home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0)
Loading cached split indices for dataset at /home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0/cache-298a54784863252c.arrow and /home/guodong.li/.cache/huggingface/datasets/json/default-3c2be6958ca766f9/0.0.0/cache-e827ad98bd5ab470.arrow
Splitting train dataset in train and validation according to `eval_dataset_size`
Found cached dataset json (/home/guodong.li/.cache/huggingface/datasets/json/default-a08e5825b0ce557e/0.0.0/e347ab1c932092252e717ff3f949105a4dd28b27e842dd53157d2f72e276c2e4)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 704.21it/s]
torch.bfloat16 1323843584 0.039277144217520744
torch.uint8 32380026880 0.9606837249535352
torch.float32 1318912 3.913082894407767e-05
{'loss': 1.5995, 'learning_rate': 0.0001, 'epoch': 0.0}
{'loss': 1.6043, 'learning_rate': 0.0001, 'epoch': 0.01}
{'loss': 1.7943, 'learning_rate': 0.0001, 'epoch': 0.01}
{'loss': 1.9854, 'learning_rate': 0.0001, 'epoch': 0.01}
{'loss': 2.5809, 'learning_rate': 0.0001, 'epoch': 0.02}
{'eval_loss': 2.077033519744873, 'eval_runtime': 101.312, 'eval_samples_per_second': 1.263, 'eval_steps_per_second': 1.263, 'epoch': 0.02}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [38:19<00:00,  1.50s/it]
{'mmlu_loss': 0.5965077246348893, 'mmlu_eval_accuracy_professional_accounting': 0.5483870967741935, 'mmlu_eval_accuracy_business_ethics': 0.7272727272727273, 'mmlu_eval_accuracy_international_law': 0.8461538461538461, 'mmlu_eval_accuracy_high_school_world_history': 0.6538461538461539, 'mmlu_eval_accuracy_college_physics': 0.45454545454545453, 'mmlu_eval_accuracy_public_relations': 0.6666666666666666, 'mmlu_eval_accuracy_management': 0.7272727272727273, 'mmlu_eval_accuracy_marketing': 0.88, 'mmlu_eval_accuracy_high_school_microeconomics': 0.5, 'mmlu_eval_accuracy_anatomy': 0.5714285714285714, 'mmlu_eval_accuracy_high_school_european_history': 0.7777777777777778, 'mmlu_eval_accuracy_high_school_government_and_politics': 0.7619047619047619, 'mmlu_eval_accuracy_college_mathematics': 0.2727272727272727, 'mmlu_eval_accuracy_logical_fallacies': 0.7222222222222222, 'mmlu_eval_accuracy_high_school_computer_science': 0.5555555555555556, 'mmlu_eval_accuracy_high_school_us_history': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_biology': 0.625, 'mmlu_eval_accuracy_formal_logic': 0.2857142857142857, 'mmlu_eval_accuracy_computer_security': 0.5454545454545454, 'mmlu_eval_accuracy_security_studies': 0.5185185185185185, 'mmlu_eval_accuracy_human_sexuality': 0.5833333333333334, 'mmlu_eval_accuracy_astronomy': 0.5625, 'mmlu_eval_accuracy_elementary_mathematics': 0.34146341463414637, 'mmlu_eval_accuracy_machine_learning': 0.45454545454545453, 'mmlu_eval_accuracy_moral_scenarios': 0.49, 'mmlu_eval_accuracy_college_chemistry': 0.125, 'mmlu_eval_accuracy_sociology': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_statistics': 0.2608695652173913, 'mmlu_eval_accuracy_high_school_chemistry': 0.3181818181818182, 'mmlu_eval_accuracy_philosophy': 0.7647058823529411, 'mmlu_eval_accuracy_virology': 0.5555555555555556, 'mmlu_eval_accuracy_electrical_engineering': 0.3125, 'mmlu_eval_accuracy_prehistory': 0.6, 'mmlu_eval_accuracy_high_school_mathematics': 0.20689655172413793, 'mmlu_eval_accuracy_professional_law': 0.4176470588235294, 'mmlu_eval_accuracy_high_school_macroeconomics': 0.6046511627906976, 'mmlu_eval_accuracy_world_religions': 0.8421052631578947, 'mmlu_eval_accuracy_college_biology': 0.625, 'mmlu_eval_accuracy_college_computer_science': 0.36363636363636365, 'mmlu_eval_accuracy_college_medicine': 0.36363636363636365, 'mmlu_eval_accuracy_miscellaneous': 0.7093023255813954, 'mmlu_eval_accuracy_professional_medicine': 0.5483870967741935, 'mmlu_eval_accuracy_nutrition': 0.5757575757575758, 'mmlu_eval_accuracy_jurisprudence': 0.5454545454545454, 'mmlu_eval_accuracy_us_foreign_policy': 0.9090909090909091, 'mmlu_eval_accuracy_global_facts': 0.4, 'mmlu_eval_accuracy_medical_genetics': 0.9090909090909091, 'mmlu_eval_accuracy_moral_disputes': 0.5526315789473685, 'mmlu_eval_accuracy_abstract_algebra': 0.18181818181818182, 'mmlu_eval_accuracy_conceptual_physics': 0.38461538461538464, 'mmlu_eval_accuracy_econometrics': 0.5, 'mmlu_eval_accuracy_human_aging': 0.7391304347826086, 'mmlu_eval_accuracy_professional_psychology': 0.5217391304347826, 'mmlu_eval_accuracy_high_school_physics': 0.23529411764705882, 'mmlu_eval_accuracy_clinical_knowledge': 0.4482758620689655, 'mmlu_eval_accuracy_high_school_geography': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_psychology': 0.85, 'mmlu_eval_accuracy': 0.5572183480994843, 'epoch': 0.02}
{'loss': 1.6049, 'learning_rate': 0.0001, 'epoch': 0.02}
{'loss': 1.5043, 'learning_rate': 0.0001, 'epoch': 0.02}
{'loss': 1.5604, 'learning_rate': 0.0001, 'epoch': 0.03}
{'loss': 1.6828, 'learning_rate': 0.0001, 'epoch': 0.03}
{'loss': 2.3214, 'learning_rate': 0.0001, 'epoch': 0.03}
{'eval_loss': 1.8286590576171875, 'eval_runtime': 157.8957, 'eval_samples_per_second': 0.811, 'eval_steps_per_second': 0.811, 'epoch': 0.03}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [39:05<00:00,  1.53s/it]
{'mmlu_loss': 0.6160509743618856, 'mmlu_eval_accuracy_professional_accounting': 0.4838709677419355, 'mmlu_eval_accuracy_business_ethics': 0.7272727272727273, 'mmlu_eval_accuracy_international_law': 0.8461538461538461, 'mmlu_eval_accuracy_high_school_world_history': 0.7307692307692307, 'mmlu_eval_accuracy_college_physics': 0.45454545454545453, 'mmlu_eval_accuracy_public_relations': 0.5833333333333334, 'mmlu_eval_accuracy_management': 0.7272727272727273, 'mmlu_eval_accuracy_marketing': 0.84, 'mmlu_eval_accuracy_high_school_microeconomics': 0.5384615384615384, 'mmlu_eval_accuracy_anatomy': 0.5714285714285714, 'mmlu_eval_accuracy_high_school_european_history': 0.8333333333333334, 'mmlu_eval_accuracy_high_school_government_and_politics': 0.8095238095238095, 'mmlu_eval_accuracy_college_mathematics': 0.36363636363636365, 'mmlu_eval_accuracy_logical_fallacies': 0.7222222222222222, 'mmlu_eval_accuracy_high_school_computer_science': 0.5555555555555556, 'mmlu_eval_accuracy_high_school_us_history': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_biology': 0.46875, 'mmlu_eval_accuracy_formal_logic': 0.2857142857142857, 'mmlu_eval_accuracy_computer_security': 0.45454545454545453, 'mmlu_eval_accuracy_security_studies': 0.5555555555555556, 'mmlu_eval_accuracy_human_sexuality': 0.5, 'mmlu_eval_accuracy_astronomy': 0.6875, 'mmlu_eval_accuracy_elementary_mathematics': 0.43902439024390244, 'mmlu_eval_accuracy_machine_learning': 0.5454545454545454, 'mmlu_eval_accuracy_moral_scenarios': 0.4, 'mmlu_eval_accuracy_college_chemistry': 0.0, 'mmlu_eval_accuracy_sociology': 0.8181818181818182, 'mmlu_eval_accuracy_high_school_statistics': 0.2608695652173913, 'mmlu_eval_accuracy_high_school_chemistry': 0.2727272727272727, 'mmlu_eval_accuracy_philosophy': 0.8235294117647058, 'mmlu_eval_accuracy_virology': 0.5555555555555556, 'mmlu_eval_accuracy_electrical_engineering': 0.3125, 'mmlu_eval_accuracy_prehistory': 0.6, 'mmlu_eval_accuracy_high_school_mathematics': 0.20689655172413793, 'mmlu_eval_accuracy_professional_law': 0.38235294117647056, 'mmlu_eval_accuracy_high_school_macroeconomics': 0.5348837209302325, 'mmlu_eval_accuracy_world_religions': 0.7894736842105263, 'mmlu_eval_accuracy_college_biology': 0.75, 'mmlu_eval_accuracy_college_computer_science': 0.18181818181818182, 'mmlu_eval_accuracy_college_medicine': 0.45454545454545453, 'mmlu_eval_accuracy_miscellaneous': 0.6976744186046512, 'mmlu_eval_accuracy_professional_medicine': 0.5806451612903226, 'mmlu_eval_accuracy_nutrition': 0.6060606060606061, 'mmlu_eval_accuracy_jurisprudence': 0.5454545454545454, 'mmlu_eval_accuracy_us_foreign_policy': 0.9090909090909091, 'mmlu_eval_accuracy_global_facts': 0.3, 'mmlu_eval_accuracy_medical_genetics': 1.0, 'mmlu_eval_accuracy_moral_disputes': 0.5526315789473685, 'mmlu_eval_accuracy_abstract_algebra': 0.36363636363636365, 'mmlu_eval_accuracy_conceptual_physics': 0.34615384615384615, 'mmlu_eval_accuracy_econometrics': 0.5, 'mmlu_eval_accuracy_human_aging': 0.8260869565217391, 'mmlu_eval_accuracy_professional_psychology': 0.5507246376811594, 'mmlu_eval_accuracy_high_school_physics': 0.058823529411764705, 'mmlu_eval_accuracy_clinical_knowledge': 0.41379310344827586, 'mmlu_eval_accuracy_high_school_geography': 0.8636363636363636, 'mmlu_eval_accuracy_high_school_psychology': 0.8666666666666667, 'mmlu_eval_accuracy': 0.5582642812271578, 'epoch': 0.03}
 50%|███████████████████████████████████████████████████████████████████████████████                                                                               | 100/200 [2:29:20<1:21:41, 49.01s/it]Saving PEFT checkpoint...
{'loss': 1.5671, 'learning_rate': 0.0001, 'epoch': 0.04}
{'loss': 1.468, 'learning_rate': 0.0001, 'epoch': 0.04}
{'loss': 1.6495, 'learning_rate': 0.0001, 'epoch': 0.04}
{'loss': 1.6844, 'learning_rate': 0.0001, 'epoch': 0.05}
{'loss': 2.384, 'learning_rate': 0.0001, 'epoch': 0.05}
{'eval_loss': 1.7745016813278198, 'eval_runtime': 105.4656, 'eval_samples_per_second': 1.214, 'eval_steps_per_second': 1.214, 'epoch': 0.05}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [38:49<00:00,  1.52s/it]
{'mmlu_loss': 0.6844602518810469, 'mmlu_eval_accuracy_professional_accounting': 0.4838709677419355, 'mmlu_eval_accuracy_business_ethics': 0.7272727272727273, 'mmlu_eval_accuracy_international_law': 0.8461538461538461, 'mmlu_eval_accuracy_high_school_world_history': 0.6923076923076923, 'mmlu_eval_accuracy_college_physics': 0.45454545454545453, 'mmlu_eval_accuracy_public_relations': 0.5833333333333334, 'mmlu_eval_accuracy_management': 0.7272727272727273, 'mmlu_eval_accuracy_marketing': 0.84, 'mmlu_eval_accuracy_high_school_microeconomics': 0.46153846153846156, 'mmlu_eval_accuracy_anatomy': 0.5, 'mmlu_eval_accuracy_high_school_european_history': 0.8333333333333334, 'mmlu_eval_accuracy_high_school_government_and_politics': 0.8095238095238095, 'mmlu_eval_accuracy_college_mathematics': 0.2727272727272727, 'mmlu_eval_accuracy_logical_fallacies': 0.7222222222222222, 'mmlu_eval_accuracy_high_school_computer_science': 0.5555555555555556, 'mmlu_eval_accuracy_high_school_us_history': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_biology': 0.46875, 'mmlu_eval_accuracy_formal_logic': 0.2857142857142857, 'mmlu_eval_accuracy_computer_security': 0.45454545454545453, 'mmlu_eval_accuracy_security_studies': 0.5925925925925926, 'mmlu_eval_accuracy_human_sexuality': 0.5833333333333334, 'mmlu_eval_accuracy_astronomy': 0.6875, 'mmlu_eval_accuracy_elementary_mathematics': 0.4878048780487805, 'mmlu_eval_accuracy_machine_learning': 0.6363636363636364, 'mmlu_eval_accuracy_moral_scenarios': 0.4, 'mmlu_eval_accuracy_college_chemistry': 0.125, 'mmlu_eval_accuracy_sociology': 0.8181818181818182, 'mmlu_eval_accuracy_high_school_statistics': 0.2608695652173913, 'mmlu_eval_accuracy_high_school_chemistry': 0.2727272727272727, 'mmlu_eval_accuracy_philosophy': 0.7941176470588235, 'mmlu_eval_accuracy_virology': 0.5555555555555556, 'mmlu_eval_accuracy_electrical_engineering': 0.375, 'mmlu_eval_accuracy_prehistory': 0.6, 'mmlu_eval_accuracy_high_school_mathematics': 0.20689655172413793, 'mmlu_eval_accuracy_professional_law': 0.38235294117647056, 'mmlu_eval_accuracy_high_school_macroeconomics': 0.5116279069767442, 'mmlu_eval_accuracy_world_religions': 0.8421052631578947, 'mmlu_eval_accuracy_college_biology': 0.625, 'mmlu_eval_accuracy_college_computer_science': 0.2727272727272727, 'mmlu_eval_accuracy_college_medicine': 0.4090909090909091, 'mmlu_eval_accuracy_miscellaneous': 0.6976744186046512, 'mmlu_eval_accuracy_professional_medicine': 0.5806451612903226, 'mmlu_eval_accuracy_nutrition': 0.6060606060606061, 'mmlu_eval_accuracy_jurisprudence': 0.5454545454545454, 'mmlu_eval_accuracy_us_foreign_policy': 0.9090909090909091, 'mmlu_eval_accuracy_global_facts': 0.3, 'mmlu_eval_accuracy_medical_genetics': 1.0, 'mmlu_eval_accuracy_moral_disputes': 0.5263157894736842, 'mmlu_eval_accuracy_abstract_algebra': 0.36363636363636365, 'mmlu_eval_accuracy_conceptual_physics': 0.38461538461538464, 'mmlu_eval_accuracy_econometrics': 0.3333333333333333, 'mmlu_eval_accuracy_human_aging': 0.8260869565217391, 'mmlu_eval_accuracy_professional_psychology': 0.5507246376811594, 'mmlu_eval_accuracy_high_school_physics': 0.11764705882352941, 'mmlu_eval_accuracy_clinical_knowledge': 0.41379310344827586, 'mmlu_eval_accuracy_high_school_geography': 0.8181818181818182, 'mmlu_eval_accuracy_high_school_psychology': 0.8166666666666667, 'mmlu_eval_accuracy': 0.5564941809356316, 'epoch': 0.05}
{'loss': 1.4593, 'learning_rate': 0.0001, 'epoch': 0.05}
{'loss': 1.4768, 'learning_rate': 0.0001, 'epoch': 0.06}
{'loss': 1.4924, 'learning_rate': 0.0001, 'epoch': 0.06}
{'loss': 1.6138, 'learning_rate': 0.0001, 'epoch': 0.07}
{'loss': 2.2459, 'learning_rate': 0.0001, 'epoch': 0.07}
{'eval_loss': 1.798527479171753, 'eval_runtime': 101.7857, 'eval_samples_per_second': 1.258, 'eval_steps_per_second': 1.258, 'epoch': 0.07}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [38:25<00:00,  1.51s/it]
{'mmlu_loss': 0.6745707825225292, 'mmlu_eval_accuracy_professional_accounting': 0.4838709677419355, 'mmlu_eval_accuracy_business_ethics': 0.6363636363636364, 'mmlu_eval_accuracy_international_law': 0.8461538461538461, 'mmlu_eval_accuracy_high_school_world_history': 0.6923076923076923, 'mmlu_eval_accuracy_college_physics': 0.36363636363636365, 'mmlu_eval_accuracy_public_relations': 0.6666666666666666, 'mmlu_eval_accuracy_management': 0.8181818181818182, 'mmlu_eval_accuracy_marketing': 0.8, 'mmlu_eval_accuracy_high_school_microeconomics': 0.6153846153846154, 'mmlu_eval_accuracy_anatomy': 0.5714285714285714, 'mmlu_eval_accuracy_high_school_european_history': 0.7777777777777778, 'mmlu_eval_accuracy_high_school_government_and_politics': 0.8095238095238095, 'mmlu_eval_accuracy_college_mathematics': 0.36363636363636365, 'mmlu_eval_accuracy_logical_fallacies': 0.7222222222222222, 'mmlu_eval_accuracy_high_school_computer_science': 0.5555555555555556, 'mmlu_eval_accuracy_high_school_us_history': 0.8181818181818182, 'mmlu_eval_accuracy_high_school_biology': 0.53125, 'mmlu_eval_accuracy_formal_logic': 0.21428571428571427, 'mmlu_eval_accuracy_computer_security': 0.6363636363636364, 'mmlu_eval_accuracy_security_studies': 0.6296296296296297, 'mmlu_eval_accuracy_human_sexuality': 0.5833333333333334, 'mmlu_eval_accuracy_astronomy': 0.75, 'mmlu_eval_accuracy_elementary_mathematics': 0.3902439024390244, 'mmlu_eval_accuracy_machine_learning': 0.5454545454545454, 'mmlu_eval_accuracy_moral_scenarios': 0.41, 'mmlu_eval_accuracy_college_chemistry': 0.125, 'mmlu_eval_accuracy_sociology': 0.7727272727272727, 'mmlu_eval_accuracy_high_school_statistics': 0.34782608695652173, 'mmlu_eval_accuracy_high_school_chemistry': 0.3181818181818182, 'mmlu_eval_accuracy_philosophy': 0.7941176470588235, 'mmlu_eval_accuracy_virology': 0.5555555555555556, 'mmlu_eval_accuracy_electrical_engineering': 0.25, 'mmlu_eval_accuracy_prehistory': 0.6285714285714286, 'mmlu_eval_accuracy_high_school_mathematics': 0.2413793103448276, 'mmlu_eval_accuracy_professional_law': 0.4294117647058823, 'mmlu_eval_accuracy_high_school_macroeconomics': 0.6046511627906976, 'mmlu_eval_accuracy_world_religions': 0.8421052631578947, 'mmlu_eval_accuracy_college_biology': 0.625, 'mmlu_eval_accuracy_college_computer_science': 0.18181818181818182, 'mmlu_eval_accuracy_college_medicine': 0.45454545454545453, 'mmlu_eval_accuracy_miscellaneous': 0.7209302325581395, 'mmlu_eval_accuracy_professional_medicine': 0.5161290322580645, 'mmlu_eval_accuracy_nutrition': 0.6060606060606061, 'mmlu_eval_accuracy_jurisprudence': 0.45454545454545453, 'mmlu_eval_accuracy_us_foreign_policy': 0.8181818181818182, 'mmlu_eval_accuracy_global_facts': 0.5, 'mmlu_eval_accuracy_medical_genetics': 0.9090909090909091, 'mmlu_eval_accuracy_moral_disputes': 0.6052631578947368, 'mmlu_eval_accuracy_abstract_algebra': 0.2727272727272727, 'mmlu_eval_accuracy_conceptual_physics': 0.34615384615384615, 'mmlu_eval_accuracy_econometrics': 0.5, 'mmlu_eval_accuracy_human_aging': 0.8260869565217391, 'mmlu_eval_accuracy_professional_psychology': 0.5652173913043478, 'mmlu_eval_accuracy_high_school_physics': 0.35294117647058826, 'mmlu_eval_accuracy_clinical_knowledge': 0.5172413793103449, 'mmlu_eval_accuracy_high_school_geography': 0.8181818181818182, 'mmlu_eval_accuracy_high_school_psychology': 0.85, 'mmlu_eval_accuracy': 0.5715981488410985, 'epoch': 0.07}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [4:57:24<00:00, 38.35s/it]Saving PEFT checkpoint...
{'train_runtime': 17857.1127, 'train_samples_per_second': 0.179, 'train_steps_per_second': 0.011, 'train_loss': 1.7639566564559936, 'epoch': 0.07}
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [4:57:37<00:00, 89.29s/it]
Saving PEFT checkpoint...
***** train metrics *****
  epoch                    =       0.07
  train_loss               =      1.764
  train_runtime            = 4:57:37.11
  train_samples_per_second =      0.179
  train_steps_per_second   =      0.011
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 128/128 [01:40<00:00,  1.27it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1531/1531 [38:35<00:00,  1.51s/it]
***** eval metrics *****
  epoch                   =       0.07
  eval_loss               =     1.7985
  eval_runtime            = 0:01:42.39
  eval_samples_per_second =       1.25
  eval_steps_per_second   =       1.25

```

显存占用：

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     16138      C   python                          44515MiB |
+-----------------------------------------------------------------------------+
```

可以看到，对于显存的占用不到48G，当然使用QLoRA微调模型的速度要慢于使用LoRA进行微调。具体原因请查看大模型参数高效微调技术原理综述（五）-LoRA、AdaLoRA、QLoRA一文了解其技术原理。

本文讲述了使用高效微调技术QLoRA训练LLaMA大模型，并讲述了如何将lora权重合并到原始模型权重以及模型推理。

------
------
## 6. Alpaca： A Strong, Replicable Instruction-Following Model

<!-- https://crfm.stanford.edu/2023/03/13/alpaca.html -->

<!-- https://zhuanlan.zhihu.com/p/621592728 -->

<!-- https://blog.csdn.net/LZL2020LZL/article/details/130145555 -->

<!-- https://zhuanlan.zhihu.com/p/615227156 -->

<!-- https://zhuanlan.zhihu.com/p/631811216 -->

<!-- https://zhuanlan.zhihu.com/p/609636289 -->

<!-- https://blog.csdn.net/v_JULY_v/article/details/129709105 -->

<!-- https://blog.csdn.net/qq_41185868/article/details/129775107 -->

!> 原文地址： https://crfm.stanford.edu/2023/03/13/alpaca.html

LLaMA是Meta于2023年2月发布的模型集合（参数量7B/13B/33B/65B），其中LLaMA-13B在大多数数据集上超过了GPT3（175B），LLaMA-65B达到了和Chinchilla-70B、PaLM-540B相当的水平。初此之外，LLaMA模型所使用的训练语料都是开源语料（1.4T tokens）；模型结构上，LLaMA在Transformer-Decoder-Only基础上引入预归一（参考GPT3）、SwiGLU激活函数（参考PaLM）和旋转位置编码（参考GPTNeo）；算力资源上，65B模型使用2048张A100 80G，按照每张卡每秒处理380个token来算，训完1.4T token需要21天。LLaMA暂不支持商用。

### 6.1 概述

诸如GPT-3.5(text-davinci-003)、ChatGPT、Claude、Bing Chat等指令跟随模型已经快速的变强。很多用户经常与这种模型进行交互，甚至已经使用在了工作中。然而，尽管得到了广泛传播，指令跟随模型仍然存在很多不足，比如：它们可能生成错误信息、传播社会刻板印象、生产有毒的语言。

为了在这些紧迫性问题上取得最大进展，学术界的参与是至关重要的。不幸的是，学术界在指令跟随模型上很难进行学术研究，因为没有在能力上接近闭源模型（比如OpenAI的text-davinci-003）的可访问模型。

我们正在发布一个关于指令跟随的语言模型的研究成果，又称作Alpaca，它是在Meta的LLaMA 7B模型上进行微调得到的。Alpaca采用52K的指令跟随说明，使用self-instruct方法在text-davinci-003上生成的。在很多的[self-instruct](https://arxiv.org/abs/2212.10560)评测集上，Alpaca表现出与OpenAI的text-davinci-003相当的效果，但是Alpaca模型对比Text-davinci-003小很多且容易得到。

我们正在发布训练细节和数据，并且未来会开源模型的权重。为了研究界能够更好地体验Alpaca的效果，同时也提供了一个[交互式的界面](https://alpaca-ai.ngrok.io/)。交互式访问可以暴露出意外的功能和故障，这将引导我们未来对这些模型的评测。

!> 我们重申Alpaca仅仅用作学术研究，禁止任何形式的商业应用，这样的决定主要有三点考虑：

+ Alpaca是基于LLaMA，LLaMA是没有商业版权的；
+ instruction数据是基于OpenAI的text-davinci-003，其禁止用作OpenAI的竞争；
+ 还没有设计足够的安全策略，Alpaca还没有准备好作为通用工具。


### 6.2 训练方法

在学术领域训练一个高质量的指令跟随模型主要有两项重要的挑战：一个强大的预训练语言模型和高质量的指令跟随数据。第一个挑战是通过Meta的新模型LLaMA解决的，第二个挑战，通过self-instruct论文中提出的通过现有大语言模型进行自动生成指令数据的方法。特别地，Alpaca使用LLaMA-7B模型进行有监督的微调，指令数据使用OpenAI的text-davinci-003生成的52K的指令跟随说明。

下图表明了我们如何得到的Alpaca模型。数据方面，借鉴了self-instruct的方法，使用self-instruct seed set中175个人工撰写的instruction-output对。然后使用text-davinci-003通过in-context-learning生成更多的指令。最终生成了52K不同的指令和相应的输出，使用OpenAI API花费不超过`$500`。

<div align=center>
    <img src="zh-cn/img/ch2/4-5/p1.jpg" /> 
</div>

有了指令跟随数据，我们使用`Hugging Face`的训练框架微调LLaMA模型，此训练框架支持数据并行和混合精度训练。整个训练过程使用`8张`80GB的`A100`训练了3个小时，总共花费大概也就`$100`。


### 6.3 初步评价

为了评价Alpaca，我们在self-instruct的验证集上进行了人工评价。这个评价数据集是self-instruct作者收集的，覆盖了一系列面向用户的指令，包括：写邮件、社交媒体和生产力工具。评价方法是在text-davinci-003和Alpaca 7B上进行成对盲评。我们发现在这两个模型上的效果非常相似，Alpaca对比text-davinci-003的结果是90:89。

我们对这样的结果感到很惊讶，在更小模型尺寸下使用少量的指令跟随数据竟然可以取得相当的效果。此外，我们在很多交互式的示例上也发现Alpaca 7B与text-davinci-003效果相当。同时，我们也承认在评价数据规模上和多样性上有局限，所以开放demo面向大众测试。如下是一些Alpaca 7B上的效果展示。

<div align=center>
    <img src="zh-cn/img/ch2/4-5/p2.png" /> 
</div>


### 6.4 已知的局限性

Alpaca表现出了语言模型上的几个常见的缺陷，包括：幻觉、有毒性、模式化思维。其中幻觉是Alpaca的常见问题，甚至于text-davinci-003对比也是如此。

例如，在下图中，Alpaca错误地说坦桑尼亚的首都是达累斯萨拉姆，达累斯萨拉姆是坦桑尼亚最大的城市.

<div align=center>
    <img src="zh-cn/img/ch2/4-5/p3.png" /> 
</div>

此外，Alpaca也容易一本正经的传播错误信息。比如

<div align=center>
    <img src="zh-cn/img/ch2/4-5/p4.png" /> 
</div>

### 6.5 未来方向

未来Alpaca可能的研究方向主要包括：

+ 评估：我们需要更严格地评估羊驼。
+ 安全性：未来可能进一步研究Alpaca的风险，提升其安全性，方法可能包括：automatic red teaming, auditing, and adaptive testing.
+ 理解：从训练范式上希望更深入的理解如何提升能力。


### 6.6 使用LoRA对Chinese-LLaMa-Alpaca进行微调
<!-- https://zhuanlan.zhihu.com/p/631811216 -->

!> github: https://github.com/taishan1994/Chinese-LLaMA-Alpaca-LoRA-Tuning

**1.  Chinese-LLaMA-Alpaca-LoRA-Tuning**

使用LoRA对Chinese-LLaMA-Alpaca进行微调。整体的结构非常简单，构造好相应格式的数据后就可以开始训练。

Facebook官方发布的[LLaMA模型](https://github.com/facebookresearch/llama)禁止商用，并且官方没有正式开源模型权重（虽然网上已经有很多第三方的下载地址）。为了遵循相应的许可，目前暂时无法发布完整的模型权重，敬请各位理解（目前国外也是一样）。自行搜索下载地址。

训练好的实体识别LoRA权重已经位于checkpoint下。

**2. 依赖**

Linux操作系统为Ubantu，GPU为A40-48G显存。

```
mpi4py
transformers==4.28.1
peft==0.3.0
icetk
deepspeed==0.9.2
accelerate
cpm_kernels
sentencepiece==0.1.99
peft=0.3.0
torch=2.0.0 
```

**3. 说明**

+ 目录结构

```
--checkpoint：保存模型
----msra：数据集名称
--------model_adapter
------------train_deepspeed
----------------adapter_model
--------------------adapter_config.json
--------------------adapter_model.bin
--------------------train_args.json
------------train_trainer
----------------adapter_model
--------------------adapter_config.json
--------------------adapter_model.bin
--------------------train_args.json
--model_hub：预训练模型
----7B：英文LLaMA原始权重
----7B-hf：英文权重转换为hugging face格式权重
----chinese-llama-plus-lora-7b：中文llama-7b的lora权重
----chinese-alpaca-plus-lora-7b：中文alpaca-7b的lora权重
----chinese-alpaca-7b：合并lora后的最终的模型
----tokenizer.model：7B文件
----convert_llama_weights_to_hf.py
----merge_llama_with_chinese_lora.py
--data：数据
----msra：数据集名称
--------instruct_data：指令数据
------------dev.txt
------------train.txt
--------ori_data：原始数据
--chat_ner.py：闲聊
--train_deepspeed.py：使用原生deepspeed训练
--train_trainer.py： 使用transformers的Trainer进行训练
--test.py：测试训练好的模型
--predict.py：预测
--process.py：处理数据为instruct_data
--dataset.py：加载数据为相应的格式
--deepspeed.json：deepspeed配置文件，用于trasnformers的Trainer
--config_utils.py：用于用字典定义配置，并接收命令行参数

```

+ 转换得到中文alpaca

1、下载好7B、[llama-lora](https://huggingface.co/ziqingyang/chinese-llama-plus-lora-7b)、[alpaca-lora](https://huggingface.co/ziqingyang/chinese-alpaca-plus-lora-7b)到model_hub下。进入到model_hub目录下。

2、将llama转换为hugging face支持的格式：`python convert_llama_weights_to_hf.py --input_dir ./ --model_size 7B --output_dir ./7B-hf`。如果报错：`If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0`则可以`pip install --upgrade protobuf==3.20.1`，然后：`python convert_llama_weights_to_hf.py --input_dir ./ --model_size tokenizer_only --output_dir ./7B-hf`。最终我们可以得到`7B-hf`。

3、合并lora到llama上：`python merge_llama_with_chinese_lora.py --base_model "./7B-hf" --lora_model "./chinese-llama-plus-lora-7b,chinese-alpaca-plus-lora-7b" --output_type "huggingface" --output_dir "./chinese-alpaca-7b" `。最终我们可以得到`chinese-alpaca-7b`。

4、回到主目录，进行闲聊验证是否得到正确的模型：`python chat_ner.py --base_model "./model_hub/chinese-alpaca-7b" --tokenizer_path "./model_hub/chinese-alpaca-7b" --with_prompt --interactive`

+ 数据格式

这里我们以命名实体识别任务为例，数据在data/msra下，其中ori_data为原始数据,instruct_data为处理后的数据，数据格式为一条一个样本，具体是：

```
{"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。", "query": "文本：因有关日寇在京掠夺文物详情，藏界较为重视，也是我们收藏北京史料中的要件之一。", "answer": "日_地名\n京_地名\n北京_地名"}
```

可以按照自己的任务自行构建。

+ 一般过程

1、data下新建数据集，用process.py处理数据为instruct_data下的数据。

2、这里使用`train_trainer.py`进行训练，为了能够让transformers的Trainer在训练的过程中保存lora权重，对Trainer进行相应的修改，参考：[https://github.com/huggingface/peft/issues/96](https://github.com/huggingface/peft/issues/96) 。因为有了`config_utils.py`，我们可以在字典里面定义相关参数，然后可以在命令行修改参数的值（嵌套参数之间用`_`分隔）。

```
args = {
    "data_name": "msra",  # 数据集名称
    "model_dir": "/root/autodl-tmp/chatglm-6b/",  # chatglm-6b地址，修改为自己的路径
    "lora_r": 8,  # lora参数
    "max_seq_length": 128+64,  # 句子最大长度
    "instruct_column": "instruct",  # instruct列名
    "query_column": "query",  # query列名
    "response_column": "answer",  # answer列名
    "train_path": "data/msra/instruct_data/train.txt", # 训练数据，修改为自己数据
    "dev_path": "data/msra/instruct_data/dev.txt",  # 测试数据，修改为自己数据
    "train_batch_size": 12,  # 训练batch_size
    "gradient_accumulation_steps": 1,  # 默认就好
    "save_dir": "。/checkpoint/msra/train_trainer/",  # 保存模型位置，修改为自己的路径
    "num_train_epochs": 1,  # 训练epoch
    "local_rank": -1,  # deepspeed所需，默认就好
    "log_steps": 10,  # 多少步打印一次结果
    "save_steps": 50,  # 多少步保存一次模型
    "deepspeed_json_path": "deepspeed.json" # deepspeed配置
}
```

需要注意的是，Trainer中使用deepspeed要保持deepspeed定义的参数和Trainer里面参数保持一致，比如：`deepspeed.json`：

```
{
  "train_micro_batch_size_per_gpu": 12,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 1e-05,
      "betas": [
        0.9,
        0.95
      ],
      "eps": 1e-08,
      "weight_decay": 0.0005
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 1,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000.0,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000.0,
    "contiguous_gradients": true
  }
}

```

train_micro_batch_size_per_gpu和per_device_train_batch_size

lr和learning_rate

betas里面和adam_beta1、adam_beta2

weight_decay和weight_decay

fp16和fp16

默认的话不用修改这些。

+ 训练

```
deeepspeed train_deepspeed.py 或者 deepspeed train_trainer.py
```

+ 测试

修改data_name，运行：`python test.py`

```
预测： ['_地名\n中国_地名\n朝鲜_地名\n台湾_地名\n', '释迦_人名', '邓小平_人名\n日本_地名', '最高人民法院_机构名', '铁道部_机构名', '元_地名', '美_地名\n台_地名', '长安_地名', '京沪高速铁路_地名\n沪沪铁路_地名', '人民大会堂_地名 ', '_地名\n中国共产党_机构名', '长城_机构名\n王朝公司_机构名', '玉峰_地名', '_地名', '_地名', '开来_人名\n北京开来律师事务所_机构名', '_地名女儿_地名', '海道新干线_机构名', '瑞士_地名\n西班牙_地名\n比利时_地名\n丹麦_地名', '玉峰山_地名\n栋梁河_地名']

真实： ['日本_地名\n中国_地名\n朝鲜_地名\n台湾_地名', '释迦_人名', '邓小平_人名\n日本_地名', '最高人民法院_机构名', '铁道部_机构名', '岭南_地名', '美_地名\n台_地名', '长安_地名', '京沪高速铁路_地名\n京沪铁路_地名', '人民大会堂_地名', '中国_地名\n中国共产党_机构名', '长城_机构名\n王朝公司_机构名', '玉峰_地名', '中国_地名', '中国_地名', '开来_人名\n北京开来律师事务所_机构名', '美国_地名\n中_地名', '东海道新干线_地名', '瑞士_地名\n西班牙_地名\n比利时_地名\n丹麦_地名', '玉峰山_地名\n栋梁河_地名']
```


+  预测

修改data_name，运行：`python predict.py`

```
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。"
预测 >>>  郑振铎_人名
阿英_人名
国民党_机构名
真实 >>>  郑振铎_人名
阿英_人名
国民党_机构名
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。"
预测 >>>  市_地名
真实 >>>  北京市_地名
文本 >>>  "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。"
预测 >>>  姜德明_人名
真实 >>>  姜德明_人名

```

+ 闲聊

修改data_name，运行：`python chat_ner.py --base_model "./model_hub/chinese-alpaca-7b" --tokenizer_path "./model_hub/chinese-alpaca-7b" --lora_model "./checkpoint/msra/train_trainer/adapter_model" --with_prompt --interactive`

```
+ 该模式下仅支持单轮问答，无多轮对话能力。
+ 如要进行多轮对话，请使用llama.cpp或llamachat工具。
-------------------------------------------------------------------------------------
+ This mode only supports single-turn QA.
+ If you want to experience multi-turn dialogue, please use llama.cpp or llamachat.
=====================================================================================
Input:你好  
Response:  Hello!


Input:你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。
Response:  郑振铎_人名
阿英_人名


Input:你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。
Response:  北京市_地名


Input:你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。
Response:  姜德明_人名
```

原始模型也并没有退化。

+ 报错解决

安装mpi4py报错

```
sudo apt-get update
sudo apt-get install openmpi-bin libopenmpi-dev
pip install mpi4py
```

**4. 补充**

+ 怎么训练自己的数据？ 按照instruct_data下的数据结构构造数据，定义好相关参数运行即可。
+ 怎么进行预测？ 在test.py中，预测时可根据自己的任务进行解码。
+ 为什么不进行评价指标的计算？ 只是作了初步的训练，难免效果不太好就不进行评价指标的计算了，可以在test.py里面自行定义。


**参考**

[从0到1复现斯坦福羊驼（Stanford Alpaca 7B）](https://mp.weixin.qq.com/s/I4h3WXGwqEPVKbgy-BmpoA)

### 6.7 visual-med-alpaca
<!-- https://github.com/cambridgeltl/visual-med-alpaca -->

!> github: https://github.com/cambridgeltl/visual-med-alpaca

!> Blog: https://cambridgeltl.github.io/visual-med-alpaca/

**1.Abstract**

本文提出Visual Med-Alpaca，是一个开源的、参数高效的生物医学基础模型，可以与医学“视觉专家”集成以进行多模态生物医学任务。该模型建立在LLaMa-7B架构上，使用由GPT-3.5-Turbo和人类专家协作策划的指令集进行训练。利用几个小时的指令调整和即插即用的视觉模块，Visual Med-Alpaca 可以执行各种医学任务。


<div align=center>
    <img src="zh-cn/img/ch2/4-6/demo.gif" /> 
</div>

!> 警告：仅供学术使用，不适用于真实的临床场景！

**2.Overview**

领域特定基础模型在生物医学领域发挥着关键作用，因为生物医学文本中使用的语言是高度专业化的，通常包含维基百科和图书等一般领域文本语料库中没有的领域特定概念和关系。经验证据表明，与现有的公开预训练语言模型（PLM）相比，对大量生物医学文本进行预训练显著提高了语言模型在各种生物医学文本挖掘任务中的性能（Lee等人，2019；Gururangan等人，2020，Gu等人，2021）。

现代大型语言模型（LLM）需要前所未有的计算资源水平来进行全模型微调。
对于大多数学术机构来说，即使是在PubMed上专门微调一个70亿参数的LLM，其成本也高得令人望而却步。
在广泛的医学图像数据集上预训练模型以获得多模态能力会导致更高的成本。
因此，研究人员正在探索更具成本效益的技术，例如Adapter, Instruct-Tuning, and Prompt Augmentation 
用来开发可以在消费级显卡上训练和部署的模型，同时保持足够的性能。
在为多模态应用连接文本和视觉的背景下，训练也可能同样昂贵（Alayrac等人，2022）。此外，据我们所知，目前还没有专门为生物医学应用设计的公开的多模态生成基础模型。

为了应对这些挑战，我们引入了Visual Med-Alpaca，这是一个开源、参数高效的生物医学基础模型，具有即插即用的视觉扩展框架。
为了开发Visual Med-Alpaca模型，我们最初通过从BigBIO存储库中的各种医学数据集中提取医学问题来创建生物医学指令集（Fries等人，2022）。
随后，我们提示GPT-3.5-Turbo综合这些问题的答案。
执行多轮人工过滤和编辑以细化问答对，从而生成包括54k个数据点的高质量指令集。接下来，我们将文本模型与“视觉医学专家”（即专业的医学计算机视觉模型）连接起来，将Med-Alpaca扩展为Visual Med-Alpaca。例如，在放射学领域的应用程序中，我们训练了了一个名为Med-GIT的内部放射学图像字幕模型（详见下文）。当给出输入图像时，分类器确定是否或哪个医学视觉专家负责该图像。然后指定的医学专家将图像转换为文本提示。提示管理器随后将转换后的视觉信息与文本查询合并，使Med Alpaca能够生成适当的响应。

**正在进行的工作**未来的首要目标是彻底评估Visual Med-Alpaca的医疗能力和潜在缺陷，包括误导性医疗建议和不正确的医疗信息等问题。超越传统的基准测试和手动评估方法，我们的目标是关注不同的用户群体，包括医生和患者，并通过以用户为中心的方法评估模型的各个方面。这一全面评估将使我们能够确保Visual Med-Alpaca在解决各种生物医学任务和满足用户不同需求方面的可靠性和有效性。

!> 同样重要的是要注意，Visual Med-Alpaca严格用于学术研究目的，在任何国家都没有合法批准用于医疗用途。

**3.Model Architecture and Training Pipeline**

<div align=center>
    <img src="zh-cn/img/ch2/4-6/model.png" /> 
</div>

Visual Med-Alpaca通过prompt augmentation方法将文本和视觉模式连接起来。首先，将图像输入输入到类型分类器中，以识别用于将视觉信息转换为中间文本格式的适当模块，然后将其附加到文本输入中，用于随后的推理过程。例如，通过使用[DePlot](https://huggingface.co/docs/transformers/main/model_doc/deplot)[arxiv](https://arxiv.org/abs/2212.10505)模块，将医学图转换为中间线性化表。然后，prompt manager将从图像和文本输入中提取的文本信息合并到Med-Alpaca的提示中，这是一个大型语言模型，用于利用生物医学领域的专业知识生成响应。

为了将生物医学知识和视觉模态纳入基础模型LLaMA-7B，我们使用两个不同的数据集进行了微调。最初，我们使用模型生成的数据集对LLaMA-7B模型进行了标准微调和低秩自适应（LoRA）微调，该数据集包括54000个生物医学示例，用于指令调整目的。其次，我们在上下文中的放射对象（ROCO）数据集上对Microsoft GIT模型进行了微调，以纳入视觉模态。

**4.Domain Adaptation: Self-Instruct in the Biomedical Domain**

我们的研究实现了从各种医学问答数据集（MEDIQA RQE、MedQA、MedDialog、MEDIQA QA、PubMedQA）收集查询的过程。这种方法旨在增加数据集的多样性和彻底性，并提高所获得结果的准确性和全面性。

我们用GPT-3.5-Turbo以自学的方式综合这些问题的答案。GPT-3.5-Turbo模型配备了先进的自然语言处理功能，使其能够理解并生成对各种问题的类似人类的回答。这使它成为生成结构化和信息性答案的可靠工具。

筛选和编辑问答对的过程是手动执行的。考虑到平衡和多样性的标准，总共仔细选择了54000个.


**5.Visual Experts: Radiology Image Captioning, DePlot, and More**

视觉输入是医学领域的重要组成部分，在医疗环境中提供不可或缺的信息。医疗保健专业人员广泛依赖视觉提示来诊断、监测和治疗患者。医学成像技术，如X射线、CT扫描和核磁共振成像，可以对内脏提供无与伦比的洞察力，检测肉眼看不见的疾病和异常。此外，科学数字和医疗记录，包括图表和表格，在医学领域很普遍。

我们建议将视觉专家与Med-Alpaca联系起来，因为基础模型链接提供了一个模块化且适应性强的框架，用于整合各种视觉模块。在这个框架内，任何多模态任务都可以分为两个基本阶段：（1）图像到文本的转换，以及（2）基于衍生文本的认知推理。在我们的上下文中，视觉专家（即视觉基础模型）将医学图像转换为中间文本表示。然后，将转换后的数据用于提示（ prompt ）预训练（pre-trained)的LLM，利用LLM固有的few-shot推理能力来生成适当的响应。

目前，我们的平台支持两个不同的视觉专家：Med-GIT和DePlot，之所以选择这两个专家，是因为放射学图像和绘图在医学领域中广泛存在。该系统的架构也旨在实现替代医学视觉专家的无缝集成，我们计划在不久的将来将额外的医学视觉基础模型作为视觉专家。

Med-GIT模型代表了一种[GIT](https://github.com/microsoft/GenerativeImage2Text)：用于视觉和语言的生成图像到文本转换器，专门在ROCO数据集上进行了微调，以促进专门的放射学图像字幕。该模型的训练程序在我们可公开访问的Github存储库中进行了全面详细的概述。

**6.Case Study**


!> Input 1: What are the chemicals that treat hair loss? Image: No image.

+ **Visual Med-Alpaca:** There are several chemicals that can be used to treat hair loss, including: 1. Minoxidil: a medication that is applied to the scalp to promote hall growth. 2. Finasteride: a medication that is taken orally to slow down the production of dihydrotestosterone (DHT), a hormone that can cause hair loss. 3. Corticosteroids: topical medications that can be applied to the scalp to reduce inflammation and promote hair growth. 4. Anthralin: a medication that is applied to the scalp to promote hair growth by stimulating the production of blood cells that nourish the hair follicles. 5, Ketoconazole: a medication that is applied to the scalp to treat fungal infections that can cause hair loss. It is important to consult with a healthcare professional before using any of these chemicals to treat hair loss, as they can have potential side effects and may not be suitable for everyone.

+ **[Alpaca-LoRA](https://huggingface.co/spaces/tloen/alpaca-lora):** The chemicals that treat hair loss include minoxidil, finasteride, and ketoconazole.
+ **[LLama-Adapter](https://huggingface.co/spaces/csuhan/LLaMA-Adapter):** Chemicals that treat hair loss include minoxidil, finasteride, and ketoconazole. Minoxidil is a topical solution that is applied directly to the scalp and is used to stimulate hair growth. Finasteride is a pill that is taken orally and is used to block the production of the hormone dihydrotestosterone, which can cause hair loss. Ketoconazole is an antifungal medication that is used to treat scalp infections that can cause hair loss.
+ **[ChatGPT (GPT-3.5-Turbo)](https://chat.openai.com/chat):** There are several chemicals that are used to treat hair loss, including: 1. Minoxidil: This is an over-the-counter medication that is applied topically to the scalp. It works by increasing blood flow to the hair follicles, which can promote hair growth. 2. Finasteride: This is a prescription medication that is taken orally. It works by blocking the conversion of testosterone to dihydrotestosterone (DHT), which is a hormone that can cause hair loss. 3. Ketoconazole: This is an antifungal medication that is sometimes used to treat hair loss. It works by reducing inflammation and blocking the production of DHT. 4. Corticosteroids: These are anti-inflammatory medications that can be injected into the scalp. They are sometimes used to treat hair loss caused by autoimmune disorders or inflammation. 5. Spironolactone: This is a medication that is often used to treat high blood pressure, but it can also be effective in treating hair loss in women. It works by blocking the production of androgens, which can contribute to hair loss. It's important to note that the effectiveness of these treatments can vary depending on the cause and severity of hair loss. It's best to consult with a healthcare provider or a dermatologist to determine the most appropriate treatment for you.

!> Input 3: What is seen in the X-ray and what should be done? image:

<div align=center>
    <img src="zh-cn/img/ch2/4-6/p2.png" /> 
</div>


+ **Visual Med-Alpaca:** Bilateral infiltrates and bilateral pleural effusion seen in a chest x-ray may indicate a respiratory infection or other respiratory conditions such as pneumonia pulmonary edema, or pleural effusion, it is important to consult a medical professional for a proper diagnosis and treatment plan.

!> Input 3: How effective is this treatment on papule? Image:

<div align=center>
    <img src="zh-cn/img/ch2/4-6/p3.png" /> 
</div>


+ **Visual Med-Alpaca:** Based on the information provided, it appears that the treatment has been effective in reducing the size of the papule, The before and after images show a significant reduction in the size of the papule, indicating that the treatment has been successful, However, it is important to note that the control group may not have received any treatment, which could have also contributed to the reduction in size. It would be best to consult with a medical professional for a more accurate assessment of the effectiveness of the treatment.

**7.Future Work**

正在进行的最关键的工作之一是对Visual Med-Alpaca以及生物医学领域内的其他NLP模型进行系统评估。随着医学数据的结构和类型的变化，评估NLP模型的有效性及其在不同数据集上的可推广性至关重要。

我们还期望对医学数据进行预训练可以提高NLP模型在生物医学领域的性能。它应该有助于识别和推理疾病表型、药物机制和临床概念的表达。

添加基因组蛋白质模式也可能有助于在LLM中实现更好的推理。鉴于遗传和蛋白质信息对理解疾病过程至关重要，LLM可以帮助分析大量基因组数据，从而有可能识别各种疾病过程中涉及的新突变。因此，将基因组信息纳入LLM将在生物医学领域有更广泛的应用。


------
------
## 7. Vicuna:An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality

<!-- https://blog.csdn.net/qq_41185868/article/details/130876638 -->

!> blog: https://lmsys.org/blog/2023-03-30-vicuna/
!> github: https://github.com/lm-sys/FastChat

作者提出了一个开源的聊天机器人Vicuna-13B。它是通过训练从ShareGPT收集的用户共享对话，然后在LLaMA基础模型上进行调整而产生的。根据初步的GPT-4评估，Vicuna-13B的质量达到了ChatGPT和Bard 90%的质量，超过其他开源模型如LLaMA和Alpaca。作者提出利用GPT-4作为评估工具来评估不同聊天机器人的有效性，通过它产生的答案和分数。尽管存在局限性，但这证明了自动化评估的潜力。Vicuna-13B的训练成本很低，大约只有300美元，采用了内存优化、多轮对话的改进方法，并通过Spot实例降低了成本。该模型的代码、参数和在线演示向公众开放。最后，作者强调Vicuna存在的限制，如在涉及推理和数学的任务方面存在问题，缺少安全优化。但它可以作为未来研究解决这些限制的开端。该项目的作者：The Vicuna Team这是与来自多个机构的合作者的共同努力，包括加州大学伯克利分校、CMU、斯坦福大学、加州大学圣地亚哥分校和MBZUAI。

我们推出了Vicuna-13B，这是一个通过在LLaMA上调整ShareGPT收集的用户共享对话进行训练的开源聊天机器人。利用GPT-4作为评判，初步评估显示Vicuna-13B达到OpenAI ChatGPT和谷歌Bard 90%* 的质量，在90%* 的情况下超过LLaMA和斯坦福Alpaca等其他模型。训练Vicuna-13B的成本约为300美元。

!> 代码、参数以及在线演示均向公众开放用于非商业用途。根据GPT-4的有趣和非科学评估。需要进一步严格评估。

How Good is Vicuna? Vicuna-13B的性能有多好？

<div align=center>
    <img src="zh-cn/img/ch2/4-7/chart.svg" /> 
</div><p align =center>  Figure 1. Relative Response Quality Assessed by GPT-4*</p>

在对Vicuna进行7万用户共享ChatGPT对话的调整后,我们发现与Alpaca相比，Vicuna能够生成更详细和结构更好的答案(见下例),质量与ChatGPT相当。

然而，评估聊天机器人从来不是一件简单的任务。随着GPT-4的最新进展，我们好奇其能力是否达到了人类级别，能否实现基准生成和性能评估的自动化评估框架。我们的初步发现表明，GPT-4可以产生高度一致的排名和详细的评估，以比较聊天机器人的答案(见GPT-4判断的上例)。基于GPT-4的初步评估总结在图1中，显示Vicuna达到Bard/ChatGPT的90%* 能力。虽然这种提议的框架显示出自动化评估聊天机器人的潜力，但这还不是一个严格的方法。建立聊天机器人的评估系统仍然是一个需要进一步研究的开放问题。更多详情在评估部分提供。

<div align=center>
    <img src="zh-cn/img/ch2/4-7/p1.gif" /> 
</div>

Online Demo在线演示:

<div align=center>
    <img src="zh-cn/img/ch2/4-7/p2.gif" /> 
</div>

**Overview概述**

<div align=center>
    <img src="zh-cn/img/ch2/4-7/overview.png" /> 
</div><p align =center> Figure 2. Workflow Overview</p>

大规模语言模型(LLM)的快速发展彻底改变了聊天机器人系统，表现出前所未有的智能，如OpenAI的ChatGPT。然而，**尽管性能令人印象深刻，ChatGPT的训练和架构细节仍不清楚**，阻碍了该领域的研究和开源创新。受Meta LLaMA和斯坦福Alpaca项目的启发，我们推出了Vicuna-13B，这是一个由增强数据集和易于使用的可扩展基础设施支持的开源聊天机器人。通过在LLaMA基础模型上调整从ShareGPT.com收集的用户共享对话，Vicuna-13B已经展示出与其他开源模型(如斯坦福Alpaca)相媲美的性能。本博客文章对Vicuna-13B的性能进行初步评估，并描述了其训练和服务基础设施。我们还邀请社区与我们的在线演示互动，测试此聊天机器人的能力。

图2概述了我们的工作。首先，我们从ShareGPT.com网站收集了约7万段对话，用户可以在该网站上共享他们的ChatGPT对话。其次，我们改进了Alpaca提供的训练脚本，更好地处理多轮对话和长序列。训练在`8个A100 GPU上一天内完成`，使用`PyTorch FSDP`。为了演示服务，我们实现了一个轻量级的分布式服务系统。我们通过创建80个多样化的问题，并利用GPT-4判断模型输出来对模型质量进行初步评估。为了比较两个不同的模型，我们将每个模型的输出组合在每个问题的单个提示中。然后将提示发送给GPT-4，它会评估哪个模型提供更好的回答。LLaMA，Alpaca，ChatGPT和Vicuna的详细比较见下表1

<div align=center>
    <img src="zh-cn/img/ch2/4-7/p3.png" /> 
</div>


**Training训练：训练方法基于alpaca构建+内存优化+通过Spot实例降低成本**

Vicuna是通过调整LLaMA基础模型来创建的，使用从ShareGPT.com收集的约7万段用户共享对话。为确保数据质量，我们将HTML转换回markdown，并过滤掉一些不适当或低质量的样本。此外，我们将较长的对话分成较小的段，以符合模型的最大上下文长度。

我们的训练方法基于斯坦福大学Stanford’s alpaca构建，具有以下改进。

**内存优化：**为了使Vicuna理解长上下文，我们将alpaca中的最大上下文长度从512扩展到2048，这大大增加了GPU内存需求。我们通过使用梯度检查点gradient checkpointing和闪光注意力flash attention来解决内存压力。

**多轮对话：**我们调整训练损失，以考虑多轮对话，并仅根据聊天机器人的输出计算调整损失。

**通过Spot实例降低成本：**数据集扩大40倍和序列长度增加4倍的训练会带来相当大的训练费用挑战。我们采用SkyPilot托管的Spot实例SkyPilot managed spot，利用更便宜的Spot实例与自动恢复预防和自动区域切换来降低成本。该解决方案将7B模型的训练成本从500美元降低到约140美元，13B模型的训练成本从约1,000美元降低到300美元。

**Serving服务：分布式工作节点+灵活添加GPU节点**

我们构建了一个能够使用分布式工作节点服务多个模型的服务系统。它支持从本地集群和云中灵活添加GPU工作节点。通过利用容错控制器和SkyPilot中的托管Spot功能，此服务系统可以与来自多个云的更便宜的Spot实例很好地配合使用，以降低服务成本。这目前是一个轻量级实现，我们正在努力将我们最新的研究成果集成进去。

**How To Evaluate a Chatbot?如何评估聊天机器人？——提出了一种基于GPT-4的评估框架来自动评估聊天机器人的性能**


<div align=center>
    <img src="zh-cn/img/ch2/4-7/p4.png" /> 
</div>


评估AI聊天机器人是一项具有挑战性的任务，因为它需要检查语言理解、推理和上下文意识。随着AI聊天机器人变得更加先进，目前的开放基准可能不再足够。例如，斯坦福大学Alpaca使用的评估数据集self-instruct可以被当前最先进的聊天机器人有效回答，这使人类难以 辨别性能差异。更多限制包括训练/测试数据污染和潜在的创建新基准的高成本。为解决这些问题，我们提出了一种基于GPT-4的评估框架来自动评估聊天机器人的性能。

首先，我们设计了八个问题类别，如费米问题、角色扮演场景和编码/数学任务，以测试聊天机器人性能的各个方面。通过精心设计提示，GPT-4能够生成基线模型难以应对的多样化和具有挑战性的问题。我们从五个聊天机器人中选择每个类别十个问题的答案：LLaMA、Alpaca、ChatGPT、Bard和Vicuna。然后我们要求GPT-4根据有用性、相关性、准确性和细节评价它们的答案质量。我们发现GPT-4不仅可以产生相对一致的分数，而且能够详细解释为什么给出这样的分数(详细示例链接)。然而，我们也注意到GPT-4在判断编码/数学任务方面不是很好。

图3显示了所有基准线和Vicuna之间的比较结果。在80%的问题中，GPT-4更喜欢Vicuna而不是最先进的开源模型(LLaMA，Alpaca)，并达到专有模型(ChatGPT，Bard)的竞争性能。在45%的问题中，GPT-4将Vicuna的回答评为优于或等于ChatGPT的回答。由于GPT-4在10点量表上为每个回答分配一个定量得分，我们通过将每个模型在80个问题上获得的得分相加来计算每个(基准，Vicuna)比较对的总得分。如表2所示，Vicuna的总得分是ChatGPT的92%。尽管最近有所进展，但这些聊天机器人仍面临一些限制，如难以应对基本的数学问题或具有有限的编码能力。

虽然此提出的评估框架展示了评估聊天机器人的潜力，但由于大语言模型容易产生幻觉，所以这还不是一个严格或成熟的方法。开发全面标准化的聊天机器人评估系统仍然是一个需要进一步研究的开放问题。

<div align=center>
    <img src="zh-cn/img/ch2/4-7/p5.png" /> 
</div>

**Limitations局限性—不擅长涉及推理或数学**

我们注意到，与其他大型语言模型一样，Vicuna也存在某些限制。例如，它不擅长涉及推理或数学的任务，并且在准确识别自己或确保其输出的事实准确性方面可能存在限制。此外，它还没有得到足够的优化以确保安全性或减轻潜在的 toxicity或偏见。为了解决安全问题，**我们在在线演示中使用OpenAI调解API过滤掉不适当的用户输入**。尽管如此，我们预计Vicuna可以作为未来研究解决这些限制的开放起点。

**Release发行**

在我们的首次发布中，我们将在GitHub repo：[https：//github.com/lm-sys/FastChat](https：//github.com/lm-sys/FastChat)上共享训练，服务和评估代码。我们还发布了Vicuna-13B模型权重，请在这里找相关说明。暂无计划发布数据集。加入我们的Discord服务器并关注我们的Twitter以获取最新动态。

**License许可证**

在线演示仅供非商业用途，受LLaMA模型许可证、OpenAI生成的数据使用条款和ShareGPT的隐私实践的约束。如果发现任何潜在违规行为，请联系我们。代码根据Apache许可证2.0版发布。


**The Team团队**

这是与来自多个机构的合作者的共同努力，包括加州大学伯克利分校、CMU、斯坦福大学、加州大学圣地亚哥分校和MBZUAI。

```
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://lmsys.org/blog/2023-03-30-vicuna/},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}

```

!> 中文的Vicuna-LoRA微调项目参考：<https://zhuanlan.zhihu.com/p/624592383>,<https://github.com/Facico/Chinese-Vicuna>

!> 大模型也内卷，Vicuna训练及推理指南，效果碾压斯坦福羊驼: <https://www.zhihu.com/tardis/zm/art/624012908?source_id=1005>


------
------

## 8. MiniGPT-4:Enhancing Vision-Language Understanding with Advanced Large Language Models

<!-- https://blog.csdn.net/beingstrong/article/details/130659313?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-130659313-blog-130508898.235%5Ev38%5Epc_relevant_anti_vip_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-2-130659313-blog-130508898.235%5Ev38%5Epc_relevant_anti_vip_base&utm_relevant_index=3 -->

<!-- https://zhuanlan.zhihu.com/p/627671257 -->

### 8.1 简介

MiniGPT-4 是前段时间由KAUST（沙特阿卜杜拉国王科技大学）开源的多模态大模型，去网站上体验了一下功能，把论文粗略的看了一遍，也做个记录。

最近发布的GPT-4展示了非凡的多模态能力，例如直接从手写文本生成网站，识别图像中的幽默元素。这些特性在以前的视觉语言模型中很少被观察到。我们认为GPT-4先进的多模态生成功能的主要原因在于使用了更先进的大型语言模型（LLM）。为了验证这一现象，我们提出了MiniGPT-4，它只使用一个投影层将冻结的视觉编码器与冻结的LLM Vicuna对齐。我们的研究结果表明，MiniGPT-4具有许多与GPT-4类似的功能，如生成详细的图像描述以及通过手写草稿来创建网站。此外，我们还观察到MiniGPT-4中的其他涌现能力，包括用给定的图像创作故事和诗歌，为图像中显示的问题提供解决方案，根据食物照片教用户如何烹饪等。在我们的实验中，我们发现只使用原始图像-文本对进行预训练，会产生缺乏连贯性的包括重复和碎片句子的不自然的输出。为了解决这个问题，我们在第二阶段创建了一个高质量、对齐良好的数据集，以使用对话模板微调我们的模型。事实证明，这一步骤对于增强模型的生成可靠性和整体可用性至关重要。值得注意的是，我们的模型计算效率很高，因为我们只使用大约500万对对齐的图像-文本对来训练一个投影层。我们的代码、预训练模型和收集的数据集可在[MiniGPT-4](https://minigpt-4.github.io/) 获取。

### 8.2 实现方法


<div align=center>
    <img src="zh-cn/img/ch2/6-8/p1.png" /> 
</div>

MiniGPT-4的框架如上图，它的主要目的是为了对齐预训练的视觉编码器和先进的大语言模型(LLM)。 使用大语言模型Vicuna来做语言编码器，视觉感知使用BLIP-2一样的视觉编码器：ViT + 预训练Q-Former。使用一个线性投影层将视觉编码器和LLM关联起来，也就是视觉编码器的结果经过线性投影层之后作为Vicuna的输入。MiniGPT-4的训练有两个阶段：

**阶段一：** 

+ 预训练视觉编码器和LLM的权重都是冻结的，只有线性投影层被训练
+ 使用组合数据集来训练，数据集由Conceptual Caption、SBU、LAION 构成，共约500万的图像-文本对。
+ batch size 为256，共训练了20000步， 整个过程使用4个 A100(80GB) GPU，共花了10个小时
+ 阶段一训练完的模型能够理解图像的含义，但是生成的连贯的描述文本有困难，会出现重复单词或句子，不相关的内容等。

因为经过阶段一的训练后模型生成效果不好，所以作者们构建了一个数据集：

+ 从Conceptual Caption 数据集中随机选择了5000张图片，首先使用阶段一的模型来对这些给定的图片生成详细描述，设计了如下与Vicuna对话形式一致的prompt，prompt 中的`<ImageFeature>`是由前面提到的线性投影层生成的。 

```
 ###Human: <Img><ImageFeature></Img> Describe this image in detail. Give as many details as possible. Say everything you see. ###Assistant:
```

+ 为了识别到不完整的句子，会检查模型生成的句子的token数是否超过80，如果没有超过80，会使用额外的`prompt: "###Human: Continue ###Assistant:" `让模型扩展生成的内容，将两部分prompt得到的结果拼成一个更详细的图像描述。
+ 前面也提到阶段一后的模型效果不理想，为了去掉错误信息，使用ChatGPT来对生成的描述进行完善，对ChatGPT使用的prompt 如下：

```
Fix the error in the given paragraph. Remove any repeating sentences, meaningless characters, not English sentences, and so on. Remove unnecessary repetition. Rewrite any incomplete sentences. Return directly the results without explanation. Return directly the input paragraph if it is already correct without explanation.
```
+ 执行上面的后处理步骤后，再通过人工确认每个图像描述的正确性以保证质量。主要是检查生成图像描述是否是想要的格式，也将chatGPT没有检测到的冗余单词和句子给去掉，最终生成了3500个满足要求的图像-句子对。

**阶段二：**

+ 使用创建的数据集来finetune 阶段一的模型，使用了如下模板的prompt, 里面的`<instruction>`是从定义好的指令集里随机选择的，指令集是类似于`“ Describe this image in detail”`的 `“Could you describe the contents of this image for me” `的变化形式。

```
###Human: <Img><ImageFeature></Img> <Instruction> ###Assistant:
```
+ 作者强调对上述text-image prompt 没有计算回归损失 （do not calculate the regression loss for this specific text-image prompt)
+ 经过阶段二后，MiniGPT-4可以生成更自然和可靠的回应，并且这个finetune 过程非常高效，batch size 为12， 训练400 步，使用一个A100 GPU 只需要训练7分钟

### 8.3 效果及局限


miniGPT-4 可以达到的效果（论文图2-图13展示了案例）：

+ 生成详细的图片描述
+ 识别图片中有趣的点，比如猫穿了衣服躺着的照片
+ 识别图片中不寻常的点，比如仙人掌不会出现在冰川之类
+ 从手写文字生成网站
+ 识别图片中的问题并给出解决方案
+ 根据图片内容创造诗歌和rap歌曲
+ 为图片写故事
+ 为图片中的产品打广告
+ 识别出图片中的名人
+ 提供有洞察的图片评论
+ 抽取跟图片相关的事实
+ 根据给定的照片，教用户如何做菜

局限性：

+ 语言幻想，这个主要是由于LLM模型的局限性导致的，可能通过在更高质量的图像文本对或者对齐更好的LLM来减轻。
+ 不充足的感知能力，对识别图像中的文字、空间定位等有困难。可能因为这几个因素：
   - 1. 缺少充足的对齐的关于空间信息和文本注释的图像-文本对，通过更多数据集来减轻； 
   - 2. 视觉编码器中的Q-former可能会丢失一些关键特征，替换更强的视觉感知模型来提高效果； 
   - 3. 只使用一个投影层可能没有足够的空间来学习视觉-文本对齐信息。

!> arxiv: https://arxiv.org/abs/2304.10592

### 8.4 大杀器，多模态大模型MiniGPT-4入坑指南

<!-- https://zhuanlan.zhihu.com/p/627671257 -->

ChatGPT的发布给大家带来了不少的震撼，而随后发布的GPT-4更是展现了非凡的多模态能力。但是，ChatGPT和GPT4官方公布的细节很少，OpenAI俨然走上了闭源之路，让广大AI从业者又爱又恨。

最近，来自沙特阿拉伯阿卜杜拉国王科技大学的研究团队开源了GPT-4的平民版 MiniGPT-4。他们认为，GPT-4 具有先进的多模态生成能力的主要原因在于利用了更先进的大型语言模型（LLM）。为了研究这一现象，他们提出了 MiniGPT-4。

MiniGPT-4 仅使用一个投影层将一个冻结的视觉编码器（BLIP-2）与一个冻结的 LLM（Vicuna）对齐。MiniGPT-4 产生了许多类似于 GPT-4 中新兴的视觉语言能力。比如：根据给定的图像创作故事和诗歌，为图像中显示的问题提供解决方案，教用户如何根据食物照片烹饪，给个手绘草图直接写出网站的代码等。

除此之外，此方法计算效率很高，因为它仅使用大约 500 万个对齐的图像-文本对和额外的 3,500 个经过精心策划的高质量图像-文本对来训练一个投影层。

>BLIP-2 简介

BLIP-2是一种通用且高效的视觉-语言预训练方法，它可以从现成的冻结预训练图像编码器和冻结大型语言模型中引导视觉-语言预训练。BLIP-2通过一个轻量级的Querying Transformer来弥合模态差距，并在两个阶段进行预训练。第一个阶段从冻结图像编码器引导视觉-语言表示学习。第二个阶段从冻结语言模型中引导视觉-语言生成学习。尽管比现有方法具有显著较少的可训练参数，但BLIP-2在各种视觉-语言任务上实现了最先进的性能。在零样本 VQAv2 上，BLIP-2 相较于 80 亿参数的 Flamingo 模型，使用的可训练参数数量少了 54 倍，性能却提升了 8.7 %。

**MiniGPT-4 模型训练原理**

MiniGPT-4 的模型架构遵循 BLIP-2，因此，训练 MiniGPT-4 分两个阶段。

+ 第一个传统预训练阶段使用 4 张 A100 卡在 10 小时内使用大约 500 万个对齐的图像-文本对进行训练。 在第一阶段之后，Vicuna 虽然能够理解图像。 但是Vicuna的生成能力受到了很大的影响。

为了解决这个问题并提高可用性，MiniGPT-4 提出了一种通过模型本身和 ChatGPT 一起创建高质量图像文本对的新方法。 基于此，MiniGPT-4 随后创建了一个小规模（总共 3500 对）但高质量的数据集。

+ 第二个微调阶段在对话模板中对该数据集进行训练，以显著提高其生成的可靠性和整体的可用性。 令人惊讶的是，这个阶段的计算效率很高，使用单个 A100 只需大约 7 分钟即可完成。

**环境搭建**

基础环境配置如下：

+ 操作系统: Ubuntu 18.04
+ CPUs: 单个节点具有 384GB 内存的 Intel CPU，物理CPU个数为2，每颗CPU核数为20
+ GPUs: 4 卡 A800 80GB GPUs
+ Python: 3.10 (需要先升级OpenSSL到1.1.1t版本（点击下载OpenSSL），然后再编译安装Python)
+ NVIDIA驱动程序版本: 525.105.17，根据不同型号选择不同的驱动程序。
+ CUDA工具包: 11.6
+ cuDNN: 8.8.1.3_cuda11

本文选择使用Doker镜像进行环境搭建。首先，下载对应版本的Pytorch镜像。

```
docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

镜像下载完成之后，创建容器。

```
docker run -dt --name minigpt4_env_dev --restart=always --gpus all \
--network=host \
--shm-size 4G \
-v /home/gdong/workspace/code:/workspace/code \
-v /home/gdong/workspace/data:/workspace/data \
-v /home/gdong/workspace/model:/workspace/model \
-v /home/gdong/workspace/output:/workspace/output \
-v /home/gdong/workspace/package:/workspace/package \
-w /workspace \
pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel \
/bin/bash
```

进入容器。

```
docker exec -it minigpt4_env_dev bash
```

安装 cv2 的依赖项。

```
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
```

安装其他依赖包。

```
pip install -r requirements.txt
```
其中，`requirements.txt`文件内容如下：

```
accelerate==0.16.0
aiohttp==3.8.4
aiosignal==1.3.1
async-timeout==4.0.2
attrs==22.2.0
bitsandbytes==0.37.0
cchardet==2.1.7
chardet==5.1.0
contourpy==1.0.7
cycler==0.11.0
filelock==3.9.0
fonttools==4.38.0
frozenlist==1.3.3
huggingface-hub==0.13.4
importlib-resources==5.12.0
kiwisolver==1.4.4
matplotlib==3.7.0
multidict==6.0.4
openai==0.27.0
packaging==23.0
psutil==5.9.4
pycocotools==2.0.6
pyparsing==3.0.9
python-dateutil==2.8.2
pyyaml==6.0
regex==2022.10.31
tokenizers==0.13.2
tqdm==4.64.1
transformers==4.28.0
timm==0.6.13
spacy==3.5.1
webdataset==0.2.48
scikit-learn==1.2.2
scipy==1.10.1
yarl==1.8.2
zipp==3.14.0
omegaconf==2.3.0
opencv-python==4.7.0.72
iopath==0.1.10
decord==0.6.0
tenacity==8.2.2
peft
pycocoevalcap
sentence-transformers
umap-learn
notebook
gradio==3.24.1
gradio-client==0.0.8
wandb
```

接下来，安装img2dataset库，用于后续下载数据集使用。

```
pip install img2dataset -i https://pypi.tuna.tsinghua.edu.cn/simple  --trusted-host pypi.tuna.tsinghua.edu.cn

```

**数据集、模型权重及训练推理代码下载**

+ 下载模型训练及推理代码

```
# commit id: 22d8888ca2cf0aac862f537e7d22ef5830036808
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
```

+ 模型权重准备

预先准备好 Vicuna 权重，详情请查看[官方文档](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md)。

在之前的文章大模型也内卷，[Vicuna训练及推理指南，效果碾压斯坦福羊驼中](https://www.zhihu.com/tardis/zm/art/624012908?source_id=1005)，有讲解过如何合并Vicuna模型权重，在这里我直接使用之前合并好的Vicuna权重文件。

准备好 Vicuna 权重之后，在模型配置文件 minigpt4.yaml 中的第 16 行设置 Vicuna 权重的路径。

```
model:
  arch: mini_gpt4

  # vit encoder
  image_size: 224
  drop_path_rate: 0
  use_grad_checkpoint: False
  vit_precision: "fp16"
  freeze_vit: True
  freeze_qformer: True

  # Q-Former
  num_query_token: 32

  # Vicuna
  llama_model: "/workspace/model/vicuna-7b-all-v1.1"

  # generation configs
  prompt: ""

preprocess:
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224
        eval:
          name: "blip2_image_eval"
          image_size: 224
    text_processor:
        train:
          name: "blip_caption"
        eval:
          name: "blip_caption"

```

然后，下载预训练的 MiniGPT-4 检查点（checkpoint），用于模型推理。下载地址：[与 Vicuna 7B 对齐的checkpoint(prerained_minigpt4_7b.pth)](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) 或 [与 Vicuna 13B 对齐的checkpoint(pretrained_minigpt4_13b.pth)](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link)

如果服务器无法访问外网，需要预先下载好 VIT([eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pt))、Q-Former ([blip2_pretrained_flant5xxl.pth](https://link.zhihu.com/?target=https%3A//storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth))的权重以及Bert([bert-base-uncased](https://huggingface.co/bert-base-uncased))的Tokenizer。如果服务器可以访问外网且网络状况良好，可以直接忽略以下步骤。

eva_vit_g.pth和blip2_pretrained_flant5xxl.pth下载好之后，格式如下：

```
> ls -al hub/checkpoints/ --block-size=K
total 2401124K
drwxr-xr-x 2 root root       4K May  5 02:09 .
drwxr-xr-x 3 root root       4K May  7 02:34 ..
-rw------- 1 root root  423322K May  5 02:09 blip2_pretrained_flant5xxl.pth
-rw------- 1 root root 1977783K May  5 02:08 eva_vit_g.pth
```

同时需要设置环境变量：

```
# export TORCH_HOME=/workspace/model/cache/torch
export TORCH_HOME=/root/.cache/torch

```
bert-base-uncased下载好之后，格式如下：

```
> ls -al bert-base-uncased --block-size=K                
total 244K
drwxr-xr-x 2 root root   4K May  7 09:03 .
drwxrwxrwx 9 root root   4K May  7 09:02 ..
-rw-r--r-- 1 root root   1K May  7 09:03 config.json
-rw-r--r-- 1 root root   1K May  7 09:03 tokenizer_config.json
-rw-r--r-- 1 root root 227K May  7 09:03 vocab.txt
```

同时，需要修改`/workspace/code/MiniGPT-4/minigpt4/models/blip2.py`文件，改为本地加载Tokenizer：
```python
class Blip2Base(BaseModel):
    @classmethod
    def init_tokenizer(cls):
        # TODO
        #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("/workspace/model/bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    ...

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        # TODO
        #encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config = BertConfig.from_pretrained("/workspace/model/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
```

+ 数据集准备

下面准备数据集，MiniGPT-4 的训练包含两个阶段，每个阶段使用的数据集不一样。

首先，准备第一阶段数据集。

<div align=center>
    <img src="zh-cn/img/ch2/6-8/p2.PNG" /> 
</div>

下载`ccs_synthetic_filtered_large.json`和`laion_synthetic_filtered_large.json`文件，并移动到对应的目录。

```
export MINIGPT4_DATASET=/workspace/data/blip
mkdir ${MINIGPT4_DATASET}/cc_sbu
mkdir ${MINIGPT4_DATASET}/laion
mv ccs_synthetic_filtered_large.json ${MINIGPT4_DATASET}/cc_sbu
mv laion_synthetic_filtered_large.json ${MINIGPT4_DATASET}/laion
```

进入MiniGPT-4项目的dataset目录，并拷贝转换数据格式和下载数据集的脚本。

```
cd dataset/
cp convert_cc_sbu.py ${MINIGPT4_DATASET}/cc_sbu
cp download_cc_sbu.sh ${MINIGPT4_DATASET}/cc_sbu

cp convert_laion.py ${MINIGPT4_DATASET}/laion
cp download_laion.sh ${MINIGPT4_DATASET}/laion
```

由于数据集太大，进入`${MINIGPT4_DATASET}/cc_sbu`和`${MINIGPT4_DATASET}/laion`文件夹，修改`convert_cc_sbu.py`和`convert_laion.py`脚本，改为仅下载一部分数据。

```python
#rows = [x.values() for x in data]

rows = []

for i, x in enumerate(data):
    if i >= 1000:
        break
    rows.append(x.values())
```

然后，将laion和cc_sbu标注文件格式转换为img2dataset格式。

```
cd ${MINIGPT4_DATASET}/cc_sbu
python convert_cc_sbu.py

cd ${MINIGPT4_DATASET}/laion
python convert_laion.py
```

进入`${MINIGPT4_DATASET}/cc_sbu`和`${MINIGPT4_DATASET}/laion`文件夹，修改下载数据集脚本`download_cc_sbu.sh`和`download_laion.sh`，将`--enable_wandb`配置项改为`False`。

然后，执行脚本，使用img2dataset下载数据集。

```
cd ${MINIGPT4_DATASET}/cc_sbu
sh download_cc_sbu.sh

cd ${MINIGPT4_DATASET}/laion
sh download_laion.sh
```

下载完成之后的最终数据集结构如下所示：

```
> tree
.
|-- cc_sbu
|   |-- cc_sbu_dataset
|   |   |-- 00000.parquet
|   |   |-- 00000.tar
|   |   `-- 00000_stats.json
|   |-- ccs_synthetic_filtered_large.json
|   |-- ccs_synthetic_filtered_large.tsv
|   |-- convert_cc_sbu.py
|   `-- download_cc_sbu.sh
`-- laion
    |-- convert_laion.py
    |-- download_laion.sh
    |-- laion_dataset
    |   |-- 00000.parquet
    |   |-- 00000.tar
    |   `-- 00000_stats.json
    |-- laion_synthetic_filtered_large.json
    `-- laion_synthetic_filtered_large.tsv

4 directories, 14 files
```

之后，修改数据集配置文件。

修改配置文件`minigpt4/configs/datasets/laion/defaults.yaml`的第五行设置LAION数据集加载路径，具体如下所示：

```
datasets:
  laion:
    data_type: images
    build_info:
      storage: /workspace/data/blip/laion/laion_dataset/00000.tar
```

修改配置文件`minigpt4/configs/datasets/cc_sbu/defaults.yaml`的第五行设置 Conceptual Captoin 和 SBU 数据集加载路径，具体如下所示：

```
datasets:
  cc_sbu:
    data_type: images
    build_info:
      storage: /workspace/data/blip/cc_sbu/cc_sbu_dataset/00000.tar
```
接下来，准备第二阶段数据集，具体在此处[下载](https://link.zhihu.com/?target=https%3A//drive.google.com/file/d/1nJXhoEcy3KTExr17I7BXqY5Y9Lx_-n-9/view%3Fusp%3Dshare_link)，数据集文件夹结构如下所示。

```
cc_sbu_align
├── filter_cap.json
└── image
    ├── 2.jpg
    ├── 3.jpg
    ...   
```
下载完成之后，在数据集配置文件`minigpt4/configs/datasets/cc_sbu/align.yaml`中的第 5 行设置数据集路径。

```
datasets:
  cc_sbu_align:
    data_type: images
    build_info:
      storage: /workspace/data/cc_sbu_align/
```


**代码结构**

MiniGPT-4 项目基于 [BLIP2](https://huggingface.co/docs/transformers/main/model_doc/blip-2)、[Lavis](https://github.com/salesforce/LAVIS) 和 [Vicuna](v) 进行构建，使用 OmegaConf 基于 YAML 进行分层系统配置，整个代码结构如下所示：

```
.
|-- LICENSE.md
|-- LICENSE_Lavis.md
|-- MiniGPT_4.pdf
|-- PrepareVicuna.md
|-- README.md
|-- dataset # 数据集预处理
|   |-- README_1_STAGE.md
|   |-- README_2_STAGE.md
|   |-- convert_cc_sbu.py # 转换标注数据格式
|   |-- convert_laion.py
|   |-- download_cc_sbu.sh # 下载数据集
|   `-- download_laion.sh
|-- demo.py    # 模型测试/推理
|-- environment.yml
|-- eval_configs # 模型评估配置文件
|   `-- minigpt4_eval.yaml
|-- minigpt4
|   |-- __init__.py
|   |-- common
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- dist_utils.py # 模型权重缓存文件路径
|   |   |-- gradcam.py
|   |   |-- logger.py
|   |   |-- optims.py
|   |   |-- registry.py
|   |   `-- utils.py
|   |-- configs 
|   |   |-- datasets # 数据集配置文件
|   |   |   |-- cc_sbu
|   |   |   |   |-- align.yaml # cc_sbu对齐数据集配置文件
|   |   |   |   `-- defaults.yaml # cc_sbu数据集配置文件
|   |   |   `-- laion
|   |   |       `-- defaults.yaml # laion数据集配置文件
|   |   |-- default.yaml
|   |   `-- models # 模型配置文件
|   |       `-- minigpt4.yaml
|   |-- conversation
|   |   |-- __init__.py
|   |   `-- conversation.py
|   |-- datasets
|   |   |-- __init__.py
|   |   |-- builders
|   |   |   |-- __init__.py
|   |   |   |-- base_dataset_builder.py
|   |   |   `-- image_text_pair_builder.py
|   |   |-- data_utils.py
|   |   `-- datasets
|   |       |-- __init__.py
|   |       |-- base_dataset.py
|   |       |-- caption_datasets.py
|   |       |-- cc_sbu_dataset.py
|   |       |-- dataloader_utils.py
|   |       `-- laion_dataset.py
|   |-- models
|   |   |-- Qformer.py
|   |   |-- __init__.py
|   |   |-- base_model.py
|   |   |-- blip2.py # 初始化Bert Tokenizer 和 Qformer等
|   |   |-- blip2_outputs.py
|   |   |-- eva_vit.py
|   |   |-- mini_gpt4.py
|   |   `-- modeling_llama.py
|   |-- processors
|   |   |-- __init__.py
|   |   |-- base_processor.py
|   |   |-- blip_processors.py
|   |   `-- randaugment.py
|   |-- runners
|   |   |-- __init__.py
|   |   `-- runner_base.py
|   `-- tasks 
|       |-- __init__.py
|       |-- base_task.py 
|       `-- image_text_pretrain.py
|-- prompts 
|   `-- alignment.txt
|-- train.py # 模型训练
`-- train_configs # 模型训练配置文件
    |-- minigpt4_stage1_pretrain.yaml # 第一阶段预训练配置
    `-- minigpt4_stage2_finetune.yaml # 第二阶段微调配置
```

**模型推理**

首先，在评估配置文件`eval_configs/minigpt4_eval.yaml`中的第 11 行设置预训练checkpoint的路径(即刚刚下载的预训练的 MiniGPT-4 检查点)。

```
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  max_txt_len: 160
  end_sym: "###"
  low_resource: True
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: '/workspace/model/minigpt/prerained_minigpt4_7b.pth'

datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name: "blip2_image_eval"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"

run:
  task: image_text_pretrain
```

执行如下命令启动模型推理服务：

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
```

为了节省 GPU 内存，Vicuna 默认以 8 bit 进行加载，beam search 宽度为 1。此配置对于 Vicuna-13B 需要大约 23G GPU 内存、对于 Vicuna-7B 需要大约 11.5G GPU 内存。 如果你有更强大的 GPU，您可以通过在配置文件 minigpt4_eval.yaml 中将 low_resource 设置为 False 以 16 bit运行模型并使用更大的beam search宽度。

运行过程：

```
> python demo.py --cfg-path eval_configs/minigpt4_eval.yaml  --gpu-id 0
Initializing Chat
Loading VIT
Loading VIT Done
Loading Q-Former
Loading Q-Former Done
Loading LLAMA

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████| 2/2 [01:02<00:00, 31.47s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Take a look at this image and describe what you notice. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/model/minigpt/prerained_minigpt4_7b.pth
Initialization Finished
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://71e239f43b078ebe0b.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces
```

模型推理测试：

<div align=center>
    <img src="zh-cn/img/ch2/6-8/p3.png" /> 
</div>

<div align=center>
    <img src="zh-cn/img/ch2/6-8/p4.png" /> 
</div>

**模型训练**

MiniGPT-4 的训练包含两个对齐阶段。

+ 第一阶段：预训练
+ 
在预训练阶段，模型使用来自 Laion 和 CC 数据集的图像文本对进行训练，以对齐视觉和语言模型。

第一阶段之后，视觉特征被映射，可以被语言模型理解。 MiniGPT-4 官方在实验时使用了 4 个 A100。 除此之外，您还可以在配置文件 ·train_configs/minigpt4_stage1_pretrain.yaml· 中更改保存路径，具体内容如下：

```
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True


datasets:
  laion:
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"
    sample_ratio: 115
  cc_sbu:
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224
    text_processor:
        train:
          name: "blip_caption"
    sample_ratio: 14


run:
  task: image_text_pretrain
  # optimizer
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 8e-5
  warmup_lr: 1e-6

  weight_decay: 0.05
  max_epoch: 3
  batch_size_train: 16
  batch_size_eval: 2
  num_workers: 4
  warmup_steps: 500
  iters_per_epoch: 500

  seed: 42
  output_dir: "/workspace/output/minigpt4_stage1_pretrain"

  amp: True
  resume_ckpt_path: null

  evaluate: False 
  train_splits: ["train"]

  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
```

接下来，通过以下命令启动第一阶段训练。

```
CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
```

运行过程：


```
> CUDA_VISIBLE_DEVICES=0,1,2,3  torchrun --nproc_per_node 4 train.py --cfg-path train_configs/minigpt4_stage1_pretrain.yaml
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
| distributed init (rank 1, world 4): env://
| distributed init (rank 0, world 4): env://
| distributed init (rank 2, world 4): env://
| distributed init (rank 3, world 4): env://
2023-05-07 11:36:36,497 [INFO] 
=====  Running Parameters    =====
2023-05-07 11:36:36,498 [INFO] {
    "amp": true,
    "batch_size_eval": 2,
    "batch_size_train": 16,
    "device": "cuda",
    "dist_backend": "nccl",
    "dist_url": "env://",
    "distributed": true,
    "evaluate": false,
    "gpu": 0,
    "init_lr": 0.0001,
    "iters_per_epoch": 500,
    "lr_sched": "linear_warmup_cosine_lr",
    "max_epoch": 3,
    "min_lr": 8e-05,
    "num_workers": 4,
    "output_dir": "/workspace/output/minigpt4_stage1_pretrain",
    "rank": 0,
    "resume_ckpt_path": null,
    "seed": 42,
    "task": "image_text_pretrain",
    "train_splits": [
        "train"
    ],
    "warmup_lr": 1e-06,
    "warmup_steps": 500,
    "weight_decay": 0.05,
    "world_size": 4
}
2023-05-07 11:36:36,498 [INFO] 
======  Dataset Attributes  ======
2023-05-07 11:36:36,498 [INFO] 
======== laion =======
2023-05-07 11:36:36,499 [INFO] {
    "build_info": {
        "storage": "/workspace/data/blip/laion/laion_dataset/00000.tar"
    },
    "data_type": "images",
    "sample_ratio": 115,
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 11:36:36,499 [INFO] 
======== cc_sbu =======
2023-05-07 11:36:36,499 [INFO] {
    "build_info": {
        "storage": "/workspace/data/blip/cc_sbu/cc_sbu_dataset/00000.tar"
    },
    "data_type": "images",
    "sample_ratio": 14,
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 11:36:36,499 [INFO] 
======  Model Attributes  ======
2023-05-07 11:36:36,500 [INFO] {
    "arch": "mini_gpt4",
    "drop_path_rate": 0,
    "freeze_qformer": true,
    "freeze_vit": true,
    "image_size": 224,
    "llama_model": "/workspace/model/vicuna-7b-all-v1.1",
    "model_type": "pretrain_vicuna",
    "num_query_token": 32,
    "prompt": "",
    "use_grad_checkpoint": false,
    "vit_precision": "fp16"
}
2023-05-07 11:36:36,501 [INFO] Building datasets...
2023-05-07 11:36:36,503 [INFO] Building datasets...
Loading VIT
2023-05-07 11:36:58,812 [INFO] Downloading: "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth" to /root/.cache/torch/hub/checkpoints/eva_vit_g.pth

100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 1.89G/1.89G [02:33<00:00, 13.2MB/s]
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
2023-05-07 11:39:41,878 [INFO] freeze vision encoder
Loading VIT Done
Loading Q-Former
2023-05-07 11:39:45,781 [INFO] Downloading: "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth" to /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 413M/413M [00:31<00:00, 13.8MB/s]
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
2023-05-07 11:40:18,140 [INFO] load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
2023-05-07 11:40:18,155 [INFO] freeze Qformer
Loading Q-Former Done
Loading LLAMA
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.79s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:15<00:00,  7.94s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:16<00:00,  8.21s/it]
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:16<00:00,  8.13s/it]
Loading LLAMA Done
2023-05-07 11:43:21,365 [INFO] Start training
2023-05-07 11:43:28,791 [INFO] dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).
2023-05-07 11:43:28,791 [INFO] Loaded 0 records for train split from the dataset.
module.llama_proj.weight
module.llama_proj.bias
2023-05-07 11:43:30,005 [INFO] number of trainable parameters: 3149824
2023-05-07 11:43:30,008 [INFO] Start training epoch 0, 500 iters per inner epoch.
Train: data epoch: [0]  [  0/500]  eta: 0:35:50  lr: 0.000001  loss: 7.4586  time: 4.3018  data: 0.0000  max mem: 20913
2023-05-07 11:43:34,313 [INFO] Reducer buckets have been rebuilt in this iteration.
Train: data epoch: [0]  [ 50/500]  eta: 0:03:04  lr: 0.000011  loss: 4.9250  time: 0.3323  data: 0.0000  max mem: 22076
Train: data epoch: [0]  [100/500]  eta: 0:02:29  lr: 0.000021  loss: 3.6569  time: 0.3376  data: 0.0000  max mem: 22076
Train: data epoch: [0]  [150/500]  eta: 0:02:06  lr: 0.000031  loss: 2.8653  time: 0.3415  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [200/500]  eta: 0:01:47  lr: 0.000041  loss: 2.5771  time: 0.3417  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [250/500]  eta: 0:01:28  lr: 0.000051  loss: 3.0763  time: 0.3375  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [300/500]  eta: 0:01:10  lr: 0.000060  loss: 2.3269  time: 0.3369  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [350/500]  eta: 0:00:52  lr: 0.000070  loss: 2.5431  time: 0.3403  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [400/500]  eta: 0:00:34  lr: 0.000080  loss: 2.6711  time: 0.3383  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [450/500]  eta: 0:00:17  lr: 0.000090  loss: 2.3690  time: 0.3426  data: 0.0000  max mem: 22193
Train: data epoch: [0]  [499/500]  eta: 0:00:00  lr: 0.000100  loss: 1.5752  time: 0.3424  data: 0.0000  max mem: 22193
Train: data epoch: [0] Total time: 0:02:53 (0.3466 s / it)
2023-05-07 11:46:23,294 [INFO] Averaged stats: lr: 0.0001  loss: 3.2105
2023-05-07 11:46:23,297 [INFO] No validation splits found.
2023-05-07 11:46:23,334 [INFO] Saving checkpoint at epoch 0 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_0.pth.
2023-05-07 11:46:23,402 [INFO] Start training
2023-05-07 11:46:23,443 [INFO] Start training epoch 1, 500 iters per inner epoch.
Train: data epoch: [1]  [  0/500]  eta: 0:03:00  lr: 0.000095  loss: 1.9775  time: 0.3606  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [ 50/500]  eta: 0:02:34  lr: 0.000094  loss: 1.3029  time: 0.3486  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [100/500]  eta: 0:02:16  lr: 0.000093  loss: 1.1404  time: 0.3374  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [150/500]  eta: 0:01:59  lr: 0.000092  loss: 0.8192  time: 0.3376  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [200/500]  eta: 0:01:42  lr: 0.000091  loss: 0.4934  time: 0.3415  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [250/500]  eta: 0:01:25  lr: 0.000090  loss: 0.4390  time: 0.3402  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [300/500]  eta: 0:01:08  lr: 0.000089  loss: 0.2317  time: 0.3421  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [350/500]  eta: 0:00:51  lr: 0.000088  loss: 0.1960  time: 0.3413  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [400/500]  eta: 0:00:34  lr: 0.000087  loss: 2.0755  time: 0.3420  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [450/500]  eta: 0:00:17  lr: 0.000086  loss: 0.0773  time: 0.3405  data: 0.0000  max mem: 22193
Train: data epoch: [1]  [499/500]  eta: 0:00:00  lr: 0.000085  loss: 0.1692  time: 0.3387  data: 0.0000  max mem: 22193
Train: data epoch: [1] Total time: 0:02:50 (0.3404 s / it)
2023-05-07 11:49:13,623 [INFO] Averaged stats: lr: 0.0001  loss: 0.7745
2023-05-07 11:49:13,625 [INFO] No validation splits found.
2023-05-07 11:49:13,660 [INFO] Saving checkpoint at epoch 1 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_1.pth.
2023-05-07 11:49:13,722 [INFO] Start training
2023-05-07 11:49:13,763 [INFO] Start training epoch 2, 500 iters per inner epoch.
Train: data epoch: [2]  [  0/500]  eta: 0:03:00  lr: 0.000085  loss: 0.2226  time: 0.3614  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [ 50/500]  eta: 0:02:34  lr: 0.000084  loss: 0.1156  time: 0.3454  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [100/500]  eta: 0:02:16  lr: 0.000083  loss: 0.0512  time: 0.3396  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [150/500]  eta: 0:01:59  lr: 0.000083  loss: 0.1134  time: 0.3421  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [200/500]  eta: 0:01:42  lr: 0.000082  loss: 0.0489  time: 0.3412  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [250/500]  eta: 0:01:25  lr: 0.000081  loss: 0.0693  time: 0.3409  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [300/500]  eta: 0:01:08  lr: 0.000081  loss: 0.0316  time: 0.3433  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [350/500]  eta: 0:00:51  lr: 0.000080  loss: 0.0372  time: 0.3464  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [400/500]  eta: 0:00:34  lr: 0.000080  loss: 0.0404  time: 0.3386  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [450/500]  eta: 0:00:17  lr: 0.000080  loss: 0.0523  time: 0.3396  data: 0.0000  max mem: 22193
Train: data epoch: [2]  [499/500]  eta: 0:00:00  lr: 0.000080  loss: 0.0471  time: 0.3378  data: 0.0000  max mem: 22193
Train: data epoch: [2] Total time: 0:02:50 (0.3402 s / it)
2023-05-07 11:52:03,847 [INFO] Averaged stats: lr: 0.0001  loss: 0.2326
2023-05-07 11:52:03,849 [INFO] No validation splits found.
2023-05-07 11:52:03,885 [INFO] Saving checkpoint at epoch 2 to /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth.
2023-05-07 11:52:03,946 [INFO] No validation splits found.
2023-05-07 11:52:03,946 [INFO] Training time 0:08:42
```

显存占用:

```
Sun May  7 19:48:54 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   68C    P0   297W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A800 80G...  Off  | 00000000:5E:00.0 Off |                    0 |
| N/A   65C    P0   322W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A800 80G...  Off  | 00000000:AF:00.0 Off |                    0 |
| N/A   69C    P0   218W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A800 80G...  Off  | 00000000:D8:00.0 Off |                    0 |
| N/A   69C    P0   335W / 300W |  32439MiB / 81920MiB |     97%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11425      C   /opt/conda/bin/python           32436MiB |
|    1   N/A  N/A     11426      C   /opt/conda/bin/python           32436MiB |
|    2   N/A  N/A     11427      C   /opt/conda/bin/python           32436MiB |
|    3   N/A  N/A     11428      C   /opt/conda/bin/python           32436MiB |
+-----------------------------------------------------------------------------+

```

模型权重输出：

```
> tree minigpt4_stage1_pretrain/
minigpt4_stage1_pretrain/
`-- 20230507113
    |-- checkpoint_0.pth
    |-- checkpoint_1.pth
    |-- checkpoint_2.pth
    |-- log.txt
    `-- result

2 directories, 4 files
```

你也可以直接下载只有第一阶段训练的 MiniGPT-4 的 checkpoint，具体下载地址：[13B](https://drive.google.com/file/d/1u9FRRBB3VovP1HxCAlpD9Lw4t4P6-Yq8/view?usp=share_link) 或 [7B](https://drive.google.com/file/d/1HihQtCEXUyBM1i9DQbaK934wW3TZi-h5/view?usp=share_link)。

与第二阶段之后的模型相比，第一阶段的checkpoint经常生成不完整和重复的句子。

+ 第二阶段：微调

在第二阶段，我们使用自己创建的小型高质量图文对数据集并将其转换为对话格式以进一步对齐 MiniGPT-4。

要启动第二阶段对齐，需先在`train_configs/minigpt4_stage2_finetune.yaml` 中指定第一阶段训练的checkpoint文件的路径。 当然，您还可以自定义输出权重路径，具体文件如下所示。
```
model:
  arch: mini_gpt4
  model_type: pretrain_vicuna
  freeze_vit: True
  freeze_qformer: True
  max_txt_len: 160
  end_sym: "###"
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: '/workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth'


datasets:
  cc_sbu_align:
    vis_processor:
      train:
        name: "blip2_image_train"
        image_size: 224
    text_processor:
      train:
        name: "blip_caption"

run:
  task: image_text_pretrain
  # optimizer
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 3e-5
  min_lr: 1e-5
  warmup_lr: 1e-6

  weight_decay: 0.05
  max_epoch: 5
  iters_per_epoch: 200
  batch_size_train: 12
  batch_size_eval: 12
  num_workers: 4
  warmup_steps: 200

  seed: 42
  output_dir: "/workspace/output/minigpt4_stage2_finetune"

  amp: True
  resume_ckpt_path: null

  evaluate: False 
  train_splits: ["train"]

  device: "cuda"
  world_size: 1
  dist_url: "env://"
  distributed: True
```

然后，第二阶段微调的运行命令如下所示。 MiniGPT-4官方在实验中，仅使用了 1 卡 A100。

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
```

运行过程：

```

> CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 train.py --cfg-path train_configs/minigpt4_stage2_finetune.yaml
| distributed init (rank 0, world 1): env://
2023-05-07 12:03:11,908 [INFO] 
=====  Running Parameters    =====
2023-05-07 12:03:11,909 [INFO] {
    "amp": true,
    "batch_size_eval": 12,
    "batch_size_train": 12,
    "device": "cuda",
    "dist_backend": "nccl",
    "dist_url": "env://",
    "distributed": true,
    "evaluate": false,
    "gpu": 0,
    "init_lr": 3e-05,
    "iters_per_epoch": 200,
    "lr_sched": "linear_warmup_cosine_lr",
    "max_epoch": 5,
    "min_lr": 1e-05,
    "num_workers": 4,
    "output_dir": "/workspace/output/minigpt4_stage2_finetune",
    "rank": 0,
    "resume_ckpt_path": null,
    "seed": 42,
    "task": "image_text_pretrain",
    "train_splits": [
        "train"
    ],
    "warmup_lr": 1e-06,
    "warmup_steps": 200,
    "weight_decay": 0.05,
    "world_size": 1
}
2023-05-07 12:03:11,909 [INFO] 
======  Dataset Attributes  ======
2023-05-07 12:03:11,909 [INFO] 
======== cc_sbu_align =======
2023-05-07 12:03:11,910 [INFO] {
    "build_info": {
        "storage": "/workspace/data/cc_sbu_align/"
    },
    "data_type": "images",
    "text_processor": {
        "train": {
            "name": "blip_caption"
        }
    },
    "vis_processor": {
        "train": {
            "image_size": 224,
            "name": "blip2_image_train"
        }
    }
}
2023-05-07 12:03:11,910 [INFO] 
======  Model Attributes  ======
2023-05-07 12:03:11,910 [INFO] {
    "arch": "mini_gpt4",
    "ckpt": "/workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth",
    "drop_path_rate": 0,
    "end_sym": "###",
    "freeze_qformer": true,
    "freeze_vit": true,
    "image_size": 224,
    "llama_model": "/workspace/model/vicuna-7b-all-v1.1",
    "max_txt_len": 160,
    "model_type": "pretrain_vicuna",
    "num_query_token": 32,
    "prompt": "",
    "prompt_path": "prompts/alignment.txt",
    "prompt_template": "###Human: {} ###Assistant: ",
    "use_grad_checkpoint": false,
    "vit_precision": "fp16"
}
2023-05-07 12:03:11,910 [INFO] Building datasets...
Loading VIT
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
2023-05-07 12:03:37,018 [INFO] freeze vision encoder
Loading VIT Done
Loading Q-Former
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
2023-05-07 12:03:40,903 [INFO] load checkpoint from https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth
2023-05-07 12:03:40,916 [INFO] freeze Qformer
Loading Q-Former Done
Loading LLAMA
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:10<00:00,  5.13s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Describe this image in detail. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/output/minigpt4_stage1_pretrain/20230507113/checkpoint_2.pth
2023-05-07 12:06:34,005 [INFO] Start training
2023-05-07 12:06:40,005 [INFO] dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline).
2023-05-07 12:06:40,005 [INFO] Loaded 3439 records for train split from the dataset.
module.llama_proj.weight
module.llama_proj.bias
2023-05-07 12:06:40,029 [INFO] number of trainable parameters: 3149824
2023-05-07 12:06:40,030 [INFO] Start training epoch 0, 200 iters per inner epoch.
Train: data epoch: [0]  [  0/200]  eta: 0:15:02  lr: 0.000001  loss: 1.6358  time: 4.5127  data: 0.0000  max mem: 35512
2023-05-07 12:06:44,545 [INFO] Reducer buckets have been rebuilt in this iteration.
Train: data epoch: [0]  [ 50/200]  eta: 0:01:47  lr: 0.000008  loss: 1.3364  time: 0.6420  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [100/200]  eta: 0:01:07  lr: 0.000015  loss: 1.2098  time: 0.6466  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [150/200]  eta: 0:00:33  lr: 0.000023  loss: 1.0652  time: 0.6472  data: 0.0000  max mem: 36093
Train: data epoch: [0]  [199/200]  eta: 0:00:00  lr: 0.000030  loss: 1.0278  time: 0.6460  data: 0.0000  max mem: 36093
Train: data epoch: [0] Total time: 0:02:12 (0.6627 s / it)
2023-05-07 12:08:52,563 [INFO] Averaged stats: lr: 0.0000  loss: 1.2121
2023-05-07 12:08:52,565 [INFO] No validation splits found.
2023-05-07 12:08:52,601 [INFO] Saving checkpoint at epoch 0 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_0.pth.
2023-05-07 12:08:52,668 [INFO] Start training
2023-05-07 12:08:52,708 [INFO] Start training epoch 1, 200 iters per inner epoch.
Train: data epoch: [1]  [  0/200]  eta: 0:02:14  lr: 0.000028  loss: 0.9808  time: 0.6744  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [ 50/200]  eta: 0:01:35  lr: 0.000027  loss: 0.9252  time: 0.6336  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [100/200]  eta: 0:01:07  lr: 0.000026  loss: 1.0419  time: 0.7971  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [150/200]  eta: 0:00:33  lr: 0.000025  loss: 1.0150  time: 0.6486  data: 0.0000  max mem: 36093
Train: data epoch: [1]  [199/200]  eta: 0:00:00  lr: 0.000023  loss: 0.9695  time: 0.6472  data: 0.0000  max mem: 36093
Train: data epoch: [1] Total time: 0:02:11 (0.6576 s / it)
2023-05-07 12:11:04,223 [INFO] Averaged stats: lr: 0.0000  loss: 0.9785
2023-05-07 12:11:04,227 [INFO] No validation splits found.
2023-05-07 12:11:04,264 [INFO] Saving checkpoint at epoch 1 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_1.pth.
2023-05-07 12:11:04,332 [INFO] Start training
2023-05-07 12:11:04,370 [INFO] Start training epoch 2, 200 iters per inner epoch.
Train: data epoch: [2]  [  0/200]  eta: 0:02:13  lr: 0.000023  loss: 1.1459  time: 0.6684  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [ 50/200]  eta: 0:01:38  lr: 0.000022  loss: 1.0003  time: 0.6580  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [100/200]  eta: 0:01:04  lr: 0.000020  loss: 0.8605  time: 0.6367  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [150/200]  eta: 0:00:32  lr: 0.000018  loss: 0.8841  time: 0.6445  data: 0.0000  max mem: 36093
Train: data epoch: [2]  [199/200]  eta: 0:00:00  lr: 0.000017  loss: 0.8462  time: 0.6380  data: 0.0000  max mem: 36093
Train: data epoch: [2] Total time: 0:02:11 (0.6588 s / it)
2023-05-07 12:13:16,139 [INFO] Averaged stats: lr: 0.0000  loss: 0.9272
2023-05-07 12:13:16,143 [INFO] No validation splits found.
2023-05-07 12:13:16,178 [INFO] Saving checkpoint at epoch 2 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_2.pth.
2023-05-07 12:13:16,247 [INFO] Start training
2023-05-07 12:13:16,286 [INFO] Start training epoch 3, 200 iters per inner epoch.
Train: data epoch: [3]  [  0/200]  eta: 0:02:14  lr: 0.000017  loss: 0.8447  time: 0.6750  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [ 50/200]  eta: 0:01:37  lr: 0.000015  loss: 0.9082  time: 0.6517  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [100/200]  eta: 0:01:04  lr: 0.000014  loss: 0.9476  time: 0.6380  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [150/200]  eta: 0:00:32  lr: 0.000013  loss: 0.8131  time: 0.6443  data: 0.0000  max mem: 36093
Train: data epoch: [3]  [199/200]  eta: 0:00:00  lr: 0.000012  loss: 0.8718  time: 0.6550  data: 0.0000  max mem: 36093
Train: data epoch: [3] Total time: 0:02:09 (0.6460 s / it)
2023-05-07 12:15:25,492 [INFO] Averaged stats: lr: 0.0000  loss: 0.9053
2023-05-07 12:15:25,495 [INFO] No validation splits found.
2023-05-07 12:15:25,530 [INFO] Saving checkpoint at epoch 3 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_3.pth.
2023-05-07 12:15:25,592 [INFO] Start training
2023-05-07 12:15:25,631 [INFO] Start training epoch 4, 200 iters per inner epoch.
Train: data epoch: [4]  [  0/200]  eta: 0:01:56  lr: 0.000012  loss: 0.8907  time: 0.5827  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [ 50/200]  eta: 0:01:37  lr: 0.000011  loss: 1.0402  time: 0.6489  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [100/200]  eta: 0:01:07  lr: 0.000010  loss: 0.9383  time: 0.6434  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [150/200]  eta: 0:00:33  lr: 0.000010  loss: 1.0148  time: 0.6435  data: 0.0000  max mem: 36093
Train: data epoch: [4]  [199/200]  eta: 0:00:00  lr: 0.000010  loss: 0.7553  time: 0.6397  data: 0.0000  max mem: 36093
Train: data epoch: [4] Total time: 0:02:11 (0.6594 s / it)
2023-05-07 12:17:37,503 [INFO] Averaged stats: lr: 0.0000  loss: 0.8906
2023-05-07 12:17:37,507 [INFO] No validation splits found.
2023-05-07 12:17:37,543 [INFO] Saving checkpoint at epoch 4 to /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth.
2023-05-07 12:17:37,612 [INFO] No validation splits found.
2023-05-07 12:17:37,612 [INFO] Training time 0:11:03
```

显存占用：

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A800 80G...  Off  | 00000000:3B:00.0 Off |                    0 |
| N/A   69C    P0   311W / 300W |  40041MiB / 81920MiB |     94%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     62283      C   /opt/conda/bin/python           40038MiB |
+-----------------------------------------------------------------------------+

```

模型权重输出：

```
> tree minigpt4_stage2_finetune/
minigpt4_stage2_finetune/
`-- 20230507120
    |-- checkpoint_0.pth
    |-- checkpoint_1.pth
    |-- checkpoint_2.pth
    |-- checkpoint_3.pth
    |-- checkpoint_4.pth
    |-- log.txt
    `-- result

2 directories, 6 files

```
经过第二阶段对齐之后，MiniGPT-4 能够连贯地和用户友好地讨论图像。

至此，整个模型训练过程结束。接下来进行对训练的模型进行评估。

**模型评估**

首先，在评估配置文件`eval_configs/minigpt4_eval.yaml`中的第 11 行设置待评估模型的checkpoint路径，同模型推理。

```
model:
  arch: mini_gpt4
  ...
  low_resource: True
  prompt_path: "prompts/alignment.txt"
  prompt_template: '###Human: {} ###Assistant: '
  ckpt: '/workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth'
...
```
执行如下命令启动模型推理服务进行评估：

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
```

如果出现`Could not create share link. Please check your internet connection or our status page: https://status.gradio.app`这个问题，通常是由于网络环境不稳定造成的。可修改`demo.py`文件如下的代码，使用IP:端口访问即可。

```
#demo.launch(share=True, enable_queue=True)
demo.launch(server_name='0.0.0.0', share=True, enable_queue=True)
```
运行过程：

```
python demo.py --cfg-path eval_configs/minigpt4_eval.yaml --gpu-id 0
Initializing Chat
Loading VIT
cache_file_path: /root/.cache/torch/hub/checkpoints/eva_vit_g.pth
Loading VIT Done
Loading Q-Former
cache_file_path: /root/.cache/torch/hub/checkpoints/blip2_pretrained_flant5xxl.pth
Loading Q-Former Done
Loading LLAMA

===================================BUG REPORT===================================
Welcome to bitsandbytes. For bug reports, please submit your error trace to: https://github.com/TimDettmers/bitsandbytes/issues
================================================================================
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████| 2/2 [00:35<00:00, 17.65s/it]
Loading LLAMA Done
Load 4 training prompts
Prompt Example 
###Human: <Img><ImageHere></Img> Take a look at this image and describe what you notice. ###Assistant: 
Load BLIP2-LLM Checkpoint: /workspace/output/minigpt4_stage2_finetune/20230507120/checkpoint_4.pth
Initialization Finished
Running on local URL:  http://0.0.0.0:7860

```

模型评估测试：

<div align=center>
    <img src="zh-cn/img/ch2/6-8/p5.png" /> 
</div>

本文给大家分享了多模态大模型MiniGPT-4的基本原理及模型训练推理方法，希望能够给大家带来帮助。

!> 中文版MiniGPT-4: https://github.com/RiseInRose/MiniGPT-4-ZH

------
------
## 9. 本草[原名：华驼(HuaTuo)]: 基于中文医学知识的LLaMA微调模型

**BenTsao (original name: HuaTuo): Tuning LLaMA Model With Chinese Medical Instructions**

!> arxiv:https://arxiv.org/pdf/2304.06975.pdf

!> github: https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese

<!-- https://blog.csdn.net/qq_27590277/article/details/130818290 -->
<!-- https://hub.baai.ac.cn/view/27127 -->

<!-- https://zhuanlan.zhihu.com/p/631732569 -->

<!-- https://mp.weixin.qq.com/s/iuQANmwCS7AXQRik7HwQPg -->

### 9.1 背景

当前大规模语言模型(Large Language Model)在通用域自然语言处理任务上已获得巨大的成功[1,2]，然而，由于训练预料占比有限，在部分垂直领域，如生物医学，大规模语言模型的效果仍有限，为缓解上述问题，我们提出了一种基于医学知识增强的大型语言模型指令微调方法，一种让通用大模型适配行业并改造成行业大模型的方法。

<div align=center>
    <img src="zh-cn/img/ch2/4-9/p1.png" /> 
</div>

### 9.2 指令微调

指令微调(Instruct-Tuning)是一种在自然语言处理（Natural Language Processing）领域中结合预训练模型（Pre-trained Model）应用的方法，主要用于改进预训练语言模型，如 FLAN[3]、GPT-4[2]的性能。指令微调的关键在于找到能有效触发模型正确响应的文本提示,利用有标签数据对大型预训练语言模型进行微调。通过在输入中添加指令，可以引导模型生成特定任务所需的输出。这种方法可以将模型适应于各种不同任务，如文本摘要、情感分析、问答系统等。指令微调的主要思路是将任务指令与原始输入文本结合，将它们一起作为模型输入。这使得模型可以将指令作为上下文信息，从而更好地生成与任务相关的输出。

### 9.3 医学知识增强的大型语言模型指令微调

医学知识包含各种类型的医学信息，如疾病诊断和治疗、药品信息、临床指南、研究文献、病例报告和医学图像等。语言模型在通用域上进行预训练时，由于医学知识占比相对有限，进行下游任务推理时，模型表现受制于知识储备。当前，医学知识可以通过以下几种方法融入预训练语言模型：（1）基于医学领域的数据进行微调：将预训练模型的权重用于医学领域数据的微调，以适应医学任务的需求。例如，将预训练的语言模型Fine-tune到医学文本数据上，可以提高医学文本分类、命名实体识别、关系提取等任务的性能。（2）在医学文本上进行预训练：收集医学相关的文本如医学文献、教科书、电子病历、医患对话等医学相关的文本，通过增加预训练过程中医学文本的比重，预训练模型可在预训练阶段学习医学领域的文本特征。

然而，在针对更大规模的语言模型而或当前对话式语言模型（如ChatGPT）时，额外预训练所需的开销较大，一般的微调过程在任务形式上和对话形式存在差异，因此，我们提出了基于指令微调的医学知识增强方法，具体过程如下。

首先，在知识库构建方面，我们汇集大量中文医学资料，包括医学病历、教材以及现有医学知识库等，以构建一个全面且丰富的中文医学知识库。这一知识库的建立为模型提供了丰富的背景知识，有助于提高其在医疗领域的专业性和准确性。

其次，在数据生成方面，我们受Self-Instruct[4]和Alpaca[5]启发，运用GPT-3.5接口，通过设计合适的提示模板，生成了基于医学知识的中文对话数据。这些数据涵盖了医患对话、问诊场景、病症描述和治疗建议等多个方面的内容。这一步骤为模型提供了大量拟真实场景下的医学对话样本，有助于模型更好地理解和适应实际应用场景。

在模型方面，我们选择了大型语言模型LLaMA[1]以及针对中文进行词表扩充与二次预训练的中文Alpaca-7B [7]进行指令微调，旨在提升模型在中文医学对话任务上的表现。微调过程包括构建适当的指令模板，进行数据增强，以及高效地调整参数。这些改进使得模型能够更加准确地捕捉用户的意图，并针对性地提供有价值的信息和建议。

### 9.4 基于医学文献的多轮对话大型语言模型指令微调

医学文献是医学信息的重要载体之一。先前工作通常选择利用MLM等预训练任务将医学文献信息“隐式”的融入语言模型中，从而提升语言模型在下游任务上的性能。例如BioBERT、PubemdBERT等模型均在大量医学文献信息上进行预训练。随着ChatGPT的横空出世，交互式对话引起了广泛关注。在交互式对话中，语言模型通过指令微调等方式与用户指令相对齐，遵循用户的指令给出反馈。为了扩展语言模型在交互式对话中知识的丰富面，我们以多轮对话为载体，将医学文献信息放入多轮对话中，通过指令微调的方式训练模型。在交互中，语言模型能够将医学文献信息“显式”的反馈给用户。

在构建多轮对话数据中，我们调用了GPT-3.5的接口。构建完成的多轮对话数据样例如下图所示，其中蓝色部分是融入到多轮对话中的文献信息，绿色部分则是ChatGPT自身补充的外部知识。

<div align=center>
    <img src="zh-cn/img/ch2/4-9/p2.png" /> 
</div>

经过我们自身的测试，我们观察到模型有足够的能力将文献信息进行整合来反馈给用户，但仍存在一些较严重的问题。例如，即便模型学习到了我们给定的医学文献信息，仍然无法给出相关指令的准确答案，相反，会给出许多不属于给定文献的答案（所谓的“幻觉”）。目前，我们对含幻觉在内的多种负面情况做了初步的分析，后续我们将开源数据集并进一步介绍我们的工作。

### 9.5 本草（原名华驼）模型

通过以上的指令微调方法，我们训练并发布了基于LLaMA-7B [1]以及中文Alpaca-7B [7]的本草模型(点击文末“阅读原文”直达仓库地址)，经过上述改进，本草在中文医疗问诊场景中展现出一定的性能提升，模型能够根据用户的问题，提供较准确、可靠的诊断建议和治疗方案，满足非专业人士对医学知识的需求。此外，模型具备一定的自适应能力，能够在不同的中文问诊场景中进行有效的应答，提升用户体验。在Github上获得一定的关注（截止发稿日已获得2500+ stars）。后续多种高阶版本的“本草”中文医学大模型正在研发中，敬请期待！

<div align=center>
    <img src="zh-cn/img/ch2/4-9/p3.png" /> 
</div>

我们的研究成果表明，结合知识的指令微调是提高大型语言模型在中文医学对话场景上性能的有效途径。这种方法具有广泛的应用前景，可以推广至其他对事实知识有需求的领域，如法律、金融和教育等，为各行业提供智能化知识服务。

### 9.6 效果与展望

本项目于2023年3月31日发布并开源模型参数与训练数据，目前已获得超过2500个star，我们以此提供了一种将通用领域大模型快速适配医学场景实现行业迁移的思路和方法，未来我们将不断迭代医学场景下的大型语言模型，并集成哈工大“活字”对话大模型。

我们也十分关注模型在医疗伦理和隐私保护方面的问题。当前，我们采取了多种措施，确保模型在提供医疗建议时遵循伦理原则，尊重用户隐私。例如，在数据处理过程中，我们对涉及个人隐私的信息进行了脱敏处理，以保护病患的隐私权益。同时，我们还设立了相应的审核机制，确保生成的医学建议符合医疗伦理要求。

同时，我们还将关注其在医疗教育领域的应用，以期模型可以协助医学生进行学术研究，提供针对性的学习建议，帮助医学生更好地掌握医学知识。此外，该项目还可以用于普及医学知识，提高公众的健康意识和自我保健能力。

### 9.7 总结

通过对中文大型语言模型的指令微调，我们成功地实现了一套端到端的中文医疗问诊平台。这一平台不仅有助于为非专业人士提供个性化、准确可信的医学知识服务，还能推动医疗行业的智能化发展，提高医疗服务的质量和可及性。在未来，我们将继续拓展其在医疗领域的应用范围，为更多人带来智能化的医疗服务体验。

**参考文献**

[1] Touvron, Hugo, et al. "Llama: Open and efficient foundation language models." arXiv preprint arXiv:2302.13971 (2023).

[2] OpenAI. "GPT-4 Technical Report." arXiv preprint arXiv:2303.08774 (2023).

[3] Wei, Jason, et al. "Finetuned Language Models are Zero-Shot Learners." International Conference on Learning Representations.

[4] Wang, Yizhong, et al. "Self-Instruct: Aligning Language Model with Self Generated Instructions." arXiv preprint arXiv:2212.10560 (2022).

[5] Alpaca. https://crfm.stanford.edu/2023/03/13/alpaca.html 

[6] ChatGLM. https://github.com/THUDM/ChatGLM-6B  

[7] 中文Alpaca. https://github.com/ymcui/Chinese-LLaMA-Alpaca 

[8] Wang, Haochun, et al. "Huatuo: Tuning llama model with chinese medical knowledge." arXiv preprint arXiv:2304.06975(2023).


------
------
## 10. Firefly

<!-- https://github.com/yangjianxin1/Firefly -->

<!-- https://mp.weixin.qq.com/s/94Cf7e8OZ9GX-TGBmqWwKA -->

<!-- https://mp.weixin.qq.com/s/3CKEfGrFP3OuCMomjffZoA -->

<!-- https://mp.weixin.qq.com/s/O1QV32QRJtYjtvu6ZCDc7Q -->

<!-- https://mp.weixin.qq.com/s/tyH9Ifcvw4DKqoIoYjT6Kg -->

<!-- bloom -->

<!-- https://zhuanlan.zhihu.com/p/617608656 -->
<!-- https://zhuanlan.zhihu.com/p/625911234?utm_id=0 -->
<!-- https://huggingface.co/blog/bloom -->


!> Firefly项目地址：https://github.com/yangjianxin1/Firefly

!> LLMPruner项目地址：https://github.com/yangjianxin1/LLMPruner

!> Firefly权重地址：https://huggingface.co/YeungNLP/firefly-1b4 ; https://huggingface.co/YeungNLP/firefly-2b6

!> firefly-train-1.1M 数据集：https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M

!> Belle-train_0.5M_CN数据集：https://huggingface.co/datasets/BelleGroup/train_0.5M_CN

Firefly的项目基于Bloom指令微调（Instruction Tuning）得到，我们将先介绍Bloom然后介绍基于Blomm进行的指令微调后的Firefly.

### 10.1 BLOOM： A 176B-Parameter Open-Access Multilingual Language Model


!> Bloom arxiv: https://arxiv.org/abs/2211.05100

!> code: https://huggingface.co/bigscience/bloom

<!-- https://zhuanlan.zhihu.com/p/603518061 -->

<!-- https://zhuanlan.zhihu.com/p/632780188 -->

#### 1.简介

预训练语言模型已经成为了现代自然语言处理pipeline中的基石，因为其在少量的标注数据上产生更好的结果。随着ELMo、ULMFiT、GPT和BERT的开发，使用预训练模型在下游任务上微调的范式被广泛使用。随后发现预训练语言模型在没有任何额外训练的情况下任务能执行有用的任务，进一步证明了其实用性。此外，根据经验观察，语言模型的性能随着模型的增大而增加(有时是可预测的，有时是突然的)，这也导致了模型规模越来越多的趋势。抛开环境的问题，训练大语言模型(LLM)的代价仅有资源丰富的组织可以负担的起。此外，直至最终，大多数LLM都没有公开发布。因此，大多数的研究社区都被排除在LLM的开发之外。这在不公开发布导致的具体后果：例如，大多数LLM主要是在英文文本上训练的。

为了解决这些问题，我们提出了BigScience Large Open-science Open-access Multilingual Language Model(BLOOM)。BLOOM是在46种自然语言和13种编程语言上训练的1760亿参数语言模型，其是由数百名研究人员合作开发和发布的。训练BLOOM的计算力是由来自于法国公共拨款的GENCI和IDRIS，利用了IDRIS的Jean Zay超级计算机。为了构建BLOOM，对于每个组件进行了详细的设计，包括训练数据、模型架构和训练目标、以及分布式学习的工程策略。我们也执行了模型容量的分析。我们的总体目标不仅是公开发布一个能够和近期开发的系统相媲美的大规模多语言的语言模型，而且还记录其开发中的协调过程。

#### 2.BLOOM

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p1.png" /> 
</div>

##### 2.1 训练数据

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p2.png" /> 
</div>

BLOOM是在一个称为ROOTS的语料上训练的，其是一个由498个Hugging Face数据集组成的语料。共计1.61TB的文本，包含46种自然语言和13种编程语言。上图3展示了该数据集的高层概览，上表1则详细列出了每种语言及其语属、语系和宏观区域。除了产生了语料库之外，该过程也带来了许多组织和技术工具的开发和发布。


###### 2.1.1 数据管理

大型文本语料库是由人创建且关于人的。不同的人或者机构可以"合法"地拥有这些数据，称为这些数据的权利所有任。随着机器学习开发人员将这些数据收集并整理成为越来越大的数据集时，考虑数据相关的利益方对于开发也越来越重要，包括：开发这、数据主体和权利所有人。

BigScience旨在结合技术、法律、社会学等多学科知识来解决这些问题。该组织在两个不同时间尺度上关注两个主要的目标：设计一个长期的国际数据治理结构，该结构会优先考虑数据权利所有人，并为BigScience项目直接使用的数据提供具体建议。第一个目标的进展在Jernite et al.工作中展示，其进一步激发了数据管理的需求，并描述了一个由数据托管人、权利所有人和其他参与方组成的网络。这些参与者的交互旨在考虑数据和算法上的隐私、知识产权和用户权利。特别地，这种方法依赖于数据提供者和数据主机之间的结构化协议，从而指定数据的用途。

虽然无法在项目开始到模型训练这相对短暂时间内建立一个完整的国际组织，但是我们也努力从这个过程中吸取了经验教训：(1) BigScience会尽量从数据提供者那里获得明确的数据使用许可；(2) 在预处理的最终阶段之前，保持单源独立并维护其可追溯性。(3) 对构成整个语料库的各个数据源采用一种组合发布的方式，从而促进可复用性和后续的研究。在Hugging Face的组织"BigScience Data"中可以访问并可视化ROOTS语料库资源。

###### 2.1.2 数据源

确定了数据管理策略，接下来就是决定训练语言的构成。本阶段由若干个目标驱动，这些目标有着内在的冲突。这些内存的冲突包括：构建一个使世界上尽可能多的人可以访问的语言模型，同时也需要有足够知识来管理与先前规模相当数据集的语言来改善标准文档，以及遵循数据和算法的主体权利。

+ 语言的选择
+ 
基于这些考虑，我们采用渐进的方式来选择语料库中包含的语言。首先列出8个世界上使用人数最多的语言，在项目早期积极推广这些语言并邀请该语言的流利使用者加入项目。然后，根据社区的建议将原始选择中的Swahili扩展至Niger-Congo语言类别，Hindi和Urdu扩展至Indic languages。最终，我们提出若某个语言有多于3个流利语言使用者参与，则可以添加至支持列表。

+ 源的选择

语料库的最大部分是由研讨会参与者和研究团队策划的，他们共同编写了"BigScience Catalogue"：涵盖了各种处理后或者未处理的语言列表。这采用了由Machine Learning Tokyo、Masakhane和LatinX in AI等社区所组织的hackathons形式。作为这些源的补充，其他的工作组参与者编译了特定语言的资源，例如Arabic-focused Masader repository。这种自下而上的方法共确定了252个源，每种语言至少有21个源。此外，为了增加西班牙语、中文、法语和英语资源的地理覆盖范围，参与者通过pseudocrawl确定了被添加至语料中语言的本地相关网址。

+ GitHub代码

通过Google's BigQuer上的GitHub数据集合来进一步补充该目录中的编程语言数据集，然后使用精准匹配进行去重。

+OSCAR

为了不偏离使用网页作为预训练数据源的标准研究，并且满足BLOOM尺寸计算代价的数据量需求，我们进一步使用版本为21.09的OSCAR作为数据源，对应于2021年2月的Common Crawl快照，其占用了最终语料的38%。

###### 2.1.3 数据预处理

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p3.png" /> 
</div>

在确定了数据源之后，数据处理涉及多个数据管理的步骤。上图2可以看到构建ROOTS的pipeline总体视图。在这个过程中开发的所有工具都可以在GitHub上找到。

+ 获得源数据

第一步涉及到从确定的数据源中获得文本数据，这包含从各种格式的NLP数据集中下载和提取文本字段、从档案中抓取和处理大量的PDF文件、从目录中的192个网站条目和数据工作组成员选择的另一些地理上不同的456个网站中提取和预处理文本。后者需要开发新工具来从Common Crawl WARC文件中的HTML中抽取文本。我们能够从539个网络的所有URL中找到并提取可用的数据。

+ 质量过滤

在获得文本后，我们发现大多数源中包含了大量的非自然语言，例如预处理错误、SEO页面或者垃圾。为了过滤非自然语言，我们定义了一组质量指标，其中高质量文本被定义为“由人类为人类编写的”，不区分内容或者语法的先验判断。重要的是，这些指标以两种主要的方法来适应每个源的需求。首先，它们的参数，例如阈值和支持项列表是由每个语言的流利使用者单独选择的。第二、我们首先检测每个独立的源来确定哪些指标最有可能确定出非自然语言。这两个过程都是由工具进行支持来可视化影响。

+ 去重和隐私编辑

最终，我们使用两种重复步骤来移除几乎重复的文档，并编辑了从OSCAR语料中确定出的个人身份信息。因为其被认为是最高隐私风险的来源，这促使我们使用基于正则表达式的编辑，即使表达式有一些假阳性的问题。

###### 2.1.4 Prompted数据集

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p4.png" /> 
</div>

多任务提示微调(也称为instruction tuning)涉及到对预训练语言模型的微调，微调的数据集由通过自然语言提示构成的大量不同任务组成。T0证明了在多任务混合的prompted数据集上微调的模型具有强大的zero-shot泛化能力。此外，T0优于那些数量级大但是没有经过这种微调的语言模型。受这些结果启发，我们探索了使用现有自然语言数据集来进行多任务prompted微调。

T0是在Public Pool of Prompt(P3)子集上进行训练的，其是一个各种现有的、开源的应用自然语言数据集的prompt集合。该prompt集合是通过BigScience合作者参与的一系列黑客马拉松创建的，其中黑客马拉松参与者为170+数据集编写了2000+的prompt。P3中的数据集覆盖了各种自然语言任务，包括情感分析、问答、自然语言推理，并且排除了有害的内容或者非自然语言。PromptSource，一个开源工具包促进了自然语言prompt的创建、共享和使用。

对BLOOM预训练之后，我们应用相同的大规模多任务微调，使BLOOM具有多语言zero-shot任务泛化能力。我们称得到的模型为BLOOMZ。为了训练BLOOMZ，我们扩展了P3来包含非英语中新数据集和新任务，例如翻译。这产生了xP3，它是83个数据集的提升集合，覆盖46种语言和16中任务。正如上图4所述，xP3反映了ROOTS的语言分布。xP3中的任务包含跨语言和单语言。我们使用PromptSource来收集这些prompts，为prompt添加额外的元数据，例如输入和目标语言。为了研究多语言prompt的重要性，我们还将xP3中的英语提示用机器翻译为相应的数据集语言，来生成一个称为xP3mt的集合。


##### 2.2 模型架构

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p5.png" /> 
</div>

###### 2.2.0 ALiBi
<!-- https://zhuanlan.zhihu.com/p/632780188 -->

这是在看faceformer，以及bloom，还有度小满的“轩辕2.0”金融大模型的时候，看到的。他们都用了ALiBi，即给注意力加上线性偏置的方法，来处理“训练的序列长度可以开到2k，而推理的序列长度可以开到4k”的情况。思想核心：
不给词向量加入位置嵌入向量，而是用一个和query, key之间的距离成比例的一个“惩罚项”来偏置query-key的注意力得分。效果：可以加快11%的训练速度，以及减少11%的内存使用。和其他已有的位置编码的对比：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p6.png" /> 
</div>

上面的rotary，是旋转位置编码，来自苏剑林大神的论文，比较有名的是，gpt-J，复旦的moss，都使用了这个旋转位置编码了。

**方法核心**

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p7.png" /> 
</div>

+ 左边的第一项，是自注意力得分，关于q和k的内积的。这个和传统transformer里面的一样了。
+ 第二项是一个相对距离的矩阵，例如q1, k1之间的距离是0，所以对应位置就是0；
+ q2和k1，是相对位置偏移为k的索引1 - q的索引2，得到1-2 = -1，就对应到了中间矩阵的取值为-1了。

以此类推，相对距离矩阵的中间对角线上都是0，然后左下角的取值都是对应的k的索引-q的索引了。
+ 第三项，是个坡度m。按照论文中的描述，其做法是：例如，8个heads的时候，m的取值为：
$$1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256$$
如果是16个heads，则m的取值为：
$$1/\sqrt(2), 1/2, 1/(2*\sqrt(2)), 1/4, ..., 1/256$$
相当于追加了一半的$1/\sqrt(2)$到原来的8个head的每个m的取值。

扩展到一般情况就是：对于$n$个head的话，$m$的取值就是 $2^{\frac{-8}{n}}$,$2^{\frac{-8}{1}},2^{\frac{-8}{2}},...,2^{\frac{-8}{n}}$,这样的$n$个坡度了。整体公式就是：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p8.png" /> 
</div>

对于第$i$个query来说，其可见范围，在自回归设置下，只能是从左边开始$1,2,...,i$个位置，而他们之间的相对距离就是：$0, -1, -2, ..., 1-i$了,即：$k$的索引 $- q$的索引。
$k$的索引，遍历$1, 2, ..., i$；而$q$的索引，取值为$i$。

如此，就有了上面的公式了。

###### 2.2.1 设计方法

架构设计的选择空间非常大，不可能完全探索。一种选择是完全复制现有大模型的架构。另一方面，大量改进现有架构的工作很少被采纳，采用一些推荐的实践可以产生一个更好的模型。我们采用中间立场，选择已经被证明了可扩展性良好的模型家族，以及在公开可用的工具和代码库中合理支持的模型家族。我们针对模型的组件和超参数进行了消融实验，寻求最大限度的利用我们最终的计算预算。

+ 消融实验设计

LLM的主要吸引力是其以"zero/few-shot"的方式执行任务的能力：足够大的模型可以简单的从in-context指令和例子执行新的任务，不需要在监督样本上训练。由于对100B+模型微调很麻烦，我们评估架构决策专注在zero-shot泛化能力上，并且不考虑迁移学习。具体来说，我们衡量了不同任务集合的zero-shot表现：29个任务来自于EleutherAI Language Model Evaluation Harness(EAI-Eval)，9个任务来自T0的验证集(T0-Eval)。两者之间有很大的重叠：T0-Eval中仅有一个任务是不在EAI-Eval，尽管两者的所有prompt都不同。
此外，也使用更小的模型进行了消融实验。使用6.7B模型对预训练目标进行消融实验，使用1.3B模型对位置嵌入、激活函数和layer normalization进行消融实验。近期，Dettmers在大于6.7B的模型上发现了相变，观察到了"异常特征"出现。那么在1.3B规模上是否能够外推自最终模型尺寸上？

+ 超出范围的架构

我们没有考虑mixture-of-experts(MoE)，因为缺乏适合大规模训练它的广泛使用的基于GPU的代码库。类似地，我们也没有考虑state-space模型。在设计BLOOM时，它们在自然语言任务中一种表现不佳。这两种方法都很有前景，现在证明了在大规模MoE上有竞争力的结果，并在较小规模上使用具有H3的state-space模型。

###### 2.2.2 架构和预训练目标

虽然大多数现代语言模型都是基于Transformer架构，但是架构实现之间存在着显著的不同。显然，原始的Transformer是基于encoder-decoder架构的，许多流行的模型仅选择encoder-only或者decoder-only方法。当前，所有超过100B参数的state-of-the-art模型都是decoder-only模型。这与Raffel等人的发现相反，在迁移学习方面encoder-decoder模型显著优于decoder-only模型。

在我们工作之前，文献缺乏不同架构和预训练目标的系统性评估zero-shot泛化能力。我们在Wang et al.(2022a)等人的工作中探索了这个问题，其探索了encoder-decoder和decoder-only架构以及与causal、prefix和masked language modeling预训练模型的相互作用。我们的结果显示，经过预训练之后，causal decoder-only模型的表现最好，验证了state-of-the-art LLM的选择。

###### 2.2.3 建模细节

在选择架构和预训练目标之外，对原始Transformer架构提出了许多的更改。例如，可选的位置嵌入方案或者新颖的激活函数。我们执行了一系列的实验来评估每个修改，在Le Scao et al.的causal decoder-only模型上。我们在BLOOM中采用了两种变化：

+ ALiBi位置嵌入

相比于在embedding层添加位置信息，ALiBi直接基于keys和queries的距离来衰减注意力分数。虽然ALiBi的最初动机是它能够外推至更长的序列，我们发现其在原始序列长度上也能够带来更平衡的训练以及更好的下游表现，超越了可学习embeddings和旋转embeddings。

+ Embedding LayerNorm

在训练104B参数模型的初步试验中，我们尝试在嵌入层后立即进行layer normalization，正如bitsandbytes库及其StableEmbedding层所推荐的那样。我们发现这可以显著的改善训练稳定性。尽管我们在Le Scao et al.工作中发现其对zero-shot泛化有惩罚，但我们还是在BLOOM的第一个embedding层后添加了额外的layer normalization层来避免训练不稳定性。注意初步的104B实验中使用float16，而最终的训练上使用bfloat16。因为float16一直被认为是训练LLM时观察的许多不稳定的原因。bfloat16有可能缓解对embedding LayerNorm的需要。
上图5中展示了BLOOM的全部架构。

##### 2.3 Tokenization

tokenizer的设计选择通常被忽略，倾向于"默认"设置。举例来说，OPT和GPT-3都使用GPT-2的tokenizer，训练用于English。由于BLOOM训练数据的多样性本质，需要谨慎的设计选择来确保tokenizer以无损的方式来编码句子。

###### 2.3.1 验证

我们将本文使用的tokenizer(Acs,2019)与现有的单语言tokenizer进行比较，作为完整性检测的指标。Fertility被定义为每个单词或者每个数据集被tokenizer创造的subword数量，我们使用感兴趣语言的Universal Dependencies 2.9和OSCAR子集来衡量。在一个语言上有非常高的Fertility相比于单语言tokenizer可能表明在下游多语言上的性能下降。我们的目标是在将我们的多语言tokenizer与对应但语言tokenizer进行比较时，确保每种语言的fertility能力较低不超过10个百分点。在所有的实验中，Hugging Face Tokenizers库被用来设计和训练测试的tokenizers。


###### 2.3.2  tokenizer训练数据

我们最初使用ROOTS的非重复子集。然而，一项在tokenizer的词表上的定性研究揭示了训练数据的问题。例如，在早期版本的tokenizer上，我们发现完整URLs存储为tokens，这是由几个包含大量重复的文档导致的。这个问题促使我们移除tokenizer训练数据中的重复行。

###### 2.3.3 词表大小

大的词表尺寸能够降低过度分割某些句子的风险，特别是对低资源语言。我们使用150k和250k词表尺寸来执行验证实验，以便与现有的多语言建模文献进行比较。与单语言tokenizer相比，我们最终确定的词表尺寸是250k tokens来达到最初的fertility目标。因为，词表尺寸决定了embedding矩阵的尺寸，为了GPU效率embedding尺寸必须被128整除，为了使用张量并行必须被4整除。我们最终使用了250680词表尺寸，具有200个为未来应用保留的token，例如使用占位token剔除私有信息。


###### 2.3.4 Byte-level BPE

tokenizer是一个使用Byte Pair Encoding(BPE)算法进行训练的、可学习的子词tokenzier。为了在tokenization的过程中不丢失信息，tokenizer从bytes开始创建合并，而不是以字符作为最小单位。这种方式，tokenization永远不会产生未知的tokens，因为所有256个字节都可以被包含在tokenizer的词表中。此外，Byte-level BPE最大化了语言之间的词表共享。

###### 2.3.5 规范化

在BPE算法上游，为了尽可能地获得最通用的模型，没有对文本进行规范化。在所有情况下，添加诸如NFKC这样的unicode规范化并不能减少fertility超过0.8%，但是代价是使模型不那么通用。例如，导致2^2和22以相同方法被编码。

###### 2.3.6 Pre-tokenizer

我们的pre-tokenization有两个目标：产生文本的第一个划分，并且限制由BPE算法产生的令牌序列的最大长度。pre-tokenization规模使用的是下面的正则表达式：?[^(\S|[.,!?...。，、|_])]+，其将单词分开同时保留所有的字符，特别是对编程语言至关重要的空格和换行符序列。我们不使用在其他tokenizers中常见的以英文为中心的划分。我们也没有在数字上使用划分，这导致了Arabic和code的问题。


##### 2.4 工程

###### 2.4.1 硬件

模型在Jean Zay上训练，其是由法国政府资助的超级计算机，归属于GENCI所有，由法国国家科学研究中心(CNRS)的国家计算中心IDRIS运行。训练BLOOM花费了3.5个月才完成，并消耗了1082990计算小时。在48个节点上进行训练，每个有8 NVIDIA A100 80GB GPUs(总共384个GPUs)；由于在训练过程中硬件可能损坏，我们也保留了4个备用节点。这些节点装备了2x AMD EPYC 7543 32-Core CPUs和512 GB的RAM，而存储采用混合全闪存和硬盘驱动的SpectrumScale(GPFS)并行文件系统。

###### 2.4.2 框架

BLOOM使用Megatron-DeepSpeed训练，一个用于大规模分布式训练的框架。其由两部分组成：Megatron-LM提供Transformer实现、张量并行和数据加载原语，而DeepSpeed提供ZeRO优化器、模型流水线、通过分布式训练组件。这个框架允许我们使用3D并行来高效训练---融合了三种互补的分布式深度学习方法。这些方法描述如下：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p9.png" /> 
</div>

+ 数据并行(Data parallelism, DP)

复制多份模型，每个副本被放置在不同设备上，并输入数据分片。该过程是并行完成的，所有模型副本在每个训练step结束时同步。

+ 张量并行(Tensor parallelism, TP)

跨多个设备来划分模型的独立层。这种方式，我们不把整个激活张量或者梯度张量放在单个GPU上，而是把这个张量的碎片放在单个GPU上。该技术有时被称为水平并行或者层内模型并行。

+ 流水线并行(Pipeline parallelism, PP)

在多个GPU上划分模型的层，每个GPU仅放置模型层的一小部分。这有时也称为垂直并行。

最终，Zero Redundancy Optimizer(ZeRO)运行不同的进程仅持有部分数据(参数、梯度和优化器状态)以及一个训练step所需要的数据。我们使用ZeRO stage 1，意味着仅优化器状态以这种方法进行分片。

上面描述的四个组件组合在一起，可以扩展至数百个GPU，具有极高的GPU利用率。我们能在A100 GPU的最快配置下实现156 TFLOPs，实现了理论峰值312 TFLOPs的一半。

###### 2.4.3 浮点数格式

在初步的实验中，104B参数模型在NVIDIA V100 GPUs，我们观察到数值不稳定，导致不可逆的训练发散。我们假设这些不稳定来自于最初使用的IEEE float16，动态范围非常有限的16-bit浮点数格式，可能导致溢出。我们最终获得了支持bfloat16格式的权限，其具有同float32相同的动态范围。另一方面，bfloat16精度仍然低很多，这促使我们使用混合精度训练。该技术在float32精度上执行精度敏感的操作，例如梯度累积和softmax，余下的操作则使用低精度，这允许实现高表现和训练稳定性之间的平衡。最终，我们以bfloat16混合精度执行最终的训练，其被证明解决了训练不稳定的问题。

###### 2.4.4 融合CUDA核

一般来说，GPU无法在检索数据同时执行这些计算。此外，现代GPU的计算性能远远高于每个操作(被称为GPU编程中的核)所需的内存传输速度。核融合是一种基于GPU计算的优化方法，通过在一次内核调用中执行多个连续操作。该方法提供了一种最小化数据传输的方法：中间结果留在GPU寄存器中，而不是复制到VRAM，从而节省开销。

我们使用了Megatron-LM提供了几个定制化融合CUDA核。首先，我们使用一个优化核来执行LayerNorm，以及用核来融合各种缩放、掩码和softmax操作的各种组合。使用Pytorch的JIT功能将一个偏差项添加至GeLU激活中。作为一个使用融合核的例子，在GeLU操作中添加偏差项不会增加额外的时间，因为该操作受内存限制：与GPU VRAM和寄存器之间的数据传输相比，额外的计算可以忽略不计。因此融合这两个操作基本上减少了它们的运行时间。

###### 2.4.5 额外的挑战

扩展至384个GPU需要两个修改：禁止异步CUDA内核启动(为了方便调试和防止死锁)，并将参数组划分至更小的子组(以避免过多的CPU内存分配)。

在训练过程中，我们面临硬件故障的问题：平均来说，每周有1-2个GPU故障。由于备份节点可用并自动使用，并且每三个小时保存一次checkpoint，因此这不会显著影响训练吞吐量。在数据loader中Pytorch死锁bug和磁盘空间故障会导致5-10h的停机时间。考虑到工程问题相对稀疏，而且由于只有一次损失峰值，该模型很快就恢复了，因此人工干预的必要性低于类似项目。

##### 2.5 训练

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p10.png" /> 
</div>

+ 预训练模型

我们使用上表3中详细描述的超参数来训练BLOOM的6个尺寸变体。架构和超参数来自于我们的实验结果(Le Scao et al.)和先前的训练大语言模型(Brown et al.)。非176B模型的深度和宽度大致遵循先前的文献(Brown et al.)，偏离的3B和7.1B只是为了更容易适合我们训练设置。由于更大的多语言词表，BLOOM的embedding参数尺寸更大。在开发104B参数模型的过程中，我们使用了不同的Adam 
 参数、权重衰减和梯度裁剪来对目标稳定性进行实验，但没有发现其有帮助。对于所有模型，我们在410B tokens使用cosine学习率衰减调度，在计算允许的情况下，将其作为训练长度的上限，并对375M tokens进行warmup。我们使用权重衰减、梯度裁剪，不使用dropout。ROOTS数据集包含341B tokens的文本。然而，基于训练期间发布的修订scaling laws，我们决定在重复数据上对大模型进行额外25B tokens的训练。由于warmup tokens + decay tokens大于总的token数量，所以学习率衰减始终未达到终点。

+ 多任务微调

微调的BLOOMZ模型维持了与BLOOM模型相同架构超参数。微调的超参数大致基于T0和FLAN。学习率则是将对应预训练模型的最小学习率加倍，然后再四舍五入。对于较小的变体，全局batch size乘以4来增加吞吐量。模型在13B tokens上进行微调，最优checkpoint根据独立的验证集选择。经过1-6B tokens微调后，性能趋于平稳。

+ 对比微调

我们还使用了SGPT Bi-Encoder方案对1.3B和7.1B参数的BLOOM模型进行对比微调，以训练产生高质量文本嵌入的模型。我们创建了用于多语言信息检索的SGPT-BLOOM-1.7B-msmarco，以及用于多语言语义相似度的SGPT-BLOOM-1.7B-nli。然而，近期的基准测试发现，这种模型也能够推广到各种其他的嵌入任务，例如bitext挖掘、重排或者下游分类的特征抽取。

##### 2.6. 发布

开放性是BLOOM开发的核心，并且我们希望确保社区可以轻易的使用它。


###### 2.6.1 Model Card

遵循发布机器学习模型的最优实践，BLOOM模型连同详细的Model Card一起发布，其描述了技术规范、训练细节、预期用途、范围外用途和模型的局限。跨工作组的参与者共同来产生最终的Model Card和每个checkpoint的card。

###### 2.6.2 Licensing

考虑到BLOOM可能带来的潜在有害用例，我选择了不受限制的开发访问和负责任的使用之间取得平衡，包括行为使用准则来限制模型对潜在有害用例的应用。这些条款通常被包含在"Responsible AI Licenses(RAIL)"，社区在发布模型时所采用的Licenses。与BLOOM开始时采用的RAIL license显著区别是，它分离了"源代码"和"模型"。它还包括了模型的"使用"和“派生工作”的详细定义，来确保明确识别通过prompting、微调、蒸馏、logits使用和概率分布的下游使用。该license包含13项行为使用限制，这些限制根据BLOOM Model Card描述的预期用途和限制，以及BigScience道德章程来确定。该license免费提供模型，用户只要遵守条款，就可以自由使用模型。BLOOM的源代码已经在Apache 2.0开源许可证下提供访问。

#### 3.评估

评估主要专注在zero-shot和few-shot设置。我们的目标是呈现出BLOOM与现有LLMs相比的准确图景。由于这些模型的规模，基于prompt的方法和few-shot "in-context learning"要比微调更加常见。

关于相关评估结果请参考原paper!

### 10.2 Firefly(流萤): 中文对话式大语言模型

<!-- https://mp.weixin.qq.com/s/O1QV32QRJtYjtvu6ZCDc7Q -->

在本文中，笔者将介绍关于Firefly（流萤）模型的工作，一个中文对话式大语言模型。我们使用较小的模型参数量，如1.4B和2.6B，实现了不错的生成效果

!> 项目地址：https://github.com/yangjianxin1/Firefly

#### 1.项目简介

Firefly（流萤）是一个开源的中文对话式大语言模型，使用指令微调（Instruction Tuning）在中文数据集上进行调优。同时使用了词表裁剪、ZeRO、张量并行等技术，有效降低显存消耗和提高训练效率。在训练中，我们使用了更小的模型参数量，以及更少的计算资源。

我们构造了许多与中华文化相关的数据，以提升模型在这方面的表现，如对联、作诗、文言文翻译、散文、金庸小说等。

流萤（萤火虫的别称）是中华传统文化的一个符号，虽说腐草为萤，带有悲悯意味，但萤火虽小，也能凭借其淡淡荧光，照亮夜空。本项目的名称取自杜牧的《秋夕》：银烛秋光冷画屏，轻罗小扇扑流萤。也希望本项目能够像流萤一般发出淡淡微光，为中文NLP开源社区尽绵薄之力，添砖加瓦。

```
《咏萤火》 
 唐.李白
雨打灯难灭，
风吹色更明。
若飞天上去，
定作月边星
```

本项目的主要工作如下：

+ a.数据集：firefly-train-1.1M ，一份高质量的包含1.1M中文多任务指令微调数据集，包含23种常见的中文NLP任务的指令数据。对于每个任务，由人工书写若干指令模板，保证数据的高质量与丰富度。

+ b.模型裁剪：我们开发了LLMPruner项目-大语言模型裁剪工具 。使用词表裁剪技术对多语种大语言模型进行权重裁剪，保留预训练知识的前提下，有效减少模型参数量，提高训练效率，并分享裁剪后的多种参数规模的Bloom模型权重。

+ c.权重分享：在bloom-1b4-zh和bloom-2b6-zh的基础上，进行指令微调，获得两种参数规模的中文模型：firefly-1b4和firefly-2b6。

+ d.训练代码：开源训练代码，支持张量并行、ZeRO、Gemini异构内存空间管理等大模型训练策略。可实现仅使用一张显卡，训练1B-2B参数量的模型。

```python
from transformers import BloomTokenizerFast, BloomForCausalLM
device = 'cuda'
path = 'YenugNLP/firefly-1b4'

tokenizer = BloomTokenizerFast.from_pretrained(path)
model = BloomForCausalLM.from_pretrained(path)
model.eval()
model = model.to(device)
text = input('User：')
while True:
    text = '<s>{}</s></s>'.format(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids, max_new_tokens=200, do_sample=True, top_p=0.8, temperature=0.35,
                             repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
    rets = tokenizer.batch_decode(outputs)
    output = rets[0].strip().replace(text, "").replace('</s>', "")
    print("Firefly：{}".format(output))
    text = input('User：')
```

#### 2.方法介绍

**模型裁剪**

关于LLMPruner，详见文章：[LLMPruner：大语言模型裁剪工具](https://mp.weixin.qq.com/s?__biz=MzA3MTgwODE1Ng==&mid=2247484078&idx=1&sn=690adaac3decb8447bf5a254bcd8856f&chksm=9f26a75da8512e4b673787d7d58c52970096b306232934d7fdce598347571241e443a74b6089&scene=21#wechat_redirect)

LLMPruner项目链接：https://github.com/yangjianxin1/LLMPruner

本项目首先使用LLMPruner项目对原始的Bloom模型进行词表裁剪，仅取出常用的中英文词表，大大降低了模型参数量，然后再对其进行指令微调。

Bloom是个多语言模型，由于需要兼容多语言，所以词表有25w之多，在中文领域中，大部分词表并不会被用到。我们通过删减冗余的词表，从多语言模型中提取常用的中英文词表，最终词表从25w减少到46145，缩减为原来的18.39%，在保留预训练知识的同时，有效减少参数量，提高训练效率。

我们在 bloom-1b4-zh与bloom-2b6-zh的基础上，进行指令微调，获得两种参数规模的中文模型：firefly-1b4和firefly-2b6，具有不错的效果。

裁剪后的模型如下表所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p11.png" /> 
</div>

**数据集**

我们收集了23个常见的中文数据集，对于每个任务，由人工书写若干种指令模板，保证数据的高质量与丰富度，数据量为115万，形成训练集firefly-train-1.1M。数据分布如下图所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p12.png" /> 
</div>

在此基础上，我们添加了Belle-train_0.5M_CN，最终得到165万的训练数据。每条数据的格式如下，包含任务类型、输入、目标输出：

```

{
  "kind": "ClassicalChinese", 
  "input": "将下面句子翻译成现代文：\n石中央又生一树，高百余尺，条干偃阴为五色，翠叶如盘，花径尺余，色深碧，蕊深红，异香成烟，著物霏霏。",
  "target": "大石的中央长着一棵树，一百多尺高，枝干是彩色的，树叶有盘子那样大，花的直径有一尺宽，花瓣深蓝色，花中飘出奇异的香气笼罩着周围，如烟似雾。"
}
```

训练数据集的token长度分布如下图所示，绝大部分数据的长度都小于600：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p13.png" /> 
</div>

**训练策略**

在训练时，对于每条样本，我们将input与target拼接成如下格式，然后输出模型中，进行训练。

```

<s>input</s></s>target</s>
```
我们将模型的最大输入设置为512，input部分不参与计算loss，只计算target部分的损失函数。训练的超参数设置如下表所示。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p14.png" /> 
</div>

#### 3.生成效果

下面展示的是firefly-1b4的部分生成效果。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p15.png" /> 
</div>

下面是Firefly模型更具特色的一些例子，Firefly对于文言文、诗词、对联、武侠小说、散文、歌词、文案生成、情感分析等中文任务具有非常优秀的表现。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p16.png" /> 
</div>

#### 4.结语

经过词表裁剪后，我们的模型参数量仅为1.4B和2.6B，参数量远远小于ChatGPT和LLaMA等上百亿上千亿的模型，甚至远远小于当前主流如Belle、ChatGLM等7B左右的模型。所以在效果上仍存在以下问题：

+ 对于事实性知识，容易产生错误的回复。
+ 由于未经过无害化微调，可能会产生歧视、危害、违背伦理道德的言论。
+ 在代码、推理上的能力仍有欠缺。

基于以上模型的局限性，我们要求本项目的代码、数据、模型等仅用于学术研究，不得用于商业用途，且不得用于对社会造成危害的用途。

后续笔者将从以下方向对项目进行迭代：丰富训练数据且保证数据质量，优化训练流程，尝试更大参数量的模型。

若使用本项目的数据、代码或模型，请引用本项目。
```

@misc{Firefly,
  author = {Jianxin Yang},
  title = {Firefly(流萤): 中文对话式大语言模型},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yangjianxin1/Firefly}},
}
```

Firefly项目地址：

https://github.com/yangjianxin1/Firefly


LLMPruner项目地址：

https://github.com/yangjianxin1/LLMPruner


Firefly权重地址：

https://huggingface.co/YeungNLP/firefly-1b4;

https://huggingface.co/YeungNLP/firefly-2b6


firefly-train-1.1M 数据集：

https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M


Belle-train_0.5M_CN数据集：

https://huggingface.co/datasets/BelleGroup/train_0.5M_CN



### 10.3 QLoRA实战:使用单卡高效微调bloom-7b1，效果惊艳

<!-- https://mp.weixin.qq.com/s/94Cf7e8OZ9GX-TGBmqWwKA -->

在上一节中我们介绍了关于Firefly（流萤）模型的工作。对大模型进行全量参数微调需要大量GPU资源，所以我们通过对Bloom进行词表裁剪，在`4*32G`的显卡上，勉强训练起了`2.6B`的firefly模型。

在本文中，我们将介绍QLoRA，由华盛顿大学提出的一种高效微调大模型的方法，可在单张A100上对LLaMA-65B进行微调。在论文中，作者的实验表明使用QLoRA微调的LLaMA-65B，可达到ChatGPT性能水平的99.3%（由GPT-4进行评价），并且QLoRA的性能可以逼近全量参数微调。作者做了丰富的实验证明这一结论。

在本文中我们将对QLoRA的基本原理进行介绍，并且在Firefly项目中进行实践。我们在bloom-7b1的基础上，使用QLoRA进行中文指令微调，获得firefly-7b1-qlora-v0.1模型，具有不错的效果，生成效果见第三章。QLoRA确实是一种高效训练、效果优秀、值得尝试和深入研究的方法。

论文地址：https://arxiv.org/pdf/2305.14314.pdf

项目代码：https://github.com/yangjianxin1/Firefly

模型权重：https://huggingface.co/YeungNLP/firefly-7b1-qlora-v0.1

#### 1.QLoRA简介

本章节主要对LoRA与QLoRA进行介绍(我们在前面章节已经做了详细介绍），如读者已了解本章节的内容，可直接跳过，阅读项目实践部分。

**LoRA简介**

在介绍QLoRA之前，简单回顾一下LoRA。LoRA的本质是在原模型的基础上插入若干新的参数，称之为adapter。在训练时，冻结原始模型的参数，只更新adapter的参数。对于不同的基座模型，adapter的参数量一般为`几百万~几千万`。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p17.png" /> 
</div>

LoRA的优势在于能够使用较少的GPU资源，在下游任务中对大模型进行微调。在开源社区中，开发者们使用LoRA对Stable Diffusion进行微调，取得了非常不错的效果。随着ChatGPT的火爆，也涌现出了许多使用LoRA对LLM进行指令微调的工作。

此前，我们也实践过使用LoRA对LLM进行指令微调，虽然未进行定量分析，但主观感受LoRA比全量微调还是有一定的差距。实践下来，我们发现LoRA微调中存在以下三个痛点：

+ 参数空间小：LoRA中参与训练的参数量较少，解空间较小，效果相比全量微调有一定的差距。
+ 微调大模型成本高：对于上百亿参数量的模型，LoRA微调的成本还是很高。
+ 精度损失：针对第二点，可以采用int8或int4量化，进一步对模型基座的参数进行压缩。但是又会引发精度损失的问题，降低模型性能。

**QLoRA简介**

接下来便引入今天的主角QLoRA。整篇论文读下来，我们认为QLoRA中比较重要的几个做法如下：

+ 4-bit NormalFloat：提出一种理论最优的4-bit的量化数据类型，优于当前普遍使用的FP4与Int4。
+ Double Quantization：相比于当前的模型量化方法，更加节省显存空间。每个参数平均节省0.37bit，对于65B的LLaMA模型，大约能节省3GB显存空间。
+ Paged Optimizers：使用NVIDIA统一内存来避免在处理小批量的长序列时出现的梯度检查点内存峰值。
+ 增加Adapter：4-bit的NormalFloat与Double Quantization，节省了很多空间，但带来了性能损失，作者通过插入更多adapter来弥补这种性能损失。在LoRA中，一般会选择在query和value的全连接层处插入adapter。而QLoRA则在所有全连接层处都插入了adapter，增加了训练参数，弥补精度带来的性能损失。

通过上述优化，只需要41G显存即可微调LLaMA-65B模型。甚至可以直接使用一张1080Ti来微调LLaMA-13B，手中的旧卡又可以继续发挥余热了。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p18.png" /> 
</div>

作者使用GPT4对各个模型进行评价，结果显示，使用QLoRA在OASST1数据集上微调得到的Guanaco-65B模型达到了ChatGPT的99.3%的性能。

作者进一步采用了Elo等级分制度对各个模型进行评价，裁判为人类或者GPT-4。结果显示Guanaco-65B和Guanaco-33B均优于ChatGPT-3.5。

**实验分析**

QLoRA方法是否有用，其与全量参数微调的差距有多大？作者使用LLaMA-7B和Alpaca数据集进行了实验。下图结果表明，通过插入更多的adapter，能够弥补QLoRA量化带来的性能损失，复现全量参数微调的效果。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p19.jpg" /> 
</div>

除此之外，作者还将QLoRA应用于RoBERTA和T5，评测其在GLUE和Super-NaturalInstructions数据集上的表现。从下表中可以看到，QLoRA+NF4+DQ基本上复现了BF16全量微调的实验指标。

下表中LoRA+BF16基本上也复现了BF16全量微调的实验指标，如果作者能加上LoRA+FP4或者LoRA+int4的实验结果，则可以更清晰地展现LoRA与QLoRA的性能差异。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p20.png" /> 
</div>

在指令微调阶段，数据质量和数据数量，哪一个更重要？作者使用三种不同的训练集，每个数据集分别使用5万、10万、15万的数据量进行训练。对于下表，纵向来看，随着数据量的增加，指标并没有明显的提升，说明数据量不是关键因素。横向来看，对于不同的数据集，指标差距甚大，说明数据质量更关键。

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p21.png" /> 
</div>

值得一提的是，在论文中，作者仅使用了9千多条OASST1的数据训练得到Guanaco-65B，这进一步验证了，数据质量远比数量重要，模型的知识来源于预训练阶段。

模型的知识来源于预训练阶段，指令微调目的是和人类指令进行对齐。在指令微调阶段，数据的质量与丰富度，远比数量更重要。这是最近一段时间，开源社区以及各个论文强调的一个结论，在我们的实践中也深有体会。

#### 2.项目实践

在本项目中，我们使用bloom-7b1作为基座模型。数据集为moss-003-sft-no-tools，这是由MOSS项目开源的中文指令微调数据集，我们随机抽取了29万条作为训练数据，训练得到firefly-7b1-qlora-v0.1。

训练时，我们将多轮对话拼接成如下格式，然后进行tokenize。

```
<s>input1</s>target1</s>input2</s>target2</s>...
```

我们在一张32G显卡上使用QLoRA进行训练，在所有全连接层处都插入adapter，最终参与训练的参数量超过1亿，相当于一个bert-base的参数量。训练时只计算target部分的损失函数。

训练超参数如下所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p22.png" /> 
</div>

模型的训练损失的变化趋势如下图所示：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p23.png" /> 
</div>

firefly-7b1-qlora-v0.1的使用方式如下：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, BitsAndBytesConfig
import torch

model_name = 'bigscience/bloom-7b1'
adapter_name = 'YeungNLP/firefly-7b1-qlora-v0.1'
device = 'cuda'
input_pattern = '<s>{}</s>'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
model = PeftModel.from_pretrained(model, adapter_name)
model.eval()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = input('User：')
while True:
    text = input_pattern.format(text)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=250, do_sample=True, top_p=0.75, temperature=0.35,
                             repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
    rets = tokenizer.batch_decode(outputs)
    output = rets[0].strip().replace(text, "").replace('</s>', "")
    print("Firefly：{}".format(output))
    text = input('User：')
```

#### 3.生成效果

下面的样例均为firefly-7b1-qlora-v0.1模型所生成，未经修改，仅供参考。

多轮对话:

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p24.png" /> 
</div>

多轮对话2：
<div align=center>
    <img src="zh-cn/img/ch2/4-10/p25.png" /> 
</div>

邮件生成：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p26.png" /> 
</div>

商品文案生成：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p27.png" /> 
</div>

医疗问答：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p28.png" /> 
</div>

创意性写作：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p29.png" /> 
</div>

其他例子：

<div align=center>
    <img src="zh-cn/img/ch2/4-10/p30.png" /> 
</div>

#### 4.结语

在本文中，我们介绍了QLoRA的基本原理，以及论文中一些比较重要的实验结论。并且使用QLoRA对bloom-7b1模型进行中文指令微调，获得了非常不错的效果。

从firefly-7b1-qlora-v0.1的生成效果来看，虽然没有做定量的评测(对LLM做评测确实比较困难)，但就生成效果来看，丝毫不逊色于全量微调的firefly-2b6-v2。

一些碎碎念：
+ 论文中表明QLoRA能够媲美全量参数微调的效果，虽然可能需要更丰富、多角度的实验进行验证，但如果【增大基座模型的参数量+QLoRA】能够优于【全量微调较小的模型】，也是非常有意义的。
+ 对基座模型进行量化压缩，通过增加adapter来弥补量化导致性能损失，是一个非常不错的idea，论文中的实验也证实了这一点。并且从我们的实践效果看来，确实惊艳，效果远胜LoRA。
+ 最后，如果你手边的训练资源不足，QLoRA非常值得一试。

------
------
## 11. baichuan-7B 和 Aquila-7B

<!-- https://mp.weixin.qq.com/s/L6r_iKnF2U4nTTKodcvPoA -->
<!-- https://mp.weixin.qq.com/s/o4A5mEdEOEFWVQ_U651_tA -->
<!-- https://mp.weixin.qq.com/s/XkoLnFycG1jPWrNT3w_p-g -->


!> 源码: https://github.com/baichuan-inc/baichuan-7B

!> 模型: https://huggingface.co/baichuan-inc/baichuan-7B
 
!> https://modelscope.cn/models/baichuan-inc/baichuan-7B/summary


近期2大国产可商用大模型推介：baichuan-7B及AquilaChat-7B模型的数据、结构以及所需成本介绍

最近百亿参数模型受到广泛关注。本文主要介绍baichuan-7B大模型以及AquilaChat-7B两个最近的可商用模型，以其开源项目源码、issue、项目介绍等信息为基础，进行介绍。
其中，我们重点关注训练数据、模型结构（重点优化点）、运行资源、是否可以商用以及实际评估等几个方面的内容，这有利于我们沉淀出大模型训练的一些经验。


### 11.1 baichuan-7B大模型及其评估

baichuan-7B是由百川智能开发的一个开源可商用的大规模预训练语言模型。

基于Transformer结构，在大约1.2万亿tokens上训练的70亿参数模型，支持中英双语，上下文窗口长度为4096。baichuan-7B 支持商用。在32G V100上跑推理，大约占用28G显存。根据issue的现实，目前开源了的是pretrain版本。

**1、训练数据**

原始数据包括开源的中英文数据和自行抓取的中文互联网数据，以及部分高质量知识性数据。

在数据处理上，基于启发式规则和质量模型打分，对原始数据集进行篇章和句子粒度的过滤。在全量数据上，利用局部敏感哈希方法，对篇章和句子粒度做过滤去重。


<div align=center>
    <img src="zh-cn/img/ch2/4-12/p1.png" /> 
</div>

+ 经过不断的调整和多轮测试，最终确认了一个在下游任务上表现最好的中英文配比。
+ 我们使用了一个基于自动学习的数据权重策略，对不同类别的数据进行配比

**2、tokenizer(分词)**

我们参考学术界方案使用 SentencePiece 中的 Byte-Pair Encoding (BPE) 作为分词算法，并且进行了以下的优化：

1. 目前大部分开源模型主要基于英文优化，因此对中文语料存在效率较低的问题。我们使用 2000 万条以中英为主的多语言语料训练分词模型，显著提升对于中文的压缩率。
2. 对于数学领域，我们参考了 LLaMA 和 Galactica 中的方案，对数字的每一位单独分开，避免出现数字不一致的问题，对于提升数学能力有重要帮助。
3. 对于罕见字词（如特殊符号等），支持 UTF-8 characters 的 byte 编码，因此做到未知字词的全覆盖。
4. 我们分析了不同分词器对语料的压缩率，如下表，可见我们的分词器明显优于 LLaMA, Falcon 等开源模型，并且对比其他中文分词器在压缩率相当的情况下，训练和推理效率更高。

<div align=center>
    <img src="zh-cn/img/ch2/4-12/p2.png" /> 
</div>

**3、模型结构**

整体模型基于标准的 Transformer 结构，我们采用了和 LLaMA 一样的模型设计

<div align=center>
    <img src="zh-cn/img/ch2/4-12/p3.png" /> 
</div>


并从位置编码、激活层、前馈层、Layer-Normalization等进行了优化：
1) 位置编码，采用[rotary-embedding](https://zhuanlan.zhihu.com/p/359502624)，这一现阶段被大多模型采用的位置编码方案，具有更好的外延效果。

2) 激活层，采用SwiGLU；

3) Feedforward，变化为8/3倍的隐含层大小；
4) Layer-Normalization，layer normalization 重要的两个部分是平移不变性和缩放不变性，儿大语言模型训练很重要的一点就是提升训练的稳定性，为了提升训练稳定性，GPT3、PaLM、BLOOM、OPT等大语言模型都采用了pre layernorm。

+ post layernorm。在原始的transformer中，layer normalization是放在残差连接之后的，称为postLN。使用PostLN的深层transformer模型容易出现训练不稳定的问题，postLN随着transformer层数的加深，梯度范数逐渐增大，导致了训练的不稳定性。
+ pre layernorm。改变layer normalization的位置，将其放在残差连接的过程中，self-attention或FFN块之前，称为“PreLN”，Pre layer norm在每个transformer层的梯度范数近似相等，有利于提升训练稳定性。相比于postLN，使用preLN的深层transformer训练更稳定，可以缓解训练不稳定问题，但preLN可能会轻微影响transformer模型的性能。
5) LayerNorm替换为RMSNorm。

**4、训练稳定性和吞吐**

我们在原本的 LLaMA 框架上进行诸多修改以提升训练时的吞吐，具体包括：

1. 算子优化技术：采用更高效算子，如 Flash-Attention，NVIDIA apex 的 RMSNorm 等。
2. 算子切分技术：将部分计算算子进行切分，减小内存峰值。
3. 混合精度技术：降低在不损失模型精度的情况下加速计算过程。
4. 训练容灾技术：训练平台和训练框架联合优化，IaaS + PaaS 实现分钟级的故障定位和任务恢复。
5. 通信优化技术，具体包括：
    - 采用拓扑感知的集合通信算法，避免网络拥塞问题，提高通信效率。
    - 根据卡数自适应设置 bucket size，提高带宽利用率。
    - 根据模型和集群环境，调优通信原语的触发时机，从而将计算和通信重叠。

基于上述的几个优化技术，我们在千卡 A800 显卡上达到了 7B 模型 182 TFLOPS 的吞吐，GPU 峰值算力利用率高达 58.3%。

最终的loss如下图：

<div align=center>
    <img src="zh-cn/img/ch2/4-12/p4.png" /> 
</div>

**5、民间评测**

issue:https://github.com/baichuan-inc/baichuan-7B/issues/51 对该模型进行了对比评测(https://baichuan-vicuna-eval.pleisto.app/ )，使用FastChat的评估集由GPT4进行打分。

考虑到baichuan-vicuna-7b主要是用ShareGPT数据集做的训练，中文数据占比相对较小，因此直接拿FastChat的英文评估集进行评测，并和同样基于ShareGPT数据集训练的LLaMAVicuna13B进行横向对比。

打分数据如下：

1) 写作任务：baichuan-vicuna-7b：8.5 Vllama-vicuna-13b：9.05
2) 角色扮演任务：baichuan-vicuna-7b：7.5VSllama-vicuna-13b：9.1
3) 常识知识：baichuan-vicuna-7b：8.9VSllama-vicuna-13b：8.85
4) 费米问题：baichuan-vicuna-7b：4.8VSllama-vicuna-13b：8
5) 反常识问题：baichuan-vicuna-7b：7.7VSllama-vicuna-13b：8.9
6) 编程：baichuan-vicuna-7b：4.36VSllama-vicuna-13b：4
7) 数学：baichuan-vicuna-7b：3.33VSllama-vicuna-13b：3.67
8) 一般性开放问答：baichuan-vicuna-7b：8.7VSllama-vicuna-13b：8.65
9) 专业知识：baichuan-vicuna-7b：8.5 VSllama-vicuna-13b：8.85

结论如下：

1) 在10个任务中，llama-vicuna-13b在7个任务上的表现优于baichuan-vicuna-7b（写作任务、角色扮演任务、费米问题、反常识问题、编程、数学和专业知识），而baichuan-vicuna-7b在3个任务上的表现优于llama-vicuna-13b（常识知识、一般性开放问答和编程任务）。
2) llama-vicuna-13b的表现更优，因为其在更多任务中的平均得分高于baichuan-vicuna-7b。然而，根据提供的数据，可以看出，两个模型在某些任务上的表现相当接近，例如常识知识、一般性开放问答和专业知识。llama-vicuna-13b在费米问题、反常识问题和角色扮演任务上的表现显著优于baichuan-vicuna-7b，而baichuan-vicuna-7b在编程任务上的表现略优于llama-vicuna-13b。
3) 考虑到baichuan-vicuna-7b（7B参数）和llama-vicuna-13b（13B参数）之间的参数量差异，需要重新评估它们的性能。一般来说，参数量较大的模型在性能上可能更好，但同时计算资源消耗也更高，因此在实际应用中需要权衡。
4) 由于在上述10个任务中，模型B在7个任务上的表现优于baichuan-vicuna-7b，而baichuan-vicuna-7b在3个任务上的表现优于llama-vicuna-13b。尽管llama-vicuna-13b在多数任务中表现较好，但在某些任务上，如常识知识、一般性开放问答和编程任务，两者的表现相差不大。这意味着在这些任务中，baichuan-vicuna-7b在性价比方面可能更具优势。
5) 对于不同的应用场景，可以根据以下建议选择合适的模型：

如果计算资源充足，且需要在各个任务上都获得较好的性能，可以选择参数量较大的模型；

如果计算资源有限，或者需要在特定任务（如常识知识、一般性开放问答和编程任务）上优化性价比，可以考虑选择参数量较小的baichuan-vicuna-7b；

对于费米问题、反常识问题和角色扮演任务等，llama-vicuna-13b具有明显优势，因此在这些任务上可以优先考虑llama-vicuna-13b；

**6、衍生微调实现**

https://github.com/jianzhnie/Efficient-Tuning-LLMs 项目实现了百川模型的低资源量化训练和部署， LoRA微调可在单块3090GPU上运行，同时支持QLoRA方法，最低8G显存。

### 11.2 AquilaChat-7B

AquilaChat-7B模型由智源研究院研发，AquilaChat-7B是在Aquila-7B模型的基础上，进行SFT微调后的支持中英双语的对话式语言模型。

运行Aquila-7B系列需要内存30G, 显存18G， 生成最大长度2048tokens。

<div align=center>
    <img src="zh-cn/img/ch2/4-12/p5.png" /> 
</div>

**1、训练数据**

暂未公开，只知道是中英双语。

**2、tokenizer**

Aquila模型所采用的tokenizer支持中英双语。在处理英文、中文以及代码数据时，采用了不同的分词器对一万个样本进行了抽取，对比结论如下：

<div align=center>
    <img src="zh-cn/img/ch2/4-12/p6.png" /> 
</div>

可以看到，Aquila模型词表较大，为10W级别， 采用bpe方式进行切分。

**3、模型结构**

借鉴了GPT-3, LLaMA的架构设计

地址：https://github.com/FlagAI-Open/FlagAI/tree/master/examples/Aquila/Aquila-chat

<!-- ------
------
## 12. MedicalGPT -->


<!-- ------
------
## 13. Multimodal-GPT -->
<!-- https://github.com/open-mmlab/Multimodal-GPT -->




<!-- ------
------
## 11. Falcon

------
------
## 12. CaMA

------
------
## 13. Guanaco -->


------
------
## 14. 其他大模型微调算法：BitFit, AdaLoRA, MAM Adapter, UniPELT

### 1.BitFit

<!-- https://mp.weixin.qq.com/s/2xHBE8c3hTClfAt93Z-4Rg -->

#### 1.1 背景

虽然对每个任务进行全量微调非常有效，但它也会为每个预训练任务生成一个独特的大型模型，这使得很难推断微调过程中发生了什么变化，也很难部署， 特别是随着任务数量的增加，很难维护。

理想状况下，我们希望有一种满足以下条件的高效微调方法：
+ 到达能够匹配全量微调的效果。
+ 仅更改一小部分模型参数。
+ 使数据可以通过流的方式到达，而不是同时到达，便于高效的硬件部署。
+ 改变的参数在不同下游任务中是一致的。

上述的问题取决于微调过程能多大程度引导新能力的学习以及暴露在预训练LM中学到的能力。

虽然，之前的高效微调方法Adapter-Tuning、Diff-Pruning也能够部分满足上述的需求。但是，作者提出了一种参数量更小的稀疏的微调方法BitFit，来满足上述的需求。

#### 1.2 技术原理

BitFit（论文：BitFit: Simple Parameter-efficient Fine-tuning or Transformer-based Masked Language-models）是一种稀疏的微调方法，它训练时只更新bias的参数或者部分bias参数。

对于Transformer模型而言，冻结大部分 transformer-encoder 参数，只更新bias参数跟特定任务的分类层参数。涉及到的bias参数有attention模块中计算query,key,value跟合并多个attention结果时涉及到的bias，MLP层中的bias，LayerNormalization层的bias参数。

在`Bert-Base/Bert-Large`这种模型里，bias参数仅占模型全部参数量的`0.08%～0.09%`。但是通过在Bert-Large模型上基于GLUE数据集进行了 BitFit、Adapter和Diff-Pruning的效果对比发现，BitFit在参数量远小于Adapter、Diff-Pruning的情况下，效果与Adapter、Diff-Pruning想当，甚至在某些任务上略优于Adapter、Diff-Pruning。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p1.png" /> 
</div>

同时，通过实验结果还可以看出，BitFit微调结果相对全量参数微调而言, 只更新极少量参数的情况下，在多个数据集上都达到了不错的效果，虽不及全量参数微调，但是远超固定全部模型参数的Frozen方式。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p2.png" /> 
</div>

同时，通过对比BitFit训练前后的参数，发现很多bias参数并没有太多变化（例如：跟计算key所涉及到的bias参数）。发现计算query和将特征维度从N放大到4N的FFN层（intermediate）的bias参数变化最为明显，只更新这两类bias参数也能达到不错的效果，反之，固定其中任何一者，模型的效果都有较大损失。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p3.png" /> 
</div>

### 2.AdaLoRA

<!-- https://mp.weixin.qq.com/s/N_N6RqKB9pjZ1tozfM5f5A -->
#### 2.1 背景

在NLP领域，对于下游任务进行大型预训练语言模型的微调已经成为一种重要的做法。一般而言，我们会采用对原有的预训练模型进行全量微调的方法来适配下游任务，但这种方法存在两个问题。

+ 训练阶段。对于预训练模型进行微调的时候，为了更新权重参数，需要大量的显存来存储参数的梯度和优化器信息，在当今预训练模型的参数变得越来越大的情况下，针对下游任务微调门槛变得越来越高。
+ 推理阶段。由于我们训练的时候是对于模型参数进行全量的更新，所以多个下游任务需要为每个任务维护一个大型模型的独立副本，这样就导致我们在实际应用的时候浪费了不必要的存储。

为了解决这些问题，研究者提出了两个主要研究方向，以减少微调参数的数量，同时保持甚至提高预训练语言模型的性能。

+ **方向一：添加小型网络模块：**将小型网络模块添加到PLMs中，保持基础模型保持不变的情况下仅针对每个任务微调这些模块，可以用于所有任务。这样，只需引入和更新少量任务特定的参数，就可以适配下游的任务，大大提高了预训练模型的实用性。如：Adapter tuning、Prefix tuning、Prompt Tuning等，这类方法虽然大大减少了内存消耗。但是这些方法存在一些问题，比如：Adapter tuning引入了推理延时；Prefix tuning或Prompt tuning直接优化Prefix和Prompt是非单调的，比较难收敛，并且消耗了输入的token。
+ **方向二：下游任务增量更新：**对预训练权重的增量更新进行建模，而无需修改模型架构，即`W=W0+△W`。比如：Diff pruning、LoRA等， 此类方法可以达到与完全微调几乎相当的性能，但是也存在一些问题，比如：Diff pruning需要底层实现来加速非结构化稀疏矩阵的计算，不能直接使用现有的框架，训练过程中需要存储完整的∆W矩阵，相比于全量微调并没有降低计算成本。 LoRA则需要预先指定每个增量矩阵的秩 r 相同，忽略了在微调预训练模型时，权重矩阵的重要性在不同模块和层之间存在显著差异，并且只训练了Attention，没有训练FFN，事实上FFN更重要。

基于以上问题进行总结：
+ 第一，我们不能预先指定矩阵的秩，需要动态更新增量矩阵的R，因为权重矩阵的重要性在不同模块和层之间存在显著差异。
+ 第二，需要找到更加重要的矩阵，分配更多的参数，裁剪不重要的矩阵。找到重要的矩阵，可以提升模型效果；而裁剪不重要的矩阵，可以降低参数计算量，降低模型效果差的风险。

为了弥补这一差距，作者提出了AdaLoRA，它根据权重矩阵的重要性得分，在权重矩阵之间自适应地分配参数预算。

#### 2.2 技术原理

AdaLoRA（论文：ADAPTIVE BUDGET ALLOCATION FOR PARAMETEREFFICIENT FINE-TUNING），是对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵。具体做法如下：
+ **调整增量矩分配**。AdaLoRA将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。
+ **以奇异值分解的形式对增量更新进行参数化，并根据重要性指标裁剪掉不重要的奇异值，同时保留奇异向量**。由于对一个大矩阵进行精确SVD分解的计算消耗非常大，这种方法通过减少它们的参数预算来加速计算，同时，保留未来恢复的可能性并稳定训练。
$$W=W^{(0)}+\detla=W^{(0)}+P\LambdaQ$$
+ **在训练损失中添加了额外的惩罚项**，以规范奇异矩阵P和Q的正交性，从而避免SVD的大量计算并稳定训练。

通过实验证明，AdaLoRA 实现了在所有预算、所有数据集上与现有方法相比，性能更好或相当的水平。 例如，当参数预算为 0.3M 时，AdaLoRA 在RTE数据集上，比表现最佳的基线（Baseline）高 1.8%。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p4.png" /> 
</div>

### 3.MAM Adapter

<!-- https://mp.weixin.qq.com/s/M2nds_FJBXooi08qDU-4yA -->
#### 3.1 背景

近年来提出了多种参数高效的迁移学习方法，这些方法仅微调少量（额外）参数即可获得强大的性能。虽然有效，但人们对为什么有效的关键要素以及各种高效微调方法之间的联系知之甚少。

下图展示了不同的微调方法，在Xsum数据集上做英文文本摘要任务的效果（ROUGE-2是该任务的评价指标（越大越好））以及其他高效微调方法参数量相对于全参数微调参数量的百分比。图中的左上角的位置是理想化的方法。从图中发现，Adapter，Prefix Tuning和LoRA都是性能比较好的方法。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p5.png" /> 
</div>

为什么看起来Adapter、Prefix Tuning、LoRA（在结构上和公式上）都不太一样，尤其是Prefix Tuning，但是这三种方法有近似的效果？

基于此，作者分解了当下最先进的参数高效迁移学习方法（Adapter、Prefix Tuning和LoRA）的设计，并提出了一种新方法MAM Adapter，一个在它们之间建立联系的统一框架。具体来说，将它们重新构建为对预训练模型中特定隐藏状态的修改，并定义一组设计维度，不同的方法沿着这些维度变化。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p6.png" /> 
</div>

首先，作者通过对Prefix Tuning变换，发现Prefix Tuning和Adapters的公式高度相似。

然后，分析不同微调方法的内部结构和结构插入形式的相似之处。下图展示了高效微调方法Adapter、Prefix Tuning、LoRA以及新变体（通过更换一些元素，设计了前人的工作里没有的变体） Parallel Adapter、 Scaled PA的结构。  

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p7.png" /> 
</div>

下表展示了高效微调方法Adapter、Prefix Tuning、LoRA以及新变体在新增可训练参数结构形式（functional form）、结构插入形式（Insertion form）、新增结构在PLM修改的具体位置（modified representation）、新增结构与PLM的组合函数（composition function）。其中，新增可训练参数结构形式为需要学习的部分（注：Prefix Tuning为经过转换后的格式）；插入形式有串联或并联；模型修改的具体位置有Attention、FFN层。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p8.png" /> 
</div>

#### 3.2 技术原理

MAM Adapter（论文：TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING），一个在Adapter、Prefix Tuning和LoRA之间建立联系的统一方法。

具体实现：

作者对Adapter的放置和软提示（soft prompt）进行了详细的调查。得出如下结论：

+ 并行放置的Adapter优于顺序放置的Adapter，并且与 FFN 并行放置的Adapter优于多头注意力（MHA）并行放置的Adapter（模型修改的位置如下图中所示，蓝色表示修改Attention、红色表示修改FFN）。
+ 软提示可以通过仅更改 0.1% 的参数来有效地修改注意力。
<div align=center>
    <img src="zh-cn/img/ch2/4-11/p9.png" /> 
</div>

然后，提出了“mix-and-match”（MAM）。 因此，**最终模型 MAM Adapter 是用 FFN 层的并行Adapter和软提示的组合**。

通过最终的实验结果，可以看到 MAM Adapter 在仅用了6.7%参数量（相比全量微调）的情况下，在Xsum和MT这两个任务上达到了和全量微调相近的效果，并且该方法大大优于 BitFit 和 Prompt Tuning，并始终优于 LoRA、Adapter 和 Prefix Tuning。  

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p10.png" /> 
</div>

### 4.UniPELT

#### 4.1 背景

近年来，涌现出了许多针对语言模型的参数高效微调（PELT）方法，在模型训练参数极大的减少的情况下，模型效果与全量微调相当。但是不同的PELT方法在同一个任务上表现差异可能都非常大，这让针对特定任务选择合适的方法非常繁琐。

基于此，作者提出了UniPELT方法，将不同的PELT方法作为子模块，并通过门控机制学习激活最适合当前数据或任务的方法。

#### 4.2 技术原理

UniPELT（论文： UNIPELT: A Unified Framework for Parameter-Efficient Language Model Tuning）是 LoRA、Prefix Tuning和Adapter的门控组合。

更具体地说，LoRA 重新参数化用于 WQ 和 WV 注意力矩阵，Prefix Tuning应用于每一Transformer层的key和value，并在Transformer块的feed-forward子层之后添加Adapter。 对于每个模块，门控被实现为线性层，通过GP参数控制Prefix-tuning方法的开关，GL控制LoRA方法的开关，GA控制Adapter方法的开关。可训练参数包括 LoRA 矩阵 WA（Down）和WB（Up），提示调优参数Pk和Pv、Adapter参数和门函数权重。即图中蓝颜色的参数为可学习的参数。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p11.png" /> 
</div>

UniPELT 仅用 100 个示例就在低数据场景中展示了相对于单个 LoRA、Adapter 和 Prefix Tuning 方法的显著改进。在更高数据的场景中，UniPELT 的性能与这些方法相当或更好。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p12.png" /> 
</div>

实验还对不同 PELT 方法训练时间和推理时间进行了分析。

+ 从训练速度来看，UniPELT比之前微调的方法多一些，但是还在能接受的范围，
+ 从推理时间来看，BitFit方法增加的最少，UniPELT方法时间增加了27%。
+ 从训练参数量来看，LoRA，BitFit，Prefix-tuning都比较小，UniPELT参数量相对会多一些。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p13.png" /> 
</div>

总之，本方法始终优于常规的全量微调以及它在不同设置下包含的子模块，通常超过在每个任务中单独使用每个子模块的最佳性能的上限；并且，通过研究结果表明，多种 PELT 方法的混合涉及到PLM 的不同部分可能对模型有效性和鲁棒性都有好处。


### 5.大模型高效微调技术最佳实践、总结

<!-- https://mp.weixin.qq.com/s/P_AmTa4s8dOyc_0fZBgNPA -->

之前对一些常见的高效微调技术进行了背景介绍及技术原理剖析，下面对每一种高效微调技术的特点进行简要的总结。

**BitFit**

对微调机制的一种积极探索，也很简单，通过仅调整bias效果就能有不错的效果，但没有具体阐述原理，就是通过猜测加实验得到的结果。同时，作者提出一个观点：微调的过程不是让模型适应另外的数据分布，而是让模型更好的应用出本身的表征能力。

特点：

+ 训练参数量极小（约0.1%）。
+ 在大部分任务上效果会差于LoRA、Adapter等方法。

**Prefix Tuning**

在每一个Transformer层都带上一些virtual token作为前缀，以适应不同的任务。

特点：

+ 前缀Token会占用序列长度，有一定的额外计算开销。
+ Prefix Tuning的线性插值是比较复杂的。

**Prompt Tuning**

该方法可以看着是Prefix Tuning的简化版本，针对不同的任务，仅在输入层引入virtual token形式的软提示（soft prompt）。

特点：

+ 相对于Prefix Tuning，参与训练的参数量和改变的参数量更小，更节省显存。
+ 对一些简单的NLU 任务还不错，但对硬序列标记任务（即序列标注）表现欠佳。


**P-Tuning**

将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。相比Prefix Tuning，仅在输入层加入的可微的virtual token；另外，virtual token的位置也不一定是前缀，插入的位置是可选的。

特点：

+ 引入一个prompt encoder（由一个双向的LSTM+两层MLP组成）来建模virtual token的相互依赖会收敛更快，效果更好。

**P-Tuning v2**

该方法在每一个Transformer层都加入了prompt token作为输入，引入多任务学习，针对不同任务采用不同的提示长度。并且回归传统的分类标签范式，而不是映射器。

特点：

+ 解决了Prompt Tuning无法在小模型上有效提升的问题。
+ 移除了对模型效果改进较小的重参数化的编码器（如：Prefix Tuning中的MLP、P-Tuning中的LSTM）。
+ 对于一些复杂的硬序列标记任务（即序列标注）取得了不错的效果。

**Adapter Tuning**

该方法设计了Adapter结构，并将其嵌入Transformer的结构里面，针对每一个Transformer层，增加了两个Adapter结构，在训练时，固定住原来预训练模型的参数不变，只对新增的Adapter结构和Layer Norm 层进行微调。

特点：

+ 通过在Transformer层中嵌入Adapter结构，在推理时会额外增加推理时长。

**AdapterFusion**

一种融合多任务信息的Adapter的变体，在 Adapter 的基础上进行优化，通过将学习过程分为两阶段来提升下游任务表现。

**AdapterDrop**


该方法在不影响任务性能的情况下，对Adapter动态高效的移除，尽可能的减少模型的参数量，提高模型在反向传播（训练）和正向传播（推理）时的效率。

特点：

+ 通过从较低的 Transformer 层删除可变数量的Adaper来提升推理速度。 当对多个任务执行推理时，动态地减少了运行时的计算开销，并在很大程度上保持了任务性能。

**LoRA**

该方法通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

特点：

+ 将BA加到W上可以消除推理延迟。
+ 可以通过可插拔的形式切换到不同的任务。
+ 设计的比较好，简单且效果好。

**AdaLoRA**

对LoRA的一种改进，它根据重要性评分动态分配参数预算给权重矩阵，将关键的增量矩阵分配高秩以捕捉更精细和任务特定的信息，而将较不重要的矩阵的秩降低，以防止过拟合并节省计算预算。

**QLoRA**

使用一种新颖的高精度技术将预训练模型量化为 4 bit，然后添加一小组可学习的低秩适配器权重，这些权重通过量化权重的反向传播梯度进行微调。

特点：

+ 使用 QLoRA 微调模型，可以显著降低对于显存的要求。同时，模型训练的速度会慢于LoRA。

**MAM Adapter**

一种在 Adapter、Prefix Tuning 和 LoRA 之间建立联系的统一方法。最终的模型 MAM Adapter 是用于 FFN 的并行 Adapter 和 软提示的组合。

特点：

+ 整体上来说，最终的模型MAM Adapter效果会优于单个高效微调方法。

**UniPELT**

一种将不同的PELT方法LoRA、Prefix Tuning和Adapter作为子模块，并通过门控机制学习激活最适合当前数据或任务的方法。

特点：

+ 相对于LoRA，BitFit，Prefix-tuning，训练的参数量更大；同时，推理更耗时；并且，输入会占用额外的序列长度。
+ 多种 PELT 方法的混合涉及PLM 的不同部分对模型有效性和鲁棒性都有好处。

!> 多种不同的高效微调方法对比

总的来说，像P-Tuning v2、LoRA等都是综合评估很不错的高效微调技术。如果显存资源有限可以考虑QLoRA；如果只是解决一些简单任务场景，可以考虑P-Tuning、Prompt Tuning也行。

下表从参数高效方法类型、是否存储高效和内存高效、以及在减少反向传播成本和推理开销的计算高效五个维度比较了参数高效微调方法。

<div align=center>
    <img src="zh-cn/img/ch2/4-11/p14.png" /> 
</div>

下表展示了各种参数高效方法的参与训练的参数量、最终模型与原始模型的改变参数（delta值）以及论文中参与评估的模型的范围（<1B、<20B、>20B）。


<div align=center>
    <img src="zh-cn/img/ch2/4-11/p15.png" /> 
</div>

从表中可以看到，Prompt Tuning、Prefix Tuning、LoRA等少部分微调技术针对不同参数规模的模型进行过评估，同时，这几种方式也是目前应用比较多的高效微调方法。

!> 当前高效微调技术存在的一些问题

当前的高效微调技术很难在类似方法之间进行直接比较并评估它们的真实性能，主要的原因如下所示：

+ **参数计算口径不一致**：参数计算可以分为三类：可训练参数的数量、微调模型与原始模型相比改变的参数的数量、微调模型和原始模型之间差异的等级。例如，DiffPruning更新0.5%的参数，但是实际参与训练的参数量是200%。这为比较带来了困难。尽管可训练的参数量是最可靠的存储高效指标，但是也不完美。 Ladder-side Tuning使用一个单独的小网络，参数量高于LoRA或BitFit，但是因为反向传播不经过主网络，其消耗的内存反而更小。
+ **缺乏模型大小的考虑**：已有工作表明，大模型在微调中需要更新的参数量更小（无论是以百分比相对而论还是以绝对数量而论），因此（基）模型大小在比较不同PEFT方法时也要考虑到。
+ **缺乏测量基准和评价标准**：不同方法所使用的的模型/数据集组合都不一样，评价指标也不一样，难以得到有意义的结论。
+ **代码实现可读性差**：很多开源代码都是简单拷贝Transformer代码库，然后进行小修小补。这些拷贝也不使用git fork，难以找出改了哪里。即便是能找到，可复用性也比较差（通常指定某个Transformer版本，没有说明如何脱离已有代码库复用这些方法）。

!> 高效微调技术最佳实践

针对以上存在的问题，研究高效微调技术时，建议按照最佳实践进行实施：

+ 明确指出参数数量类型。
+ 使用不同大小的模型进行评估。
+ 和类似方法进行比较。
+ 标准化PEFT测量基准。
+ 重视代码清晰度，以最小化进行实现。

**总结**

本文针对之前介绍的几种参数高效微调方法进行了简单的概述，主要有如下几类：

+ 增加额外参数，如：Prefix Tuning、Prompt Tuning、Adapter Tuning及其变体。
+ 选取一部分参数更新，如：BitFit。
+ 引入重参数化，如：LoRA、AdaLoRA、QLoRA。
+ 混合高效微调，如：MAM Adapter、UniPELT。

并比较了不同的高效微调方法之间的差异；同时，还指出当前大多数高效微调方法存在的一些问题并给出了最佳实践。


## 15. LLM的预训练策略和模型并行策略

TODO

## 16. langchain

TODO