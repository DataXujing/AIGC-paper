<!-- chatGLM,MINIGPT-4, LLaMa, Firefly,BiLLa,stable-vicuna-13B,alpaca-13b,opt,本草，langChhian(guidance),Alpaca_LoRa,Falcon-40B,CaMa,guanaco-->

<!-- chatGLM,MINIGPT-4, LLaMa, alpaca，本草，langChian-->
<!-- 下载GPT-4, InstrcutGPT,还有这些大模型的paper -->


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





