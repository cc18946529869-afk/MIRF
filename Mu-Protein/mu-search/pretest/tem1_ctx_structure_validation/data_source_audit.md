# 头孢噻肟单突变数据来源核查

核查日期：2026-11-16。本文件只记录已经读到的文字和检索结果。没有逐突变计数，就不计算富集。

## 论文

Gaszek 等人，*Saturation mutagenesis map of generalist versus specialist adaptations of β-lactamase to novel antibiotics*。bioRxiv，2026-02-28 发布的版本。DOI：https://doi.org/10.1101/2025.10.14.682469。PMC：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13160169/。预印本，这次没有把它当成已经同行评审的论文。

正文写明使用 Stiffler 的 TEM-1 单突变饱和文库，三个生物学重复。筛选浓度包括头孢噻肟 0.25 μg/mL 和头孢曲松 0.25 μg/mL。适应度写成

\[
f_{ir}=\log_{10}\left(\frac{\mathrm{freq}_{i,r,\mathrm{treated}}}{\mathrm{freq}_{i,r,\mathrm{untreated}}}\right)
\]

计数先加 0.1 的伪计数，再在每个重复内归一化成频率。这是一次筛选的频率富集，不是 MIC 剂量反应。正文另有一组单独测定的 MIC，对象是少数构建菌株，不是全位点单突变矩阵。

## 药物缩写

正文把头孢曲松写成 ceftriaxone (CFX)，头孢噻肟写成 Cefotaxime，浓度都是 0.25 μg/mL。图注的热图列了 AMP、CAZ、CFX、CTX、FEP、AZT 六列，所以 CFX 和 CTX 在这张图里是分开的两列。同一段图注又写了 “Ceftriaxone (CTX)”。这句和正文、也和热图的六列不一致。本次按正文和热图列名处理：CTX 是头孢噻肟，CFX 是头孢曲松。μSearch checkpoint 文件名里的 CFX 仍是头孢噻肟模型，和这篇论文的 CFX 不是同一种药。

## 公开数据

正文说每个药物有 `mut_fit` 计数矩阵，行是突变，列是三个生物学重复。PMC 全文里能读到的 Supplement 1 是图注，不是这张计数表。bioRxiv v2 的 supplementary-material 页面没有给出 csv、xlsx 或 zip。NCBI SRA 用 “Gaszek AND TEM-1 AND beta-lactamase” 返回 0 条；BioProject 用 “Gaszek TEM-1 saturation” 没有命中 Gaszek。GEO 检索没有把 Gaszek 识别成记录。GitHub 仓库 `msadikyildiz/Gaszek_Yildiz_Meng_2025` 的说明指向 2025 年的另一项研究，不作为这份 2026 年饱和突变计数表。

这次没有找到作者公开的逐突变计数、测序 accession，也没有在已读全文里找到单独的数据可用性段落。图注不能用来重建全位点矩阵。因此不生成 `ctx_single_mutant_labels.csv`。
