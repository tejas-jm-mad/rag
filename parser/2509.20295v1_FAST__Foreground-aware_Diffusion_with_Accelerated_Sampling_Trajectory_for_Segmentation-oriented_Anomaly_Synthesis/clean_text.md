

# [PAGE 1]
FAST: Foreground-aware Diffusion with Accelerated
Sampling Trajectory for Segmentation-oriented
Anomaly Synthesis
Xichen Xu1
Yanshu Wang1
Jinbao Wang2
Xiaoning Lei3
Guoyang Xie3∗
Guannan Jiang3∗
Zhichao Lu4
1Global Institute of Future Technology, Shanghai Jiao Tong University, Shanghai, China
2School of Artificial Intelligence, Shenzhen University, Shenzhen, China
3Department of Intelligent Manufacturing, CATL, Ningde, China
4Department of Computer Science, City University of Hong Kong, Hong Kong, China
neptune_2333@sjtu.edu.cn
isaac_wang@sjtu.edu.cn
wangjb@szu.edu.cn
leixn01, jianggn@catl.com
guoyang.xie@ieee.org
zhichao.lu@cityu.edu.hk †
Abstract
Industrial anomaly segmentation relies heavily on pixel-level annotations, yet
real-world anomalies are often scarce, diverse, and costly to label. Segmentation-
oriented industrial anomaly synthesis (SIAS) has emerged as a promising alter-
native; however, existing methods struggle to balance sampling efficiency and
generation quality. Moreover, most approaches treat all spatial regions uniformly,
overlooking the distinct statistical differences between anomaly and background ar-
eas. This uniform treatment hinders the synthesis of controllable, structure-specific
anomalies tailored for segmentation tasks. In this paper, we propose FAST, a
foreground-aware diffusion framework featuring two novel modules: the Anomaly-
Informed Accelerated Sampling (AIAS) and the Foreground-Aware Reconstruction
Module (FARM). AIAS is a training-free sampling algorithm specifically designed
for segmentation-oriented industrial anomaly synthesis, which accelerates the
reverse process through coarse-to-fine aggregation and enables the synthesis of
state-of-the-art segmentation-oriented anomalies in as few as 10 steps. Mean-
while, FARM adaptively adjusts the anomaly-aware noise within the masked
foreground regions at each sampling step, preserving localized anomaly signals
throughout the denoising trajectory. Extensive experiments on multiple industrial
benchmarks demonstrate that FAST consistently outperforms existing anomaly
synthesis methods in downstream segmentation tasks. We release the code in
https://anonymous.4open.science/r/NeurIPS-938.
1
Introduction
Motivation. Industrial anomaly segmentation plays a vital role in modern manufacturing, aiming to
localize abnormal regions at the pixel level. Unlike traditional anomaly detection, which typically
performs binary classification at the image or region level, anomaly segmentation requires more
fine-grained and precise localization of abnormal patterns. However, real-world anomalies are
inherently scarce, diverse, and non-repeatable, making it difficult to collect data that fully captures
the range of possible abnormal types. Moreover, acquiring high-quality pixel-level annotations is
labor-intensive and costly, especially in industrial scenarios. To address these limitations, recent
studies have increasingly explored the use of synthetic anomalies to expand the training data space
and improve downstream performance.
Limitations. Despite recent advances, current anomaly synthesis methods face three fundamental
limitations that hinder their effectiveness for segmentation tasks [31]. (i) Lack of controllability.
∗Corresponding author.
†This work was supported by the National Natural Science Foundation of China (Grant No. 62206122) and
the Tencent “Rhinoceros Birds” — Scientific Research Foundation for Young Teachers of Shenzhen University.
Preprint. Under review.
arXiv:2509.20295v1  [cs.CV]  24 Sep 2025

[EQ eq_p0_000] -> see eq_p0_000.tex


# [PAGE 2]
Most existing methods provide limited control over the structure, location, or extent of synthesized
anomalies. This limitation is particularly evident in GAN-based approaches [20, 37, 5]. These
methods typically adopt a one-shot generation paradigm, offering little flexibility in specifying where
and how anomalies should appear. (ii) Neglect of segmentation-relevant properties. Training-free
methods such as patch replacement or texture corruption [14, 36] may produce visible anomalies,
but the synthesized patterns often lack the structural consistency and complexity of real-world
industrial anomalies, which are critical for improving segmentation performance. (iii) Uniform
treatment of spatial regions and inefficiency. Although recent diffusion-based methods [9, 12, 24]
have mitigated the above issues, they still treat all spatial regions uniformly during both forward
and reverse processes, without explicitly modeling the distinct statistical properties of anomaly
regions [38, 34]. This absence of region-aware modeling prevents the model from preserving
abnormal regions throughout the synthesis trajectory. Moreover, these models typically require
hundreds to thousands of denoising steps [10, 25], resulting in a significant computational cost. While
recent training-free methods [15] aim to accelerate sampling, they fail to incorporate anomaly-aware
cues, making them less effective for segmentation-oriented industrial anomaly synthesis (SIAS).
These limitations motivate the need for SIAS models that support controllable anomaly synthesis,
explicit modeling of anomaly regions, and efficient, task-aligned sampling strategies.
FAST. To address these issues, we propose FAST, a novel foreground-aware diffusion framework
with two complementary modules: Anomaly-Informed Accelerated Sampling (AIAS) and the
Foreground-Aware Reconstruction Module (FARM). (i) AIAS is a training-free sampling strategy
that reduces the number of denoising steps by up to 99% (from 1000 to as few as 10), resulting in over
100× speedup for SIAS tasks. Despite this drastic acceleration, FAST achieves an average mIoU of
76.72% and accuracy of 83.97% on MVTec-AD, outperforming all prior state-of-the-art methods. (ii)
FARM explicitly models abnormal regions by reconstructing pseudo-clean anomalies and generating
anomaly-aware noise at each step in both the forward and reverse processes. Incorporating FARM
boosts performance from 65.33% to 76.72% in mIoU (↑11.39), and from 71.24% to 83.97% in
accuracy (↑12.73), demonstrating its critical role in enhancing anomaly salience. Detailed results
are provided in Sec. 4.3. Together, AIAS and FARM enable FAST to generate controllable and
segmentation-aligned anomalies that significantly improve downstream performance.
Contributions. In summary, our contributions are three-fold: (1) To mitigate the inefficiency and
semantic misalignment of existing diffusion sampling, we introduce a training-free Anomaly-Informed
Accelerated Sampling (AIAS) strategy that aggregates multiple denoising steps into a small number of
coarse-to-fine analytical updates. (2) To address the lack of persistent anomaly-region representation,
we propose a Foreground-Aware Reconstruction Module (FARM) that reconstructs pseudo-clean
anomalies and reintegrates anomaly-aware noise at each step. (3) To support segmentation-oriented
industrial anomaly synthesis, we design FAST, a controllable and efficient model. Extensive ex-
periments on MVTec-AD and BTAD datasets demonstrate that it significantly outperforms existing
methods in downstream segmentation tasks.
2
Related work
Industrial Anomaly Synthesis. Industrial anomaly synthesis aims to mitigate the scarcity of labeled
abnormal samples in real-world inspection scenarios. Existing methods can be categorized into hand-
crafted and DL-based approaches. Hand-crafted methods typically apply training-free manipulations
to normal images, such as patch pasting [21, 23] or external texture blending [36, 32, 39] from
sources like DTD [3], but they suffer from distributional deviation and limited realism. Deep learning-
based methods alleviate these limitations by learning from real anomaly patterns. GAN-based
methods [6, 28] can synthesize visually realistic anomalies but lack fine-grained controllability
over anomaly shape and location. Diffusion-based methods [6, 13, 33] offer stronger generative
capacity via large-scale pretrained models, yet treat all regions uniformly and lack explicit control
over anomaly localization, which is essential for segmentation. To this end, we propose FAST, which
integrates foreground-aware reconstruction and efficient, segmentation-oriented anomaly synthesis
into a unified diffusion framework.
Acceleration of Discrete-Time Diffusion Models. Diffusion models can be categorized into
continuous-time and discrete-time frameworks.
Continuous formulations [16, 17, 41] adopt
SDE/ODE-based parameterizations and leverage high-order solvers for efficient sampling. In contrast,
standard DDPMs [10] model a discrete-time Markov chain with fixed variance schedules and require
thousands of iterative denoising steps. While continuous-time solvers achieve notable speedups, they
rely on continuously parameterized noise or score functions, which requires reformulating training
2


# [PAGE 3]
objectives or interface in discrete-time models. Therefore, various acceleration techniques have been
developed specifically for discrete-time diffusion. Some methods modify the generative process
to reduce steps: DDGAN [29] integrates GAN-based decoding, TLDM [40] and ES-DDPM [18]
truncate the forward process, and Blurring Diffusion Models [11] operate in the frequency domain.
However, these methods require retraining and show limited generalization. In contrast, training-free
approaches such as DDIM [25], PLMS [15], and GGDM [27] accelerate sampling without model
modification. Yet, they treat all spatial regions uniformly and lack task-specific guidance essential for
SIAS. Recent work like CUT [26] introduces external prompts for localized control for anomalies,
but at the cost of multiple iterations per sampling step. In comparison, FAST proposes a novel
training-free strategy that aggregates multiple denoising steps into coarse-to-fine segments while
injecting mask-aware structural guidance, enabling efficient SIAS.
3
Methods
FAST for Anomaly Segmentation. The proposed FAST framework is built upon the Latent Diffusion
Model (LDM) [22] of T steps. For notational simplicity, we denote the encoded latent of the original
image as x0, and its predicted reconstruction from the network as ˆx0. We define xts as the noisy
latent at timestep ts, and ˆxts as the FARM-adjusted, anomaly-aware latent at the same step. Let
M ∈{0, 1}H×W denote the binary anomaly mask, and [ts, te] represent a coarse-to-fine segment in
AIAS, where te < ts. Fig. 1 illustrates a single forward-reverse process at step ts. In the forward
phase, noise is added up to timestep ts, yielding a noisy latent xts. FARM (Fϕ in Algorithm 1) then
predicts a pseudo-clean anomaly latent ˆxan
0 , and adds noise to it up to timestep ts to obtain an anomaly-
aware latent ˆxts, which aims to match the observed xts in masked regions during training. In the
corresponding reverse process, we divide the full denoising process into S segments, each spanning
[ts, te]. Within each segment, AIAS approximates the posterior transition using: q(xte | xts, ˆx0).
This formulation aggregates multiple DDPM steps into a single numerical update. FARM is also
applied to refine xte, ensuring the preservation of anomaly cues throughout the reverse process. More
details can be seen in Algorithms 1 and 2.In addition, for the textual conditioning component of
LDM, we follow the configuration of Anomaly Diffusion [13]; more implementation details can be
found there.
time steps
Sinusoidal embedding
Linear Projection
MLP+Sigmoid
×
＋
Resize
Forward diffusion
Forward Process
Reverse  Process
Foreground-aware 
Reconstruction Module
AIAS
Update 
𝐭𝐭𝐬𝐬= 𝐭𝐭𝐞𝐞
LDM
Foregrond-aware process 
in both directions
Forward diffsion
Reverse sample
New
Figure 1: Illustration of a single forward–reverse process in FAST. AIAS accelerates sampling by
aggregating multiple denoising steps into a small number of coarse-to-fine segments, achieving up to
100× speedup while preserving semantic alignment under anomaly mask guidance. FARM extracts
anomaly-only content from the noisy latent xt at each timestep t and transforms it into anomaly-aware
noise by re-applying forward diffusion.
Algorithm 1 FAST Training
1: repeat
2:
x0 ∼q(x0), M, and weights λ1, λ2
3:
ts ∼Uniform({1, . . . , T}), ϵ ∼N(0, I)
4:
xts = √¯αtsx0 + √1 −¯αtsϵ
5:
ˆxts = √¯αtsFϕ(xts, M) + √1 −¯αtsϵ
6:
Take gradient descent step on:
∇θ ∥ϵ −ϵθ(ˆxts, ts)∥2
+∇ϕ ∥(M ⊙x0 −Fϕ(xts, ts, M))∥2
7: until converged
Algorithm 2 FAST Sampling
(Details are shown in Supplementary Material A.4)
1: Initialize xT ∼N(0, I)
2: for each segment [ts, te] from T →0 do
3:
ˆϵ = ϵθ(xts, ts)
4:
ˆx0 =
1
√¯αts (xts −√1 −¯αts · ˆϵ)
5:
AIAS:
xte = Fϕ(q(xte | xts, ˆx0), te, M))
6: end for
7: return x0
3
[FIGURE img_p2_000]
[FIGURE img_p2_001]
[FIGURE img_p2_002]
[FIGURE img_p2_003]
[FIGURE img_p2_004]
[FIGURE img_p2_005]
[FIGURE img_p2_006]
[FIGURE img_p2_007]
[FIGURE img_p2_008]
[FIGURE img_p2_009]
[FIGURE img_p2_010]
[FIGURE img_p2_011]
[FIGURE img_p2_012]
[FIGURE img_p2_013]
[FIGURE img_p2_014]
[FIGURE img_p2_015]
[FIGURE img_p2_016]
[FIGURE img_p2_017]
[FIGURE img_p2_018]
[FIGURE img_p2_019]
[FIGURE img_p2_020]
[FIGURE img_p2_021]
[FIGURE img_p2_022]

[EQ eq_p2_000] -> see eq_p2_000.tex

[EQ eq_p2_001] -> see eq_p2_001.tex

[EQ eq_p2_002] -> see eq_p2_002.tex

[EQ eq_p2_003] -> see eq_p2_003.tex

[EQ eq_p2_004] -> see eq_p2_004.tex

[EQ eq_p2_005] -> see eq_p2_005.tex

[EQ eq_p2_006] -> see eq_p2_006.tex


# [PAGE 4]
3.1
Anomaly-Informed Accelerated Sampling
Standard DDPM allows us to directly compute the marginal distribution of xt given a clean sample
x0 and additive noise ϵ. Therefore, the one-step posterior distribution of xt−1 can be expressed as:
q(xt−1 | xt, x0) = N(Atx0 + Btxt, σ2
t I),
(1)
where the coefficients are derived from the variance schedule as follows:
At =
√¯αt−1βt
1 −¯αt
,
Bt =
√αt(1 −¯αt−1)
1 −¯αt
,
σ2
t = 1 −¯αt−1
1 −¯αt
βt,
and αt = 1 −βt, ¯αt = Qt
s=1 αs. All are closed-form coefficients derived from a predefined
noise schedule. In practice, the true sample x0 is not accessible during inference, and is typically
replaced by a model prediction ˆx0 obtained via denoising estimation. Equation 1 thus serves as
the foundation for approximate posterior sampling, provided that ˆx0 is a sufficiently accurate
estimate of the ground truth x0.
Theoretically, if we assume ˆx0 = x0 holds exactly (i.e., the prediction perfectly matches the ground-
truth image), then the entire reverse process becomes fully deterministic and analytically tractable,
with the only source of stochasticity being the injected noise at each step. In this idealized setting,
the reverse sampling trajectory is fully governed by closed-form probabilistic transitions. This forms
the basis for Lemma. 1 (For brevity, the full proof is provided in the Supplementary Material A.1).
Lemma 1 (Linear–Gaussian closure). Let {xk}K
k=0 ⊂Rd satisfy the recursion
xk−1 = Ck xk + dk + εk,
εk ∼N(0, Σk),
εk ⊥{xk, εk+1, . . .},
(2)
where Ck ∈Rd×d, dk ∈Rd, and Σk ∈Rd×d are deterministic. Then, for every integer m with
1≤m≤k, xk−m is again an affine–Gaussian function of xk:
xk−m =
m−1
Y
i=0
Ck−i

|
{z
}
=:C(m)
k
xk +
m−1
X
i=0
 iY
j=1
Ck−j

dk−i
|
{z
}
=:d(m)
k
+ ε(m)
k
,
(3)
where
ε(m)
k
∼N
 0, Σ(m)
k

,
Σ(m)
k
=
m−1
X
i=0
 iY
j=1
Ck−j

Σk−i
 iY
j=1
Ck−j
⊤
.
(4)
While the ideal condition ˆx0 = x0 rarely holds in practice, the following properties justify the use of
ˆx0 in the multi-step formulation:
(i) The training objective of standard DDPM is explicitly designed to minimize the discrep-
ancy between the predicted noise and the true noise. Consequently, the denoising model
ϵθ(xt, t) implicitly learns to reconstruct a close approximation of x0 through the reverse
reparameterization formula.
(ii) Both empirical observations and theoretical analyses suggest that ˆx0 varies slowly with
respect to t at large diffusion steps. That is, for a segment [ts, te] with ts > te and moderate
length (e.g., ts −te ≪T), we haveˆx0(xts, ts) ≈ˆx0(xt, t)
for all t ∈[te, ts], due to the
temporal smoothness of model predictions in the noise-dominated regime.
Therefore, it is reasonable to treat ˆx0 as fixed within a short temporal window. Under this assump-
tion, multiple single-step reverse transitions can be analytically composed into a single multi-step
affine–Gaussian kernel. This approximation and Lemma. 1 form the basis for Theorem 2, which
characterizes the closed-form reverse process from ts to te (For brevity, the full proof is provided in
the Supplementary Material A.2 ).
Lemma 2 (Closed-form reverse from ts →te). Fix indices 0 ≤te < ts ≤T, and let the single-
step coefficients (At, Bt, σ2
t ) be defined as in Eq. 13. Then the aggregated reverse kernel over
ts →· · · →te is affine–Gaussian:
xte = Πts
te xts + Σts
te ˆx0 + εte,
(5)
where
Πts
te :=
ts
Y
i=te+1
Bi,
Σts
te :=
ts
X
i=te+1
Ai
ts
Y
j=i+1
Bj,
εte ∼N


0,
ts
X
i=te+1


ts
Y
j=i+1
Bj


2
σ2
i I


.
4

[EQ eq_p3_000] -> see eq_p3_000.tex

[EQ eq_p3_001] -> see eq_p3_001.tex

[EQ eq_p3_002] -> see eq_p3_002.tex

[EQ eq_p3_003] -> see eq_p3_003.tex

[EQ eq_p3_004] -> see eq_p3_004.tex

[EQ eq_p3_005] -> see eq_p3_005.tex

[EQ eq_p3_006] -> see eq_p3_006.tex

[EQ eq_p3_007] -> see eq_p3_007.tex

[EQ eq_p3_008] -> see eq_p3_008.tex

[EQ eq_p3_009] -> see eq_p3_009.tex

[EQ eq_p3_010] -> see eq_p3_010.tex

[EQ eq_p3_011] -> see eq_p3_011.tex


# [PAGE 5]
Therefore, it can be observed that in the limited segments (e.g., ts →te), there are the three
scalars
 Πts
te, Σts
te, εte

, allowing us to precompute them once and re-use them during sampling.
Lemma. 2 enables theoretical computation of posterior transitions between any two timesteps
ts and te, allowing multi-step sampling in a manner distinct from DDIM. However, while the
affine–Gaussian transition provides an efficient coarse approximation for the reverse path xts →xte,
the approximation may introduce residual artifacts in parctice. It is caused by the strong noise
attenuation and the fixed ˆx0 assumption. Moreover, since xt inherently entangles both the foreground
and the background content, direct sampling through the affine-Gaussian kernel will ignore the critical
spatial structure discrepancies for SIAS.
To better preserve anomaly-localized information while ensuring smooth global composition, we
explicitly decompose the clean sample x0 into two disjoint components:
x0 = xan
0 + xbg
0 ,
(6)
where xan
0 is the anomaly-only region (masked by M), and xbg
0 is the background. The background
is independently forward-diffused:
xbg
te ∼q(xbg
te | xbg
0 ),
(7)
while the anomaly foreground is refined by the learned FARM module (introduced later in Sec. 3.2),
and merged with the background through spatial masking:
xR
te = FARM(xte),
xte = M ⊙xR
te + (1 −M) ⊙xbg
te .
(8)
This foreground-aware fusion ensures consistent noise levels between anomalous and normal regions
at each step, preserving local anomaly salience while maintaining global visual coherence. In practice,
we also introduce a final fine-grained refinement stage using standard DDPM posterior sampling
for small t (e.g., t = 1 or t = 2) to restore the alignment between the coarse trajectory and the
ground-truth posterior, and to enhance fine-scale texture fidelity. The complete sampling algorithm is
summarized in Algorithm 3.
3.2
Foreground-Aware Reconstruction Module
time steps
Sinusoidal embedding
Linear Projection
MLP+Sigmoid
×
＋
Resize
Forward diffusion
Forward Process
Reverse  Process
Foreground-aware 
Reconstruction Module
AIAS
Update 
𝐭𝐭𝐬𝐬= 𝐭𝐭𝐞𝐞
LDM
Foregrond-aware process 
in both directions
Forward diffsion
Reverse sample
New
Figure 2: The architecture of FARM. Given noisy latent xts and
mask M, the encoder fenc extracts features zts, which is also
modulated by a background-adaptive soft mask ˜
M and related
timestep embedding τ ts. The decoder fdec then reconstructs the
anomaly-only latent ˆxan
0 , which is forward-diffused to produce
anomaly-aware noise.
As discussed above, conven-
tional diffusion models treat all
spatial regions uniformly, which
limits their ability to synthe-
size localized anomalies.
To
address this, we propose the
Foreground-Aware Reconstruc-
tion Module (FARM), which re-
constructs clean anomaly-only
content from noisy latent inputs
under both temporal and spa-
tial guidance.
As illustrated
in Fig. 2, FARM adopts an en-
coder–decoder architecture. The
encoder fenc extracts deep repre-
sentations from the noisy latent
xts, while the decoder fdec progressively upsamples and integrates the binary mask M at multiple
resolutions, ensuring spatial alignment with anomaly regions throughout the hierarchy.
To encode temporal context, we initialize sinusoidal timestep embeddings τ ts ∈Rd and project
them into latent space via a learned linear layer. These embeddings are added to the encoder output,
modulating feature responses based on the current noise level and allowing the decoder to reconstruct
temporally consistent structures.
In addition, to modulate background activation, we introduce a background-adaptive soft mask:
˜
M = Md + (1 −Md) · σ(fbg(τ ts)),
(9)
where Md is a downsampled binary mask aligned with encoder resolution, and fbg is a lightweight
MLP. This design allows FARM to suppress irrelevant background features while adapting to the
current timestep.
The encoded feature is computed as:
zts = ˜
M · fenc(xts) + Proj(τ ts),
(10)
5
[FIGURE img_p4_023]
[FIGURE img_p4_024]
[FIGURE img_p4_025]
[FIGURE img_p4_026]
[FIGURE img_p4_027]
[FIGURE img_p4_028]
[FIGURE img_p4_029]
[FIGURE img_p4_030]
[FIGURE img_p4_031]
[FIGURE img_p4_032]
[FIGURE img_p4_033]
[FIGURE img_p4_034]
[FIGURE img_p4_035]
[FIGURE img_p4_036]
[FIGURE img_p4_037]
[FIGURE img_p4_038]
[FIGURE img_p4_039]
[FIGURE img_p4_040]
[FIGURE img_p4_041]
[FIGURE img_p4_042]
[FIGURE img_p4_043]
[FIGURE img_p4_044]
[FIGURE img_p4_045]

[EQ eq_p4_000] -> see eq_p4_000.tex

[EQ eq_p4_001] -> see eq_p4_001.tex

[EQ eq_p4_002] -> see eq_p4_002.tex

[EQ eq_p4_003] -> see eq_p4_003.tex

[EQ eq_p4_004] -> see eq_p4_004.tex

[EQ eq_p4_005] -> see eq_p4_005.tex


# [PAGE 6]
and decoded into an anomaly-only latent: ˆxan
0 = fdec(zts, M).
To inject anomaly-aware noise into the sampling trajectory, the reconstructed anomaly is forward-
diffused:
ˆxan
ts =
p
¯αts · ˆxan
0 +
p
1 −¯αts · ϵ,
ϵ ∼N(0, I),
(11)
and replaces the original noise in masked regions:
ˆxts = (1 −M) · xts + M · ˆxan
ts .
(12)
During training, FARM is supervised to ensure that the reconstructed anomalies match the masked
regions of the noisy inputs. During inference, temporal and spatial guidance together enable FARM
to introduce localized and temporally coherent anomaly signals into the reverse trajectory, ensuring
alignment with the global generative process while enhancing fine-grained control.
4
Experiments
4.1
Implementation Details.
Datasets. We evaluate FAST on two widely-used industrial anomaly segmentation benchmarks:
MVTec-AD [1] and BTAD [19]. For each anomaly class, we synthesize image–mask pairs using
normal images, binary masks, and text prompts describing anomaly semantics. A total of 500 samples
are generated per class, with approximately one-third used for training and the remainder reserved
for evaluation. Evaluation Metrics. We report performance using mean Intersection over Union
(mIoU) and pixel-wise accuracy (Acc), following standard practice in anomaly segmentation. Base-
lines. FAST is compared against six representative anomaly synthesis approaches: CutPaste [14],
DRAEM [36], GLASS [2], the GAN-based SOTA method DFMGAN [7], and diffusion-based SOTA
models Anomaly Diffusion [13] and RealNet [38]. To simulate realistic deployment scenarios, we
pair all generation methods with lightweight segmentation networks, including Segformer [30],
BiSeNet V2 [35], and STDC [8]. As our method adopts the same prompt-driven synthesis setup as
AnomalyDiffusion [13], we omit the details here for brevity. Full specifications of the textual config-
uration, as well as other implementation details, including dataset preprocessing, sampling schedules,
loss weights, and hyperparameter settings, are provided in the Supplementary Materials A.5.
4.2
Comparison Studies
Table 1: Evaluation of pixel-level segmentation accuracy on extended MVTec data using real-time
Segformer. Detailed per-category results for other real-time segmentation model, such as BiseNet V2
and STDC are reported in Supplementary Material A.6.
Category
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
AnomalyDiffusion
FAST
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
bottle
75.11
79.49
79.51
84.99
70.26
76.30
75.45
80.39
77.96
83.90
76.39
83.54
86.86
90.90
cable
55.40
60.49
64.52
70.77
58.81
62.32
62.10
64.87
62.51
69.27
62.49
74.48
73.71
77.94
capsule
35.15
40.29
51.39
62.32
34.12
38.04
41.29
15.83
46.76
51.91
37.73
44.72
63.22
71.12
carpet
66.34
77.59
72.57
81.28
70.11
77.56
71.33
83.69
68.84
79.15
64.67
73.59
73.84
83.53
grid
29.90
46.72
47.75
67.85
37.43
46.30
37.73
54.13
37.55
48.86
38.70
51.82
52.45
70.70
hazel_nut
56.95
60.72
84.22
89.74
55.51
57.43
83.43
86.03
60.18
63.49
59.33
67.48
90.81
94.79
leather
57.23
63.49
64.12
71.49
62.05
73.38
60.96
68.02
68.29
77.16
56.45
62.51
66.60
74.18
metal_nut
88.78
90.94
93.51
96.10
88.15
90.52
92.77
94.93
91.28
94.09
88.00
91.10
94.65
96.88
pill
43.28
47.11
46.99
49.76
41.52
43.54
87.19
90.05
47.32
58.31
83.21
89.00
90.17
94.07
screw
25.10
31.35
46.96
59.03
35.94
42.37
46.65
50.79
47.12
55.17
38.47
49.49
49.94
57.48
tile
85.33
91.60
89.21
93.74
85.67
90.28
88.87
91.96
83.53
87.30
84.29
89.72
90.13
93.77
toothbrush
39.40
63.93
65.35
79.43
53.75
60.46
61.00
70.50
57.68
72.03
48.68
64.41
74.98
88.63
transistor
65.03
71.05
59.96
62.18
29.28
30.67
73.56
78.48
63.71
66.79
79.27
91.74
91.80
94.50
wood
49.64
60.47
67.52
73.28
50.91
53.16
67.00
80.84
61.84
89.54
60.16
74.62
78.77
86.31
zipper
65.39
71.89
69.29
79.36
69.98
79.31
66.34
70.50
68.78
78.50
65.36
72.66
72.80
84.73
Average
55.87
63.81
66.86
74.75
56.23
61.44
67.71
72.07
62.89
71.70
62.88
72.06
76.72
83.97
Table 2: Evaluation of pixel-level segmentation accuracy on extended BTAD data using real-time
Segformer, BiseNet V2 and STDC.
Backbone
Category
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
AnomalyDiffusion
FAST
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
Segformer
01
66.94
78.20
67.86
80.14
68.02
79.57
67.02
78.03
67.17
80.20
66.55
76.31
75.93
86.12
02
65.04
83.64
69.52
82.96
69.99
83.58
68.75
84.92
70.64
83.90
68.06
84.74
70.63
81.63
03
50.96
60.41
50.39
54.30
51.77
53.53
38.95
41.55
48.76
57.50
54.85
80.20
79.40
85.64
BiseNet V2
01
57.15
69.88
49.16
63.48
44.09
50.57
49.49
59.20
45.45
57.65
46.66
55.18
58.74
68.98
02
59.45
82.05
66.46
80.29
66.37
79.46
66.02
79.21
66.11
81.67
65.57
84.00
68.02
82.40
03
31.84
40.62
36.15
39.04
30.80
37.15
20.12
21.48
29.55
33.11
42.27
74.41
77.87
92.49
STDC
01
48.06
59.86
42.17
65.36
45.51
60.12
44.68
51.71
32.91
49.21
44.85
55.29
44.95
53.47
02
59.80
77.57
64.96
84.32
65.02
81.94
64.85
75.32
64.00
82.64
64.73
78.93
67.76
82.16
03
19.76
25.20
36.14
38.80
17.04
28.01
14.67
16.55
22.57
24.79
41.71
65.45
84.04
92.36
Anomaly Segmentation Table. 1 and 2 report pixel-level segmentation results on various datasets
using Segformer trained with FAST-augmented data. We observe that FAST achieves an average mIoU
of 76.72% and accuracy of 83.97%, significantly outperforming the strongest prior method, DRAEM
(74.75% Acc), by 9.22 points, respectively. Improvements are particularly notable in challenging
categories: in capsule, FAST increases mIoU from 51.39% (DRAEM) to 63.22% (↑11.83); on
grid, from 47.75% to 52.45% (↑4.70); and on transistor, from 84.22% to 91.80% (↑7.58). Even in
6

[EQ eq_p5_000] -> see eq_p5_000.tex

[EQ eq_p5_001] -> see eq_p5_001.tex

[EQ eq_p5_002] -> see eq_p5_002.tex

[EQ eq_p5_003] -> see eq_p5_003.tex

[EQ eq_p5_004] -> see eq_p5_004.tex


# [PAGE 7]
relatively easier categories such as bottle and tile, FAST still yields consistent improvements of 7.35
and 0.92 mIoU points, respectively. These results demonstrate that the combination of mask-aware
noise injection via FARM and coarse-to-fine accelerated sampling via AIES enables more realistic
and structurally coherent anomaly synthesis, leading to superior segmentation performance. Similar
trends are observed when replacing Segformer with other real-time backbones such as BiseNetV2
and STDC, as shown in Supplementary Materials A.6, confirming the generalizability of FAST across
different segmentation architectures.
Mvtec AD
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
Anomaly Diffusion
FAST
Hazel_nut
Transistor
Tile
Metal_nut
Cable
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
Figure 3: Visualization results of different anomaly synthesis
methods on the MVTec dataset. Columns correspond to
synthesis methods (from left to right: MVTec AD, CutPaste,
DRAEM, GLASS, DFMGAN, RealNet, Anomaly Diffusion,
FAST), and rows correspond to product categories (from top
to bottom: hazel_nut, transistor, tile, metal_nut, cable).
Qualitative Comparison. Fig. 3 vi-
sually compares anomaly samples
synthesized by different anomaly
synthesis methods across several
MVTec-AD categories. It can be
observed that traditional unsuper-
vised methods such as CutPaste
and DRAEM generate anomalies
by overlaying arbitrary textures or
patches without any semantic guid-
ance. For instance, in the Cable cat-
egory, anomalies produced by Cut-
Paste appear as artificial, block-like
overlays lacking meaningful texture
or structure.
Similarly, DRAEM
and GLASS introduce unrealistic
color distortions and incoherent pat-
terns in the Transistor category,
which deviate significantly from
typical industrial anomalies. DL-
based approaches (DFMGAN, Re-
alNet, and AnomalyDiffusion) gen-
erate more visually plausible results, but still exhibit noticeable shortcomings. For instance, RealNet
often introduces color shifts and boundary artifacts, as seen in the tile and cable cases, where anoma-
lies appear overly smooth or blurred. DFMGAN and AnomalyDiffusion are able to synthesize more
coherent shapes (e.g., spray-paint-like anomalies in hazel_nut), yet they suffer from inaccurate bound-
aries and structural mismatches, as is especially evident in the tile (AnomalyDiffusion) and cable
(DFMGAN) categories. In contrast, FAST consistently produces anomalies that closely resemble
realistic defects while maintaining precise alignment with the annotated masks. In the metal_nut and
hazel_nut cases, FAST is the only method that preserves both texture and shape fidelity within the
intended regions, demonstrating superior controllability and structural consistency. These results
validate the effectiveness of the proposed FAST in segmentation-oriented anomaly synthesis.
4.3
Ablation Studies
Hazel_nut
Pill
Screw
Mvtec AD
DDPM
DDIM
PLMS
AIAS
Figure 4: SIAS results with other sampling
strategies.
Columns correspond to sam-
pling strategies (from left to right: ground
truth, DDPM (1000 steps), DDIM (50 steps),
PLMS (50 steps), AIAS (50 steps), and rows
correspond to categories (from top to bot-
tom: hazelnut, pill, screw). Further qualita-
tive results (trained on MVTec and BTAD) are
provided in the Supplementary Materials A.8.
The Impact of AIAS. We compare our proposed
AIAS strategy with several widely-used training-
free samplers, including DDPM [10] with 1000
steps, DDIM [25] with 50 steps and PLMS [15]
with 50 steps. These methods represent state-of-the-
art discrete-time sampling approaches for diffusion-
based models.
To ensure fairness, we exclude
continuous-time solvers, as they rely on a funda-
mentally different formulation based on ODEs or
SDEs, which necessitates a distinct training paradigm
and architectural adjustments incompatible with our
discrete-time framework. Quantitative results are
reported in Fig. 5. While DDPM achieves compet-
itive results on certain categories (e.g., carpet, tile),
it requires 1000 iterative steps, making it over 20×
slower than AIAS in practice. DDIM and PLMS,
though more efficient, exhibit inconsistent perfor-
mance across categories and often underperform
AIAS, particularly on challenging textures such as
7
[FIGURE img_p6_046]
[FIGURE img_p6_047]
[FIGURE img_p6_048]
[FIGURE img_p6_049]
[FIGURE img_p6_050]
[FIGURE img_p6_051]
[FIGURE img_p6_052]
[FIGURE img_p6_053]
[FIGURE img_p6_054]
[FIGURE img_p6_055]
[FIGURE img_p6_056]
[FIGURE img_p6_057]
[FIGURE img_p6_058]
[FIGURE img_p6_059]
[FIGURE img_p6_060]
[FIGURE img_p6_061]
[FIGURE img_p6_062]
[FIGURE img_p6_063]
[FIGURE img_p6_064]
[FIGURE img_p6_065]
[FIGURE img_p6_066]
[FIGURE img_p6_067]
[FIGURE img_p6_068]
[FIGURE img_p6_069]
[FIGURE img_p6_070]
[FIGURE img_p6_071]
[FIGURE img_p6_072]
[FIGURE img_p6_073]
[FIGURE img_p6_074]
[FIGURE img_p6_075]
[FIGURE img_p6_076]
[FIGURE img_p6_077]
[FIGURE img_p6_078]
[FIGURE img_p6_079]
[FIGURE img_p6_080]
[FIGURE img_p6_081]
[FIGURE img_p6_082]
[FIGURE img_p6_083]
[FIGURE img_p6_084]
[FIGURE img_p6_085]
[FIGURE img_p6_086]
[FIGURE img_p6_087]
[FIGURE img_p6_088]
[FIGURE img_p6_089]
[FIGURE img_p6_090]
[FIGURE img_p6_091]
[FIGURE img_p6_092]
[FIGURE img_p6_093]
[FIGURE img_p6_094]
[FIGURE img_p6_095]
[FIGURE img_p6_096]
[FIGURE img_p6_097]
[FIGURE img_p6_098]
[FIGURE img_p6_099]
[FIGURE img_p6_100]
[FIGURE img_p6_101]
[FIGURE img_p6_102]
[FIGURE img_p6_103]
[FIGURE img_p6_104]
[FIGURE img_p6_105]
[FIGURE img_p6_106]
[FIGURE img_p6_107]
[FIGURE img_p6_108]
[FIGURE img_p6_109]
[FIGURE img_p6_110]
[FIGURE img_p6_111]
[FIGURE img_p6_112]
[FIGURE img_p6_113]
[FIGURE img_p6_114]
[FIGURE img_p6_115]
[FIGURE img_p6_116]
[FIGURE img_p6_117]
[FIGURE img_p6_118]
[FIGURE img_p6_119]
[FIGURE img_p6_120]
[FIGURE img_p6_121]
[FIGURE img_p6_122]
[FIGURE img_p6_123]
[FIGURE img_p6_124]
[FIGURE img_p6_125]
[FIGURE img_p6_126]
[FIGURE img_p6_127]
[FIGURE img_p6_128]
[FIGURE img_p6_129]
[FIGURE img_p6_130]
[FIGURE img_p6_131]
[FIGURE img_p6_132]
[FIGURE img_p6_133]
[FIGURE img_p6_134]
[FIGURE img_p6_135]
[FIGURE img_p6_136]
[FIGURE img_p6_137]
[FIGURE img_p6_138]
[FIGURE img_p6_139]
[FIGURE img_p6_140]
[FIGURE img_p6_141]
[FIGURE img_p6_142]
[FIGURE img_p6_143]

[EQ eq_p6_000] -> see eq_p6_000.tex

[EQ eq_p6_001] -> see eq_p6_001.tex

[EQ eq_p6_002] -> see eq_p6_002.tex


# [PAGE 8]
capsule, grid, and transistor. In contrast, AIAS achieves the best results on the majority of cate-
gories and consistently provides competitive or superior performance in both mIoU and accuracy,
demonstrating its ability to generate segmentation-aligned anomalies with significantly fewer steps.
It further indicates that by analytically aggregating multiple DDPM transitions into coarse-to-fine
segments, AIAS reduces the discretization error inherent in single-step samplers (e.g., DDIM) or
fixed multistep solvers (e.g., PLMS), allowing a closer approximation of the true posterior within
just 50 steps. Fig. 4 further illustrates the qualitative advantage. For example, in the hazel_nut class,
the anomalies produced by DDPM, DDIM, and PLMS display noticeable color inconsistencies near
the anomaly boundary, resulting from distributional mismatch with the background. In comparison,
FAST-produced anomalies that are well blended into the context, with sharper and more realistic
structural alignment.
Table 3: Comparison of pixel-level anomaly segmentation using different steps on the MVTec dataset.
Category
Step 2
Step 5
Step 10
Step 30
Step 50
Step 100
Step 200
Step 500
Step 1000
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
bottle
77.03
80.96
80.55
85.08
83.26
85.90
84.59
87.89
86.86
90.90
83.75
86.95
84.04
88.54
83.52
88.19
81.65
84.83
cable
47.39
48.66
69.58
73.11
71.23
75.07
73.34
77.59
73.71
77.94
72.99
76.50
72.83
76.51
75.23
79.32
73.45
78.06
capsule
43.56
48.58
49.81
54.22
54.85
59.31
61.12
67.08
63.22
71.12
63.15
71.17
62.12
71.76
62.83
70.88
60.01
66.87
carpet
70.24
80.98
73.22
83.18
73.10
84.06
73.56
80.50
73.84
83.53
73.41
82.92
73.17
81.90
73.27
82.49
75.99
84.14
grid
48.15
61.75
50.03
63.28
50.89
71.35
48.76
61.17
52.45
70.70
50.03
65.41
52.06
67.28
49.18
63.63
50.91
63.19
hazel_nut
76.16
78.75
84.16
86.50
90.45
94.04
90.49
94.04
90.81
94.79
90.82
94.16
90.87
94.27
90.77
94.71
89.81
93.31
leather
62.11
66.86
66.74
76.16
67.09
76.51
65.44
72.41
66.60
74.18
66.88
74.22
65.87
87.88
67.95
83.62
71.03
80.32
metal_nut
92.06
93.57
93.94
95.72
94.71
96.98
94.47
96.31
94.65
96.88
94.74
97.19
94.50
96.59
94.72
96.80
94.63
97.18
pill
50.03
55.46
80.01
82.53
90.07
93.80
90.02
94.24
90.17
94.07
89.82
94.10
89.80
93.22
90.15
94.34
89.36
93.79
screw
46.07
52.01
47.92
56.55
50.04
56.21
50.11
60.85
49.94
57.48
50.06
58.66
48.41
61.05
47.71
54.90
49.35
59.18
tile
87.26
93.92
89.46
94.96
89.72
93.92
89.58
93.68
90.13
93.77
89.93
94.45
90.02
93.73
89.71
93.38
91.01
94.72
toothbrush
58.54
67.15
76.65
87.41
76.96
90.29
74.36
90.78
74.98
88.63
74.17
87.29
73.32
86.49
75.66
89.50
76.10
91.25
transistor
66.42
71.59
66.08
70.23
77.27
79.66
89.45
92.65
91.80
94.50
91.39
94.66
89.67
93.50
90.32
93.21
89.59
93.41
wood
68.69
78.28
74.23
81.07
75.97
81.18
78.76
84.99
78.77
86.31
77.00
83.95
77.60
82.85
77.71
83.45
80.03
85.30
zipper
68.85
75.26
70.92
81.44
72.44
84.99
73.08
81.91
72.80
84.73
71.99
81.94
71.71
82.21
71.73
83.47
72.45
82.35
Average
64.17
70.25
71.55
78.10
74.54
81.55
75.81
82.41
76.72
83.97
76.01
82.90
75.73
83.85
76.03
83.46
76.36
83.19
Mvtec AD
T = 2
T = 5
T = 10
T = 30
T = 50
T = 100
T = 200
T = 500
T = 1000
Capsule
Bottle
Carpet
Figure 6: Segmentation-oriented industrial anomaly synthesis results at different steps of AIAS.
Columns correspond to increasing sampling steps T (from left to right), and rows correspond to
product categories (from top to bottom: capsule, bottle, carpet).
DDPM (1000 steps)
DDIM (50 steps)
PLMS (50 steps)
AIAS (50 steps)
(a1)
(b1)
(c1)
(a2)
(b2)
(c2)
Figure 5: The effect of different sampling methods on SIAS
in the MVTec dataset. Top row shows per-category segmen-
tation performance using mIoU; bottom row shows perfor-
mance using Acc. Detailed per-category results of AIAS are
reported in Supplementary Material A.6.
The Impact of AIAS under differ-
ent step sizes.
We further investi-
gate the segmentation performance of
AIAS under varying numbers of re-
verse steps, ranging from 2 to 1000,
as reported in Table. 3.
Remark-
ably, AIAS approximates the perfor-
mance of full-step DDPM using only
10 steps, and reaches near-optimal re-
sults by 50 steps, demonstrating the
effectiveness of our coarse-to-fine ag-
gregation strategy. Performance im-
proves rapidly as t increases from 2
to 50, since early segments capture
the global layout and coarse structure
of anomalies, which are most relevant
for segmentation. This trend is also
visually confirmed in Fig. 6. Beyond
this point, performance gains gradu-
ally saturate, indicating that additional
steps primarily refine high-frequency
details with limited impact on segmen-
tation accuracy. Notably, when t = 1000, AISA degenerates to the original DDPM sampling process,
where each segment [te, ts] corresponds to a single denoising step. The convergence of performance
8
[FIGURE img_p7_144]
[FIGURE img_p7_145]
[FIGURE img_p7_146]
[FIGURE img_p7_147]
[FIGURE img_p7_148]
[FIGURE img_p7_149]
[FIGURE img_p7_150]
[FIGURE img_p7_151]
[FIGURE img_p7_152]
[FIGURE img_p7_153]
[FIGURE img_p7_154]
[FIGURE img_p7_155]
[FIGURE img_p7_156]
[FIGURE img_p7_157]
[FIGURE img_p7_158]
[FIGURE img_p7_159]
[FIGURE img_p7_160]
[FIGURE img_p7_161]
[FIGURE img_p7_162]
[FIGURE img_p7_163]
[FIGURE img_p7_164]
[FIGURE img_p7_165]
[FIGURE img_p7_166]
[FIGURE img_p7_167]
[FIGURE img_p7_168]
[FIGURE img_p7_169]
[FIGURE img_p7_170]
[FIGURE img_p7_171]
[FIGURE img_p7_172]
[FIGURE img_p7_173]
[FIGURE img_p7_174]
[FIGURE img_p7_175]
[FIGURE img_p7_176]
[FIGURE img_p7_177]
[FIGURE img_p7_178]

[EQ eq_p7_000] -> see eq_p7_000.tex

[EQ eq_p7_001] -> see eq_p7_001.tex

[EQ eq_p7_002] -> see eq_p7_002.tex


# [PAGE 9]
at this point validates that our multi-step analytical updates provide a faithful approximation of
the full diffusion trajectory, preserving both global semantics and fine-grained anomaly cues while
significantly reducing sampling cost. Furthermore, excessive denoising steps may introduce over-
smoothing or amplify reconstruction inconsistencies, potentially weakening the alignment between
synthesized anomalies and segmentation-relevant structures. Overall, these results highlight that
AIAS not only accelerates sampling, but also introduces an inductive structural bias beneficial for
anomaly segmentation. In practice, the optimal balance between quality and efficiency is achieved
within 10–50 steps.
Wood_color
Hazel_nut_crack
Tile_oil
Cable_missing_wire
(w/o FARM)
(w/ FARM)
Figure 7: Qualitative ablation results with and without
FARM on MVtec dataset. Columns correspond to cat-
egory–anomaly pairs (from left to right: Wood_color,
Tile_oil, Hazel_nut_crack, Cable_missing_wire; and
rows correspond to sampling configurations (from
top to bottom: without FARM (w/o FARM) and with
FARM (w/ FARM).
The Impact of FARM. To evaluate the
effectiveness of FARM, we conduct an ab-
lation study by comparing the model’s per-
formance with (w/ FARM) and without
(w/o FARM) FARM under identical AIAS
settings. Results on the MVTec dataset
are reported in Fig. 8. The inclusion of
FARM leads to substantial improvements
in segmentation performance, with average
mIoU increasing from 65.33 to 76.42 and
accuracy increasing from 71.24 to 83.97.
The performance gains are particularly pro-
nounced in challenging categories charac-
terized by fine-grained or complex struc-
tures, such as capsule (↑14.1 mIoU), grid
(↑14.7 mIoU), and transistor (↑29.5 mIoU).
Even in relatively easier categories like
tile and hazel_nut, FARM consistently en-
hances accuracy and boundary localization, as shown in Fig. 7. More detailed analysis of FARM can
be found in Supplementary Material A.7.
Figure 8: Qualitative ablation results with and without FARM on MVtec dataset. Columns corre-
spond to product categories and rows correspond to mIou and Acc). Detailed per-category results
for ablation study of FARM are reported in Supplementary Material A.6.
5
Conclusion
In this work, we proposed FAST, a segmentation-oriented foreground-aware diffusion framework
tailored for anomaly synthesis. To address the limitations of existing anomaly synthesis methods,
specifically their limited controllability and lack of structural awareness, we introduced two key
components: the Foreground-Aware Reconstruction Module (FARM), which adaptively injects
anomaly-aware noise at each sampling step, and the Anomaly-Informed Efficient Sampling (AIAS), a
training-free strategy that accelerates sampling via coarse-to-fine aggregation. Built upon a discrete-
time latent diffusion backbone, FAST enables the synthesis of segmentation-aligned anomalies with
as few as 10 denoising steps. Extensive experiments on MVTec-AD and BTAD demonstrate that
FAST outperforms existing baselines in downstream segmentation. FAST represents a promising
step toward controllable and efficient segmentation-oriented industrial anomaly synthesis.
9
[FIGURE img_p8_179]
[FIGURE img_p8_180]
[FIGURE img_p8_181]
[FIGURE img_p8_182]
[FIGURE img_p8_183]
[FIGURE img_p8_184]
[FIGURE img_p8_185]
[FIGURE img_p8_186]
[FIGURE img_p8_187]
[FIGURE img_p8_188]
[FIGURE img_p8_189]
[FIGURE img_p8_190]
[FIGURE img_p8_191]
[FIGURE img_p8_192]
[FIGURE img_p8_193]
[FIGURE img_p8_194]
[FIGURE img_p8_195]
[FIGURE img_p8_196]
[FIGURE img_p8_197]
[FIGURE img_p8_198]

[EQ eq_p8_000] -> see eq_p8_000.tex

[EQ eq_p8_001] -> see eq_p8_001.tex

[EQ eq_p8_002] -> see eq_p8_002.tex

[EQ eq_p8_003] -> see eq_p8_003.tex

[EQ eq_p8_004] -> see eq_p8_004.tex

[EQ eq_p8_005] -> see eq_p8_005.tex

[EQ eq_p8_006] -> see eq_p8_006.tex


# [PAGE 10]
References
[1] Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. Mvtec ad–a comprehen-
sive real-world dataset for unsupervised anomaly detection. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 9592–9600, 2019.
[2] Qiyu Chen, Huiyuan Luo, Chengkan Lv, and Zhengtao Zhang. A unified anomaly synthesis
strategy with gradient ascent for industrial anomaly detection and localization. In European
Conference on Computer Vision, pages 37–54. Springer, 2025.
[3] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, , and A. Vedaldi. Describing textures in the wild.
In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014.
[4] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of
deep bidirectional transformers for language understanding. In Jill Burstein, Christy Doran, and
Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies, Volume 1
(Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association
for Computational Linguistics.
[5] Zongwei Du, Liang Gao, and Xinyu Li. A new contrastive gan with data augmentation for
surface defect recognition under limited data. IEEE Transactions on Instrumentation and
Measurement, 72:1–13, 2022.
[6] Yuxuan Duan, Yan Hong, Li Niu, and Liqing Zhang. Few-shot defect image generation
via defect-aware feature manipulation. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 37, pages 571–578, 2023.
[7] Yuxuan Duan, Yan Hong, Li Niu, and Liqing Zhang. Few-shot defect image generation
via defect-aware feature manipulation. In Proceedings of the AAAI Conference on Artificial
Intelligence, volume 37, pages 571–578, 2023.
[8] Mingyuan Fan, Shenqi Lai, Junshi Huang, Xiaoming Wei, Zhenhua Chai, Junfeng Luo, and
Xiaolin Wei. Rethinking bisenet for real-time semantic segmentation. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 9716–9725, 2021.
[9] Shidan He, Lei Liu, and Shen Zhao. Anomalycontrol: Learning cross-modal semantic features
for controllable anomaly synthesis. arXiv preprint arXiv:2412.06510, 2024.
[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances
in neural information processing systems, 33:6840–6851, 2020.
[11] Emiel Hoogeboom and Tim Salimans. Blurring diffusion models. In The Eleventh International
Conference on Learning Representations, 2023.
[12] Jie Hu, Yawen Huang, Yilin Lu, Guoyang Xie, Guannan Jiang, Yefeng Zheng, and Zhichao
Lu.
Anomalyxfusion: Multi-modal anomaly synthesis with diffusion.
arXiv preprint
arXiv:2404.19444, 2024.
[13] Teng Hu, Jiangning Zhang, Ran Yi, Yuzhen Du, Xu Chen, Liang Liu, Yabiao Wang, and
Chengjie Wang. Anomalydiffusion: Few-shot anomaly image generation with diffusion model.
In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 8526–8534,
2024.
[14] Chun-Liang Li, Kihyuk Sohn, Jinsung Yoon, and Tomas Pfister. Cutpaste: Self-supervised
learning for anomaly detection and localization. In Proceedings of the IEEE/CVF conference
on computer vision and pattern recognition, pages 9664–9674, 2021.
[15] Luping Liu, Yi Ren, Zhijie Lin, and Zhou Zhao. Pseudo numerical methods for diffusion models
on manifolds. In International Conference on Learning Representations, 2022.
[16] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver:
A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in
Neural Information Processing Systems, 35:5775–5787, 2022.
10


# [PAGE 11]
[17] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu.
Dpm-
solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint
arXiv:2211.01095, 2022.
[18] Zhaoyang Lyu, Xudong Xu, Ceyuan Yang, Dahua Lin, and Bo Dai. Accelerating diffusion
models via early stop of the diffusion process. arXiv preprint arXiv:2205.12524, 2022.
[19] Pankaj Mishra, Riccardo Verk, Daniele Fornasier, Claudio Piciarelli, and Gian Luca Foresti.
VT-ADL: A vision transformer network for image anomaly detection and localization. In 30th
IEEE/IES International Symposium on Industrial Electronics (ISIE), June 2021.
[20] Shuanlong Niu, Bin Li, Xinggang Wang, and Hui Lin. Defect image sample generation with gan
for improving defect recognition. IEEE Transactions on Automation Science and Engineering,
17(3):1611–1622, 2020.
[21] Mingjing Pei, Ningzhong Liu, Bing Zhao, and Han Sun. Self-supervised learning for indus-
trial image anomaly detection by simulating anomalous samples. International Journal of
Computational Intelligence Systems, 16(1):152, 2023.
[22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-
resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, pages 10684–10695, 2022.
[23] Hannah M Schlüter, Jeremy Tan, Benjamin Hou, and Bernhard Kainz. Natural synthetic
anomalies for self-supervised anomaly detection and localization. In European Conference on
Computer Vision, pages 474–489. Springer, 2022.
[24] Qingfeng Shi, Jing Wei, Fei Shen, and Zhengtao Zhang. Few-shot defect image generation
based on consistency modeling. In European Conference on Computer Vision, pages 360–376.
Springer, 2025.
[25] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In
International Conference on Learning Representations, 2021.
[26] Han Sun, Yunkang Cao, and Olga Fink. Cut: A controllable, universal, and training-free visual
anomaly generation framework. arXiv preprint arXiv:2406.01078, 2024.
[27] Daniel Watson, William Chan, Jonathan Ho, and Mohammad Norouzi. Learning fast samplers
for diffusion models by differentiating through sample quality. In International Conference on
Learning Representations, 2022.
[28] Long Wen, You Wang, and Xinyu Li. A new cycle-consistent adversarial networks with attention
mechanism for surface defect classification with small samples. IEEE Transactions on Industrial
Informatics, 18(12):8988–8998, 2022.
[29] Zhisheng Xiao, Karsten Kreis, and Arash Vahdat. Tackling the generative learning trilemma
with denoising diffusion gans. In International Conference on Learning Representations, 2022.
[30] Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo.
Segformer: Simple and efficient design for semantic segmentation with transformers. Advances
in neural information processing systems, 34:12077–12090, 2021.
[31] Xichen Xu, Yanshu Wang, Yawen Huang, Jiaqi Liu, Xiaoning Lei, Guoyang Xie, Guannan Jiang,
and Zhichao Lu. A survey on industrial anomalies synthesis. arXiv preprint arXiv:2502.16412,
2025.
[32] Minghui Yang, Peng Wu, and Hui Feng. Memseg: A semi-supervised method for image surface
defect detection using differences and commonalities. Engineering Applications of Artificial
Intelligence, 119:105835, 2023.
[33] Shuai Yang, Zhifei Chen, Pengguang Chen, Xi Fang, Yixun Liang, Shu Liu, and Yingcong
Chen. Defect spectrum: a granular look of large-scale defect datasets with rich semantics. In
European Conference on Computer Vision, pages 187–203. Springer, 2024.
11


# [PAGE 12]
[34] Hang Yao, Ming Liu, Zhicun Yin, Zifei Yan, Xiaopeng Hong, and Wangmeng Zuo. Glad:
towards better reconstruction with global and local adaptive diffusion models for unsupervised
anomaly detection. In European Conference on Computer Vision, pages 1–17. Springer, 2024.
[35] Changqian Yu, Changxin Gao, Jingbo Wang, Gang Yu, Chunhua Shen, and Nong Sang. Bisenet
v2: Bilateral network with guided aggregation for real-time semantic segmentation. Interna-
tional journal of computer vision, 129:3051–3068, 2021.
[36] Vitjan Zavrtanik, Matej Kristan, and Danijel Skoˇcaj. Draem-a discriminatively trained re-
construction embedding for surface anomaly detection. In Proceedings of the IEEE/CVF
international conference on computer vision, pages 8330–8339, 2021.
[37] Gongjie Zhang, Kaiwen Cui, Tzu-Yi Hung, and Shijian Lu. Defect-gan: High-fidelity defect
synthesis for automated defect inspection. In Proceedings of the IEEE/CVF Winter Conference
on Applications of Computer Vision, pages 2524–2534, 2021.
[38] Ximiao Zhang, Min Xu, and Xiuzhuang Zhou. Realnet: A feature selection network with
realistic synthetic anomaly for anomaly detection. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pages 16699–16708, 2024.
[39] Xuan Zhang, Shiyu Li, Xi Li, Ping Huang, Jiulong Shan, and Ting Chen. Destseg: Segmentation
guided denoising student-teacher for anomaly detection. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 3914–3923, 2023.
[40] Huangjie Zheng, Pengcheng He, Weizhu Chen, and Mingyuan Zhou. Truncated diffusion prob-
abilistic models and diffusion-based adversarial auto-encoders. In The Eleventh International
Conference on Learning Representations, 2023.
[41] Kaiwen Zheng, Guande He, Jianfei Chen, Fan Bao, and Jun Zhu. Diffusion bridge implicit
models. In The Thirteenth International Conference on Learning Representations, 2025.
12


# [PAGE 13]
A
Supplementary Materials
A.1
Proof of Lemma 1
Lemma 1 [Linear–Gaussian closure] Let {xk}K
k=0 ⊂Rd satisfy the recursion
xk−1 = Ck xk + dk + εk,
εk ∼N(0, Σk),
εk ⊥{xk, εk+1, . . .},
(13)
where Ck ∈Rd×d, dk ∈Rd, and Σk ∈Rd×d are deterministic. Then, for every integer m with
1≤m≤k, xk−m is again an affine–Gaussian function of xk:
xk−m =
m−1
Y
i=0
Ck−i

|
{z
}
=:C(m)
k
xk +
m−1
X
i=0
 iY
j=1
Ck−j

dk−i
|
{z
}
=:d(m)
k
+ ε(m)
k
,
(14)
where
ε(m)
k
∼N
 0, Σ(m)
k

,
Σ(m)
k
=
m−1
X
i=0
 iY
j=1
Ck−j

Σk−i
 iY
j=1
Ck−j
⊤
.
(15)
Proof. :
Base case (m = 1).
• Eq. 14 with m = 1 is exactly the recursion Eq. 13.
Induction step.
• Assume Eq. 14 and .15 hold for m = r with 1 ≤r < k:
xk−r = C(r)
k xk + d(r)
k
+ ε(r)
k ,
ε(r)
k
∼N
 0, Σ(r)
k

,
ε(r)
k ⊥xk.
• Apply Eq. 13 once more:
xk−(r+1) = Ck−rxk−r + dk−r + εk−r
= Ck−r
 C(r)
k xk + d(r)
k
+ ε(r)
k

+ dk−r + εk−r
= Ck−rC(r)
k
|
{z
}
C(r+1)
k
xk + Ck−rd(r)
k
+ dk−r
|
{z
}
d(r+1)
k
+ Ck−rε(r)
k
+ εk−r
|
{z
}
ε(r+1)
k
.
(16)
Since ε(r)
k
and εk−r are independent zero–mean Gaussians, their linear combination ε(r+1)
k
remains
Gaussian with covariance Σ(r+1)
k
= Ck−rΣ(r)
k C⊤
k−r +Σk−r, exactly matching Eq. 15 for m = r +1.
Hence the statement holds for all m by induction.
Remark 1. The empty product convention Q0
j=1 Ck−j = Id is used in Eq. 14.
A.2
Proof of Lemma 2
Lemma 2 [Closed-form reverse from ts →te] Fix indices 0 ≤te < ts ≤T, and let the single-
step coefficients (At, Bt, σ2
t ) be defined as in Eq. 13. Then the aggregated reverse kernel over
ts →· · · →te is affine–Gaussian:
xte = Πts
te xts + Σts
te ˆx0 + εte,
(17)
where
Πts
te :=
ts
Y
i=te+1
Bi,
Σts
te :=
ts
X
i=te+1
Ai
ts
Y
j=i+1
Bj,
εte ∼N


0,
ts
X
i=te+1


ts
Y
j=i+1
Bj


2
σ2
i I


.
13

[EQ eq_p12_000] -> see eq_p12_000.tex

[EQ eq_p12_001] -> see eq_p12_001.tex

[EQ eq_p12_002] -> see eq_p12_002.tex

[EQ eq_p12_003] -> see eq_p12_003.tex

[EQ eq_p12_004] -> see eq_p12_004.tex

[EQ eq_p12_005] -> see eq_p12_005.tex

[EQ eq_p12_006] -> see eq_p12_006.tex

[EQ eq_p12_007] -> see eq_p12_007.tex

[EQ eq_p12_008] -> see eq_p12_008.tex

[EQ eq_p12_009] -> see eq_p12_009.tex

[EQ eq_p12_010] -> see eq_p12_010.tex

[EQ eq_p12_011] -> see eq_p12_011.tex

[EQ eq_p12_012] -> see eq_p12_012.tex

[EQ eq_p12_013] -> see eq_p12_013.tex

[EQ eq_p12_014] -> see eq_p12_014.tex

[EQ eq_p12_015] -> see eq_p12_015.tex

[EQ eq_p12_016] -> see eq_p12_016.tex

[EQ eq_p12_017] -> see eq_p12_017.tex

[EQ eq_p12_018] -> see eq_p12_018.tex

[EQ eq_p12_019] -> see eq_p12_019.tex


# [PAGE 14]
Proof. Apply Lemma 1 with Ck = Bk, dk = Akˆx0, Σk = σ2
kI, and m = ts−te. Equations Eq. 17
coincide with the general expressions Eq. 14–Eq. 15, so the result follows directly.
A.3
Loss function
The training objective of FAST consists of two components: the standard denoising loss and the
reconstruction loss. The denoising loss encourages accurate noise prediction across all spatial regions,
while the reconstruction loss ensures that FARM accurately reconstructs anomaly-only content,
and allowes the inserted noise to remain compatible with the global sampling dynamics, thereby
preserving the stability of the overall generation process.
LFAST = λ1 · Ex0,ϵ,t
h
∥ϵ −ϵθ(xt, t)∥2
2
i
(18)
+ λ2 · Exan
0 ,xt,M
h
∥Fϕ(xt, M, t) −xan
0 ∥2
2
i
,
where ϵ ∼N(0, I) is the reference noise for the denoising target, the xan
0 is the anomaly-only content
with pure background, ϵθ(xt, t) and Fϕ denote LDM and FARM, respectively. The scalar weights λ1
and λ2 balance the contributions of the two losses
A.4
Pseudo-code of AIAS
Algorithm 3 Anomaly-Informed Accelerated Sampling
Input: Mask M, clean background xbg
full, clean background latent xbg
0 , prediction ˆx0 from ϵθ
boundary schedule B = {t1 < t2 < · · · < tK = T} and t1 = 2 in our experiments
Output: Syntheised image xfull
Initialize noisy latent xtK ∼N(0, I)
for k = K to 1 do
ts ←tk,
te ←tk−1
# Coarse multi-step reverse from ts →te
Define cofficients At =
√¯αt−1βt
1−¯αt
, Bt =
√αt(1−¯αt−1)
1−¯αt
, and σ2
t = 1−¯αt−1
1−¯αt βt
Compute µ ←(Qts
i=te+1 Bi)xts + (Pts
i=te+1 Ai
Qts
j=i+1Bj)ˆx0
Sample noise ε ∼N(0, (Pts
i=te+1
Qts
j=i+1 Bj
2
σ2
i )I)
xte ←µ + ε
# Forward diffuse background to te
xbg
te ∼N(√¯αtexbg
0 , (1 −¯αte)I)
xR
te ←FARM(xte)
xte ←M ⊙xR
te + (1 −M) ⊙xbg
te
end for
# Fine posterior refinement
for t = t1 down to 0 do
Predict ˆx0 ←fθ(xt, t)
xt−1 ←q(xt−1 | xt, ˆx0)
end for
xfull ←M ⊙Decode(x0) + (1 −M) ⊙xbg
full
A.5
Training Configuration
To synthesize abnormal data, we utilize the complete set of normal images, their corresponding
masks, and associated textual descriptions for each type of anomaly within every category of products.
Notably, the original GLASS framework comprises three branches,a normal-sample branch, a feature-
level anomaly synthesis branch guided by gradient ascent, and an image-level branch that overlays
external textures. Therefore, its output is unsuitable directly for pixel-level anomaly segmentation and
other downstream sgementation models. Accordingly, we revised its synthesis process to align with
our segmentation-based evaluation protocol. We release the modified implementation together
with the FAST to ensure fairness.
14

[EQ eq_p13_000] -> see eq_p13_000.tex

[EQ eq_p13_001] -> see eq_p13_001.tex

[EQ eq_p13_002] -> see eq_p13_002.tex

[EQ eq_p13_003] -> see eq_p13_003.tex

[EQ eq_p13_004] -> see eq_p13_004.tex

[EQ eq_p13_005] -> see eq_p13_005.tex

[EQ eq_p13_006] -> see eq_p13_006.tex

[EQ eq_p13_007] -> see eq_p13_007.tex

[EQ eq_p13_008] -> see eq_p13_008.tex

[EQ eq_p13_009] -> see eq_p13_009.tex

[EQ eq_p13_010] -> see eq_p13_010.tex

[EQ eq_p13_011] -> see eq_p13_011.tex

[EQ eq_p13_012] -> see eq_p13_012.tex

[EQ eq_p13_013] -> see eq_p13_013.tex

[EQ eq_p13_014] -> see eq_p13_014.tex

[EQ eq_p13_015] -> see eq_p13_015.tex

[EQ eq_p13_016] -> see eq_p13_016.tex

[EQ eq_p13_017] -> see eq_p13_017.tex

[EQ eq_p13_018] -> see eq_p13_018.tex

[EQ eq_p13_019] -> see eq_p13_019.tex

[EQ eq_p13_020] -> see eq_p13_020.tex

[EQ eq_p13_021] -> see eq_p13_021.tex

[EQ eq_p13_022] -> see eq_p13_022.tex

[EQ eq_p13_023] -> see eq_p13_023.tex


# [PAGE 15]
• Model Settings. We set the total number of diffusion steps during training to T = 1000.
For sampling, the range from step 2 to 1000 is uniformly divided into 50 steps, followed by
a fine-grained adjustment phase over the initial steps [0, 2] to enhance reconstruction fidelity.
The model is trained with a batch size of 4 and a learning rate of 1.5e-4. The text embedding
E consists of 8 tokens.
• Prompt Construction. For the MVTec dataset, prompts are formed by appending the
anomaly type to the product category name. For BTAD, due to anonymized category
labels, we use a generic prompt: “damaged". Textual embeddings follow the protocol of
AnomalyDiffusion, where each prompt is tokenized into 8 discrete units and embedded
using a pre-trained BERT encoder [4].
• Hardware and Runtime. All models are trained on a setup of eight NVIDIA A100 GPUs
(40GB each), with training proceeding for roughly 80k iterations.
A.6
Other quantitative experiments
We provide extended evaluation results to complement the findings reported in the main manuscript.
We present detailed, category-wise performance metrics on the MVTec and BTAD benchmarks,
employing BiseNet V2 and STDC as the segmentation backbones. Moreover, we further analyze the
influence of different sampling strategies—except our AIAS method—on downstream segmentation
performance using Segformer.
All experiments are conducted under identical settings to those used in the main study. The results
consistently demonstrate that our proposed FAST framework significantly outperforms existing
anomaly synthesis techniques in enhancing segmentation accuracy across diverse categories.
Table 4: Evaluation of pixel-level segmentation accuracy on extended MVTec data using real-time
BiseNet V2.
Category
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
AnomalyDiffusion
FAST
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
bottle
71.77
78.57
75.13
79.17
57.81
60.79
64.28
71.31
72.16
75.55
75.28
85.11
78.48
83.18
cable
46.00
57.08
53.88
60.96
16.63
16.65
57.09
63.25
51.22
62.32
60.55
74.96
70.91
75.77
capsule
25.97
37.04
36.82
42.19
19.53
51.89
28.40
31.18
35.97
39.39
26.77
32.87
48.56
54.22
carpet
58.98
72.22
68.42
77.21
64.77
73.93
62.13
67.98
8.98
9.01
58.18
64.69
68.94
77.20
grid
24.68
44.17
42.81
63.34
6.50
6.91
10.17
15.23
10.61
11.47
18.98
24.30
39.15
51.78
hazel_nut
47.93
53.57
74.83
81.35
71.54
75.62
79.78
84.37
60.16
65.93
57.26
70.41
88.08
93.45
leather
31.11
58.36
55.07
61.58
57.98
71.84
31.77
34.82
53.77
63.85
50.02
61.60
67.18
74.23
metal_nut
82.95
87.73
91.58
94.73
83.82
85.42
91.17
93.57
88.38
90.73
85.52
90.20
93.62
95.82
pill
55.62
67.04
45.23
48.99
23.88
24.15
82.40
84.30
72.59
86.32
80.87
87.02
85.12
89.60
screw
4.88
6.63
25.08
35.77
12.32
13.11
38.14
40.36
22.35
23.78
23.23
29.91
33.49
41.12
tile
76.25
85.75
86.17
90.45
77.32
80.28
85.69
90.12
77.16
84.84
79.32
85.63
86.86
92.12
toothbrush
35.69
50.45
57.66
79.15
38.86
51.97
48.83
58.76
32.38
37.88
44.33
69.32
73.04
87.34
transistor
44.48
51.79
59.88
65.96
44.93
53.04
76.52
82.13
61.68
68.59
76.34
89.94
91.10
93.81
wood
35.51
46.00
49.82
62.09
36.41
51.10
51.84
63.70
47.29
61.35
52.06
72.75
68.15
72.69
zipper
51.61
63.09
66.88
75.75
61.99
70.07
60.61
71.11
66.09
77.54
57.86
67.64
66.59
78.16
Average
46.23
57.30
59.28
67.91
44.95
52.45
57.92
63.48
50.72
57.24
56.44
67.09
70.62
77.37
Table 5: Evaluation of pixel-level segmentation accuracy on extended MVTec data using real-time
STDC.
Category
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
AnomalyDiffusion
FAST
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
bottle
71.37
82.19
73.31
78.23
63.22
69.25
67.66
76.52
69.44
75.68
72.66
84.94
76.82
80.65
cable
42.88
54.74
50.02
58.38
49.38
57.80
57.74
62.86
35.97
38.81
59.43
74.22
54.85
60.26
capsule
21.73
30.72
36.31
41.68
22.91
27.18
25.60
27.96
31.08
34.25
22.90
26.06
49.35
55.29
carpet
50.79
66.68
66.28
76.70
63.18
77.85
58.58
71.83
57.48
68.51
56.16
68.47
64.52
75.02
grid
15.24
25.75
30.29
41.50
19.89
24.72
1.39
1.39
5.37
5.85
16.20
24.63
20.82
25.60
hazel_nut
58.48
65.59
78.75
83.66
68.57
85.83
81.77
84.66
70.16
82.40
61.83
92.42
87.96
93.99
leather
38.12
58.63
44.63
56.84
57.53
73.90
21.29
22.28
36.76
53.88
46.98
59.89
60.38
75.90
metal_nut
81.13
86.63
91.12
94.08
83.97
89.37
90.68
92.73
86.85
91.45
85.81
90.06
93.01
95.32
pill
50.00
60.28
55.47
61.05
44.48
48.11
80.41
82.55
63.96
65.96
78.23
84.35
82.15
86.48
screw
2.80
4.98
16.16
23.05
16.81
19.33
34.93
38.76
17.93
18.76
1.27
2.00
17.82
21.25
tile
69.86
78.18
84.75
91.31
79.86
88.65
85.36
89.72
70.29
77.70
76.96
84.07
86.29
93.89
toothbrush
41.19
52.81
53.72
76.55
37.46
40.91
36.78
38.94
33.85
43.03
35.39
48.93
75.76
87.32
transistor
58.24
68.80
65.57
80.31
62.64
69.32
78.38
87.23
62.57
72.45
71.96
83.28
93.01
96.05
wood
31.75
43.27
55.25
60.82
36.31
45.67
26.36
33.13
37.23
43.37
48.90
62.57
72.27
78.06
zipper
47.51
59.24
61.03
68.53
59.07
69.39
44.42
51.83
60.04
71.52
56.77
66.66
52.03
67.69
Average
45.41
55.90
57.51
66.18
51.02
59.15
52.76
58.81
49.27
56.24
52.76
63.50
65.80
72.85
A.7
More analysis of FARM
These improvements of FARM are not only empirically significant, but also consistent with intuitive
understanding. Without FARM, the segmentation-oriented industrial anomaly synthesis relies
15

[EQ eq_p14_000] -> see eq_p14_000.tex

[EQ eq_p14_001] -> see eq_p14_001.tex

[EQ eq_p14_002] -> see eq_p14_002.tex

[EQ eq_p14_003] -> see eq_p14_003.tex


# [PAGE 16]
Table 6: Ablation study of FARM on the MVTec dataset using the real-time Segformer.
Category
mIoU (w/o FARM) ↑
Acc (w/o FARM) ↑
mIoU (w/ FARM) ↑
Acc (w/ FARM) ↑
bottle
80.65
83.46
86.86
90.90
cable
65.99
70.50
73.71
77.94
capsule
49.08
53.25
63.22
71.12
carpet
72.46
80.84
73.84
83.53
grid
37.79
42.61
52.45
70.70
hazelnut
69.20
72.55
90.81
94.79
leather
61.42
65.91
66.60
74.18
metal_nut
89.59
94.31
94.65
96.88
pill
46.73
48.44
90.17
94.07
screw
46.48
54.42
49.94
57.48
tile
88.91
93.28
90.13
93.77
toothbrush
66.29
81.40
74.98
88.63
transistor
62.35
67.46
91.80
94.50
wood
71.55
79.47
78.77
86.31
zipper
71.40
80.76
72.80
84.73
Average
65.33
71.24
76.72
83.97
Table 7: Ablation Study of AIAS with other training-free sampling Methods on MVTec-AD data via
Segformer.
Category
DDPM (1000 steps)
DDIM (50 steps)
PLMS (50 steps)
AIAS (50 steps)
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
mIoU ↑
Acc ↑
bottle
81.65
84.83
82.87
86.03
81.49
84.44
86.86
90.90
cable
73.45
78.06
74.21
78.41
74.78
78.91
73.71
77.94
capsule
60.01
66.87
58.02
64.03
56.92
61.90
63.22
71.12
carpet
75.99
84.14
75.33
83.58
75.41
82.39
73.84
83.53
grid
50.91
63.19
50.85
67.91
50.43
61.42
52.45
70.70
hazel_nut
89.81
93.31
89.69
93.03
89.42
92.96
90.81
94.79
leather
71.03
80.32
66.00
72.48
71.85
81.47
66.60
74.18
metal_nut
94.63
97.18
94.50
96.47
93.93
96.69
94.65
96.88
pill
89.36
93.79
89.84
93.03
89.93
93.66
90.17
94.07
screw
49.35
59.18
48.89
57.26
48.78
55.62
49.94
57.48
tile
91.01
94.72
89.23
92.90
89.96
93.25
90.13
93.77
toothbrush
76.10
91.25
74.79
88.48
76.02
91.00
74.98
88.63
transistor
89.59
93.41
89.35
92.37
89.17
91.99
91.80
94.50
wood
80.03
85.30
79.29
84.03
79.61
84.65
78.77
86.31
zipper
72.45
82.35
71.01
83.00
72.06
81.02
72.80
84.73
Average
76.36
83.19
75.59
82.20
75.98
82.09
76.72
83.97
solely on frozen pre-trained weights and weak conditioning from learned textual embeddings.
This limits the model’s ability to capture the structural characteristics of industrial anomalies, often
leading to visually perturbed but semantically uninformative results. In contrast, FARM explicitly
reconstructs anomaly-only content from noisy latents and produce spatially localized, anomaly-aware
noise into the sampling process. Additionally, by incorporating both spatial masking and timestep
encoding, FARM guides the model to focus on abnormal regions—information that would
otherwise be uniformly treated in the absence of FARM. Together, these mechanisms improve the
structural fidelity, localization precision, and segmentation relevance of synthesized anomalies.
A.8
Other qualitative experiments
We also provide additional qualitative results to supplement the main paper. Specifically, we present
synthesized anomalies across multiple categories from MVTec and BTAD, along with comparisons
against CutPaste, DRAEM, GLASS, RealNet, DFMGAN, and AnomalyDiffusion. Each figure
includes both the generated images and their corresponding segmentation masks.
16

[EQ eq_p15_000] -> see eq_p15_000.tex

[EQ eq_p15_001] -> see eq_p15_001.tex

[EQ eq_p15_002] -> see eq_p15_002.tex


# [PAGE 17]
Mvtec AD
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
Anomaly Diffusion
FAST
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
Bottle
Cable
Capsule
Carpet
Grid
Figure 9: Visualization results of different anomaly synthesis methods on the MVTec dataset.
Columns correspond to synthesis methods (from left to right: MVTec AD, CutPaste, DRAEM,
GLASS, DFMGAN, RealNet, Anomaly Diffusion, FAST), and rows correspond to product cate-
gories (from top to bottom: bottle, cable, capsule, carpet, grid).
Mvtec AD
CutPaste
DRAEM
GLASS
DFMGAN
RealNet
Anomaly Diffusion
FAST
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(h)
Hazel_nut
Leather
Metal_nut
Pill
Screw
Figure 10: Visualization results of different anomaly synthesis methods on the MVTec dataset.
Columns correspond to synthesis methods (from left to right: MVTec AD, CutPaste, DRAEM,
GLASS, DFMGAN, RealNet, Anomaly Diffusion, FAST), and rows correspond to product cate-
gories (from top to bottom: hazel_nut, leather, metal_nut, pill, screw).
17
[FIGURE img_p16_199]
[FIGURE img_p16_200]
[FIGURE img_p16_201]
[FIGURE img_p16_202]
[FIGURE img_p16_203]
[FIGURE img_p16_204]
[FIGURE img_p16_205]
[FIGURE img_p16_206]
[FIGURE img_p16_207]
[FIGURE img_p16_208]
[FIGURE img_p16_209]
[FIGURE img_p16_210]
[FIGURE img_p16_211]
[FIGURE img_p16_212]
[FIGURE img_p16_213]
[FIGURE img_p16_214]
[FIGURE img_p16_215]
[FIGURE img_p16_216]
[FIGURE img_p16_217]
[FIGURE img_p16_218]
[FIGURE img_p16_219]
[FIGURE img_p16_220]
[FIGURE img_p16_221]
[FIGURE img_p16_222]
[FIGURE img_p16_223]
[FIGURE img_p16_224]
[FIGURE img_p16_225]
[FIGURE img_p16_226]
[FIGURE img_p16_227]
[FIGURE img_p16_228]
[FIGURE img_p16_229]
[FIGURE img_p16_230]
[FIGURE img_p16_231]
[FIGURE img_p16_232]
[FIGURE img_p16_233]
[FIGURE img_p16_234]
[FIGURE img_p16_235]
[FIGURE img_p16_236]
[FIGURE img_p16_237]
[FIGURE img_p16_238]
[FIGURE img_p16_239]
[FIGURE img_p16_240]
[FIGURE img_p16_241]
[FIGURE img_p16_242]
[FIGURE img_p16_243]
[FIGURE img_p16_244]
[FIGURE img_p16_245]
[FIGURE img_p16_246]
[FIGURE img_p16_247]
[FIGURE img_p16_248]
[FIGURE img_p16_249]
[FIGURE img_p16_250]
[FIGURE img_p16_251]
[FIGURE img_p16_252]
[FIGURE img_p16_253]
[FIGURE img_p16_254]
[FIGURE img_p16_255]
[FIGURE img_p16_256]
[FIGURE img_p16_257]
[FIGURE img_p16_258]
[FIGURE img_p16_259]
[FIGURE img_p16_260]
[FIGURE img_p16_261]
[FIGURE img_p16_262]
[FIGURE img_p16_263]
[FIGURE img_p16_264]
[FIGURE img_p16_265]
[FIGURE img_p16_266]
[FIGURE img_p16_267]
[FIGURE img_p16_268]
[FIGURE img_p16_269]
[FIGURE img_p16_270]
[FIGURE img_p16_271]
[FIGURE img_p16_272]
[FIGURE img_p16_273]
[FIGURE img_p16_274]
[FIGURE img_p16_275]
[FIGURE img_p16_276]
[FIGURE img_p16_277]
[FIGURE img_p16_278]
[FIGURE img_p16_279]
[FIGURE img_p16_280]
[FIGURE img_p16_281]
[FIGURE img_p16_282]
[FIGURE img_p16_283]
[FIGURE img_p16_284]
[FIGURE img_p16_285]
[FIGURE img_p16_286]
[FIGURE img_p16_287]
[FIGURE img_p16_288]
[FIGURE img_p16_289]
[FIGURE img_p16_290]
[FIGURE img_p16_291]
[FIGURE img_p16_292]
[FIGURE img_p16_293]
[FIGURE img_p16_294]
[FIGURE img_p16_295]
[FIGURE img_p16_296]
[FIGURE img_p16_297]
[FIGURE img_p16_298]
[FIGURE img_p16_299]
[FIGURE img_p16_300]
[FIGURE img_p16_301]
[FIGURE img_p16_302]
[FIGURE img_p16_303]
[FIGURE img_p16_304]
[FIGURE img_p16_305]
[FIGURE img_p16_306]
[FIGURE img_p16_307]
[FIGURE img_p16_308]
[FIGURE img_p16_309]
[FIGURE img_p16_310]
[FIGURE img_p16_311]
[FIGURE img_p16_312]
[FIGURE img_p16_313]
[FIGURE img_p16_314]
[FIGURE img_p16_315]
[FIGURE img_p16_316]
[FIGURE img_p16_317]
[FIGURE img_p16_318]
[FIGURE img_p16_319]
[FIGURE img_p16_320]
[FIGURE img_p16_321]
[FIGURE img_p16_322]
[FIGURE img_p16_323]
[FIGURE img_p16_324]
[FIGURE img_p16_325]
[FIGURE img_p16_326]
[FIGURE img_p16_327]
[FIGURE img_p16_328]
[FIGURE img_p16_329]
[FIGURE img_p16_330]
[FIGURE img_p16_331]
[FIGURE img_p16_332]
[FIGURE img_p16_333]
[FIGURE img_p16_334]
[FIGURE img_p16_335]
[FIGURE img_p16_336]
[FIGURE img_p16_337]
[FIGURE img_p16_338]
[FIGURE img_p16_339]
[FIGURE img_p16_340]
[FIGURE img_p16_341]
[FIGURE img_p16_342]
[FIGURE img_p16_343]
[FIGURE img_p16_344]
[FIGURE img_p16_345]
[FIGURE img_p16_346]
[FIGURE img_p16_347]
[FIGURE img_p16_348]
[FIGURE img_p16_349]
[FIGURE img_p16_350]
[FIGURE img_p16_351]
[FIGURE img_p16_352]
[FIGURE img_p16_353]
[FIGURE img_p16_354]
[FIGURE img_p16_355]
[FIGURE img_p16_356]
[FIGURE img_p16_357]
[FIGURE img_p16_358]

[EQ eq_p16_000] -> see eq_p16_000.tex

[EQ eq_p16_001] -> see eq_p16_001.tex


# [PAGE 18]
CutPaste
DRAEM
DFMGAN
Anomaly Diffusion
FAST
01
Btad
02
03
GLASS
RealNet
Figure 11: Visualization results of different anomaly synthesis methods on the BTAD dataset.
Columns correspond to synthesis methods (from left to right: MVTec AD, CutPaste, DRAEM,
GLASS, DFMGAN, RealNet, Anomaly Diffusion, FAST), and rows correspond to product cate-
gories.
18
[FIGURE img_p17_359]
[FIGURE img_p17_360]
[FIGURE img_p17_361]
[FIGURE img_p17_362]
[FIGURE img_p17_363]
[FIGURE img_p17_364]
[FIGURE img_p17_365]
[FIGURE img_p17_366]
[FIGURE img_p17_367]
[FIGURE img_p17_368]
[FIGURE img_p17_369]
[FIGURE img_p17_370]
[FIGURE img_p17_371]
[FIGURE img_p17_372]
[FIGURE img_p17_373]
[FIGURE img_p17_374]
[FIGURE img_p17_375]
[FIGURE img_p17_376]
[FIGURE img_p17_377]
[FIGURE img_p17_378]
[FIGURE img_p17_379]
[FIGURE img_p17_380]
[FIGURE img_p17_381]
[FIGURE img_p17_382]
[FIGURE img_p17_383]
[FIGURE img_p17_384]
[FIGURE img_p17_385]
[FIGURE img_p17_386]
[FIGURE img_p17_387]
[FIGURE img_p17_388]
[FIGURE img_p17_389]
[FIGURE img_p17_390]
[FIGURE img_p17_391]
[FIGURE img_p17_392]
[FIGURE img_p17_393]
[FIGURE img_p17_394]
[FIGURE img_p17_395]
[FIGURE img_p17_396]
[FIGURE img_p17_397]
[FIGURE img_p17_398]
[FIGURE img_p17_399]
[FIGURE img_p17_400]
[FIGURE img_p17_401]
[FIGURE img_p17_402]
[FIGURE img_p17_403]
[FIGURE img_p17_404]
[FIGURE img_p17_405]
[FIGURE img_p17_406]
