

# [PAGE 1]
1
HiPerformer: A High-Performance Global–Local
Segmentation Model with Modular Hierarchical
Fusion Strategy
Dayu Tan, Zhenpeng Xu, Yansen Su, Xin Peng, Chunhou Zheng, and Weimin Zhong
Abstract—Both local details and global context are crucial in
medical image segmentation, and effectively integrating them is
essential for achieving high accuracy. However, existing main-
stream methods based on CNN-Transformer hybrid architectures
typically employ simple feature fusion techniques such as serial
stacking, endpoint concatenation, or pointwise addition, which
struggle to address the inconsistencies between features and
are prone to information conflict and loss. To address the
aforementioned challenges, we innovatively propose HiPerformer.
The encoder of HiPerformer employs a novel modular hierarchi-
cal architecture that dynamically fuses multi-source features in
parallel, enabling layer-wise deep integration of heterogeneous
information. The modular hierarchical design not only retains the
independent modeling capability of each branch in the encoder,
but also ensures sufficient information transfer between layers,
effectively avoiding the degradation of features and information
loss that come with traditional stacking methods. Furthermore,
we design a Local-Global Feature Fusion (LGFF) module to
achieve precise and efficient integration of local details and
global semantic information, effectively alleviating the feature
inconsistency problem and resulting in a more comprehensive
feature representation. To further enhance multi-scale feature
representation capabilities and suppress noise interference, we
also propose a Progressive Pyramid Aggregation (PPA) module
to replace traditional skip connections. Experiments on eleven
public datasets demonstrate that the proposed method outper-
forms existing segmentation techniques, demonstrating higher
segmentation accuracy and robustness. The code is available at
https://github.com/xzphappy/HiPerformer.
Index Terms—Medical image segmentation, modular hierar-
chical strategy, local-global feature fusion, progressive pyramid
aggregation
I. INTRODUCTION
P
RECISE medical image segmentation is essential for
accurately assessing lesion extent. The detailed informa-
tion it provides enables physicians to gain a comprehensive
understanding of the patient’s condition, thereby facilitating
the development of more targeted and effective treatment plans
This work was supported in part by the National Key Research and
Development Program of China (2021YFE0102100), in part by National
Natural Science Foundation of China (62303014, 62172002, 62322301).
(Corresponding author: Yansen Su.)
Dayu Tan, Zhenpeng Xu, Yansen Su, and Chunhou Zheng are with
the Key Laboratory of Intelligent Computing and Signal Processing,
Ministry of Education, Anhui University, Hefei 230601, China (e-mail:
suyansen@ahu.edu.cn).
Xin Peng and Weimin Zhong are with the Key Laboratory of Smart
Manufacturing in Energy Chemical Process, Ministry of Education, East
China University of Science and Technology, Shanghai 200237, China (e-
mail: wmzhong@ecust.edu.cn).
and significantly improving therapeutic outcomes. In multi-
region and fine-grained medical images, the anatomical struc-
tures are complex, and morphological differences are subtle.
Different tissues or lesions are extremely similar in grayscale
and texture features, and they are often adjacent to each
other, resulting in blurred segmentation boundaries. Target
structures such as tumors or lesions, which have small volumes
with significantly fewer pixels compared to normal tissues,
are prone to being overlooked or misclassified due to their
low resolution and limited semantic information. Furthermore,
the background of medical images often contains substantial
extraneous noise originating from scanning artifacts, device
interference, or surrounding tissues, which can lead to mis-
segmentation. The aforementioned challenges make the seg-
mentation of multi-region and fine-grained medical images
particularly difficult. Traditional machine learning methods
[1], [2] still require manual feature selection, and segmentation
performance depends heavily on how accurately those features
are chosen, making it difficult to handle complex segmentation
tasks that require rich semantic information. In recent years,
deep learning techniques are widely applied to medical image
segmentation, effectively reducing subjective errors introduced
by human judgment, improving annotation accuracy and con-
sistency, and markedly enhancing segmentation performance.
U-Net [3] adopts an encoder-decoder architecture with a
U-shaped structure, significantly enhancing the accuracy of
medical image segmentation, and has become a classic and
widely used model in this field. It is based on a pure CNN
architecture, which efficiently captures the local details and
texture information of images through the sliding window of
convolutional kernels. However, limited by CNN’s receptive
field, it struggles to model long-range dependencies. In con-
trast, the Transformer employs a self-attention mechanism that
effectively captures long-range dependencies among pixels
at a global scale. The global modeling capability has been
integrated into image segmentation tasks, resulting in the
emergence of Swin-UNet [4]. Built upon a pure Transformer-
based framework, Swin-UNet utilizes a U-shaped architecture
to incorporate global contextual information effectively, yet its
capability to perceive local details remains limited.
To fully leverage the respective advantages of CNN and
Transformer, recent studies explore the integration of both
architectures for image segmentation, aiming to preserve both
local details and global semantic context, thereby enhancing
overall performance. However, most existing mainstream hy-
brid architectures adopt simple approaches such as serial stack-
arXiv:2509.20280v1  [cs.CV]  24 Sep 2025

[EQ eq_p0_000] -> see eq_p0_000.tex


# [PAGE 2]
2
ing (e.g., TransUnet [5]), concatenation at the end of network
(e.g., FAT-Net [6]), or pointwise addition (e.g., MixFormer [7])
and therefore neglect inconsistencies between convolutional
and transformer-derived feature representations. Such neglect
hampers the ability to balance the contributions of each feature
type and can introduce feature conflicts and redundancy. In
addition, existing studies commonly lack mechanisms for
cross-level interaction, resulting in a separation between local
details and global semantics, information loss during feature
propagation, and an inability to achieve comprehensive and
efficient fusion.
To more effectively address the aforementioned problem, we
propose HiPerformer, a novel U-shaped network architecture.
The architecture employs a modular hierarchical design in the
encoder and effectively fuses CNN and Transformer features
via a Local and Global Feature Fusion module, mitigating
feature conflicts and information loss. In the output stage,
a fusion mechanism is employed to combine features from
four different stages, fully retaining multi-scale information
and enhancing spatial detail and edge representation capa-
bility. Additionally, we replace traditional skip connections
with a Progressive Pyramid Aggregation module, effectively
suppressing noise propagation. The main contributions of our
research can be summarized as follows:
• This study presents a novel modular hierarchical en-
coder that progressively integrates multi-source informa-
tion layer by layer. The hierarchical fusion mechanism
ensures effective information propagation across layers,
preventing feature degradation associated with conven-
tional stacked fusion. Meanwhile, each branch of the
encoder preserves its independent modeling capability,
avoiding the loss of important information during fusion
and preventing disruption of internal structures.
• We design a Local and Global Feature Fusion (LGFF)
module that can efficiently and accurately fuse local
detail features with global semantic information. The
module mitigates feature inconsistency and achieves a
more comprehensive and refined feature representation,
thereby significantly improving the model’s accuracy and
robustness in complex scenarios.
• We propose a Progressive Pyramid Aggregation (PPA)
module to replace traditional skip connections. The PPA
module not only performs progressive multiplicative fu-
sion of shallow and deep features to amplify feature
differences and narrow the semantic gap, but also further
enhances feature representation capability while sup-
pressing interference from irrelevant regions, enabling the
network to focus more on key regions.
II. RELATED WORK
A. Hybrid CNN-Transformer Segmentation Network
An increasing number of studies capture both local details
and global contextual information simultaneously to improve
model performance. For example, TransUNet [5] combines
convolutional neural networks and Transformers for medical
image segmentation, significantly improving segmentation ac-
curacy. DA-TransUnet [8] further introduces dual attention into
TransUNet, greatly enhancing feature extraction capability.
However, the aforementioned models are constructed by serial
stacking, and a main limitation is that each layer models
only one type of dependency (global or local), causing global
information to be lost during local modeling and making
features prone to being overwhelmed. Moreover, such stacked
architectures require very deep networks, which readily lead
to feature attenuation.
FAT-Net [6] and MixFormer [7] adopt parallel architec-
tures to optimize feature fusion strategies; however, such
methods typically perform fusion using only simple end-
stage concatenation or point-wise addition, which does not
adequately account for inconsistencies between features and
easily induces feature conflict and redundancy. The Performer
[9] model introduces an innovative interaction mechanism
designed for CNN and Transformer architectures to mutually
enhance image feature extraction capabilities. TransFuse [10]
proposes a novel fusion technique whose core component,
the BiFusion module, effectively integrates multilevel fea-
tures from the CNN and Transformer branches, enabling
capture of global information without relying on extremely
deep networks while maintaining high sensitivity to low-level
details. To further mitigate inter-feature inconsistency, it is
necessary to continue exploring new hybrid CNN-Transformer
segmentation architectures.
B. Skip connections in U-shaped architectures
Skip connections improve the recovery of spatial detail
by introducing high-resolution features from the encoder into
the decoder; they also help alleviate vanishing gradients and
stabilize model training. In recent years, many improvements
to skip connections are proposed. Perspective+ [11] designs a
Spatial Cross-Scale Integrator (SCSI) in its skip connections
to enable coherent fusion of information across stages and
better preserve fine-grained details. FSCA-Net [12] employs
a Parallel Attention Transformer (PAT) to strengthen spatial
and channel feature extraction within skip connections, further
reducing information loss caused by downsampling. SUet [13]
proposes an efficient feature fusion (EFF) module based on
multi-attention to achieve better fusion between skip connec-
tions and decoder features in U-shaped network.
UNet++ [14] addresses the semantic gap between encoder
and decoder features by introducing nested, densely con-
nected skip pathways and deep supervision, thereby producing
more compatible fused features. DenseUNet [15] significantly
enhances feature reuse, gradient propagation, and parameter
efficiency by incorporating Dense Blocks and dense connec-
tivity into both the encoder and decoder paths of U-Net.
The effectiveness of traditional skip connections is largely
constrained by the quality of the encoder’s extracted features:
when the encoder’s representational capacity is limited, sub-
stantial irrelevant background noise can be passed directly to
the decoder through the skip connections, degrading the image
reconstruction process. In addition, discrepancies in feature
distributions between the encoder and decoder exacerbate this
semantic gap. Therefore, optimizing and redesigning skip-
connection mechanisms to improve feature fusion, suppress


# [PAGE 3]
3
Fig. 1.
Illustration of the proposed HiPerformer. We utilize a modular hierarchical fusion strategy to redesign the encoder component, employ the Local
and Global Feature Fusion (LGFF) module to efficiently and accurately merge local detail features with global semantic information, and adopt Progressive
Pyramid Aggregation (PPA) to replace the traditional skip connections.
noise transmission, and achieve more precise semantic align-
ment between the encoder and the decoder constitutes an
important direction for our future research.
C. Spatial and Channel attention
Spatial attention [16] mechanisms enhance the representa-
tion of important regions by focusing on key areas in an image
and dynamically adjusting the feature weights at different
spatial locations. Channel attention mechanisms concentrate
on the interrelationships among feature channels, typically
employing squeeze-and-excitation operations to learn the im-
portance weights of individual channels and thereby amplify
responses of effective channels. For example, the core idea
of the SE [17] attention mechanism is to adaptively assign
distinct weights to each channel to strengthen useful features
and suppress irrelevant information.
CBAM [18] integrates both channel and spatial attention
mechanisms and cleverly fuses average pooling and max
pooling to aggregate information. However, two important
drawbacks remain. First, it fails to capture spatial information
at multiple scales to enrich the feature space. Second, the spa-
tial attention focuses only on local regions and cannot establish
long-range dependencies. In contrast, the PSA [19] module
can handle spatial information of multi-scale input feature
maps and effectively establish long-range dependencies among
multi-scale channel attention. The PSA module module di-
rectly computes attention weights along the spatial dimension,
typically via point-wise operations, and emphasizes capturing
fine-grained spatial information with a lightweight structure
and high computational efficiency. Nevertheless, during multi-
scale feature fusion, PSA lacks adaptive modeling of semantic
associations across different scales. Therefore, it is necessary
to further optimize its attention mechanism to enhance feature
discriminability and semantic awareness, thereby improving
the overall representational performance of the model.
III. PROPOSED METHOD
A. Overall Architecture
The overall network architecture of the proposed HiPer-
former is illustrated in Fig. 1. To fully integrate local and
global information while minimizing feature conflicts, the
encoder adopts a modular hierarchical design with three
branches, each retaining its own independent modeling capa-
bility. The Local branch is dedicated to extracting local de-
tailed information. Specifically, it first applies a convolutional
layer with a 7 × 7 kernel and a stride of 2, reducing the input
image to half its original size. It then performs four stages of
local feature extraction, each consisting of a max-pooling layer
followed by a DuChResBlock. The DuChResBlock employs a
dual-channel architecture that integrates standard convolution
with dilated convolution and incorporates a residual connec-
tion, as shown in the left-hand side of Fig. 2. Such a design
preserves local feature extraction capabilities while effectively
expanding the receptive field, thereby enhancing the model’s
ability to capture fine-grained details. The specific computation
process of DuChResBlock can be expressed as follows:
xt
s+1 = xs + fs (fs (xs)) ,
(1)
[FIGURE img_p2_000]

[EQ eq_p2_000] -> see eq_p2_000.tex


# [PAGE 4]
4
xd
s+1 = xs + f k
s
 f k
s (xs)

,
(2)
xs+1 = Conv(Concat[xt
s+1, xd
s+1]),
(3)
in Eqs. (1), (2), and (3), xd
s+1, xd
s+1, and xs+1 denote the
features captured by standard convolution, dilated convolution,
and the resulting merged features in stage s + 1, respectively.
fs and f k
s denote the standard convolution and dilated con-
volution with dilation rate k in stage s. The symbol Conv
denotes a 1 × 1 convolution operation.
Fig. 2.
Feature extraction module. On the left is the fine-grained feature
extraction module, and on the right is the coarse-grained feature extraction
module.
The Global branch consists of four stages comprising 2, 2,
18, and 2 Swin Transformer blocks, respectively, to capture
global contextual information. Unlike the traditional Multi-
Head Self-Attention (MSA) module, the Swin Transformer
block is constructed based on shifted windows, as shown in
the right-hand side of Fig. 2. ˆzl and zl represent the outputs
of the window based multi-head self attention (W-MSA) and
MLP at the i-th layer, respectively, and can be expressed by
the following formulas:
ˆzl = W-MSA
 LN(zl−1)

+ zl−1,
(4)
zl = MLP
 LN(ˆzl)

+ ˆzl.
(5)
The output after applying the shifted window partitioning
strategy, through the shifted window-based multi-head self
attention (SW-MSA) and MLP modules, can be expressed by
Eqs. (6) and (7), respectively.
ˆzl+1 = SW-MSA
 LN(zl)

+ zl,
(6)
zl+1 = MLP
 LN(ˆzl+1)

+ ˆzl+1.
(7)
Each stage of the Local-Global fusion branch consists of
an LGFF module that integrates both local and global features
from the current layer with the fused features from the previ-
ous layer, thereby enabling multi-level feature aggregation.
In the bridge layer, we propose a novel PPA module. The
PPA module integrates multi-scale features to mitigate the
semantic gap while enhancing feature representation capability
and suppressing interference from irrelevant information. To
enhance detail representation and avoid boundary blurring,
we additionally employ a multi-scale fusion strategy at the
final stage of the decoder. Specifically, we first adjust the
channel dimensions of outputs from four scales using 1 ×
1 convolutions, then upsample them to a unified resolution,
and finally fuse them through element-wise addition.
B. LGFF: Local and Global Feature Fusion
We innovatively proposed the LGFF module, as shown in
Fig. 3a. Here, Gi denotes the feature matrix output by the
Transformer global feature block, Li represents the feature
matrix output by the CNN local feature block, Fi−1 signifies
the output feature matrix from the previous stage’s LGFF
module, and Fi indicates the feature matrix generated through
fusion in the current stage. The LGFF module efficiently
integrates local features captured by the Local branch, global
dependencies obtained by the Global branch, and semantic
information from preceding layers. By effectively resolving
feature learning inconsistencies within the same stage while
mitigating feature discrepancies, it achieves a more compre-
hensive feature representation.
Fig. 3. Structure diagram of the LGFF module and its submodules. (a) Local
and global feature fusion (LGFF). (b) Adaptive channel interaction (ACI).
(c) Spatial perception enhancement (SPE). (d) Inverted residual multilayer
perceptron (IRMLP).
Each channel map of high-level features contains category-
specific semantic responses, which exhibit a certain degree
of inter-channel correlation. By modeling the dependencies
between channels, relevant feature maps can be reinforced,
thereby enhancing the representation capability of specific
semantics. To achieve that goal, we introduce a Adaptive
channel interaction (ACI) for processing global features, as
illustrated in Fig. 3b. The ACI module dynamically models
inter-channel dependencies, adjusts the distribution of chan-
nel weights, and strengthens features related to the target
semantics, ultimately improving the discriminative power of
the global semantic representation. The formula of ACI can
be expressed as follows:
ACI(x) = x + R (Softmax (R(x) · R&T(x)) · R(x)) , (8)
where Softmax is the activation function, R denotes the
reshape operation, and T denotes the transpose operation.
[FIGURE img_p3_001]
[FIGURE img_p3_002]

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


# [PAGE 5]
5
We also introduced a spatial perception enhancement (SPE)
to enhance the processing of local features, as illustrated in
Fig. 3c. The SPE module employs two 7 × 7 convolutional
layers to integrate spatial information and incorporates a
reduction ratio r to control the number of feature channels. By
focusing on key areas and suppressing irrelevant background
interference, it effectively enhances the representation of local
details, thereby improving the ability to characterize fine-
grained features. The formula of SPE can be expressed as
follows:
SPE(x) = x · Sigmoid (Conv7×7 (Conv7×7(x))) ,
(9)
where Sigmoid denotes the activation function, and Conv7×7
represents convolution operation with kernel sizes of 7 × 7.
The
inverted
residual
multilayer
perceptron
(IRMLP)
employs an inverted residual structure, combining high-
dimensional representations with depthwise separable convo-
lutions to effectively extract features with strong expressive
power, as shown in Fig. 3d. The core of IRMLP in placing
nonlinear operations in high-dimensional space, thus avoiding
information loss caused by low-dimensional activation func-
tions. The inverted residual structure not only enhances the
efficiency of feature extraction but also effectively mitigates
problems such as gradient vanishing and network degradation,
facilitating stable training and performance improvement of
deep networks. The formula of IRMLP can be expressed as
follows:
IRMLP(x) = Conv1×1(Conv1×1(DWConv3×3(x) + x)),
(10)
in Eq. (10), Conv1×1 denotes convolution operation with
kernel sizes of 1 × 1, and DWConv3×3 denotes depthwise
separable convolution operation with a kernel size of 3 × 3.
The overall process of LGFF can be expressed by the
following formula:
Fmid1 = Avgpool(Conv1×1(Fi−1)),
(11)
Fmid2 = Conv1×1(Concat[Li, Fmid1, Gi]),
(12)
Fi = IRMLP (Concat [ACI(Li), Fmid2, SPE(Gi)])+Fmid1,
(13)
herein, Avgpool denotes the average pooling operation, and
Fmid1 and Fmid2 denote the features generated at intermediate
stages.
C. PPA: Progressive Pyramid Aggregation
We propose a novel Progressive Pyramid Aggregation (PPA)
module to replace traditional skip connections in order to
reduce the semantic gap and suppress noise interference, as
shown in the middle part of Fig. 1. The PPA module is
primarily composed of a Progressive Multiplicative Integration
(PMI) module and a Pyramid Gated Attention (PGA) module.
1) Progressive Multiplicative Integration (PMI): In U-Net
architectures, features extracted by deep encoder layers are
rich in high-level semantic information, whereas features from
shallow encoder layers are better at capturing fine-grained
boundary details. To bridge the semantic gap between them,
existing methods typically fuse deep and shallow features.
However, shallow encoder features often contain substan-
tial background noise, causing most fusion strategies to be
interfered by irrelevant information and thereby degrading
final segmentation performance. To address the aforemen-
tioned issue, we design the PMI module. The PMI module
employs cascading multiplication to progressively fuse deep
and shallow features through skip connections, enabling the
aggregation of multi-scale semantic information. Specifically,
deep features first undergo an upsampling process, and then
both deep features and shallow features are simultaneously
processed through a 3 × 3 convolutional layer and a 1 × 1
convolutional layer. The outputs are then fused using element-
wise multiplication. Such multiplicative fusion amplifies the
differences between noisy and normal regions, effectively sup-
pressing background noise in shallow features and improving
feature representation. The output after processing the outputs
xi (i=1, 2, 3, 4) of the four-level encoder through the PMI
module is represented as follows:
yi =
(
f(xi) · f(Up(yi+1)),
i = 1, 2, 3
x4,
i = 4
,
(14)
in Eq. (14), f(·) represents 3 × 3 and 1 × 1 convolution
used for channel adjustment and feature enhancement, and Up
denotes upsampling.
2) Pyramid Gated Attention (PGA): We propose a novel
PGA module (Fig. 4a), which is composed of two primary sub-
modules: an EAG module and a PSA module. The EAG (Fig.
4b) module concatenates semantic features from the bridging
layer and the decoder, doubling the feature dimensionality.
A residual connection atop the AG mitigates the influence
of high-level semantics on low-level semantics when input
feature correlation is weak, thereby enhancing the semantic
expressiveness of the concatenated features and effectively
suppressing interference from irrelevant regions. The formula
for EAG can be expressed as follows:
EAG(e, d) = d + d × Sigmoid (Conv1×1 (ReLU(Fe + Fd))) ,
(15)
Fe = ReLU (BN (GpConv1×1(e))) ,
(16)
Fd = ReLU (BN (GpConv1×1(d))) ,
(17)
herein, the Sigmoid and ReLU denote activation func-
tions, and BN represents batch normalization. Conv1×1 and
GpConv1×1 refer to standard 1 × 1 convolution and group
convolution respectively, while e and d represent features from
the bridging layer and decoder.
If the concatenated features are directly processed after-
wards, it may result in a significant loss of useful information.
To further emphasize key features and extract richer feature
information, we feed the feature information obtained from
EAG into the PSA module to more effectively guide the
network in focusing on relevant areas, 0achieving accurate
localization of lesions.

[EQ eq_p4_000] -> see eq_p4_000.tex

[EQ eq_p4_001] -> see eq_p4_001.tex

[EQ eq_p4_002] -> see eq_p4_002.tex

[EQ eq_p4_003] -> see eq_p4_003.tex

[EQ eq_p4_004] -> see eq_p4_004.tex

[EQ eq_p4_005] -> see eq_p4_005.tex


# [PAGE 6]
6
Fig. 4. (a) Structure diagrams of PGA. (b) Structure diagrams of EAG.
D. Loss Function
We use a weighted sum of the cross-entropy loss and the
Dice loss as the loss function to balance pixel-level accuracy
and global region consistency. The formula is as follows:
L = αLCE + (1 −α)LDice,
(18)
the hyperparameter α controls the relative weighting between
LCE and LDice in the final loss function. The formula for
LDice is presented in Eq. (19):
LDice = 1 −2|X ∩Y |
|X| + |Y |,
(19)
where X and Y represent the cardinalities of the ground truth
and predicted values, respectively.
The formula for LCE is presented in Eq. (20):
LCE = −1
N
N
X
i=0
C
X
c=0
yi,c log(ˆyi,c),
(20)
herein, N and C represent the total number of samples and
classes, respectively. yi,c is an indicator variable that equals
1 when sample i belongs to class c, and 0 otherwise. ˆyi,c
represents the predicted probability of sample i belonging to
class c.
E. Evaluation Metrics
To assess the performance of our approach across all
datasets, we primarily employ the mean Dice Similarity Co-
efficient (DSC) and the mean Hausdorff distance at the 95th
percentile (HD95) as our evaluation metrics. The formulas are
as follows:
DSC = 2|X ∩Y |
|X| + |Y |,
(21)
H(X, Y ) = max

max
x∈X min
y∈Y d(x, y), max
y∈Y min
x∈X d(x, y)

,
(22)
where X represents the ground truth and Y represents the
predicted values, d(x, y) denotes the distance between pixel
point x and pixel point y.
For the Chase, STARE, and LES-AV datasets, we also add
two evaluation metrics: average Recall and mean IOU. The
formulas are shown below:
Recall =
TP
FN + TP,
(23)
IoU =
TP
FP + FN + TP,
(24)
herein, true positive (TP) denotes the number of pixels cor-
rectly segmented as lesions, false negative (FN) is the number
of true lesions pixels that are not segmented as normal tissue,
and false positive (FP) denotes the number of background
pixels wrongly segmented as lesions.
IV.
EXPERIMENTS
A. Datasets
We conduct experiments on eleven datasets, primarily fo-
cusing on multi-region and fine-grained segmentation. Below
is a brief description of each dataset:
1) Synapse: The dataset is derived from the MICCAI 2015
Multi-Atlas Abdomen Labeling Challenge [20]. It includes
abdominal CT scans from 30 patients, comprising a total of
3779 axial clinical CT images covering eight abdominal or-
gans. In this study, all input images are uniformly resized and
standardized to a resolution of 224 × 224 pixels. Consistent
with previous research [5], the dataset is randomly divided
into 18 cases (2212 axial slices) for training and 12 cases for
testing.
2) ACDC: The dataset is derived from the Automatic
Cardiac Diagnosis Challenge [21] and provides manually
annotated images for three regions, with pixel sizes varying
between 0.83 and 1.75 mm2. In this study, all input images are
uniformly standardized to a resolution of 224 × 224 pixels.
The dataset is randomly divided into 80 training cases (1510
axial slices) and 20 testing cases.
3) BTCV: The dataset from the BTCV Challenge [20]
focuses on the segmentation of 13 abdominal organs. The in-
plane resolution of these images varies from 0.54 × 0.54 mm2
to 0.98 × 0.98 mm2, with slice thickness ranging from 2.5
[FIGURE img_p5_003]

[EQ eq_p5_000] -> see eq_p5_000.tex

[EQ eq_p5_001] -> see eq_p5_001.tex

[EQ eq_p5_002] -> see eq_p5_002.tex

[EQ eq_p5_003] -> see eq_p5_003.tex

[EQ eq_p5_004] -> see eq_p5_004.tex

[EQ eq_p5_005] -> see eq_p5_005.tex

[EQ eq_p5_006] -> see eq_p5_006.tex

[EQ eq_p5_007] -> see eq_p5_007.tex

[EQ eq_p5_008] -> see eq_p5_008.tex

[EQ eq_p5_009] -> see eq_p5_009.tex

[EQ eq_p5_010] -> see eq_p5_010.tex


# [PAGE 7]
7
Fig. 5. The visualization results of HiPerformer and eleven comparison methods for image segmentation on eleven datasets, as well as the original image
and GroundTruth. Incorrect segmentation areas are marked with red and blue boxes.
[FIGURE img_p6_004]


# [PAGE 8]
8
mm to 5.0 mm. In this study, we select a dataset of 30 cases,
with input images standardized to a resolution of 224 × 224
pixels. The dataset is randomly divided into 24 training cases
(1740 axial slices) and 6 testing cases.
4) AMOS: The dataset originates from the 2022 Multi-
Modality Abdominal Multi-Organ Segmentation Challenge
[22]. We select Task 1—Abdominal Organ Segmentation (CT
imaging only), where each case includes voxel-level annota-
tions for 15 abdominal organs. In this study, we randomly
select 50 cases and divide them into 40 training cases (5783
axial slices) and 10 test cases. All input images in the dataset
are standardized to a resolution of 224 × 224 pixels.
5) SegTHOR: The dataset is from the ISBI 2019 Challenge
on Segmentation of Thoracic Organs at Risk in CT Images
[23]. It contains CT scans of 40 cases, with each scan sized
at 512 × 512 pixels, and each case includes voxel-level
annotations for four thoracic organs. In this study, all input
images are uniformly standardized to a resolution of 224 ×
224 pixels. The dataset is randomly divided into 32 training
cases (5995 axial slices) and 8 test cases.
6) RAOS: The dataset [24] is an abdominal multi-organ
segmentation collection that includes cases with organ absence
or morphological abnormalities. It comprises Computed To-
mography (CT) images from 413 abdominal tumor patients
who had undergone surgery or radiotherapy/chemotherapy,
with annotations for 17 (female) or 19 (male) abdominal
organs. Each case includes voxel-wise annotations for 19
abdominal organs. In this study, we randomly select a subset
of 40 cases from the dataset. Input images are standardized
to a resolution of 224 × 224 pixels. The dataset is randomly
divided into 32 cases (5171 axial slices) for training and 8
cases for testing.
7) MM-WHS: The dataset is provided by the Multi-
Modality Whole Heart Challenge 2017 [25]. Each sample
comprises seven substructures of the heart. The in-plane
average resolution of the images is 0.44 × 0.44 mm2, with
an average slice thickness of 0.60 mm. In this study, we
select 40 cases as the dataset. All input images are uniformly
standardized to a resolution of 224 × 224 pixels. The dataset
is randomly split into 32 training cases (5542 axial slices) and
8 testing cases.
8) Refuge: The dataset [26] originates from the MICCAI
2018 Retinal Fundus Glaucoma Challenge, with the main
task being optic disc and cup segmentation. It contains 1200
fundus images. In this study, all input images are uniformly
standardized to a resolution of 224 × 224 pixels. The dataset
is randomly split into 960 cases for training and 240 cases for
testing.
9) Chase: The dataset [27] contains a total of 28 retinal
fundus images from 28 subjects, each with a resolution of
999 × 960 pixels. All images are accompanied by manually
annotated vessel segmentation labels created by professional
medical personnel. In this study, all input images are uniformly
standardized to a resolution of 384 × 384 pixels. The dataset
is randomly split into 22 cases for training and 6 cases for
testing.
10) STARE: The dataset is a publicly available retinal
vessel segmentation dataset, consisting of 20 fundus images
with a resolution of 700 × 605 pixels. The images cover
various types of lesions, including macular degeneration, hy-
pertensive retinopathy, and diabetic retinopathy. Each image
is accompanied by professionally hand-annotated vessel seg-
mentation maps. In this study, all input images are uniformly
standardized to a resolution of 384 × 384 pixels. The dataset
is randomly split into 16 cases for training and 4 cases for
testing.
11) LES-AV: The dataset [28] contains fundus images from
22 different patients. Among them, 21 images have a field
of view (FOV) of 30 degrees with a resolution of 1444 ×
1620 pixels, while one image has a FOV of 45 degrees and
a resolution of 1958 × 2196 pixels. Each pixel corresponds
to a physical size of 6 micrometers. In this study, all input
images are uniformly standardized to a resolution of 384 ×
384 pixels. The dataset is randomly split into 18 cases for
training and 4 cases for testing.
B. Implementation Details
We implement our HiPerformer model using PyTorch on an
NVIDIA RTX 3090 GPU with 24GB of memory. The model
is trained using the AdamW optimizer with an initial learning
rate of 1e-4 and a weight decay of 5e-4. A cosine annealing
learning rate scheduler is employed, setting T max to 100
and eta min to 1e-6, over a total of 300 training iterations. To
prevent gradient explosion and ensure stable training, gradient
clipping is implemented. Furthermore, to mitigate overfitting,
a series of data augmentation techniques are applied to the
dataset prior to inputting images into the model, which include
random flipping and rotations at multiple angles. For the
Chase, STARE, and LES-AV datasets, the batch size is set
to 4, while for the other datasets, the batch size is set to 16.
C. Comparison Experiments
To demonstrate the effectiveness of our model in segmen-
tation tasks, we evaluate the performance of HiPerformer by
comparing it with seven state-of-the-art methods: UNet [3],
Attention UNet [29], Trans-UNet [5], UCTransNet [30], Swin-
UNet [4], SwinPA-Net [31], HTC-NET [32], CSWin-UNet
[33], SCUNet++ [34], and SWMA-UNet [35], DA-TransUnet
[8]. For each dataset, we assess the DSC for each organ as
well as the average DSC and average HD95 across all organs.
The best results are highlighted in bold.
1) Experiments on the Synapse dataset: Table I presents
the comprehensive evaluation results of our method on the
Synapse dataset. Compared to existing methods, our approach
achieves superior performance in both the average DSC and
HD95 metrics, reaching 83.93% and 11.582, respectively. In
the segmentation tasks of specific organs, our method attains
the best performance on the Gallbladder, Right Kidney, Liver,
Spleen, and Stomach.
The first column in Fig. 5 shows a qualitative comparison
of HiPerformer with existing segmentation methods on the
Synapse dataset. It can be observed that other existing methods
often suffer from mis-segmentation of background areas or
incomplete segmentation, especially for the Pancreas organ
labeled in pink. In contrast, our method’s predictions are closer


# [PAGE 9]
9
TABLE I
COMPARSION RESULT ON THE SYNAPSE DATASET
Method
DSC ↑
HD95 ↓
Aorta
Gallbladder
Kidney(L)
Kidney(R)
Liver
Pancreas
Spleen
Stomach
(%, mean)
(mm, mean)
DSC ↑
UNet
76.83
29.877
87.44
66.42
78.98
72.03
92.41
56.47
86.56
74.32
Attention UNet
79.70
31.870
89.91
71.57
82.39
73.04
94.19
59.39
90.21
76.93
Trans-UNet
79.32
22.773
87.69
61.92
84.93
82.76
94.30
60.91
88.95
73.10
Swin-UNet
78.12
19.558
84.99
64.96
83.81
80.66
93.96
54.26
89.01
73.33
UCTransNet
79.02
27.509
87.91
64.72
84.14
74.63
93.54
62.77
89.95
74.52
SwinPA-Net
81.11
25.268
86.73
70.84
82.29
80.67
94.22
68.47
88.56
77.12
HTC-NET
81.75
28.826
88.98
67.35
85.21
83.42
94.82
62.68
91.45
80.10
CSWin-UNet
82.23
20.565
87.84
69.03
87.88
81.75
94.77
64.63
90.73
81.24
SCUNet++
80.73
21.531
87.26
67.90
86.22
82.82
93.90
60.32
89.46
77.95
SWMA-Unet
79.26
25.206
87.14
68.60
84.50
81.21
94.19
55.61
87.66
75.19
DA-TransUnet
79.76
22.047
87.92
68.38
85.46
81.98
94.28
58.50
86.44
75.14
Ours
83.93
11.582
88.39
71.58
86.58
85.11
95.20
68.21
92.15
84.20
TABLE II
COMPARSION RESULT ON THE BTCV DATASET
Method
DSC ↑
HD95 ↓
Spleen Kidney(R) Kidney(L) Gallbladder Esophagus Liver Stomach Aorta IVC HPV&SV Pancreas RAG LAG
(%, mean) (mm, mean)
DSC ↑
UNet
71.74
34.054
82.93
74.65
83.69
59.57
71.41
92.58
77.47
82.35 71.94
65.31
63.28
51.28 56.05
Attention UNet
73.55
27.550
84.96
85.53
85.91
64.48
70.82
93.69
76.92
85.15 75.25
65.64
58.60
56.26 55.91
Trans-UNet
69.43
16.227
89.73
86.29
89.06
58.78
70.10
93.78
66.97
82.26 69.61
62.61
50.97
49.55 32.88
Swin-UNet
69.71
15.949
89.12
77.74
81.76
59.45
68.23
93.93
81.23
81.70 67.37
54.40
55.15
50.82 45.36
UCTransNet
68.89
32.216
78.81
76.91
81.95
50.88
68.49
92.30
72.02
84.87 66.01
62.57
56.87
49.90 54.03
SwinPA-Net
75.63
14.388
92.09
87.69
89.86
70.04
71.83
94.70
84.87
87.40 74.30
64.46
62.88
46.82 56.24
HTC-NET
74.86
18.334
92.48
81.35
84.63
65.06
71.95
94.73
83.71
86.78 74.98
64.87
62.97
50.35 59.25
CSWin-UNet
73.43
15.851
90.69
80.79
85.10
63.70
64.90
95.01
85.91
83.44 71.97
64.71
58.35
50.46 59.59
SCUNet++
72.65
14.245
92.48
79.84
84.63
63.68
69.89
94.78
84.30
84.82 67.41
64.87
58.39
46.09 56.25
SWMA-Unet
69.41
18.221
90.42
79.88
83.77
54.59
64.05
93.67
79.84
84.05 67.24
53.35
56.66
50.30 44.52
DA-TransUnet
67.79
27.134
84.25
78.11
75.58
51.06
65.96
92.23
76.34
80.23 69.11
56.82
51.03
47.65 50.95
Ours
76.39
13.621
90.62
84.28
81.66
70.93
73.39
94.72
83.82
88.27 75.37
68.46
66.40
51.75 63.38
TABLE III
COMPARSION RESULT ON THE AMOS DATASET
Method
DSC ↑
HD95 ↓
Spleen Kidney(R) Kidney(L) Gallbladder Esophagus Liver Stomach Aorta IVC Pancreas RAG LAG Duodenum Bladder Pr/Ut
(%, mean) (mm, mean)
DSC ↑
UNet
60.06
20.156
82.76
84.54
86.21
49.05
62.03
91.79
66.78
84.97 65.85
56.53
49.67 49.25
51.70
58.72 51.08
Attention-UNet
64.98
26.569
81.88
77.12
76.11
46.71
69.03
90.66
69.24
82.68 66.09
57.56
50.39 49.07
50.90
56.54 50.79
Trans-UNet
71.63
12.246
90.17
92.22
91.26
62.33
65.45
94.33
73.37
83.89 75.83
64.70
52.58 53.68
58.54
59.31 56.76
Swin-UNet
64.54
14.245
88.06
88.97
85.76
50.77
53.57
93.00
69.33
75.77 66.53
58.43
40.34 42.77
54.39
56.62 43.75
UCTransNet
66.34
18.357
84.60
80.31
83.55
48.84
61.91
92.48
66.49
82.32 67.68
57.86
51.10 42.33
49.77
70.00 55.93
SwinPA-Net
70.73
13.600
89.43
88.24
87.42
56.69
63.40
93.82
74.55
81.73 72.30
62.93
47.17 49.21
57.97
70.78 65.31
HTC-NET
71.76
16.388
89.41
90.75
89.62
62.70
68.87
94.05
74.49
85.37 78.59
65.04
53.24 45.57
58.41
60.90 59.31
CSWin-UNet
71.86
11.806
91.72
91.90
92.34
56.03
61.92
94.74
75.10
86.27 72.71
66.83
52.86 52.21
62.27
61.58 59.36
SCUUNet++
69.53
17.524
88.08
87.75
87.93
54.78
61.35
93.27
75.36
83.73 71.09
57.14
43.70 39.67
54.86
73.66 70.54
SWMA-UNet
67.56
13.786
87.31
89.63
88.27
58.16
58.93
93.48
68.58
81.92 69.48
61.54
42.34 41.97
53.56
61.44 56.79
DA-TransUnet
71.73
13.927
92.91
92.32
92.22
60.57
66.23
93.45
72.96
85.70 76.15
66.44
52.96 55.66
57.94
57.14 53.27
Ours
75.33
8.409
93.79
91.68
90.84
62.96
69.04
94.92
79.25
88.42 77.37
68.76
51.53 59.83
64.10
70.55 66.88
to the ground truth labels, demonstrating higher accuracy and
consistency.
2) Experiments on the ACDC dataset: Table VI presents a
comprehensive evaluation on the ACDC dataset. According to
the results, our method outperforms existing approaches across
all metrics, achieving an average DSC of 91.98% and an HD95
of 1.071.
The second column in Fig. 5 shows a qualitative comparison
of HiPerformer with existing segmentation methods on the
ACDC dataset. Incorrectly segmented regions are marked with
red boxes. We observe that most existing methods mistakenly
segment parts of the background as RV, whereas our proposed
method does not exhibit this issue, resulting in more accurate
segmentation.
3) Experiments on the BTCV dataset: Table II presents a
comprehensive evaluation on the BTCV dataset. We report the
results on all 13 organs along with the corresponding mean
performance over all organs. According to the results, our
model achieves an average DSC of 76.39% and an average
HD95 of 13.621, both outperforming existing methods. No-
tably, for the smaller organ LAG, our method still performs
excellently, achieving a DSC of 63.38%, which is 3.97%
higher than the second-best method, CSWin-UNet.
The third column in Fig. 5 shows a qualitative comparison
of HiPerformer with existing segmentation methods on the
BTCV dataset. Incorrectly segmented regions are marked with

[EQ eq_p8_000] -> see eq_p8_000.tex

[EQ eq_p8_001] -> see eq_p8_001.tex

[EQ eq_p8_002] -> see eq_p8_002.tex

[EQ eq_p8_003] -> see eq_p8_003.tex

[EQ eq_p8_004] -> see eq_p8_004.tex

[EQ eq_p8_005] -> see eq_p8_005.tex

[EQ eq_p8_006] -> see eq_p8_006.tex

[EQ eq_p8_007] -> see eq_p8_007.tex

[EQ eq_p8_008] -> see eq_p8_008.tex

[EQ eq_p8_009] -> see eq_p8_009.tex

[EQ eq_p8_010] -> see eq_p8_010.tex

[EQ eq_p8_011] -> see eq_p8_011.tex

[EQ eq_p8_012] -> see eq_p8_012.tex

[EQ eq_p8_013] -> see eq_p8_013.tex

[EQ eq_p8_014] -> see eq_p8_014.tex

[EQ eq_p8_015] -> see eq_p8_015.tex

[EQ eq_p8_016] -> see eq_p8_016.tex

[EQ eq_p8_017] -> see eq_p8_017.tex

[EQ eq_p8_018] -> see eq_p8_018.tex


# [PAGE 10]
10
TABLE IV
COMPARSION RESULT ON THE RAOS DATASET
Method
DSC ↑
HD95 ↓
Liver
Spleen
Kidney(L) Kidney(R) Stomach Gallbladder Esophagus Pancreas Duodenum
(%, mean) (mm, mean)
DSC ↑
UNet
68.88
28.769
91.50
88.22
70.58
69.23
78.52
61.05
65.20
63.72
40.86
Attention UNet
73.09
26.847
92.34
93.18
81.45
72.61
81.07
62.78
68.33
66.91
50.80
Trans-UNet
75.28
16.721
93.36
93.33
86.49
85.65
82.04
56.00
71.34
69.86
49.45
Swin-UNet
71.08
20.551
94.55
91.41
82.90
84.07
78.16
55.24
68.03
62.79
47.23
UCTransNet
69.86
19.927
92.33
90.18
76.09
73.67
78.32
57.03
66.05
63.35
45.85
SwinPA-Net
74.09
14.099
94.24
92.18
86.27
83.29
81.48
52.18
71.91
66.47
51.96
HTC-NET
74.63
20.857
94.82
88.68
80.78
83.83
80.08
62.52
71.78
66.29
51.59
CSWin-UNet
76.38
15.204
93.30
91.43
83.50
76.14
81.07
62.09
75.38
70.52
52.83
SCUUNet++
73.01
15.912
94.05
90.55
79.59
75.87
80.60
53.89
71.41
66.69
46.95
SWMA-UNet
70.62
19.051
93.71
91.32
79.06
76.36
79.00
48.88
69.04
62.79
41.50
DA-TransUnet
74.27
15.646
93.34
93.07
90.76
88.66
81.21
51.19
66.51
69.73
49.65
Ours
76.66
8.664
94.77
93.67
82.86
79.60
79.11
66.47
69.23
64.31
57.05
Method
Colon
Intestine
Adrenal(L) Adrenal(R)
Rectum
Bladder
HOF(L)
HOF(R)
Prostate
SV
DSC ↑
UNet
73.08
73.31
53.02
35.31
71.73
92.12
72.35
70.75
72.81
65.40
Attention-UNet
75.28
78.97
58.46
57.13
76.13
92.13
85.13
85.28
52.58
58.08
Trans-UNet
72.71
79.10
62.60
54.87
75.52
96.66
80.28
81.57
74.68
67.86
Swin-UNet
70.76
75.11
52.75
45.21
74.90
92.53
85.82
85.55
77.00
26.45
UCTransNet
71.74
74.98
56.06
55.31
74.42
92.30
83.71
83.47
56.41
35.93
SwinPA-Net
73.48
79.13
53.49
55.60
74.98
93.80
87.11
87.71
70.85
51.47
HTC-NET
75.32
77.50
62.62
64.95
77.52
91.12
89.64
86.47
62.24
53.27
CSWin-UNet
74.19
79.12
62.26
57.10
76.58
94.69
85.19
86.02
83.57
65.58
SCUUNet++
72.14
74.95
58.12
55.72
75.47
93.01
86.66
88.92
82.79
39.71
SWMA-UNet
71.06
75.45
54.56
53.75
72.60
93.04
83.91
81.88
60.93
52.93
DA-TransUnet
72.25
77.53
56.38
58.70
76.38
93.58
90.13
91.15
72.65
38.28
Ours
73.75
80.56
65.14
61.71
79.55
95.26
93.43
94.52
62.98
62.55
TABLE V
COMPARSION RESULT ON THE MM-WHS DATASET
Method
DSC ↑
HD95 ↓
LVM
LABC
LVBC
RABC
RVBV
AATH
PA
(%, mean)
(mm, mean)
DSC ↑
UNet
83.20
21.035
88.92
81.39
87.31
86.23
84.33
77.93
76.31
Attention UNet
82.43
20.494
87.40
80.28
84.76
86.44
84.50
77.70
75.95
Trans-UNet
81.34
17.225
85.10
76.76
85.15
85.64
82.93
79.03
74.76
Swin-UNet
80.44
20.220
88.06
79.45
80.23
83.40
82.45
81.12
68.34
UCTransNet
83.09
21.582
87.67
80.48
86.07
86.09
84.00
80.22
77.09
SwinPA-Net
85.61
11.289
90.24
83.36
89.03
88.23
86.86
83.83
77.74
HTC-NET
84.20
13.327
88.41
80.77
87.69
87.26
83.75
84.49
77.01
CSWin-UNet
85.54
16.273
82.79
82.58
88.88
88.34
86.18
83.66
79.34
SCUNet++
84.29
14.882
89.45
82.12
88.32
86.32
85.36
81.15
77.33
SWMA-Unet
80.70
25.691
86.96
78.81
84.23
82.65
82.42
80.38
69.45
DA-TransUnet
81.11
19.668
84.19
79.39
85.93
85.63
80.77
79.04
72.79
Ours
86.52
10.654
90.54
84.37
90.03
88.38
86.45
86.02
79.83
TABLE VI
COMPARSION RESULT ON THE ACDC DATASET
Method
DSC ↑
HD95 ↓
RV
Myo
LV
(%, mean)
(mm, mean)
DSC ↑
UNet
91.58
1.236
90.58
89.66
94.49
Attention UNet
91.50
1.117
90.55
89.28
94.66
Trans-UNet
90.88
1.127
89.27
88.77
94.62
Swin-UNet
90.69
1.397
89.51
88.67
93.91
UCTransNet
91.26
1.107
90.02
89.34
94.43
SwinPA-Net
91.54
1.103
90.48
89.51
94.63
HTC-NET
90.79
1.195
89.51
88.89
93.97
CSWin-UNet
90.99
1.896
90.32
88.56
94.09
SCUNet++
91.08
1.097
90.32
88.88
94.03
SWMA-Unet
90.67
1.169
89.34
88.47
94.20
DA-TransUnet
90.07
1.316
88.23
88.09
93.89
Ours
91.98
1.071
90.98
90.14
94.83
TABLE VII
COMPARSION RESULT ON THE SEGTHOR DATASET
Method
DSC ↑
HD95 ↓
Esophagus Heart Trachea Aorta
(%, mean) (mm, mean)
DSC ↑
UNet
82.47
5.889
63.30
91.80
86.68
88.10
Attention UNet
83.08
6.420
65.17
92.20
85.65
89.30
Trans-UNet
82.91
5.357
64.71
90.01
86.25
88.67
Swin-UNet
79.20
6.698
56.35
91.85
84.45
84.16
UCTransNet
81.38
5.982
60.91
91.59
86.20
86.83
SwinPA-Net
82.90
6.278
64.88
92.62
85.44
88.25
HTC-NET
83.09
5.825
66.70
92.10
85.30
88.27
CSWin-UNet
82.50
5.214
62.33
92.21
86.46
88.98
SCUNet++
81.55
5.806
61.50
91.65
85.43
87.60
SWMA-Unet
77.89
8.381
55.02
90.15
84.63
81.76
DA-TransUnet
81.60
7.146
63.58
90.74
86.00
86.07
Ours
84.28
4.637
69.05
93.35
85.29
89.45

[EQ eq_p9_000] -> see eq_p9_000.tex

[EQ eq_p9_001] -> see eq_p9_001.tex

[EQ eq_p9_002] -> see eq_p9_002.tex

[EQ eq_p9_003] -> see eq_p9_003.tex

[EQ eq_p9_004] -> see eq_p9_004.tex

[EQ eq_p9_005] -> see eq_p9_005.tex

[EQ eq_p9_006] -> see eq_p9_006.tex

[EQ eq_p9_007] -> see eq_p9_007.tex

[EQ eq_p9_008] -> see eq_p9_008.tex

[EQ eq_p9_009] -> see eq_p9_009.tex

[EQ eq_p9_010] -> see eq_p9_010.tex

[EQ eq_p9_011] -> see eq_p9_011.tex

[EQ eq_p9_012] -> see eq_p9_012.tex

[EQ eq_p9_013] -> see eq_p9_013.tex

[EQ eq_p9_014] -> see eq_p9_014.tex

[EQ eq_p9_015] -> see eq_p9_015.tex

[EQ eq_p9_016] -> see eq_p9_016.tex

[EQ eq_p9_017] -> see eq_p9_017.tex

[EQ eq_p9_018] -> see eq_p9_018.tex

[EQ eq_p9_019] -> see eq_p9_019.tex

[EQ eq_p9_020] -> see eq_p9_020.tex

[EQ eq_p9_021] -> see eq_p9_021.tex

[EQ eq_p9_022] -> see eq_p9_022.tex

[EQ eq_p9_023] -> see eq_p9_023.tex

[EQ eq_p9_024] -> see eq_p9_024.tex

[EQ eq_p9_025] -> see eq_p9_025.tex

[EQ eq_p9_026] -> see eq_p9_026.tex

[EQ eq_p9_027] -> see eq_p9_027.tex

[EQ eq_p9_028] -> see eq_p9_028.tex

[EQ eq_p9_029] -> see eq_p9_029.tex

[EQ eq_p9_030] -> see eq_p9_030.tex

[EQ eq_p9_031] -> see eq_p9_031.tex


# [PAGE 11]
11
Fig. 6. The visualization images of attention weight heatmaps on multiple datasets.
TABLE VIII
COMPARSION RESULT ON THE REFUGE DATASET
Method
DSC ↑
HD95 ↓
CUP
DISC
(%, mean)
(mm, mean)
DSC ↑
UNet
89.61
3.915
89.42
89.81
Attention UNet
89.54
4.047
89.43
89.66
Trans-UNet
88.80
4.218
88.22
89.39
Swin-UNet
88.82
4.590
88.59
89.05
UCTransNet
89.66
4.099
89.40
89.91
SwinPA-Net
89.83
3.781
89.47
90.19
HTC-NET
89.72
3.777
89.20
90.23
CSWin-UNet
89.03
4.003
88.58
89.49
SCUUNet++
89.26
3.942
88.86
89.66
SWMA-UNet
88.60
4.457
88.21
88.98
DA-TransUnet
89.02
4.105
88.54
89.50
Ours
90.13
3.627
89.79
90.46
TABLE IX
COMPARSION RESULT ON THE CHASE DATASET
Method
DSC ↑
HD95 ↓
Recall ↑
IoU ↑
(%, mean)
(mm, mean)
(%, mean)
(%, mean)
UNet
63.55
48.142
59.47
47.08
Attention-UNet
74.19
23.602
72.07
60.10
Trans-UNet
76.08
19.515
73.72
62.00
Swin-UNet
72.35
18.065
71.49
57.14
UCTransNet
68.37
37.740
66.54
52.50
SwinPA-Net
76.63
13.149
77.53
62.99
HTC-NET
75.36
17.228
76.70
61.16
CSWin-UNet
77.12
13.419
78.58
63.46
SCUUNet++
75.39
14.845
74.17
60.59
SWMA-UNet
74.13
17.278
72.67
59.41
DA-TransUnet
75.62
20.937
74.73
61.13
Ours
77.70
12.792
78.71
64.40
red boxes. It can be observed that other existing methods
exhibit inaccuracies during segmentation, whereas our method
shows precise organ segmentation with improved boundary
delineation.
4) Experiments on the AMOS dataset: Table III presents
a comprehensive evaluation on the AMOS dataset. Compared
with existing methods, our approach performs better on both
TABLE X
COMPARSION RESULT ON THE STARE DATASET
Method
DSC ↑
HD95 ↓
Recall ↑
IoU ↑
(%, mean)
(mm, mean)
(%, mean)
(%, mean)
UNet
51.58
31.818
61.74
40.24
Attention UNet
57.64
28.331
55.89
44.50
Trans-UNet
64.96
20.029
68.00
51.32
Swin-UNet
68.89
11.408
67.68
55.03
UCTransNet
48.96
51.213
47.07
37.20
SwinPA-Net
71.32
9.600
72.07
57.62
HTC-NET
69.13
13.446
68.42
55.57
CSWin-UNet
72.72
10.215
72.50
59.60
SCUUNet++
69.99
11.702
73.49
56.51
SWMA-UNet
68.95
12.707
68.78
55.43
DA-TransUnet
67.96
16.726
66.79
54.42
Ours
73.26
8.405
75.41
60.29
TABLE XI
COMPARSION RESULT ON THE LES-AV DATASET
Method
DSC ↑
HD95 ↓
Recall ↑
IoU ↑
(%, mean)
(mm, mean)
(%, mean)
(%, mean)
UNet
62.30
150.187
55.91
41.86
Attention UNet
75.48
45.245
72.39
58.99
Trans-UNet
77.15
40.094
75.98
60.68
Swin-UNet
74.70
30.300
71.29
58.35
UCTransNet
66.03
96.168
60.66
47.03
SwinPA-Net
76.45
26.129
77.04
60.29
HTC-NET
74.82
30.577
73.64
58.36
CSWin-UNet
77.35
25.252
77.27
61.92
SCUUNet++
75.99
26.943
74.15
59.70
SWMA-UNet
74.93
34.573
70.77
58.34
DA-TransUnet
77.12
46.382
73.95
60.19
Ours
77.55
23.120
79.76
62.36
the mean DSC and HD95 metrics, achieving 75.33% and
8.409, respectively.
The fourth column in Fig. 5 shows a qualitative comparison
between HiPerformer and existing segmentation methods on
the AMOS dataset. Incorrectly segmented regions are marked
with red boxes. It can be observed that existing segmentation
methods often misclassify the background as the gallbladder
[FIGURE img_p10_005]

[EQ eq_p10_000] -> see eq_p10_000.tex

[EQ eq_p10_001] -> see eq_p10_001.tex

[EQ eq_p10_002] -> see eq_p10_002.tex

[EQ eq_p10_003] -> see eq_p10_003.tex

[EQ eq_p10_004] -> see eq_p10_004.tex

[EQ eq_p10_005] -> see eq_p10_005.tex

[EQ eq_p10_006] -> see eq_p10_006.tex

[EQ eq_p10_007] -> see eq_p10_007.tex

[EQ eq_p10_008] -> see eq_p10_008.tex

[EQ eq_p10_009] -> see eq_p10_009.tex

[EQ eq_p10_010] -> see eq_p10_010.tex

[EQ eq_p10_011] -> see eq_p10_011.tex

[EQ eq_p10_012] -> see eq_p10_012.tex

[EQ eq_p10_013] -> see eq_p10_013.tex

[EQ eq_p10_014] -> see eq_p10_014.tex

[EQ eq_p10_015] -> see eq_p10_015.tex

[EQ eq_p10_016] -> see eq_p10_016.tex

[EQ eq_p10_017] -> see eq_p10_017.tex

[EQ eq_p10_018] -> see eq_p10_018.tex

[EQ eq_p10_019] -> see eq_p10_019.tex

[EQ eq_p10_020] -> see eq_p10_020.tex

[EQ eq_p10_021] -> see eq_p10_021.tex

[EQ eq_p10_022] -> see eq_p10_022.tex

[EQ eq_p10_023] -> see eq_p10_023.tex

[EQ eq_p10_024] -> see eq_p10_024.tex

[EQ eq_p10_025] -> see eq_p10_025.tex

[EQ eq_p10_026] -> see eq_p10_026.tex

[EQ eq_p10_027] -> see eq_p10_027.tex

[EQ eq_p10_028] -> see eq_p10_028.tex

[EQ eq_p10_029] -> see eq_p10_029.tex

[EQ eq_p10_030] -> see eq_p10_030.tex

[EQ eq_p10_031] -> see eq_p10_031.tex

[EQ eq_p10_032] -> see eq_p10_032.tex

[EQ eq_p10_033] -> see eq_p10_033.tex


# [PAGE 12]
12
(yellow label), resulting in significant deviations in the out-
comes. Additionally, the duodenum (deep cyan label) is also
prone to missegmentation, which affects the overall quality.
In contrast, our proposed method effectively addresses these
issues, providing more precise and stable segmentation results,
and demonstrating its advantages in detail feature extraction
and boundary information capture. It can be observed that
many existing segmentation methods often misclassify back-
ground as the gallbladder (yellow label), leading to large
deviations in the results. At the same time, the duodenum (dark
cyan label) is also prone to incorrect segmentation, which
degrades overall quality. In contrast, our proposed method
effectively addresses these issues, yielding more accurate
and stable segmentation results and demonstrating superior
ability to extract fine-grained features and capture boundary
information.
5) Experiments on the MM-WHS dataset: Table V presents
a comprehensive evaluation on the MM-WHS dataset. Accord-
ing to the results, except for the RVBV organ, our method
outperforms existing methods in the segmentation performance
of all other organs, achieving an average DSC and HD95 of
86.52% and 10.654, respectively.
The fifth column in Fig. 5 shows a qualitative comparison
between HiPerformer and existing segmentation methods on
the MM-WHS dataset. Incorrectly segmented regions are
marked with red boxes. It can be observed that other exist-
ing methods produce some small and inaccurate error pix-
els during segmentation, whereas our method’s segmentation
results are closer to the ground truth. The improvement in
segmentation performance likely stems from fully capturing
and integrating both local details and global context.
6) Experiments on the RAOS dataset: Table IV presents
a comprehensive evaluation on the RAOS dataset. According
to the results, among existing methods CSWin-UNet performs
comparatively well, with an average DSC of 76.38%. Building
on this, our method achieves a further improvement, raising
the average DSC to 76.66%. Although the increase in average
DSC is small, our method shows a clear advantage on the
average HD95 metric, which is reduced by 6.54 and is
significantly better than CSWin-UNet.
The sixth column in Fig. 5 shows a qualitative comparison
between HiPerformer and existing segmentation methods on
the RAOS dataset. Incorrectly segmented regions are marked
with red boxes. It can be observed that existing methods often
produce segmentation errors for the brown-labeled seminal
vesicles (SV) and the pink-labeled head of the right femur
(HOF(R)), resulting in large deviations from the ground truth.
In contrast, our method shows a clear advantage in segmenting
these two critical regions, delivering more accurate and stable
results that better match the true anatomical structures.
7) Experiments on the SegTHOR dataset: Table VII presents
a comprehensive evaluation on the SegTHOR dataset. Accord-
ing to the results, except for the Trachea organ, the proposed
method outperforms existing methods in the segmentation
performance of all other organs, achieving an average DSC
of 84.28% and an HD95 of 4.637.
The seventh column in Fig. 5 shows a qualitative compari-
son of HiPerformer with existing segmentation methods on the
SegTHOR dataset. Incorrectly segmented regions are marked
with red boxes. We observe that existing methods exhibit
missing regions in the segmentation of the heart, resulting in
incomplete segmentation, whereas the proposed method is able
to achieve complete segmentation of the heart.
8) Experiments on the Refuge dataset: Table VIII presents
a comprehensive evaluation on the Refuge dataset. According
to the results, the proposed method outperforms existing seg-
mentation methods both in terms of performance on specific
organs and overall average performance, achieving an average
DSC of 90.13% and an average HD95 of 3.627.
The eighth column in Fig. 5 provides a qualitative compar-
ison between HiPerformer and existing segmentation methods
on the Refuge dataset. It can be observed that the proposed
method shows no significant difference from existing methods
in the visualization of segmentation results. The above results
can largely be attributed to the limited number of organ classes
in the dataset and the distinct feature differences between
categories, which reduced the complexity of the segmentation
task and resulted in high accuracy across all models. However,
it is worth highlighting that our method offers a distinct
advantage in accurately capturing organ boundary information.
Enhanced representation of boundary details directly yields
superior quantitative evaluation results.
9) Experiments on the Chase dataset: Table IX presents a
comprehensive evaluation on the Chase dataset. According to
the results, the proposed method performs excellently across
all metrics, with an average DSC of 77.70%, an average HD95
of 12.792, an average Recall of 78.71%, and an average IoU
of 64.40%, all significantly outperforming existing methods.
The ninth column in Fig. 5 shows a qualitative comparison
of HiPerformer with existing segmentation methods on the
Chase dataset. Incorrectly segmented regions are marked with
blue boxes. Due to uneven background illumination in the
Chase images, low contrast of the blood vessels, and the rel-
atively broad morphology of small arteries, the segmentation
of retinal blood vessels is particularly challenging. As shown
in the area marked by the blue box, the blood vessels are
thin and have poor contrast, making it difficult for existing
methods to achieve complete and accurate segmentation, often
resulting in breaks or omissions. In contrast, our proposed
method employs a modular hierarchical fusion strategy, which
effectively reduces feature information loss during transmis-
sion and significantly improves the recovery of tiny capillaries,
making the segmentation results closer to the ground truth.
10) Experiments on the STARE dataset: Table X presents a
comprehensive evaluation on the STARE dataset. According
to the results, our method outperforms existing methods across
all evaluation metrics, achieving an average DSC of 73.26%,
an average HD95 of 8.405, an average Recall of 75.41%, and
an average IoU of 60.29%.
The tenth column in Fig. 5 shows a qualitative comparison
of HiPerformer with existing segmentation methods on the
STARE dataset. Incorrectly segmented regions are marked
with blue boxes. We select two typical regions prone to seg-
mentation errors for analysis. The first region is the gray area
located in the center of the image, whose color is highly sim-
ilar to blood vessels, causing models such as UNet, Attention


# [PAGE 13]
13
UNet, Swin-UNet, and UCTransNet to mistakenly segment it
as blood vessels. The second region is the terminal structure
of the blood vessels in the upper right corner. Due to the low
contrast between the pixels in this area and the background, the
fine vessel terminals are difficult to completely separate from
the background. Except for SwinPA-Net, other existing meth-
ods fail to achieve complete segmentation here. In contrast,
our method introduces the Pyramid Gated Attention (PGA)
module, which enhances feature representation of key regions
while suppressing interference from irrelevant areas, thereby
enabling more accurate differentiation of pathological regions.
Furthermore, our approach efficiently integrates local details
with global contextual information, allowing precise extraction
of vessel terminal edge features and significantly improving
segmentation accuracy.
11) Experiments on the LES-AV dataset: Table XI presents
a comprehensive evaluation on the LES-AV dataset. According
to the results, CSWin-UNet performs excellently, with an
average DSC of 77.35%. Our method, however, achieves
even better performance, with the average DSC improved to
77.55%, an increase of 0.2%. In other metrics, the average
HD95 is 23.120, the average Recall reaches 79.76%, and
the average IoU is 62.36%, all of which outperform existing
methods.
The eleventh column in Fig. 5 shows a qualitative com-
parison of HiPerformer with existing segmentation methods
on the LES-AV dataset. Incorrectly segmented regions are
marked with blue boxes. The retinal vascular network structure
is extremely complex, and the uneven distribution of pixel in-
tensities leads to suboptimal performance of existing methods
when handling crosses structures in dense vascular regions.
As shown in the blue boxed area, current methods often
struggle to accurately capture edge features at vessel crossings,
which frequently results in vessel merging or partial vessel
segmentation omissions. In contrast, our method proposes
the PMI module to achieve effective fusion of multi-scale
semantic information significantly reducing boundary blurring.
our method enables more precise segmentation of intersecting
vessels, greatly enhancing the restoration and detail represen-
tation of the retinal vascular structure, and fully demonstrating
its superior performance in complex vascular environments.
D. Loss parameter experiment
To further optimize the model, we adopt a balanced joint
loss function, which is a weighted combination of cross-
entropy loss (LCE) and Dice loss (LDice). In Eq. (18), the
value of the parameter a influences the model’s optimization
direction. A larger α value causes the model to prioritize main-
taining the overall consistency of the segmentation regions,
potentially at the cost of pixel-level classification accuracy.
Conversely, a smaller α value leads to better performance in
pixel-level classification but may hinder the optimization of
overall segmentation consistency. On the Synapse dataset, we
systematically study the impact of a on segmentation accuracy
by conducting experiments with α values of 0.1, 0.3, 0.5,
0.7, and 0.9. The experimental results are shown in Fig. 7,
indicating that when α = 0.5, the model achieves the best
segmentation performance, with the highest average DSC of
83.93% and the lowest average HD95 of 11.582.
Fig. 7. The experimental results of the combined loss function with different
hyperparameters on the Synapse dataset.
E. Ablation Studies
To evaluate the impact of each key component in the
proposed method on the overall segmentation performance,
a series of ablation experiments are conducted based on the
Synapse and BTCV datasets. The results of the ablation studies
are presented in Table XII. When the encoder retains only
the Local branch, the average DSC on the Synapse and MM-
WHS datasets decreases by 5.47% and 7.62%, respectively.
When only the Global branch is retained, the average DSC
drops by 1.70% and 2.45%, respectively. The above results
fully validate the significant contribution of fusing CNN and
Transformer to performance improvement.
Furthermore, when both the PMI and PGA modules in
the skip connections are removed simultaneously, the average
DSC on the Synapse and MM-WHS datasets decreases by
1.70% and 1.71%, respectively. Removing the PMI module
alone leads to average DSC drops of 0.79% and 1.44%,
respectively, while removing the PGA module alone causes
average DSC declines of 0.94% and 0.51%, respectively. The
above results indicate that the synergistic effect between the
PMI and PGA modules is crucial, and combining them into
a PPA module to replace traditional skip connections plays a
decisive role in performance improvement.
When all modules are integrated, HiPerformer achieves the
best segmentation performance. If any module is removed,
performance decreases, which demonstrates the effectiveness
and necessity of the designed module combination.
F. Attention Mechanism Visualization
In medical image segmentation tasks, incorporating atten-
tion mechanisms not only improves the segmentation perfor-
mance of the model but also enables intuitive visualization
of the regions the model focuses on or suppresses during
segmentation through attention heatmaps. As shown in Fig. 6,
we present the segmentation heatmaps of partial organs across
[FIGURE img_p12_006]


# [PAGE 14]
14
TABLE XII
RESULTS OF ABLATION EXPERIMENTS ON THE SYNAPSE AND BTCV
DATASETS
Ablation Type
DSC ↑(%, mean)
Local
Global
LGFF
PMI
PGA
Synapse
BTCV
✓
×
×
✓
✓
78.46
68.77
×
✓
×
✓
✓
82.23
73.94
✓
✓
✓
×
×
82.83
74.68
✓
✓
✓
×
✓
83.14
74.95
✓
✓
✓
✓
×
82.99
75.88
✓
✓
✓
✓
✓
83.93
76.39
eleven datasets using the proposed method, where brighter
colors indicate higher model attention. The results show that in
the three retinal vessel segmentation datasets (Chase, STARE,
and LEA-VES), the model tends to focus on the optic disc
and cup regions, while struggling to effectively concentrate
on the small and low-contrast vascular structures. The above-
mentioned limitation is likely the main reason for its relatively
lower accuracy in the retinal vessel segmentation task. In the
ACDC dataset, the model also exhibits a small amount of
erroneous attention to background regions. Apart from the
datasets mentioned above, our model accurately focuses on
target regions and effectively suppresses background noise,
clearly demonstrating the superiority of our method.
V. CONCLUSION
In this study, we propose HiPerformer, a model specifically
designed for multi-region and fine-grained medical image
segmentation tasks. Its encoder designs a novel modular hi-
erarchical architecture and incorporates a Local and Global
Feature Fusion module that hierarchically integrates local
features with global context, effectively preventing informa-
tion loss and feature conflicts from simple concatenation
and enabling more accurate, efficient information integration.
At the skip connections, the model proposes a Progressive
Pyramid Aggregation module, which not only fuse deep and
shallow features but also employ targeted feature-enhancement
mechanisms to effectively reduce semantic gaps across scales
and suppress noise. We conduct a comprehensive evaluation
of HiPerformer on eleven different datasets and compare it
with various advanced models, validating its outstanding per-
formance and effectiveness. Our methods significantly improve
feature representation capabilities, opening new avenues for
related research. Despite its excellent performance, there are
still some limitations, such as its IRMLP structure, which
expands the dimensions by four times, resulting in a large
overall number of parameters. Future work should focus on
optimizing the structure to reduce the parameter size and
improve computational efficiency.
REFERENCES
[1] N. Otsu et al., “A threshold selection method from gray-level his-
tograms,” Automatica, vol. 11, no. 285-296, pp. 23–27, 1975.
[2] D. Tan, Z. Huang, X. Peng, W. Zhong, and V. Mahalec, “Deep adaptive
fuzzy clustering for evolutionary unsupervised representation learning,”
IEEE Transactions on Neural Networks and Learning Systems, vol. 35,
no. 5, pp. 6103–6117, 2023.
[3] O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional net-
works for biomedical image segmentation,” in International Confer-
ence on Medical image computing and computer-assisted intervention.
Springer, 2015, pp. 234–241.
[4] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, and M. Wang,
“Swin-unet: Unet-like pure transformer for medical image segmenta-
tion,” in European conference on computer vision.
Springer, 2022,
pp. 205–218.
[5] J. Chen, Y. Lu, Q. Yu, X. Luo, E. Adeli, Y. Wang, L. Lu, A. L.
Yuille, and Y. Zhou, “Transunet: Transformers make strong encoders
for medical image segmentation,” arXiv preprint arXiv:2102.04306,
2021.
[6] H. Wu, S. Chen, G. Chen, W. Wang, B. Lei, and Z. Wen, “Fat-net:
Feature adaptive transformers for automated skin lesion segmentation,”
Medical image analysis, vol. 76, p. 102327, 2022.
[7] J. Liu, K. Li, C. Huang, H. Dong, Y. Song, and R. Li, “Mixformer:
a mixed cnn-transformer backbone for medical image segmentation,”
IEEE Transactions on Instrumentation and Measurement, 2024.
[8] G. Sun, Y. Pan, W. Kong, Z. Xu, J. Ma, T. Racharak, L.-M. Nguyen,
and J. Xin, “Da-transunet: integrating spatial and channel dual attention
with transformer u-net for medical image segmentation,” Frontiers in
Bioengineering and Biotechnology, vol. 12, p. 1398237, 2024.
[9] D. Tan, R. Hao, L. Hua, Q. Xu, Y. Su, C. Zheng, and W. Zhong,
“Performer: A high-performance global-local model-augmented with
dual network interaction mechanism,” IEEE Transactions on Cognitive
and Developmental Systems, 2024.
[10] Y. Zhang, H. Liu, and Q. Hu, “Transfuse: Fusing transformers and cnns
for medical image segmentation,” in International conference on med-
ical image computing and computer-assisted intervention.
Springer,
2021, pp. 14–24.
[11] J. Hu, S. Chen, Z. Pan, S. Zeng, and W. Yang, “Perspective+ unet:
Enhancing segmentation with bi-path fusion and efficient non-local
attention for superior receptive fields,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention.
Springer, 2024, pp. 499–509.
[12] D. Tan, R. Hao, X. Zhou, J. Xia, Y. Su, and C. Zheng, “A novel skip-
connection strategy by fusing spatial and channel wise features for
multi-region medical image segmentation,” IEEE Journal of Biomedi-
cal and Health Informatics, vol. 28, no. 9, pp. 5396–5409, 2024.
[13] X. Li, X. Qin, C. Huang, Y. Lu, J. Cheng, L. Wang, O. Liu, J. Shuai,
and C.-a. Yuan, “Sunet: A multi-organ segmentation network based on
multiple attention,” Computers in Biology and Medicine, vol. 167, p.
107596, 2023.
[14] Z. Zhou, M. M. Rahman Siddiquee, N. Tajbakhsh, and J. Liang,
“Unet++: A nested u-net architecture for medical image segmentation,”
in International workshop on deep learning in medical image analysis.
Springer, 2018, pp. 3–11.
[15] S. Cai, Y. Tian, H. Lui, H. Zeng, Y. Wu, and G. Chen, “Dense-unet:
a novel multiphoton in vivo cellular image segmentation model based
on a convolutional neural network,” Quantitative imaging in medicine
and surgery, vol. 10, no. 6, p. 1275, 2020.
[16] J. Fu, J. Liu, H. Tian, Y. Li, Y. Bao, Z. Fang, and H. Lu, “Dual attention
network for scene segmentation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2019, pp.
3146–3154.
[17] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in
Proceedings of the IEEE conference on computer vision and pattern
recognition, 2018, pp. 7132–7141.
[18] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, “Cbam: Convolutional
block attention module,” in Proceedings of the European conference
on computer vision (ECCV), 2018, pp. 3–19.
[19] H. Zhang, K. Zu, J. Lu, Y. Zou, and D. Meng, “Epsanet: An efficient
pyramid split attention block on convolutional neural network,” CoRR,
2021.
[20] B. Landman, Z. Xu, J. Igelsias, M. Styner, T. Langerak, and A. Klein,
“Miccai multi-atlas labeling beyond the cranial vault–workshop and
challenge,” in Proc. MICCAI multi-atlas labeling beyond cranial
vault—workshop challenge, vol. 5.
Munich, Germany, 2015, p. 12.
[21] O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, X. Yang, P.-A.
Heng, I. Cetin, K. Lekadir, O. Camara, M. A. G. Ballester et al.,
“Deep learning techniques for automatic mri cardiac multi-structures
segmentation and diagnosis: is the problem solved?” IEEE transactions
on medical imaging, vol. 37, no. 11, pp. 2514–2525, 2018.
[22] Y. Ji, H. Bai, C. Ge, J. Yang, Y. Zhu, R. Zhang, Z. Li, L. Zhanng,
W. Ma, X. Wan et al., “Amos: A large-scale abdominal multi-organ
benchmark for versatile medical image segmentation,” Advances in

[EQ eq_p13_000] -> see eq_p13_000.tex


# [PAGE 15]
15
neural information processing systems, vol. 35, pp. 36 722–36 732,
2022.
[23] Z. Lambert, C. Petitjean, B. Dubray, and S. Kuan, “Segthor: Seg-
mentation of thoracic organs at risk in ct images,” in 2020 Tenth
International Conference on Image Processing Theory, Tools and
Applications (IPTA).
Ieee, 2020, pp. 1–6.
[24] X. Luo, Z. Li, S. Zhang, W. Liao, and G. Wang, “Rethinking abdominal
organ segmentation (raos) in the clinical scenario: A robustness evalu-
ation benchmark with challenging cases,” in International Conference
on Medical Image Computing and Computer-Assisted Intervention.
Springer, 2024, pp. 531–541.
[25] X. Zhuang, L. Li, C. Payer, D. ˇStern, M. Urschler, M. P. Heinrich,
J. Oster, C. Wang, ¨O. Smedby, C. Bian et al., “Evaluation of algorithms
for multi-modality whole heart segmentation: an open-access grand
challenge,” Medical image analysis, vol. 58, p. 101537, 2019.
[26] J. I. Orlando, H. Fu, J. B. Breda, K. Van Keer, D. R. Bathula, A. Diaz-
Pinto, R. Fang, P.-A. Heng, J. Kim, J. Lee et al., “Refuge challenge:
A unified framework for evaluating automated methods for glaucoma
assessment from fundus photographs,” Medical image analysis, vol. 59,
p. 101570, 2020.
[27] C. G. Owen, A. R. Rudnicka, C. M. Nightingale, R. Mullen, S. A.
Barman, N. Sattar, D. G. Cook, and P. H. Whincup, “Retinal arteriolar
tortuosity and cardiovascular risk factors in a multi-ethnic population
study of 10-year-old children; the child heart and health study in
england (chase),” Arteriosclerosis, thrombosis, and vascular biology,
vol. 31, no. 8, pp. 1933–1938, 2011.
[28] J. I. Orlando, J. Barbosa Breda, K. Van Keer, M. B. Blaschko, P. J.
Blanco, and C. A. Bulant, “Towards a glaucoma risk index based
on simulated hemodynamics from fundus images,” in International
Conference on Medical Image Computing and Computer-Assisted
Intervention.
Springer, 2018, pp. 65–73.
[29] O. Oktay, J. Schlemper, L. L. Folgoc, M. Lee, M. Heinrich, K. Misawa,
K. Mori, S. McDonagh, N. Y. Hammerla, B. Kainz et al., “Atten-
tion u-net: Learning where to look for the pancreas,” arXiv preprint
arXiv:1804.03999, 2018.
[30] H. Wang, P. Cao, J. Wang, and O. R. Zaiane, “Uctransnet: rethinking
the skip connections in u-net from a channel-wise perspective with
transformer,” in Proceedings of the AAAI conference on artificial
intelligence, vol. 36, no. 3, 2022, pp. 2441–2449.
[31] H. Du, J. Wang, M. Liu, Y. Wang, and E. Meijering, “Swinpa-net: Swin
transformer-based multiscale feature pyramid aggregation network for
medical image segmentation,” IEEE Transactions on Neural Networks
and Learning Systems, vol. 35, no. 4, pp. 5355–5366, 2022.
[32] H. Tang, Y. Chen, T. Wang, Y. Zhou, L. Zhao, Q. Gao, M. Du, T. Tan,
X. Zhang, and T. Tong, “Htc-net: A hybrid cnn-transformer framework
for medical image segmentation,” Biomedical Signal Processing and
Control, vol. 88, p. 105605, 2024.
[33] X. Liu, P. Gao, T. Yu, F. Wang, and R.-Y. Yuan, “Cswin-unet:
Transformer unet with cross-shaped windows for medical image seg-
mentation,” Information Fusion, vol. 113, p. 102634, 2025.
[34] Y. Chen, B. Zou, Z. Guo, Y. Huang, Y. Huang, F. Qin, Q. Li, and
C. Wang, “Scunet++: Swin-unet and cnn bottleneck hybrid architecture
with multi-fusion dense skip connection for pulmonary embolism ct im-
age segmentation,” in Proceedings of the IEEE/CVF winter conference
on applications of computer vision, 2024, pp. 7759–7767.
[35] X. Tang, J. Li, Q. Liu, C. Zhou, P. Zeng, Y. Meng, J. Xu, G. Tian,
and J. Yang, “Swma-unet: multi-path attention network for improved
medical image segmentation,” IEEE Journal of Biomedical and Health
Informatics, 2024.
[36] C. Chow and T. Kaneko, “Automatic boundary detection of the left
ventricle from cineangiograms,” Computers and biomedical research,
vol. 5, no. 4, pp. 388–410, 1972.
[37] N. Dhanachandra, K. Manglem, and Y. J. Chanu, “Image segmen-
tation using k-means clustering algorithm and subtractive clustering
algorithm,” Procedia Computer Science, vol. 54, pp. 764–771, 2015.
[38] D. Comaniciu and P. Meer, “Mean shift: A robust approach toward
feature space analysis,” IEEE Transactions on pattern analysis and
machine intelligence, vol. 24, no. 5, pp. 603–619, 2002.
[39] J. Canny, “A computational approach to edge detection,” IEEE Trans-
actions on Pattern Analysis and Machine Intelligence, vol. PAMI-8,
no. 6, pp. 679–698, 1986.
[40] A. Rosenfeld, “The max roberts operator is a hueckel-type edge detec-
tor,” IEEE transactions on pattern analysis and machine intelligence,
no. 1, pp. 101–103, 1981.
[41] N. Ibtehaz and M. S. Rahman, “Multiresunet: Rethinking the u-net
architecture for multimodal biomedical image segmentation,” Neural
networks, vol. 121, pp. 74–87, 2020.
[42] X. Huang, Z. Deng, D. Li, X. Yuan, and Y. Fu, “Missformer: An
effective transformer for 2d medical image segmentation,” IEEE trans-
actions on medical imaging, vol. 42, no. 5, pp. 1484–1494, 2022.
[43] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo,
“Swin transformer: Hierarchical vision transformer using shifted win-
dows,” in Proceedings of the IEEE/CVF international conference on
computer vision, 2021, pp. 10 012–10 022.
[44] X. Huo, G. Sun, S. Tian, Y. Wang, L. Yu, J. Long, W. Zhang,
and A. Li, “Hifuse: Hierarchical multi-scale feature fusion network
for medical image classification,” Biomedical Signal Processing and
Control, vol. 87, p. 105534, 2024.
[45] Y. Liu, Z. Shao, and N. Hoffmann, “Global attention mechanism:
Retain information to enhance channel-spatial interactions,” arXiv
preprint arXiv:2112.05561, 2021.
