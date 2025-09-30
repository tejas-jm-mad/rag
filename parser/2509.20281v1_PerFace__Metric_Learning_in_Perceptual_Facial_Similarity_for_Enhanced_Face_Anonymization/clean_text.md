

# [PAGE 1]
PERFACE: METRIC LEARNING IN PERCEPTUAL FACIAL SIMILARITY FOR ENHANCED
FACE ANONYMIZATION
Haruka Kumagai1, Leslie W¨ohler1, Satoshi Ikehata2,1, Kiyoharu Aizawa1
The University of Tokyo1 NII2
ABSTRACT
In response to rising societal awareness of privacy concerns,
face anonymization techniques have advanced, including the emer-
gence of face-swapping methods that replace one identity with an-
other. Achieving a balance between anonymity and naturalness in
face swapping requires careful selection of identities: overly similar
faces compromise anonymity, while dissimilar ones reduce natural-
ness. Existing models, however, focus on binary identity classifica-
tion “the same person or not”, making it difficult to measure nuanced
similarities such as “completely different” versus “highly similar but
different.” This paper proposes a human-perception-based face simi-
larity metric, creating a dataset of 6,400 triplet annotations and met-
ric learning to predict the similarity. Experimental results demon-
strate significant improvements in both face similarity prediction and
attribute-based face classification tasks over existing methods. Our
dataset is available at https://github.com/kumanotanin/
PerFace.
Index Terms— face anonymization, face similarity, face swap-
ping, human perception
1. INTRODUCTION
Face anonymization is a technique used to protect individual privacy
in facial images while maintaining the usability of the underlying
information. Among the various approaches such as occlusion, blur-
ring and face swapping—replacing the target’s facial identity with
the source’s while preserving target’s identity-irrelevant attributes
(e.g., pose, expression, or background) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11]—is considered an effective anonymization method. Unlike tra-
ditional methods such as occlusion or blurring, which compromise
image quality and utility while effectively concealing the identity of
the original person, face swapping ensures realism and coherence by
preserving the facial structure in images.
When employing face swapping for anonymization, it is crucial
that the source and the target share key attributes such as gender
and age [4] to avoid unnatural outcomes (See Fig. 2). To address
this, a recent study [12] leveraged distances between embedded fea-
tures from pretrained face recognition models (e.g., ArcFace [13]).
Their algorithm identifies the most distant face in latent space that
nonetheless retains similar attributes (e.g., age, gender), thereby en-
suring both anonymity and a natural appearance.
However, leveraging pretrained face recognition models [14, 15,
16, 17, 18, 19, 13, 20, 21, 22, 11, 23] for facial anonymization poses
two major challenges. First, these models were optimized via metric
learning to cluster images of the same identity and separate those
of different identities—thereby considering all different identities
This research was partially supported by JSPS 25H01164.
“A is more similar!”
B
reference
human 
evaluation
face-swapped
A
similar 
pair
dissimilar 
pair
“A is more similar!”
0.322
0.121
A and ref. 
B and ref.
>
predicted similarity 
of each pair
metric
learning
Fig. 1: We developed a model that predicts perceptual facial similar-
ity using metric learning, based on our SimCelebA dataset.
as dissimilar, even when they appear perceptually similar. In con-
trast, face swapping for anonymization necessitates an accurate as-
sessment of similarity between different identities to guarantee that
the swapped face is perceptually distinct from the original. Second,
existing models were trained exclusively on genuine face images. In
face anonymization, it is imperative to evaluate the perceptual dis-
tance between the original image and its face-swapped counterpart,
rather than merely measuring the distance between the target and
source images. This is essential because a face’s overall impression
is influenced not only by its intrinsic parts (e.g., eyes, noses) but
also by factors such as facial contours and hairstyle. Moreover, face-
swapped images often contain artifacts and attribute inconsistencies,
resulting in a significant domain gap from natural images.
In this work, we present PerFace, a feature extractor tailored
for evaluating facial similarity. Unlike traditional face recognition
models that emphasize identity matching, PerFace is trained through
metric learning on human-annotated similarity assessments specifi-
cally derived from face-swapped images. To address the challenge
of directly quantifying similarity, we introduced a pairwise compari-
son task in which annotators select the face-swapped image that most
closely resembles a reference face-swapped image (Fig. 1). Building
on PerFace, we propose a comprehensive face anonymization frame-
work that leverages these refined features. Our framework outper-
forms conventional pretrained models (e.g., [13]) on facial similarity
assessment and more effectively selects images that are perceptually
dissimilar to the original while preserving key attributes such as age
and gender. Extensive validation using our human-annotated Sim-
CelebA dataset underscores the effectiveness and specificity of our
contributions to face anonymization.
2. RELATED WORKS
2.1. Face Recognition Models
In recent years, machine learning models have come to dominate
face recognition, a technology used in applications from smartphone
authentication to law enforcement. Typically, these systems identify
the face in a database that matches the identity of a given test image.
arXiv:2509.20281v1  [cs.CV]  24 Sep 2025
[FIGURE img_p0_000]
[FIGURE img_p0_001]
[FIGURE img_p0_002]


# [PAGE 2]
Problematic cases
fs*
*face-swapping
fs*
Fig. 2: Example of unnatural face swapping. Swapping a young
child’s face with an adult male’s can produce unrealistic images with
deep wrinkles and age-inappropriate contours. Similarly, gender-
based facial differences can also lead to unnatural results.
To achieve this, feature extractors are trained to compute distances
between feature vectors—proxies for facial similarity—using losses
such as contrastive loss [14, 24, 25, 26], triplet loss [15, 16, 27],
and angular margin-based losses [13, 19, 18]. However, because
these similarity measures are optimized for verification and identi-
fication—tasks focused on confirming whether two faces share the
same identity—they fall short when it comes to effectively quantify-
ing the degree of difference between faces of different individuals.
2.2. Face Anonymization via Face Swapping
Face swapping, in the context of facial anonymization, replaces only
identity-specific facial features while preserving non-identifying at-
tributes such as expression, pose, and lighting [1, 2, 3, 4, 5, 6, 7, 8, 9,
10, 11]. In contrast to conventional anonymization methods—such
as occlusion or blurring—that obscure the entire face and often de-
grade image quality, face swapping modifies only the features crit-
ical for identity recognition. In practical applications, this process
typically involves replacing a target face with one drawn from a set
of source faces that have sufficiently different identities. Although
state-of-the-art face recognition systems can detect and quantify dif-
ferences between target and source faces through deep feature em-
beddings [28], human observers may still perceive a high degree of
similarity or even believe the target and face-swapped images be-
long to the same individual, even when the model has assigned them
different identities.
3. METHOD
The primary goal of this work is to develop an effective facial feature
extractor that accurately measures facial similarity, facilitating the
selection of appropriate source and target faces for face-swapping.
Building on this, we propose a facial anonymization framework that
replaces a target face with a source face that shares key attributes
(e.g., gender, age) while ensuring that the original and face-swapped
images remain as perceptually distinct as possible. Traditional fa-
cial anonymization methods (e.g., [12]) typically rely on pretrained
face recognition models (e.g., ArcFace [13]). However, these mod-
els are not optimized for evaluating similarity between images of
entirely different identities or for assessing similarity distances in
face-swapped images. To address these limitations, we conduct hu-
man assessments of similarity in face-swapped images and construct
a dataset, SimCelebA, to train a facial feature extractor, PerFace, us-
ing metric learning.
3.1. SimCelebA Dataset
Firstly, we conducted a human assessment to create a training dataset
containing face-swapped images annotated with perceptual facial
similarity scores. Since assessing the absolute similarity between
two face-swapped images is inherently challenging, we adopted a
triplet-based approach. Each triplet comprised three images: a refer-
ence image (C) and two comparison images (A and B). Participants
were tasked with determining which of the two, A or B, bore a closer
resemblance to C.
To prevent similarity judgments from being overly influenced by
key facial attributes of the target (e.g., hair style, contour) or artifacts
caused by face-swapping, we kept the target face constant and se-
lected three distinct source images for swapping. This ensured that
all images presented to the participants were face-swapped, elimi-
nating potential biases arising from the presence of natural images.
We utilized the CelebAMask-HQ dataset [29], a large-scale col-
lection of 30,000 high-resolution face images. From this dataset,
we manually selected 80 target images (40 male and 40 female)
and selected 240 source images (120 male and 120 female), form-
ing 80 triplets. To minimize bias, images with obstructions such as
glasses were excluded from the dataset. Using these chosen target
and source images, we applied the SimSwap method [2] to generate
face-swapped images. This process resulted in a dataset of 6,400
samples.
We recruited 18 participants, ensuring that each triplet was anno-
tated by at least three individuals to capture human-perceived facial
similarity. To enhance the reliability of the annotations, we embed-
ded multiple dummy samples within the triplets. In these dummy
samples, two images depicted the same individual, ensuring that a
careful examination would unambiguously yield the correct answer.
Consequently, only annotations from participants who answered all
of these dummy samples correctly were considered valid. As a re-
sult, each sample ultimately received three high-quality annotations.
This dataset was divided into training, validation, and test sets for
use.
3.2. PerFace: facial similarity extractor trained on SimCelebA
Our goal is for the model to acquire the ability to evaluate facial sim-
ilarity in a human-like manner, going beyond mere identity match-
ing. To achieve this, we fine-tuned ArcFace [13] on our SimCelebA
dataset, leveraging its strong discriminative power in face recogni-
tion tasks to better capture human-perceived facial similarity in face-
swapped images. Given an annotated triplet (i.e., A, B, and C) of
face-swapped images, we propose using the following triplet loss to
train the model:
L = max
 0,
 cos(xi, x−
i ) −cos(xi, x+
i )

+ m

.
(1)
Here, xi ∈Rd represents the embedded feature of the reference
image (i.e., C) in the i-th sample. x+
i ∈Rd is the embedded feature
of the image chosen as more similar to the reference image xi by
the majority of annotators (i.e.., A or B). Similarly, x−
i
∈Rd is
the feature vector of the image chosen by the minority (i.e., A or
B). The feature vector dimension d is set to 512, following prior
studies [30, 13, 18, 19]. cos(·, ·) denotes cosine similarity, and m
represents the margin. This triplet loss is minimized to ensure that
the distance between similar pairs chosen by annotators decreases,
while the distance between non-selected pairs increases. As a result,
the model is fine-tuned such that the distances between embedded
features of similar faces, as perceived by humans, become smaller.
To facilitate face similarity comparisons, our dataset is designed to
select relatively similar faces, as described in Sec.3.1. Accordingly,
the loss function is also designed to focus on relative distances.
[FIGURE img_p1_003]
[FIGURE img_p1_004]
[FIGURE img_p1_005]
[FIGURE img_p1_006]
[FIGURE img_p1_007]
[FIGURE img_p1_008]
[FIGURE img_p1_009]

[EQ eq_p1_000] -> see eq_p1_000.tex

[EQ eq_p1_001] -> see eq_p1_001.tex

[EQ eq_p1_002] -> see eq_p1_002.tex


# [PAGE 3]
Candidates
Female
Young
1. Select a set with closer attributes. (gender/age)
2. Select the items with low similarity within the chosen set.
high
Similarity using PerFace
low
Candidates
Male
Female
Query
Candidates
Young
Older
Query
Fig. 3: Overview of the method for selecting a suitable source for
face swapping with a query image.
3.3. Face Anonymization with PerFace
In most existing work (e.g., ArcFace [13]), when employing face
swapping for anonymization, only the similarity between the source
and target faces is evaluated. In contrast, we aim to assess the sim-
ilarity between the target face and the face-swapped face using our
PerFace feature extractor. However, simply comparing all pairs of
source and target faces is prohibitively resource demanding, so we
first align facial attributes before and after swapping and then com-
pare faces only within each group.
The overview of the selection method, which consists of two
steps, is shown in Fig. 3. Considering practical applications, a pre-
defined set of face-swap candidates may sometimes be prepared.
Therefore, this study assumes that the face-swap candidates have
been annotated with attributes in advance. Conversely, it is assumed
that the target attributes are unknown. Since it is impossible to pre-
dict which face the user wishes to anonymize, pre-annotating the
target is impractical.
STEP 1: Group Selection. To ensure that the face-swapping pro-
cess remains natural, suitable face-swapping candidates are identi-
fied through facial similarity evaluation. Assume there is a face im-
age that the user wants to anonymize, referred to as the query image.
The attributes of the query image are determined. Within the set of
face-swapping candidates, images are grouped based on attributes.
Following previous studies [31], age and gender are considered as
attributes. We created 4 attribute groups (male, female, young, and
older) and their intersection sets considering both gender and age
(e.g.,young∩male), resulting in a total of 8 groups. For each at-
tribute group, the similarity with the query image is calculated, and
the group with the highest similarity is selected as the face-swapping
candidates.
STEP 2: Anonymization. Next, the anonymization process is per-
formed. As long as the selected group from STEP1 is used, the face-
swapped image is expected to maintain a certain degree of natural-
ness. The goal is to achieve anonymization while maintaining this
naturalness. Using the proposed method, our model trained via met-
ric learning is employed to estimate the similarity between the im-
ages of the query and the face-swapping candidates within the group.
At this stage, an image with low similarity in the group is chosen as
the source for face swapping. By choosing less similar face from
the same attribute group, face anonymization can be achieved while
DeepID
BlendFace
GhostFaceNet
FaceNet
ArcFace
Ours
VGGFace
OpenFace
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Fig. 4: Scatter plot of predicted similarity scores for similar and
dissimilar pairs by major face recognition models. The X-axis rep-
resents the predicted similarity between similar pairs, while the Y-
axis represents the predicted similarity between dissimilar pairs. The
gray dashed line represents the graph of Y = X. Points below
Y < X indicate that the pair judged as more similar has been as-
signed a higher similarity score by the model.
ensuring naturalness.
4. EXPERIMENTS AND DISCUSSION
4.1. Training details
We adopted ArcFace [13] as the pretrained model, which we trained
using the MS1MV3 [32] dataset. For the feature extractor, we em-
ployed a ResNet50[33]. We considered two cases for training: (1)
using all triplet data in the training set (D1), and (2) using only triplet
data with consistent annotations (D2).
Common Settings for Training Data D1 and D2. The batch size
was 32, momentum was set to 0.9, weight decay to 5e-4, and the
learning rate to 0.01. SGD was used as the optimizer, and the loss
margin in Equation (1) was set to 0.1.
4.2. Evaluation method
Using the test triplet data and the annotation results, similar pairs
and dissimilar pairs were created. Similar pairs consisted of the ref-
erence image and the image selected as more similar. Dissimilar
pairs consisted of the reference image and the image not selected as
more similar. If the similarity score for the similar pair was higher
than that for the dissimilar pair, the model prediction was considered
correct for that sample. Only samples with consistent annotations
across all three responses were adopted as evaluation data.
4.3. Comparison with other methods
Here, we compare our method with existing approaches capable of
measuring the distance between faces. We used D2 to ensure the
highest annotation quality. As for BlendFace [11], we used the offi-
cial code and weights. Other similarity evaluations were conducted
through the DeepFace framework [34], with RetinaFace [35] as the
detector and cosine similarity as the metric.
Fig. 4 presents scatter plots of similarity scores for similar and
dissimilar pairs. The proportion of points below Y < X corresponds
to the accuracy values shown in Table 1.
Our proposed method
significantly outperformed existing methods in terms of accuracy.
Models such as ArcFace [13], VGG Face [16], GhostFaceNets [23],
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


# [PAGE 4]
Table 1: Similarity evaluation results in comparison with previous
methods.
Method
Acc
DeepID[14]
0.604
VGG Face[16]
0.750
FaceNet[15]
0.715
OpenFace[17]
0.660
ArcFace[13]
0.701
BlendFace[11]
0.576
GhostFaceNets[23]
0.604
Ours
0.917
Table 2: Examples of evaluating similar and dissimilar pairs in
triplet samples using our method compared to others.
For each
triplet, the central image and the left-side image form a similar pair,
while the central image and the right-side image form a dissimilar
pair.
similar
dissimilar
similar
dissimilar
Method
DeepID[14]
0.993
0.997
0.998
0.998
VGG Face[16]
0.252
0.349
0.198
0.376
FaceNet[15]
0.207
0.261
0.405
0.339
OpenFace[17]
0.905
0.948
0.864
0.823
ArcFace[13]
0.287
0.273
0.288
0.314
BlendFace[11]
0.105
0.143
0.144
0.051
GhostFaceNets[23]
0.333
0.364
0.171
0.257
Ours
0.339
-0.006
0.183
0.055
and BlendFace[11] predicted similarity scores for both similar and
dissimilar pairs around 0.00 to 0.50. In contrast, OpenFace [17]
and DeepID [14] predicted high similarity scores for both similar
and dissimilar pairs. For the triplet samples used in this evaluation,
all three face images inherited the hairstyle and skin tone of target,
making it difficult for these models to understand differences in fa-
cial parts. Specific examples are presented in Table2. Comparison
methods make incorrect predictions, often assigning higher similar-
ity scores to the dissimilar pair or consistently predicting nearly iden-
tical scores.
4.4. Analyzing the effects on source/target knowledge
Considering the task of face anonymization, there may be cases
where the source used for anonymization is known in advance.
Therefore, we investigated how prior knowledge of identity affects
face similarity evaluation. Additionally, we also examined the im-
pact of training data quality on model performance.
Evaluation Data: [i], [ii], and [iii]. We evaluated with three types
of evaluation datasets: [i] Neither the source nor the target were in-
cluded in the training data. [ii] The target was not included in the
training data, but the source was included. [iii] The source was not
included in the training data, but the target was included. As sum-
marized in Fig. 5, set [ii] achieved the highest accuracy, suggesting
that familiarity with source faces improves performance. In con-
trast, [i] and [iii], which involved unknown source, resulted in lower
accuracy. This highlights the advantage of pretraining on known
face-swapping candidates.
Table 3: Mean±SD of facial similarity prediction accuracy.
Dataset
Acc in [i]
Acc in [ii]
Acc in [iii]
- (Original) 0.690
0.600
0.644
D1
0.753 ±0.025 0.795 ±0.006 0.770 ±0.014
D2
0.862 ±0.031 0.922 ±0.007 0.864 ±0.018
Pre Fine-tuning
Post Fine-tuning
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
Sim. of similar pair
Sim. of dissimilar pair
 [i] 
 [ii] 
 [iii] 
Fig. 5: Scatter plot of the similarity scores output by the model for
similar and dissimilar pairs in triplet data.
Training Data: D1 and D2. As shown in Table 3, we found that
while D1 benefits from a larger dataset, its inconsistent annotations
hinder performance. In contrast, D2 achieves higher accuracy due
to its superior annotation quality. For the following experiments, D2
was employed.
Key Insight. High-quality training data (D2) and familiarity with
source triplets ([ii]) are critical for optimal model performance, em-
phasizing the importance of dataset curation and pretraining strate-
gies.
4.5. Selection of attribute groups
This section presents the results of the performance on attribute
group classification task. We determined the most similar attribute
group for a given face query image, based on our perceptual similar-
ity metric.
Details of the attribute classification experiment. In this experi-
ment, we defined the following eight attribute groups: male, female,
young, older, and their combinations: young∩male, older∩male,
young∩female, and older∩female. These attribute groups were de-
fined based on annotations from CelebAMask-HQ [29]. Addition-
ally, we rechecked the annotations and corrected several annotation
errors. For the four subdivided groups (young∩male, older∩male,
young∩female, older∩female), 100 face images were randomly se-
lected for each group. For the broader male, female, young, and
older groups, each group was constructed as the union of the corre-
sponding subdivided groups, containing 200 face images per group.
By dividing the dataset into these groups, we prepared to analyze
how similarity evaluation changes across attribute groups.
For an image IGi,j in an attribute group Gi and a query im-
age Iq, the distance dIq,IGi,j is defined as: dIq,IGi,j
= 1 −
cos(f(Iq), f(IGi,j)) where f(·) is the feature extractor in our pro-
posed model. The distance between an attribute group Gi and the
query image Iq is denoted as DIq,Gi. The attribute group G∗most
similar to the query image Iq is defined as: G∗= arg min
Gi
DIq,Gi
[FIGURE img_p3_023]
[FIGURE img_p3_024]
[FIGURE img_p3_025]
[FIGURE img_p3_026]
[FIGURE img_p3_027]
[FIGURE img_p3_028]
[FIGURE img_p3_029]
[FIGURE img_p3_030]
[FIGURE img_p3_031]
[FIGURE img_p3_032]
[FIGURE img_p3_033]
[FIGURE img_p3_034]


# [PAGE 5]
Pre Fine-tuning
Post Fine-tuning
Older/Male
Query
Similarity
male
female
young
older
young∩male
young∩female
older∩male
older∩female
Similarity
male
female
young
older
young∩male
young∩female
older∩male
older∩female
Fig. 6: Distributions of similarity between query images and face
swap candidates within each attribute group.
We used a total of 1000 query images, selecting 250 images
from each of the attribute groups: young∩male, young∩female,
older∩male, and older∩female. As the evaluation, for each query
image Iq, the group G∗with the minimum distance was identi-
fied, and the accuracy was evaluated by calculating the proportion
of queries where the predicted attribute label matched the actual at-
tribute label.
Results. Fig. 6 shows the similarity distribution between the query
image and each attribute group. Here, we use an image with the
attributes “male ∩older” as an example to illustrate the analysis. As
a result, in the fine-tuned model, the variance within attribute groups
and the variance between groups both increased compared to before
fine-tuning. It was observed that “male,” “older,” and “male ∩older”
had higher similarities than other groups.
Based on the similarity distribution, we classified the closest at-
tribute group to the query image using the 95% upper confidence
interval as the distance. We then verified whether the attribute of
the selected group matched the actual attribute of the query image.
The classification method was evaluated under two conditions: con-
sidering single attribute and considering multiple attributes simul-
taneously. For single attribute, we perform binary classification of
gender (male or female) and age group (young or older). When
considering multiple attributes simultaneously, we classify into four
groups: “young ∩male,” “young ∩female,” “older ∩male,” and
“older ∩female.”
As shown in Table 4, after fine-tuning, the AUC increased in
most cases compared to before fine-tuning, indicating improved
classification accuracy. Focusing on the differences in classification
accuracy by attribute, the accuracy for male/female classification is
higher than that for young/older classification. This suggests that the
model prioritizes gender (male/female) over age group (young/older)
when evaluating similarity. This can likely be traced back to the an-
notators’ judgments in the training dataset, which may have placed
greater emphasis on gender distinctions.
4.6. Selection of face swap candidates
Based on the attributes of the query image determined in Sec. 4.5,
suitable face swap candidates for anonymization were selected from
those sharing the same attributes. To select candidates with lower
similarity to the query, the face swap candidates in the selected group
were sorted by similarity to the query, as examples shown in Fig. 7.
Given the diversity of human faces, even among faces that are
not very similar to the query image, various types of “dissimilar-
ity” may exist. To provide users with more flexibility in selecting
a “dissimilar” face, this selection algorithm can effectively recom-
mend multiple face swap candidates with relatively low similarity.
Table 4: Classification accuracy when the distance DIq,Giis de-
fined as the upper limit of the 95% confidence interval of distances
dIq,IGi,j.
Category
PrecisionRecallAccuracyAUC
Pre
Fine-tuning
Male
0.930
0.936
0.933
0.933
Female
0.936
0.930
Young
0.829
0.812
0.822
0.822
Older
0.816
0.832
Young∩Male
0.698
0.812
0.865
0.847
Young∩Female
0.807
0.652
0.874
0.800
Older∩Male
0.801
0.788
0.898
0.861
Older∩Female
0.770
0.804
0.891
0.862
Post
Fine-tuning
Male
0.959
0.972
0.965
0.965
Female
0.972
0.958
Young
0.871
0.754
0.821
0.821
Older
0.783
0.888
Young∩Male
0.716
0.856
0.879
0.871
Young∩Female
0.822
0.664
0.880
0.808
Older∩Male
0.872
0.788
0.918
0.875
Older∩Female
0.791
0.864
0.909
0.894
Candidates
Query
Similarity
high
low
Fig. 7: Sorted face swap candidates in the selected attribute group,
showing the top two most similar and the bottom five least similar
candidates.
5. CONCLUSION
We propose a novel method to address limitations in face similarity
prediction for face anonymization via natural face swapping. Con-
ventional methods struggle to evaluate nuanced similarities, such as
distinguishing “completely different” from “highly similar but dif-
ferent individuals.”
To overcome this, we introduce a new task to assess “how simi-
lar a different identity is,” using our perceptual similarity model Per-
Face. We constructed an evaluation dataset via user studies and de-
veloped a transfer-learning-based model optimized to capture subtle
inter-individual similarities.
Our PerFace model significantly outperformed baseline models
in face similarity judgment tasks. Additionally, it achieved supe-
rior accuracy in attribute classification, highlighting the influence of
facial attributes on perceptual similarity judgments and offering in-
sights into perception-based evaluations.
A limitation of this work is its inclusion of subjective factors, es-
pecially since facial similarity inherently involves personal percep-
tion. Consequently, broader and more diverse perspectives are cru-
cial to capture the sense of similarity that people share. We hope this
paper will inspire the creation of larger and more inclusive datasets.
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
[FIGURE img_p4_046]
[FIGURE img_p4_047]
[FIGURE img_p4_048]
[FIGURE img_p4_049]
[FIGURE img_p4_050]
[FIGURE img_p4_051]
[FIGURE img_p4_052]
[FIGURE img_p4_053]
[FIGURE img_p4_054]
[FIGURE img_p4_055]
[FIGURE img_p4_056]
[FIGURE img_p4_057]
[FIGURE img_p4_058]
[FIGURE img_p4_059]
[FIGURE img_p4_060]
[FIGURE img_p4_061]
[FIGURE img_p4_062]
[FIGURE img_p4_063]
[FIGURE img_p4_064]
[FIGURE img_p4_065]
[FIGURE img_p4_066]
[FIGURE img_p4_067]
[FIGURE img_p4_068]
[FIGURE img_p4_069]

[EQ eq_p4_000] -> see eq_p4_000.tex

[EQ eq_p4_001] -> see eq_p4_001.tex

[EQ eq_p4_002] -> see eq_p4_002.tex

[EQ eq_p4_003] -> see eq_p4_003.tex

[EQ eq_p4_004] -> see eq_p4_004.tex

[EQ eq_p4_005] -> see eq_p4_005.tex

[EQ eq_p4_006] -> see eq_p4_006.tex

[EQ eq_p4_007] -> see eq_p4_007.tex

[EQ eq_p4_008] -> see eq_p4_008.tex


# [PAGE 6]
6. REFERENCES
[1] “deepfakes,”
https://github.com/deepfakes/
faceswap, Accessed: Dec. 26, 2024.
[2] Renwang Chen, Xuanhong Chen, Bingbing Ni, and Yanhao
Ge, “Simswap: An efficient framework for high fidelity face
swapping,” in MM, 2020, pp. 2003–2011.
[3] Kihong Kim, Yunho Kim, Seokju Cho, Junyoung Seo, Jisu
Nam, Kychul Lee, Seungryong Kim, and KwangHee Lee,
“Diffface: Diffusion-based face swapping with facial guid-
ance,” arXiv preprint arXiv:2212.13344, 2022.
[4] Leslie W¨ohler, Susana Castillo, and Marcus Magnor, “Person-
ality analysis of face swaps: can they be used as avatars?,” in
IVA. 2022, IVA ’22, Association for Computing Machinery.
[5] Yuval Nirkin, Yosi Keller, and Tal Hassner, “Fsgan: Subject
agnostic face swapping and reenactment,” in ICCV, 2019, pp.
7184–7193.
[6] Lingzhi Li, Jianmin Bao, Hao Yang, Dong Chen, and Fang
Wen, “Advancing high fidelity identity swapping for forgery
detection,” in CVPR, 2020, pp. 5073–5082.
[7] Yuhao Zhu, Qi Li, Jian Wang, Cheng-Zhong Xu, and Zhenan
Sun, “One shot face swapping on megapixels,” in CVPR, 2021,
pp. 4834–4844.
[8] Gege Gao, Huaibo Huang, Chaoyou Fu, Zhaoyang Li, and
Ran He, “Information bottleneck disentanglement for identity
swapping,” in CVPR, 2021, pp. 3404–3413.
[9] Yuhan Wang, Xu Chen, Junwei Zhu, Wenqing Chu, Ying Tai,
Chengjie Wang, Jilin Li, Yongjian Wu, Feiyue Huang, and
Rongrong Ji, “Hififace: 3d shape and semantic prior guided
high fidelity face swapping.,” in IJCAI, 2021, pp. 1136–1142.
[10] Wenliang Zhao, Yongming Rao, Weikang Shi, Zuyan Liu, Jie
Zhou, and Jiwen Lu, “Diffswap: High-fidelity and control-
lable face swapping via 3d-aware masked diffusion,” in CVPR,
2023, pp. 8568–8577.
[11] Kaede Shiohara,
Xingchao Yang,
and Takafumi Take-
tomi,
“Blendface: Re-designing identity encoders for face-
swapping,” in ICCV, 2023, pp. 7634–7644.
[12] Umur A Ciftci, Gokturk Yuksek, and Ilke Demir, “My face
my choice:
Privacy enhancing deepfakes for social media
anonymization,” in WACV, 2023, pp. 1369–1379.
[13] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou,
“Arcface: Additive angular margin loss for deep face recogni-
tion,” in CVPR, 2019, pp. 4690–4699.
[14] Yi Sun, Yuheng Chen, Xiaogang Wang, and Xiaoou Tang,
“Deep learning face representation by joint identification-
verification,” NeurIPS, vol. 27, 2014.
[15] Florian Schroff, Dmitry Kalenichenko, and James Philbin,
“Facenet: A unified embedding for face recognition and clus-
tering,” in CVPR, 2015, pp. 815–823.
[16] Omkar Parkhi, Andrea Vedaldi, and Andrew Zisserman, “Deep
face recognition,” in BMVC. British Machine Vision Associa-
tion, 2015.
[17] Tadas
Baltruˇsaitis,
Peter
Robinson,
and
Louis-Philippe
Morency, “Openface: an open source facial behavior analy-
sis toolkit,” in WACV. IEEE, 2016, pp. 1–10.
[18] Weiyang Liu, Yandong Wen, Zhiding Yu, Ming Li, Bhiksha
Raj, and Le Song, “Sphereface: Deep hypersphere embedding
for face recognition,” in CVPR, 2017, pp. 212–220.
[19] Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong,
Jingchao Zhou, Zhifeng Li, and Wei Liu,
“Cosface: Large
margin cosine loss for deep face recognition,” in CVPR, 2018,
pp. 5265–5274.
[20] Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou,
“Magface: A universal representation for face recognition and
quality assessment,” in CVPR, 2021, pp. 14225–14234.
[21] Haibo Qiu, Baosheng Yu, Dihong Gong, Zhifeng Li, Wei Liu,
and Dacheng Tao, “Synface: Face recognition with synthetic
data,” in ICCV, 2021, pp. 10880–10890.
[22] Minchul Kim, Anil K. Jain, and Xiaoming Liu,
“Adaface:
Quality adaptive margin for face recognition,” in CVPR, 2022,
pp. 18750–18759.
[23] Mohamad Alansari, Oussama Abdul Hay, Sajid Javed, Abdul-
hadi Shoufan, Yahya Zweiri, and Naoufel Werghi,
“Ghost-
facenets: Lightweight face recognition model from cheap op-
erations,” IEEE Access, vol. 11, pp. 35429–35446, 2023.
[24] Yi Sun, Xiaogang Wang, and Xiaoou Tang, “Deeply learned
face representations are sparse, selective, and robust,”
in
CVPR, 2015, pp. 2892–2900.
[25] Yi Sun, Ding Liang, Xiaogang Wang, and Xiaoou Tang,
“Deepid3: Face recognition with very deep neural networks,”
arXiv preprint arXiv:1502.00873, 2015.
[26] Dong Yi, Zhen Lei, Shengcai Liao, and Stan Z Li, “Learn-
ing face representation from scratch,”
arXiv preprint
arXiv:1411.7923, 2014.
[27] Changxing Ding and Dacheng Tao, “Robust face recognition
via multimodal deep face representation,” IEEE transactions
on Multimedia, vol. 17, no. 11, pp. 2049–2058, 2015.
[28] Jingyi Cao, Xiangyi Chen, Bo Liu, Ming Ding, Rong Xie,
Li Song, Zhu Li, and Wenjun Zhang, “Face de-identification:
State-of-the-art methods and comparative studies,”
arXiv
preprint arXiv:2411.09863, 2024.
[29] Cheng-Han Lee, Ziwei Liu, Lingyun Wu, and Ping Luo,
“Maskgan: Towards diverse and interactive facial image ma-
nipulation,” in CVPR, 2020.
[30] Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao, “A
discriminative feature learning approach for deep face recog-
nition,” in ECCV. Springer, 2016, pp. 499–515.
[31] Jiseob Kim, Jihoon Lee, and Byoung-Tak Zhang, “Smooth-
swap: A simple enhancement for face-swapping with smooth-
ness,” in CVPR, 2022, pp. 10779–10788.
[32] Jiankang Deng, Jia Guo, Debing Zhang, Yafeng Deng, Xiangju
Lu, and Song Shi, “Lightweight face recognition challenge,”
in ICCVW, 2019, pp. 2638–2646.
[33] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun,
“Deep residual learning for image recognition,”
in CVPR,
2016, pp. 770–778.
[34] Sefik Ilkin Serengil and Alper Ozpinar, “Lightface: A hybrid
deep face recognition framework,” in INISTA. IEEE, 2020, pp.
1–5.
[35] Jiankang Deng, Jia Guo, Evangelos Ververas, Irene Kotsia, and
Stefanos Zafeiriou, “Retinaface: Single-shot multi-level face
localisation in the wild,” in CVPR, 2020, pp. 5203–5212.

[EQ eq_p5_000] -> see eq_p5_000.tex

[EQ eq_p5_001] -> see eq_p5_001.tex
