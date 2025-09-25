## Feed-Forward SceneDINO for Unsupervised Semantic Scene Completion

Figure 1. SceneDINO overview. Given a single input image ( left ), SceneDINO predicts both 3D scene geometry and 3D features in the form of a feature field ( middle ) in a feed-forward manner, capturing the structure and semantics of the scene. Unsupervised distillation and clustering of SceneDINO's feature space leads to unsupervised semantic scene completion predictions ( right ).

## Abstract

Semantic scene completion (SSC) aims to infer both the 3D geometry and semantics of a scene from single images. In contrast to prior work on SSC that heavily relies on expensive ground-truth annotations, we approach SSC in an unsupervised setting. Our novel method, SceneDINO , adapts techniques from self-supervised representation learning and 2D unsupervised scene understanding to SSC. Our training exclusively utilizes multi-view consistency self-supervision without any form of semantic or geometric ground truth. Given a single input image, SceneDINO infers the 3D geometry and expressive 3D DINO features in a feed-forward manner. Through a novel 3D feature distillation approach, we obtain unsupervised 3D semantics. In both 3D and 2D unsupervised scene understanding, SceneDINO reaches state-of-the-art segmentation accuracy. Linear probing our 3D features matches the segmentation accuracy of a current supervised SSC approach. Additionally, we showcase the domain generalization and multi-view consistency of SceneDINO, taking the first steps towards a strong foundation for single image 3D scene understanding.

## 1. Introduction

Understanding the geometry and semantics of 3D scenes from image observations is a fundamental computer vision task with broad applications in robotics [26], autonomous driving [46, 65], medical image analysis [18, 112], and civil engineering [69]. The Semantic Scene Completion (SSC)

To appear in Proceedings of the IEEE/CVF International Conference on Computer Vision , Honolulu, Hawai'i, USA, 2025.

© 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.

3D Feature Field

task unifies 3D geometry and semantic prediction from limited image observations [63, 88, 95]. Recent progress in SSC has been primarily driven by utilizing supervised learning [37, 87, 95]. However, acquiring large-scale 3D annotations is highly labor-intensive [65]. While significant resources have been invested in collecting human annotations for 2D tasks [52, 84], annotating similar amounts of data in 3D remains unapproached. This motivates approaching SSC without the need for manually annotated data.

Existing SSC approaches rely on ground-truth semantic annotations and frequently utilize additional supervision from LiDAR scans [37, 45, 73, 95]. In contrast, we are the first to approach SSC in a fully unsupervised setting, i.e. without task supervision or other supervised components. In particular, we aim to approach SSC from a single image without relying on any human annotations, only learning from unlabeled multi-view images using self-supervision. This setting is extremely challenging for two reasons: first , the human-defined nature of semantic taxonomies is ambiguous, and second , a single image only provides a partial observation of the scene with many invisible areas.

We take inspiration from recent advances in self-supervised learning (SSL) of 2D representations and 3D reconstruction. 2D SSL representations, such as from DINO [11], have been shown effective for 2D unsupervised scene understanding [32, 103]. 3D reconstruction approaches successfully leveraged SSL from multi-view data to infer dense 3D geometry from a single image [33, 107].

In this paper, we present SceneDINO , to the best of our knowledge, the first approach for unsupervised semantic scene completion. Trained using 2D SSL features

from DINO [11] and multi-view self-supervision [107], SceneDINO predicts both 3D geometry and 3D features from a single image during inference in a feed-forward manner. Our general 3D feature representations enable us to approach unsupervised 3D scene understanding. Harnessing our expressive 3D features, we propose a novel 3D feature distillation approach for obtaining unsupervised semantic predictions in 3D. While we focus on the task of unsupervised SSC, SceneDINO's features are general, offering a foundation for different 3D scene-understanding tasks by building on our 3D feature field.

Specifically, we make the following contributions: (i) We introduce SceneDINO, the first approach predicting dense 3D geometry and expressive 3D features in a feedforward manner from a single image. (ii) We effectively distill SceneDINO's feature field representation in 3D, obtaining unsupervised semantic predictions. (iii) We demonstrate the first fully unsupervised SSC results. We build a simple yet competitive unsupervised SSC baseline, lifting unsupervised 2D semantic predictions. Our SceneDINO approach outperforms this SSC baseline in unsupervised SSC as well as established 2D approaches in 2D semantic segmentation. (iv) Finally, we also showcase the domain generalization ability and multi-view consistency of SceneDINO.

## 2. Related Work

Single-image scene reconstruction. Estimating 3D geometry from image observations is a fundamental task in computer vision and has been studied for decades [36]. Traditional approaches, such as structure from motion [89], as well as recent neural distance fields (NeRFs) [74], perform scene reconstruction using multiple images, as reviewed by multiple surveys [34, 108, 119]. Recently, estimating dense 3D geometry from a single image have been approached [8, 33, 80, 85, 96, 102, 107, 113]. Unlike monocular depth estimation [75], these approaches predict the depth for visible and occluded regions, reconstructing a complete scene. Behind the Sences (BTS) [107] introduced an approach for unsupervised single-image scene reconstruction using multi-view self-supervision, which infers dense 3D geometry in a feed-forward manner. Our approach extends BTS by additionally lifting self-supervised features into 3D for unsupervised 3D scene understanding.

Semantic scene completion (SSC), also known as 3D semantic occupancy prediction, aims to jointly estimate the 3D geometry and semantics of a scene [62, 63, 95, 117]. Initial approaches used 3D semantic and geometric annotations and addressed indoor scenes [6, 13, 57-59, 67, 116], outdoor scenes with LIDAR [16, 61, 86, 87, 109], or both domains [8, 73]. Using birds-eye views has been proven effective for SSC [44, 64, 99]. To overcome the need for 3D annotations, approaches for using 2D annotations have

been proposed [37, 45, 81]. While SelfOcc [45] and DerOcc [81] use multiple inference views, S4C [37] performs single-image SSC. In particular, S4C [37] employs a supervised 2D model and lifts 2D multi-view semantic predictions into 3D. In contrast to using 2D annotations, GaussTR [48] uses 2D foundation models for SSC and multiple views during inference. However, GaussTR relies on heavily supervised foundation models, including SAM [52] and MetricDv2 [42], and uses weak supervision from image/text pairs. To the best of our knowledge, there is no method for approaching SSC without the need for any ground-truth annotations. Our work presents the first unsupervised SSC approach, utilizing lifted SSL features and a single RGB input image for inference.

Self-supervised representation learning (SSL) aims to extract general features from data without annotations, facilitating various downstream tasks such as segmentation [24]. Recent SSL methods, often based on Vision Transformers (ViTs) [23], leverage clustering [2, 9, 10, 47, 60], masked modeling [20, 29, 39, 76, 106], contrastive learning [13, 2, 14, 38, 40, 41], or negative-free [4, 5, 11, 28, 79] pretext tasks [22, 78] for large-scale training. Stateof-the-art models, e.g., DINO [11], produce semantically rich, dense features, driving recent advances in 2D unsupervised scene understanding [32, 103]. We here aim to bring expressive features from DINO [11, 79] to 3D for SSC.

2D-to-feature lifting. The expressiveness of 2D visual representations has motivated lifting 2D features into 3D [93, 110]. Existing approaches utilize multi-view 2D features for 3D feature lifting [30, 43, 49, 53, 72, 82, 92, 93, 97, 100, 101, 105, 110, 115]. Lifting 2D features is effective in various tasks, including few-shot semantic occupancy prediction [110], and refining 2D representations [115]. However, existing feature-lifting approaches fit to a single scene [49, 53, 92, 93, 100, 101, 110, 115], require RGB-D inputs [30, 43, 72, 97, 105], or work on 3D point cloud inputs [82]. The only feed-forward approaches that use RGB inputs and lift 2D features, which we are aware of, are GaussTR [48]; MVSPlat630 [15]. However, both approaches utilize multiple input images during inference, and MVSPlat360 [15] only predicts low-dimensional feature representations, which are not suitable for unsupervised scene understanding. In contrast, we propose the first feedforward approach for inferring lifted high-dimensional and rich 3D features using a single input image.

2D unsupervised semantic segmentation partitions images automatically into semantically meaningful regions without any form of human annotations. Early deep learning-based methods [17, 35, 47] approach the problem via representation learning. Leveraging SSL features from DINO as an inductive prior, STEGO [32] distills the feature representation into a lower-dimensional space for unsupervised probing. Building up on STEGO, subsequent methods

Figure 2. SceneDINO architecture, rendering, and training. (a) Inference: Given a single input image I$_{0}$ during inference, a 2D encoder-decoder ξ produces the embedding E from which the local embedding e$_{u}$ is interpolated. The MLP encoder ϕ takes in e$_{u}$ and 3D position x$_{i}$ , and predicts both the density σ$_{x}$$_{i}$ and the 3D feature f$_{x}$$_{i}$ . Using a lightweight unsupervised segmentation head h , we can obtain semantic predictions χ$_{x}$$_{i}$ using f$_{x}$$_{i}$ . (b) Rendering: Our feature field allows for volume rendering by shooting rays through it, yielding depth d and f in 2D. Color c$_{i}$ is sampled from an another view (e.g., I$_{1}$ ) using u$_{s}$ and rendered to obtain the reconstructed color c . (c) Multi-view training: We render 2D views (features &amp; images) from our feature field and reconstruct the training views.

[31, 50, 91, 94] propose enhancements to the distillation. Our approach follows the idea of STEGO [32], extending it to 3D and integrating feature distillation using our 3D feature field to build the first unsupervised SSC approach.

## 3. Unsupervised Semantic Scene Completion

We approach semantic scene completion (SSC) without any form of manual supervision. To this end, we first describe SceneDINO, predicting 3D geometry and expressive 3D features from a single image in a feed-forward manner (Sec. 3.1), and SceneDINO's multi-view training (Sec. 3.2). Next, we present our 3D feature distillation approach to obtain unsupervised 3D semantic predictions (Sec. 3.3). An overview of our full pipeline, including inference, rendering, and multi-view self-supervision, is provided in Fig. 2.

Notation. Let I$_{0}$ ∈ [0 , 1] × 3 × $^{W}$be a single RGB input image (for both training &amp; inference) with corresponding pose T$_{0}$ ∈ R 4 × $^{4}$and projection matrix K$_{0}$ ∈ R 3 × $^{4}$. For training, let ( I$_{v}$ , T$_{v}$ , K$_{v}$ ) with v ∈ { 1 , 2 , . . . , n } , be n additional views for multi-view self-supervision. Assuming a pinhole camera model, any 3D point x ∈ R 3 in world coordinates can be projected onto the image plane of view v and the input view v = 0 with the perspective projection π$_{v}$ ( x ).

## 3.1. SceneDINO

Given a single input image I$_{0}$ , SceneDINO represents the dense geometric structure and features of a scene as a continuous mapping from world coordinates x ∈ R 3 to a volumetric density σ$_{x}$ ∈ R 1 + and a feature f$_{x}$ ∈ R $^{D}$. This continuous output representation is often called a feature field . While SceneDINO could represent any feature space, we aim for expressive SLS features from DINO [11, 79].

Architecture &amp; feature field inference. Our SceneDINO architecture comprises two main parts: a 2D encoder-

decoder ξ and an MLP decoder ( cf. Fig. 2a), following BTS [107]. ξ takes in I$_{0}$ and produces a per-pixel embedding E ∈$_{R}$ D × $^{H}$× W with D$_{e}$ dimensions. Intuitively, every spatial element of E represents a camera ray through a pixel, capturing both local geometry and features.

To infer the feature at a 3D position x , we employ a two-layer MLP decoder ϕ ( cf. Fig. 2a). Given a position x within the camera frustum, we project x into the camera plane, obtaining the pixel location u$_{i}$ = π$_{0}$ ( x ). We query E at the position u using bilinear interpolation, obtaining the local embedding e$_{u}$ . Based on the embedding e$_{u}$ , the pixel position u , and the distance d$_{x}$ ∈ R 1 + of x to the camera, we obtain the density σ$_{x}$ and feature prediction f$_{x}$ as

$$( \sigma _ { x }, f _ { x } ) = \phi ( e _ { u }, \gamma ( u, d _ { x } ) ), \quad \quad ( 1 )$$

where γ denotes a positional encoding [74].

Feature, depth &amp; color volume rendering. SceneDINO predicts a continuous feature field from a single image. This representation can be used to render features and depth in 2D from an arbitrary viewpoint (cf. Fig. 2b), following the discretization strategy of Max et al. [71]. Given a viewpoint ( T$_{r}$ , K$_{r}$ ), we sample L points x$_{i}$ along the ray through pixel u$_{r}$ , with distance δ$_{i}$ between x$_{i}$ and x$_{i}$$_{+}$$_{1}$ . Based on the volumetric densities σ$_{x}$$_{i}$ (cf. Eq. 1), we can compute the probabilities α$_{i}$ of the ray ending between x$_{i}$ and x$_{i}$$_{+}$$_{1}$ , and accumulate these into V$_{i}$ , the probability of x$_{i}$ being visible:

$$V _ { i } = \prod _ { j = 1 } ^ { i - 1 } \left ( 1 - \alpha _ { j } \right ), \ \ w i t h \, \alpha _ { i } = 1 - \exp ( - \sigma _ { x _ { i } } \delta _ { i } ) \. \ ( 2 )$$

Using V$_{i}$ and α$_{i}$ , we render depth d$_{u}$ and feature f$_{u}$ , from the estimated features f$_{x}$$_{i}$ from Eq. (1) and distances d$_{x}$$_{i}$ to x$_{i}$ onto the image plane at position u$_{r}$ as

$$\hat { f } _ { u _ { r } } = \sum _ { i = 1 } ^ { L } V _ { i } \alpha _ { i } f _ { x _ { i } } \quad \quad \hat { d } _ { u _ { r } } = \sum _ { i = 1 } ^ { L } V _ { i } \alpha _ { i } d _ { x _ { i } } \,. \quad ( 3 )$$

The differentiability of this rendering process enables us to self-supervise SceneDINO using multi-view images and their 2D feature representations ( e.g. , from DINO [11]). SceneDINO predicts 3D geometry and features, but does not predict color as we focus on semantic downstream tasks. To obtain color for image reconstruction during training, we follow the color sampling approach of BTS [107].

## 3.2. 3D feature field training

We train SceneDINO using multi-view self-supervision (cf. Fig. 2c), aiming to obtain an expressive and view-consistent feature field without the need for any form of manual annotations. For self-supervision, we sample n + 1 views I$_{v}$ with camera parameters T$_{1}$ , K$_{v}$ from the data and obtain dense 2D features F$_{v}$ from a self-supervised VIT (e.g., DINO [11]). Note that the 2D features entail a resolution of F$_{v}$ ∈ R D × H × p , where the VIT patch size p . The set of training views and features V = { ( I$_{v}$ , V$_{v}$ , F$_{v}$ ) | v = 0 , . . . , n } is randomly partitioned into two subsets V$_{source}$ and V$_{target}$ . Training reconstructs the views V$_{target}$ using the views of V$_{source}$ . In practice, we use a randomly sampled set of image patches that align with the VIT patches instead of the full image. In the following, we still refer to images for the sake of brevity.

Image reconstruction. We aim to learn the geometry of our feature field via multi-view photometric consistency. In particular, for every image I$_{v}$ ∈ V$_{v}$ , we derive a reconstructed image I$_{t,s}$ from every view s in V$_{source}$ using differentiable rendering and color sampling. Equipped with both the reconstructed image I$_{t,s}$ and the target image I$_{t}$ , we compute our photometric loss per patch as

$$\mathcal { L } _ { p } = \min _ { I _ { v } \in V _ { source } } \left ( \lambda _ { L 1 } ( I _ { t }, \hat { I } _ { t, s } ) + \lambda _ { S S M } \left ( I _ { t, s } ( I _ { t, s } ) \right ) \right ) \cdot ( 4 )$$

We only consider the minimum per-patch loss across the different views in V$_{source}$ . The scalars λ$_{1}$ and λ$_{Ssim}$ weight the absolute error L$_{1}$ and the SSIM loss L$_{Ssim}$ [104].

To regularize the 3D geometry prediction, we impose smoothness using an edge-aware smoothness loss [27]. Based on the estimated depth d$_{u}$ , (cf. Eq. 3), we obtain the inverse and mean-normalized depth d ∗ u , using d ∗ u , we compute the edge-aware smoothness L$_{s}$ for pixel u$_{t}$ as

$$\mathcal { L } _ { s } = \left | \nabla \mathbf x \mathbf x _ { u _ { t } } ^ { \prime } | e ^ { - \left | \nabla \mathbf x _ { u _ { t } } ^ { \prime } | } + \left | \nabla \mathbf y \mathbf x _ { u _ { t } } ^ { \prime } | e ^ { - \left | \nabla \mathbf y _ { u _ { t } } ^ { \prime } | }, \ \ ( 5 )$$

using the first spatial derivatives ∇$_{x}$ and ∇$_{y}$ at u$_{t}$ .

Feature reconstruction. We learn a multi-view consistent and expressive 3D feature field using the 2D features F$_{t}$ from V$_{target}$ . As we aim to learn a high-resolution (continuous) feature field, we render 2D features using Eq. 3 at the full image resolution F$_{t}$ ∈ R D × H × W . To compensate for

$^{1}$Note, camera poses can be obtained using unsupervised visual SLAM [7], strictly adhering to the fully unsupervised setting.

Figure 3. 3D feature distillation. Given an input image, SceneDINO predicts a 3D feature field. 3D features f$_{X}$ and f$_{Y}$ are sampled from the feature field. For f$_{X}$ , we obtain f$_{Y}$ kNN and f$_{Y}$ krand from the feature buffer. The segmentation head h distills the features into a low-dimensional space and is trained using L$_{dist}$ .

the reduced spatial dimension of F$_{t}$ , we employ the downsampler ν$_{t}$ proposed by Fu et al. [25] to our rendered features F$_{t}$ . While current 2D SSD features capture semantics, they lack multi-view consistency, i.e. , due to positional encodings used in ViTs [111], leading to different features for identical visual content at two distinct positions in an image. As we aim for multi-view consistency, we compensate for this by learning a constant decomposition F ∈ R D × H × p of features induced by positional encodings. Our feature loss is defined per feature as

$$\mathcal { L } _ { t } = 1 - \cos \lim _ { \mathbf x \to \mathbf y } ( \mathbf F _ { t } \mathbf y ) + \mathbf F _ { t } ), \quad \quad ( 6 )$$

where cosim is the cosine similarity between two features.

As image edges correlate with semantic edges and to further impose consistency, we regularize the rendered features F$_{t}$ using an edge-aware smoothness loss per feature

$$\mathcal { L } _ { f _ { s } } = | \nabla _ { x } \mathbf F _ { t } ^ { \prime } | e ^ { - | \nabla _ { x } \mathbf f _ { t } | } + | \nabla _ { y } \mathbf F _ { t } ^ { \prime } | e ^ { - | \nabla _ { y } \mathbf l _ { t } | }. \quad ( 7 )$$

Our final loss for training SceneDINO is a weighted sum of the photometric loss, the feature loss, and both smoothness losses L$_{sceneDINO}$ = λ$_{L}$cp + λ$_{S}$$_{s}$ + λ$_{L}$f$_{L}$ + λ$_{f}$L$_{fs}$ , averaged over all pixels and features.

## 3.3. 3D feature distillation for unsupervised SSC

Given the expressive feature field representation, we aim to obtain unsupervised semantic predictions for SSC. While naïve k -means [68, 70] can yield meaningful pseudo semantics, distilling features into a lower-dimensional space has been shown to be more effective in 2D semantic segmentation [32, 54]. To this end, we present a novel 3D feature distillation approach (cf. Fig. 3). We train a pointwise segmentation head h , mapping f$_{X}$ ∈ R D to a lowerdimensional distilled representation z ∈ R $^{K}$, with K ≪ D . The resulting distilled space is clustered to obtain pseudosemanic predictions p$_{X}$ ∈ [0 , 1] $^{C}$, with C pseudo classes.

Existing work in 2D unsupervised semantic segmentation has shown that SSL feature correspondence captures

semantic class co-occurrence [32]. This correspondence between two batches of N sample points X = [ x$_{1}$, . . . , x$_{N}$ ] and Y = [ y$_{1}$, . . . , y$_{N}$ ] can be expressed by pairwise feature similarity S$_{i,j}$ = cos-sim ( f$_{X}$$_{i}$, f$_{Y}$$_{j}$ ) ∈ [ - 1 , 1]. Similarly, we can express the correspondence in the distilled feature space by S h$_{i,j}$ = cos-sim ( h ( f$_{X}$$_{i}$, h ( f$_{Y}$$_{j}$ )) ∈ [ - 1 , 1]. We describe the sampling of the x$_{i}$ and y$_{j}$ below.

Feature distillation. We aim to distill features such that similar features align while dissimilar features are separated. To this end, we use the contrastive correlation loss L$_{corr}$ , introduced by STEGO [32] and defined as

$$\mathcal { L } _ { c o r t } ( f _ { X }, f _ { Y }, b ) = - \sum _ { i, j } ( S _ { i, j } - b ) \max _ { k } ( S _ { i, j }, 0 ), \ \ ( 8 )$$

where f$_{X}$ , f$_{Y}$ are the features of the two sample batches. This loss pushes S h$_{i,j}$ , towards 1 in case S$_{i,j}$ exceeds the threshold b . Otherwise, L$_{corr}$ pushes the S h$_{i,j}$ below 0 .

The correlation loss L$_{corr}$ requires informative pairs of sampled features, balancing attractive and repulsive signals. Following STEGO [32], we consider three different relations: (1) feature pairs from the same image ( f$_{X}$, f$_{X}$$_{i}$ ), (2) feature pairs from an image and its k -nearest neighbors in feature space ( f$_{X}$, f$_{Y}$ , k$_{N}$$_{i}$ ), and (3) feature pairs from an image and a randomly sampled other image ( f$_{X}$, f$_{Y}$$_{rand}$ ). Note that each pair is obtained from SceneDINO's 3D feature field, see below. Equipped with the three feature sample pairs, we compute the full distillation loss as

$$\mathcal { L } _ { d i s t } = \lambda _ { e l s t } \mathcal { L } _ { c o r t } ( f _ { X }, f _ { X } ^ { 2 }, b _ { s e l l } ) \\ + \lambda _ { N N } \mathcal { L } _ { c o r t } ( f _ { X }, f _ { X } ^ { 2 } \mathcal { K } _ { N N }, b _ { k N N } ) \quad \quad ( 9 ) \\ + \lambda _ { R a n d } \mathcal { L } _ { C o r t } ( f _ { X }, f _ { X } ^ { 2 } \mathcal { R } _ { d } ^ { 2 }, \mathrm { b } _ { r a n d } ),$$

where λ$_{self}$ , λ$_{KNN}$ , and λ$_{rand}$ denote the scalar loss weights. b$_{self}$ , b$_{KNN}$ , and b$_{rand}$ are the contrastive thresholds.

Feature sampling in 3D. While obtaining feature pairs using 2D rendered features is straightforward [32], we aim to take advantage of our learned 3D geometry of the scene. To this end, we introduce a novel 3D feature sampling approach for the distillation loss L$_{dist}$ from Eq. (9). Our goal is to sample features both similar and dissimilar in terms of the encoded semantic concept, which should capture rich semantics as well as different semantic concepts.

First, we obtain all C$_{G}$ 3D surface points V ∈ R 3 × G and their depth d v ∈ R 3 × G from the camera. To sample points that cover different semantic concepts, we use depth as a cue and sample different depth ranges. In particular, we sort the surface points V based on d v . The sorted surface points V are partitioned into M equally-sized chunks; we uniformly sample a single 3D point from each chunk, resulting in M center points X ∈ R 3 × M .

Equipped with the center points X , we aim to extract rich semantic features from the feature field. While we could just obtain the features for X , we query positions in the

Figure 4. 3D feature sampling. We first sample a center point X$_{i}$ from all visible surface points. Further points are sampled within the radius around the center point X$_{i}$ . Sampled points with sufficient density are accepted; otherwise rejected. The accepted points are used to obtain the feature batch f$_{X}$ .

neighborhood of X to increase semantic richness and better capture the 3D structure of the scene for distillation. In particular, for each center point, we randomly sample a point within a radius of r = 0 . 5 m . To account for samples falling into uncoupled regions in our feature field, we only keep samples with a sufficient density σ &gt; 0 . 5 . We repeat this sampling process until we obtain N valid samples per center point. Using these samples, we query our feature field, resulting in a feature batch f$_{X}$ ∈ R 3 × D for each of the G center points in each scene ( cf. Fig. 4).

To obtain f$_{KNN}$ and f$_{R}$ and , we utilize a feature buffer that efficiently stores the sampled features of multiple scenes. Given a new input image, we obtain G feature batches f$_{X}$ as just described. For each f$_{X}$ , we randomly sample another feature batch from the buffer to obtain f$_{Y}$$_{rand}$ . To obtain f$_{Y}$$_{KNN}$ , we search in the feature buffer for the k -nearest neighbors of f$_{X}$ , using the average feature of each batch. From these k -nearest neighbors, we randomly pick a feature batch to obtain f$_{Y}$$_{KNN}$ and compute the distillation loss L$_{dist}$ . After repeating this process for each of the current G feature batches, we add the current feature batches to the feature buffer and remove the oldest batches.

Unsupervised probing. To obtain semantic predictions, we probe the distilled feature space using k -means [68, 70]. In particular, we iteratively update cluster centers θ ∈ R K × C using cosine distance-based mini-batch k -means [90] during distillation. To infer the final semantic prediction, we compute p$_{X}$ = softmax(cos-sim ( h ( f$_{X}$ , θ )).

## 4. Experiments

We evaluate SceneDINO on SSC and compare it to a simple unsupervised SSC baseline (Sec. 4.1). We also report results for 2D unsupervised segmentation, including domain generalization results (Sec. 4.2). Finally, we explore multiview feature consistency (Sec. 4.3) and present an analysis of SceneDINO and our 3D distillation (Sec. 4.4).

Datasets. We train using KITTI-360 [65], composed of clips from a moving vehicle equipped with cameras. For

consistency, we follow S4C [37] by sampling eight views and using the dataset camera poses. We also provide results with estimated poses. We also show experiments for training on RealEstate 0k [118], composed of monocular videos. Here, we follow the setup of BTS [107], obtaining three views. If not noted differently, we report results obtained with training on KITTI-360. For SSC and 2D semantic segmentation validation, we use the SSCBench-KITTI360 test split [63]. Cityscapes [19] and BDID00K [114] val are used for domain generalization results. To enable evaluation in 3D and 2D, we use the 19-class taxonomy of Cityscapes and perform 2D evaluation on Cityscapes, BDID00K, and KITTI-360 on 19 classes. For SSCBench, we combine classes to adhere to the 15 SSCBench classes.

3D evaluation. Given our unsupervised setup, we predict pseudo-semantic classes that must be aligned with the ground truth for evaluation. We follow standard practice in 2D unsupervised semantic segmentation [17, 31, 32, 50, 91, 94] by applying Hungarian matching [56] to align our pseudo semantics. For validating the aligned semantics, we follow the standardized SSCBench [63] protocol and report both semantic performance using the mean Intersectionover-Union (mloU) and geometric performance using IoU, precision, and recall. We report all metrics on SSCBench ranges 12.8 m, 25.6 m, and 51.2 m.

2D evaluation. Following the established evaluation protocol in 2D unsupervised semantic segmentation [17, 31, 32, 50, 91, 94], we use the all-pixel accuracy (Acc) and mean Intersection-over-Union (mloU) metrics. Likewise, in line with prior work, 2D segmentation predictions of all models are refined using a dense Conditional Random Field [55] before computing Acc and mloU.

Multi-view feature consistency evaluation. We aim to evaluate the multi-view consistency of our feature field. As we are not aware of any general feed-forward 3D feature field approach, we compare against 2D SSL models. To measure multi-view consistency in 2D, we use two video frames and estimate optical flow and occlusions with RAFT [98]. We backward warp 2D features of the second frame to the first. On the aligned features, we compute the feature similarity using absolute error (L$_{1}$ ), the Euclidean distance (L$_{2}$ ), and the cosine similarity, ignoring occlusions.

Baselines. We are not aware of any existing unsupervised SSC approach. To allow for comparisons, we construct a competitive baseline for unsupervised SSC. In particular, we train the S4C approach with unsupervised semantics of the established STEGO [32] approach. For 2D semantic segmentation, we use U2Seg [77] and STEGO [32] as established unsupervised baselines. Note U2Seg is trained on ImageNet [21] and COCO [66] using STEGO pseudolabels. We use STEGO [32] with DINO [11] (ViT-B/8), DINOv2 [79] (ViT-B/14), and FiT3D [115] (ViT-B/14)

Table 1. SSCBench-KITTI-360 results. Semantic results using mloU and per class IoU, and geometric results using IoU, Precision, and Recall (all in %, ↑ ) on SSCBench-KITTI-360 test using three depth ranges. We compare our baseline S4C + STEGO to our SceneDINO. We report S4C as a 2D supervised baseline.

| Method               | S4C + STEGO                                 | | SceneDINO (Ours) |   | S4C                  |                        |       |
|----------------------|---------------------------------------------|------------------------|----------------------|------------------------|-------|
| Supervision          | Supervision                                 | Supervision            | Supervision          |                        |       |
| Range                | 12.8 m 25.6 m 51.2 m [12.8 m 25.6 m 51.2 m] | 2D supervision         | 2D supervision       |                        |       |
| Semantic validation  | Semantic validation                         | Semantic validation    | Semantic validation  |                        |       |
| mloU                 | 10.53 9.26                                  | 6.60                   | 10.76 10.01          | 8.00 16.94 13.94 10.19 |       |
| car                  | 18.57 14.90 9.22                            | 21.25 14.94 11.21      | 22.58 18.64 11.49    | 22.58 18.64 11.49      |       |
| bicycle              | 0.01 0.01                                   | 0.01                   | 0.00                 | 0.00                   | 0.00  |
| motorcycle           | 0.01 0.01                                   | 0.00                   | 0.00                 | 0.00                   | 0.00  |
| truck                | 0.11 0.04                                   | 0.02                   | 0.00                 | 0.00                   | 0.00  |
| other-v              | 0.05 0.02                                   | 0.05                   | 0.00                 | 0.00                   | 0.00  |
| p0                   | 0.01 0.01                                   | 0.01                   | 0.00                 | 0.00                   | 0.00  |
| 61.97                | 32.47                                       | 52.78                  | 51.10                | 32.98                  | 69.38 |
| sideview             | 4.29 17.44                                  | 24.81                  | 22.69                | 18.97                  | 45.30 |
| midway               | 17.45                                       | 24.81                  | 17.38                | 12.33                  | 14.32 |
| fence                | 1.41                                        | 0.01                   | 0.01                 | 0.59                   | 0.78  |
| vegetation           | 1.58 1.63                                   | 1.58                   | 1.31                 | 3.22                   | 19.88 |
| earrain              | 24.95                                       | 9.41                   | 2.35                 | 12.60                  | 15.22 |
| pole                 | 0.04                                        | 0.05                   | 0.04                 | 0.00                   | 0.05  |
| traffic-sign         | 0.00                                        | 0.00                   | 0.00                 | 0.00                   | 0.00  |
| other-obj            | 0.04                                        | 0.02                   | 0.04                 | 0.00                   | 0.00  |
| Geometric validation | Geometric validation                        | Geometric validation   | Geometric validation | Geometric validation   |       |
| IoU                  | 49.32                                       | 41.38                  | 46.39                | 49.54                  | 42.27 |
| Precision            | 54.46                                       | 42.63                  | 41.93                | 52.46                  | 41.59 |
| Recall               | 84.95                                       | 78.69                  | 73.83                | 87.61                  | 83.59 |

features. Fit3D offers multi-view refined DINOv2 features [115]. Note that Fit3D reports results, concatenating the refined features with DINOv2 features. We report results using both plain features only and the concatenation. We also use rendered 2D segmentations of our S4C + STEGO baseline for 2D validation. For multi-view feature consistency, we utilize DINO [11], DINOv2, and Fit3D [115] features as a baseline.

Implementation details. Our encoder-decoder uses a DINO-B/8 [11] backbone and a dense prediction decoder [83]. The MLP decoder φ entails two layers with 128 hidden features. As rendering features is expensive, φ predicts 64 features. We employ another MLP to upproject again to the full dimensionality D = 768. If not stated differently, our target features are obtained from DINO-B/8 [11]. We train using a batch size of 4 and extract 32 patches of size 8 × 8 from each image to compute L$_{SceneDINO}$ . Volume rendering samples each ray at L = 32 uniformly spaced points in inverse depth within [3 m, 80 m]. We train for 100 k steps using Adam [51] with a base learning rate of 10 - $^{4}$. Training takes ca. 2 days on a single V100 GPU. We distill using a batch size of 4, 5 center points, a feature batch of size 576, and cluster with K = 19. For kN sampling, we use k = 4. The feature buffer holds 256 feature batches. Refer to the supplement for more details.

## 4.1.3D semantic scene completion

We assess the unsupervised SSC and geometric accuracy of SceneDINO with our 3D feature distillation approach on SSCBench-KITTI-360. In particular, Tab. 1 com-

Figure 5. Qualitative SSC comparison on KITTI-360. We show the input image, SceneDINO's feature field using the first three principal components and SSC prediction, the prediction of our baseline S4C + STEGO, and the ground truth. We only visualize surface voxels. Qualitative results show the expressiveness of our feature field and SceneDINO's capabilities to accurately reconstruct and label a scene.

Table 2. 2D unsupervised semantic segmentation results on KITTI-360. Comparing SceneDINO to existing 2D methods and our S4C + STEGO 3D baseline, using Accuracy and mean IoU (in %, ↑) on the SSCBench-KITTI-360 test split. ↑ denotes the use of plain Fit3D features. ↓ denotes training on ImageNet and COCO.

| Method                | Features    |   Acc |   mloU |   MLO |
|-----------------------|-------------|-------|--------|-------|
| U2Seg$^{+}$ [77]      | DINO [11]   | 72.89 |  23.43 | 73.32 |
| STEGO [32]            | DINO [11]   | 73.32 |  23.57 | 24.28 |
| STEGO [32]            | DINOv2 [79] | 64.54 |  24.82 | 24.92 |
| STEGO [32]            | FIT3D [115] | 54.19 |  22.29 | 57.25 |
| STEGO [32]            | FIT3D [115] | 57.25 |  18.95 | 65.16 |
| S4C [37] + STEGO [32] | DINO [11]   | 65.16 |  21.67 | 21.75 |
| SceneDINO (Ours)      | DINO [11]   | 77.74 |  25.81 | 16.76 |

parees SceneDINO against our unsupervised SSC baseline S4C [37] + STEGO [32]. SceneDINO achieves a (semantic) mloU of 8.0 % for the range of 51.2 m, significantly improving over our unsupervised baseline (6.6 %). This demonstrates that SceneDINO effectively lifts DINO features into 3D. In terms of geometric accuracy, SceneDINO moderately improves over S4C + STEGO. Despite being fully unsupervised, SceneDINO comes within 2.2 % points mloU of the 2D-supervised S4C.

Fig. 5 provides qualitative samples on SSCBenchKITTI-360. SceneDINO's unsupervised SSC predictions are less noisy and capture finely resolved semantics compared to S4C + STEGO. Compared to the ground truth, we observe, SceneDINO captures both the geometry and general semantics of the scene. We visualize SceneDINO's feature field (before distillation) using the first three principal components. In PCA space, we observe that our feature field captures semantically meaningful regions.

## 4.2.2D semantic segmentation

Table 2 compares the semantic predictions of SceneDINO to recent 2D approaches and our 3D baseline. We obtain 2D semantic segmentations from SceneDINO and our S4C + STEGO baseline using semantic rendering [37]. SceneDINO with our 3D distillation approach outperforms STEGO with DINO features, an established 2D unsupervised

Table 3. 2D unsupervised semantic segmentation domain generalization results. Comparing SceneDINO to existing 2D unsupervised semantic segmentation methods and S4C + STEGO 3D baseline, using Accuracy and mean IoU (in %, ↑). We train on KITTI-360 images and report domain generalization results on Cityscapes and DBD-100K val. ↑ denotes plain Fit3D features.

vised semantic segmentation approach. In particular, the mloU of SceneDINO is 2.24 % points higher than for STEGO (w/ DINO). Utilizing 3D refined features from Fit3D deteriorates the baseline relative to DINO, indicating that the Fit3D refinement reduces feature expressiveness. Notably, our unsupervised 3D baseline S4C + STEGO transfers significantly worse to 2D than SceneDINO.

| Method                | Features    | Citiescapes   | BDD-100K   | Citiescapes   | BDD-100K   |
|-----------------------|-------------|---------------|------------|---------------|------------|
|                       |             |               |            |               |            |
| U2Seg$^{+}$ [77]      | DINO [11]   | 75.57         | 18.62      | 69.00         | 17.99      |
| STEGO [32]            | DINO [11]   | 71.21         | 19.42      | 75.02         | 21.41      |
| STEGO [32]            | DINOv2 [79] | 78.41         | 19.73      | 65.72         | 21.77      |
| STEGO [32]            | FIT3D [115] | 66.94         | 20.11      | 65.96         | 20.99      |
| STEGO [32]            | FIT3D [115] | 47.66         | 17.76      | 61.07         | 21.76      |
| S4C [37] + STEGO [32] | DINO [11]   | 54.80         | 14.04      | 44.98         | 11.62      |
| SceneDINO (Ours)      | DINO [11]   | 73.17         | 22.81      | 72.28         | 22.09      |

We also validate SceneDINO, trained on KITTI-360, on Cityscapes and BDD10K, demonstrating domain generalization. The results are reported in Tab. 3. SceneDINO outperforms all baselines in mloU on both datasets while only falling short in Access. S4C + STEGO falls short in generalization. We suspect this poor generalization is caused by the fact that S4C does not rely on general SSL features in the final model, while our feature field generalizes.

## 4.3. Multi-view feature consistency

We analyze the multi-view consistency of our feature field against existing 2D SSL features in Tab. 4. We report the results of SceneDINO trained on KITTI-360 and RealEstate10K. SceneDINO trained using DINO features exhibits significant improvements in multi-view feature consistency over standard DINO features. We also train SceneDINO using target features from DINOv2 [79]. Compared to standard DINOv2 and Fit3D fea-

Table 4. Multi-view consistency results. Comparing multi-view consistency of SceneDINO to existing 2D SSL features, using L1 distance ( l ), L2 distance ( l ), and cosine similarity ( t ) on KITTI360 and RealEstate10K. We compare DINO ( t ) and DINOv2based ( bottom ) features. ˆ t denotes plain Fit3D features.

| Method                |   KITTI-360 | RealEstate10K   |            |       |      |
|-----------------------|-------------|-----------------|------------|-------|------|
| DINO [11]             |       16.06 | L2              | Cos-Sim L1 |       |      |
| SceneDINO (w/ DINO)   |        6.45 | 0.33            | 0.93       | 5.87  | 0.28 |
| DINOv2 [79]           |       15.83 | 0.73            | 0.70       | 14.20 | 0.66 |
| Fit3D [115]           |       22.86 | 0.81            | 0.82       | 19.88 | 0.72 |
| Fit3D [115]           |        7.02 | 0.33            | 0.93       | 5.67  | 0.27 |
| SceneDINO (w/ DINOv2) |        5.24 | 0.24            | 0.96       | 4.87  | 0.22 |

Table 5. SceneDINO analysis. We analyze the role of decomposing positional encodings, the choice of downsampling features during training, the effectiveness of the feature smoothness loss, the effect of estimated camera poses, and the choice of target features. We report the mean IoU ( ˆ n % ) using a range of 51.2 m on SSCBench-KITTI-360 test. ˆ t MLOU reports the absolute difference in % points to our standard model with DINO target features.

Table 5. SceneDINO analysis. We analyze the role of decomposing positional encodings, the choice of downsampling features during training, the effectiveness of the feature smoothness loss, the effect of estimated camera poses, and the choice of target features. We report the mean IoU ( ˆ n % ) using a range of 51.2 m on SSCBench-KITTI-360 test. ˆ t MLOU reports the absolute difference in % points to our standard model with DINO target features.

|   Δ mloU | mloU | Configuration   |
|----------|------------------------|
|    -1.18 | -6.82 0.39 0.00        |
|    -1.17 | -6.83 0.00 0.00        |
|    -0.74 | -7.88 0.00 0.00        |
|    -0.12 | -7.88 0.00 0.00        |

tures, SceneDINO's feature field yields significantly better multi-view consistency. Notably, compared against plain 3D refined features of Fit3D, SceneDINO shows a better multi-view consistency on both datasets and all metrics while also offering more expressiveness ( cf . Tab. 2).

## 4.4. Analyzing SceneDINO

To understand what core components contribute to obtaining an expressive feature field of SceneDINO, we omit or replace individual components and report the results in Tab. 5. Replacing the downsampling approach with bilinear upsampling and multi-crop augmentations, similar to [1], to obtain high-resolution target features leads decrease SSC mloU by 1.18 %. Omitting the feature smoothness loss leads to a similar mloU drop. Abolishing the constant decomposition of positional encodings leads to a mloU drop of 0.74 %. Training using unsupervised camera poses estimated by ORB-SLAM3 [7] results in an insignificant mloU drop of only 0.12 %, over using KITTI-360 poses. Going from DINO to DINOv2 target features leads to an increased mloU of 1.08 %, demonstrating, SceneDINO can benefit from more expressive 2D target features.

In Tab. 6, we analyze our 3D distillation. Performing no distillation at all, just clustering our features, decreases mloU by 1.61 %. Omitting the k$_{NN}$-correlation loss leads to a mloU drop of 1.35 %. Distilling only with center points,

Table 6. Feature distillation analysis. We analyze the effectiveness of distilling SceneDINO's features, the kNN-correlation loss, our neighborhood sampling, and our 3D sampling approach over standard 5-crop sampling. We report the mean IoU (in % , t ) using a range of 51.2 m on SSCBench-KITTI-360 test.

| Δ mloU   |   mloU | Configuration |                                              |
|----------|------------------------|----------------------------------------------|
| -1.61    |                   6.39 | No distillation                              |
| -1.35    |                   6.65 | No k$_{NN}$-correlation loss ( k$_{NN}$ = 0) |
| -0.97    |                   7.03 | No neighborhood sampling ( cf . Fig. 4)      |
| -0.47    |                   7.53 | 5-crop sampling [32] (instead 3D sampling)   |
|          |                   8    | Full framework (SceneDINO)                   |

Table 7. Probing analysis. We analyze linear and unsupervised probing of our distilled SceneDINO features on SSCBenchKITTI-360 test using mean IoU (in % , t ). For reference, we also report S4C (2D supervised). Linear probing uses 2D annotations.

| Probing approach    | Target features   |   mloU |       |
|---------------------|-------------------|--------|-------|
| Unsupervised        | DINO [11]         |  10.76 | 10.01 |
| Unsupervised        | DINOv2 [79]       |  13.76 | 11.78 |
| Linear              | DINO [11]         |  13.63 | 12.07 |
| Linear              | DINOv2 [79]       |  15.85 | 13.7  |
| S4C (full training) | n/a               |  16.94 | 13.94 |

i.e. , not performing neighborhood sampling ( cf . Fig. 4), reduces mloU by 0.97 %. Using 5-crop feature sampling [32], instead of our proposed 3D sampling, leads to a reduced mloU of 0.47 %. This demonstrates the effectiveness of performing distillation in 3D using our novel approach.

While focusing on unsupervised SSC, we can also linearly probe our distilled feature field ( cf . Tab. 7). In particular, we train SceneDINO using different target features (DINO [11] and DINOv2 [1]), perform distillation, and probe the resulting distilled features. Using linear probing, i.e. , a single linear layer using 2D semantic labels, leads to a consistent mloU increase over unsupervised probing. SceneDINO trained using DINOv2 target features even closes the gap to S4C, trained using 2D ground-truth semantic labels. We even surpass 2D supervised S4C slightly on the full range (51.2 m), suggesting the effectiveness of SceneDINO also for weakly-supervised tasks.

## 5. Conclusion

We presented SceneDINO, to our knowledge, the first approach for unsupervised semantic scene completion. Trained using multi-view images and 2D DINO features without human supervision, SceneDINO is able to predict an expressive 3D feature field using a single input image during inference. Our novel 3D distillation approach yields state-of-the-art results in unsupervised SSC. While we focus on unsupervised SSC, our multi-view feature consistency, linear probing, and domain generalization results highlight the potential of SceneDINO as a strong foundation for various 3D scene-understanding tasks.

Acknowledgments. This project was partially supported by the European Research Council (ERC) Advanced Grant SIMULACRON, DFG project CR 250/26-1 "4D-YouTube", and GNI Project "AICC". This project has also received funding from the ERC under the European Union's Horizon 2020 research and innovation programme (grant agreement No. 866008). Additionally, this work has further been co-funded by the LOEWE initiative (Hesse, Germany) within the emergenCY center [LOEWE/1/12/519/03/05.001(0016)/72] and by the Excellence Cluster EXC3066 "The Adaptive Mind". Christoph Reich is supported by the Konrad Zusack School of Excellence in Learning and Intelligent Systems (ELIZA) through the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Education and Research. Christian Rupprecht is supported by an Amazon Research Award. Finally, we acknowledge the support of the European Laboratory for Learning and Intelligent Systems (ELLIS) and thank Mateo de Mayo as well as Igor Cviˇciˇc for help with estimating camera poses.

## References

- [1] Nikita Araslanov and Stefan Roth. Self-supervised augmentation consistency for adapting semantic segmentation. In CVPR , pages 15384-15394, 2021. 8
- [2] Yuki Markus Asano, Christian Rupprecht, and Andrea Vedaldi. Self-labelling via simultaneous clustering and representation learning. In ICLR , 2020. 2
- [3] Philip Bachman, R. Devon Hjelm, and William Buchwalter. Learning representations by maximizing mutual information across views. In NeurIPS , 2019, pages 15509-15519.
- [4] Adrien Bardes, Jean Ponce, and Yann LeCun. VICRegL: Self-supervised learning of local visual features. In NeurIPS , 2022, pages 8799-8810. 2
- [5] Adrien Bardes, Jean Ponce, and Yann LeCun. VICReg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR , 2022. 2
- [6] Yingjie Cai, Xuesong Chen, Chao Zhang, Kwan-Yee Lin, Xiaogang Wang, and Hongsheng Li. Semantic scene completion via integrating instances and scene in-the-loop. In CVPR , pages 324-333, 2021. 2
- [7] Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez, José M. Montiel, and Juan D Tartódg. ORB-SLAM3: An accurate open-source library for visual, visual-inertial and multi-map SLAM. IEEE Trans. Robot. , 37(6):1874-1890, 2021. 4, 8, VI
- [8] Anh-Quan Cao and Raoul de Charette. Monosocene: Monocular 3D semantic scene completion. In CVPR , pages 3981-3991, 2022. 2
- [9] Mathilde Caron, Ishan Misra, Julien Mairai, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS , 2020, pages 991-9924. 2
- [10] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV , pages 132-149, 2018. 2
- [11] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv'e J'egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV , pages 9650-9660, 2021. 1, 2, 3, 4, 6, 7, 8, vi
- [12] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv:2003.04297 [cs.CV], 2020. 2
- [13] Xiaokang Chen, Kwan-Yee Lin, Chen Qian, Gang Zeng, and Hongsheng Li. 3d sketch-aware semantic scene completion via semi-supervised structure prior. In CVPR , pages 4192-4201, 2020. 2
- [14] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In CVPR , pages 9640-9649, 2021. 2
- [15] Yuedong Chen, Chuanxia Zheng, Haofei Xu, Bohan Zhuang, Andrea Vedaldi, Tat-Jen Cham, and Jianfei Cai. MVSplat360: Feed-forward 360 scene synthesis from sparse views. In NeurIPS ,2024, pages 107064-107086. 2
- [16] Ran Cheng, Christopher Agia, Yuan Ren, Xinhai Li, and Bingbing Liu. S3CNet: A sparse semantic scene completion network for LiDAR point clouds. In CoRL , pages 2148-2161, 2020. 2
- [17] Jang Hyun Cho, Utkarsh Mall, Kavita Bala, and Bharath Hariharan. PiCIE: Unsupervised semantic segmentation using invariance and equivariance in clustering. In CVPR , pages 16794-16804, 2021. 2, 6, vi
- [18] Ozgin Ciek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3D U-Net: Learning dense volumetric segmentation from sparse annotation. In MICCAI , pages 424-432, 2016. 1
- [19] Marius Cordhs, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes dataset for semantic urban scene understanding. In CVPR , pages 3213-3223, 2016. 6, i
- [20] Timoth'e Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojawowski. Cluster and predict latches patches for improved masked image modeling. arXiv:2502.08769 [cs.CV], 2025. 2
- [21] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In CVPR , pages 248-255, 2009. 6
- [22] Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised visual representation learning by context prediction. In ICCV , pages 1422-1430, 2015. 2
- [23] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16 × 16 words: Transformers for image recognition at scale. In ICLR , 2021. 2
- [24] Linus Ericsson, Henry Gouk, Chen Change Loy, and Timothy M. Hospedales. Self-supervised representation learning: Introduction, advances, and challenges. IEEE Trans. Signal Process. , 39(3):42-62, 2022. 2
- [25] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. FeatUp:

| In ICLR , 2024. 4   | A model-agnostic framework for features at any resolution.                                                                                                                                                                                                               |                                                               |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| [26]                | Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The KITTI dataset. Int. J. Robot. Res. , 32 (11):1231-1237, 2013. 1                                                                                                           |                                                               |
| [27]                | Cl'ement Godard, Disin Mac Aodha, and Gabriel J. Bros- tow. Unsupervised monocular depth estimation with left- right consistency. In CVPR , pages 270-279, 2017. 4                                                                                                       |                                                               |
| [28]                | Jean-Bastien Grill, Florian Strub, Florent Altch'e, Corentin Tallec, Pierre H. Rickemond, Elena Buchatskaya, Carl Do- ersch, Bernardo Avila Pires, et al. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS '2020 , pages 21271-21284. 2 |                                                               |
| [29]                | Agri Gupta, Jiajun Wu, Jia Deng, and Li Fei-Siemi. Seame                                                                                                                                                                                                                 | Masked Autoencoders. In NeurIPS '2023 , pages 40676- 40693. 2 |
| [30]                | Huy Ha and Shuran Song. Semantic abstraction: Open- world 3D scene understanding from 2D vision-language models. In CoRL , pages 643-653, 2023. 2                                                                                                                        |                                                               |
| [31]                | Eliu Hahn, Nikita Araslanov, Simone Schaub-Meyer, and Stefan Roth. Boosting unsupervised semantic segmentation with principal mask proposals. Trans. Mach. Learn. Res. , 2024. 3, 6, i, iv                                                                               |                                                               |
| [32]                | Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised se- mantic segmentation by distilling feature correspondences. In ICLR , 2022. 1, 2, 3, 4, 5, 6, 7, 8, iv, i, iv                                                    |                                                               |
| [33]                | Keonhee Han, Dominik Muhle, Felix Wimbauer, and Daniel Cremers. Beschulp-sverifusion for single-view scene completion via knowledge distillation. In CVPR , pages 9837-9847, 2024. 1, 2                                                                                  |                                                               |
| [34]                | Xian-Feng Han, Hamid Laga, and Mohammed Ben- namoun. Image-based 3D object reconstruction: State-of- the-art and trends in the deep learning era. IEEE Trans. Pattern Anal. Mach. Intell. , 43 (5):1578-1604, 2019. 2                                                    |                                                               |
| [35]                | Robert Harb and Patrick Knoblereiner. InfoSeg: Unsupervised semantic image segmentation with mutual informa- tion maximization. In GCPR , pages 18-32, 2021. 2                                                                                                           |                                                               |
| [36]                | Richard Hartley and Andrew Zisserman. Multiple view ge- ometry in computer vision. Cambridge University Press, 2003. 2                                                                                                                                                   |                                                               |
| [37]                | Adrian Hayler, Felix Wimbauer, Dominik Muhle, Christian Rupprecht, and Daniel Cremers. S4C: Self-supervised se- mantic scene completion with neural fields. In 3DV , pages 409-420, 2024. 1, 2, 6, 7, i, iv, i, iv                                                       |                                                               |
| [38]                | Kaiming He, Haoqui Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual rep- resentation learning. In CVPR , pages 9729-9738, 2020. 2                                                                                                |                                                               |
| [39]                | Kaiming He, Xinle Chen, Saining Xie, Yanghao Li, Piotr Doll'ar, and Ross Girshick. Masked autoencoders are scal- able vision learners. In CVPR , pages 16000-16009, 2022. 2                                                                                              |                                                               |
| [40]                | Olivier Henaff. Data-efficient image pecognition with con- trastive predictive coding. In ICML , pages 4182-4192, 2020. 2                                                                                                                                                |                                                               |
| [41]                | R. Devon Hjelm, Alex Fedorov, Samuel Lavoie- Marchildon, Karan Grewal, Phil Bachman, Adam                                                                                                                                                                                |                                                               |

- Trischler, and Yoshua Bengio. Learning deep representa- tions by mutual information estimation and maximization. In ICLR , 2019. 2
- [42] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. MetricD3 v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. IEEE Trans. Pattern Anal. Mach. Intell. , 46 (12):10579-10596, 2024. 2
- [43] Rui Huang, Songyou Peng, Ayca Takmaz, Federico Tombari, Marc Pollefeys, Shiji Song, Gao Huang, and Francis Engelmann. SegmentD: Learning fine-grained class-agnostic 3D segmentation without manual labels. In ECCV , pages 278-295, 2024. 2
- [44] Yuanhui Huang, Wenzhao Zheng, Yunpeng Zhang, Jie Zhou, and Jiwen Lu. Tri-perspective view for vision-based 3D semantic occupancy prediction. In CVPR , pages 92239232, 2023. 2
- [45] Yuanhuu Huang, Wenzhao Zheng, Borui Zhang, Jie Zhou, and Jiwen Lu. SelfOcc: Self-supervised vision-based 3D occupancy prediction. In CVPR , pages 19946-19956, 2024. 1, 2
- [46] Joel Janai, Fatma G ' ueay, Seim Behil, and Andreas Geiger. Computer vision for autonomous vehicles: Problems, datasets and state of the art. Found. Trends Comput. Graph. Vis. , 12 (1-3):1-308, 2020. 1
- [47] Xu Ji, Joao F. Henriques, and Andrea Vedaldi. Invariant in- formation clustering for unsupervised image classification and segmentation. In ICCV , pages 9865-9874, 2019. 2
- [48] Haoyi Jiang, Liu Liu, Tianheng Cheng, Xinjie Wang, Tianwei Lin, Zhizhong Su, Wenyu Liu, and Xinggang Wang. GaussTR: Foundation model-aligned gaussian transformer for self-supervised 3D spatial understanding. arXiv:2412.13193 [cs.CV], 2024. 2
- [49] Justin Kerr, Chung Min Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. LERF: Language embedd radiance fields. In ICCV , pages 19729-19739, 2023. 2
- [50] Chanyong Kim, Woojung Han, Dayun Ju, and Seong Jae Hwang. EAGLE: Eigen aggregation learning for objectcentric unsupervised semantic segmentation. In CVPR , pages 3523-3533, 2024. 3, 6, i
- [51] Diederik P. Kingma and Jimmy Lei Ba. Adam: A method for stochastic optimization. In ICLR , 2015. 6
- [52] Alexander Kirillov, Eric Minut, Nikhila Ravi, Hanzi Mao, Choe Rolland, Laura Gustaison, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll'ar, and Ross Girshick. Segment Anything. In ICCV , pages 4015-4026, 2023. 1, 2
- [53] Sosuke Kobayashi, Eiichi Matsumoto, and Vincent Sitzmann. Decomposing NeRF for editing via feature field distilat ion. In NeurIPS '2022 , pages 23311-23330. 2
- [54] Alexander Koenig, Maximilian Schambach, and Johannes Otterbach. Uncovering the inner workings of STEGO for safe unsupervised semantic segmentation. In CVPR , pages 3789-3798, 2023. 4

## Feed-Forward SceneDINO for Unsupervised Semantic Scene Completion

## Supplementary Material

Aleksandar Jevti'c ∗ 1 Christoph Reich ∗ 1,2,4,5 Felix Wimbauer 1,4 Oliver Hahn 2 Christian Rupprecht 3 Stefan Roth 2,5,6 Daniel Cremers 1,4,5 $^{1}$TU Munich $^{2}$TU Darmstadt $^{3}$University of Oxford $^{4}$MCML $^{5}$ELIZA $^{6}$hessian.AI ∗ equal contribution https://visinf.github.io/scenedino

In this appendix, we provide further implementation details, including dataset properties and an overview of SceneDINO's computational complexity ( cf. Sec. A). We discuss our multi-view feature consistency evaluation approach ( cf. Sec. B). Next, we provide additional qualitative and quantitative results ( cf. Sec. C), including failure cases. Finally, we discuss the limitations of SceneDINO and suggest future research directions ( cf. Sec. D).

## A. Reproducibility

Here, we provide further implementation details, information about the utilized dataset, and computational complexity details to ensure reproducibility. Note that our code is available at https://github.com/tum-vision/ scenedino .

## A.1. Implementation details

We implement SceneDINO in PyTorch [122] and build on the code of BTS [107], STEGO [32], and S4C [37]. Our encoder-decoder (pre-trained DINO-B/8 and randomly initialized dense prediction decoder) produces per-pixel embeddings of dimensionality D$_{b}$ = 256 . Based on these embeddings, the two-layer MLP ϕ (hidden dimension 128) predicts 64 features. As rendering features is expensive, requiring multiple forward passes through the MLP, ϕ predicts 64 features. We employ another MLP to up-project again to the full dimensionality D = 768 , this MLP is learn with SceneDINO and can up-project 2D features and 2D rendered features. We train for 100k steps with a base learning rate of 10 - $^{4}$, dropping to 10 - 5 after 50 k steps. We train using a batch size of 4, extracting 32 patches of size 8 × 8 per image. These patches align with the per-patch DINO target features. For our feature field loss formulation ( cf. Sec. 3.2), we use the loss weights λ$_{p}$ = 1 , λ$_{s}$ = 0 . 001 , λ$_{f}$ = 0 . 2 , λ$_{s}$ = 0 . 25 .

The MLP head h (hidden dimension 768) produces 64 distilled features. We perform distillation for 1000 steps with a learning rate of 5 · 10 - $^{4}$. We train using a batch size of 4, 5 center points, a feature batch of size 576, and cluster with K = 19 . For kNN sampling, we use k = 4 . The feature buffer holds 256 feature batches. The loss term in Eq. (9) is parameterized with λ$_{self}$ = 0 . 08 λ$_{NN}$ = 0 . 43

λ$_{rand}$ = 0 . 67 , and b$_{rand}$ = 0 . 87 . For the similarity thresholds we use b$_{self}$ = 0 . 44 , b$_{kn}$ = 0 . 18 , and b$_{rand}$ = 0 . 87 .

We follow standard practice in 2D unsupervised semantic segmentation [17, 31, 32, 50, 77, 91, 94] by applying Hungarian matching [56] to align our pseudo semantics. For SSC validation, we map down to 15 semantic classes while following existing work [31, 32] for 2D validation and map to 19 semantic classes.

## A.2. Datasets

We provide additional details about the datasets utilized to train and evaluate SceneDINO.

KITTI-360 [63, 65] provides video sequences from a moving vehicle equipped with a forward-facing stereo pair and two side-facing fisheye cameras. In future frames, the fisheye views capture additional geometric and semantic cues of regions occluded in the forward-facing view. For training, we resample the fisheye images into perspective projection. We focus on an area approximately 50 meters ahead of the ego vehicle. Assuming an average velocity of 30 - 50 km/h, side views are randomly sampled 1 - 4 seconds into the future. Given a frame rate of 10 Hz, this translates to 10 - 40 time steps. Each training sample consists of eight images: four forward-facing views (including the input image) and four side-facing views.

To evaluate our predicted field in SSCBench-KITTI360, we follow the evaluation procedure of S4C [37]. The voxel predictions are evaluated in three different ranges: 12 . 8 × 12 . 8 × 12 . 8 m × 6 . 4 m, 25 . 6 × 25 . 6 m × 6 . 4 m, and the full range 51 . 2 m × 51 . 2 m × 6 . 4 m. For each voxel, multiple evenly distributed points are sampled from the semantic field. The predictions are aggregated per voxel by taking the maximum occupancy and weighting the class predictions accordingly.

Citiescapes [19] consists of 500 high-resolution and densely annotated validation images of ego-centric driving scenes. For validation, Cityscapes uses a 19-class taxonomy. We leverage the Cityscapes validation samples at a resolution of 640 × 192 for our domain generalization experiments (2D semantic segmentation).

BDD-100K [114] is a driving scene dataset obtained from urban areas in the US. BDD-100K contains 1000 semantic

Figure 6. 3D qualitative SSC comparison on KITTI-360. We provide additional qualitative results, visualizing the input image, SceneDINO's predicted feature field using the first three principal components, and SSC prediction, the SSC prediction of our baseline S4C+STEGO, and the SSC ground truth. We only visualize surface voxels within the field of view for the sake of clarity.

segmentation validation images. The semantic taxonomy follows the 19-class Cityscapes definition. For domain generalization experiments, we utilize BDD-100K images at a resolution of 640 × 192.

RealEstate10K [118] is a large-scale dataset containing videos of real-world indoor and outdoor scenes, primarily sourced from YouTube. For our experiments, we train with a resolution of 512 × 288 . Each training samples of three frames, separated by a randomly sampled time offset. There are no semantic annotations provided with the dataset. We evaluate the multi-view consistency of our model in this setting.

## A.3. Computational complexity

SceneDINO requires only a single GPU for training and inference. In SSCBench (51.2 m range), SceneDINO requires 0.76 ± 0.1 s to infer a full scene on a V100 GPU. The peak VRAM usage during inference is 11 GB. For reference, S4C requires 0.32 ± 0.13 s. Considering our expressive and high-dimensional feature field and ViT encoder, this is a moderate runtime increase. SceneDINO has 100 M parameters and is trained for approximately 2 days on a single V100 32 GB GPU. All results are reported using automatic mixed precision.

We aim to measure the multi-view consistency of 2D and 3D features. Note, we are not aware of any standardized approach for multi-view feature consistency. To this end, we employ a straightforward approach. Given two video frames with a temporal stride of 3, forward optical flow is computed using RAFT large [98]. We estimate occlusion by forward-backward consistency [124]; for this, we also compute the backward optical flow. The 2D feature maps obtained using the second frame are backward warped to the features of the first frame. We compute different similarity metrics between the aligned features ( L$_{1}$ , L$_{2}$ , and cos-sim). Note that we ignore occlusions. While features from DINO, DINOv2 , and Fit3D possess a lower resolution than our 2D rendered SceneDINO features, we upscale these features to the image resolution before warping. This evaluation approach utilizes optical flow correspondences and captures both ego motion as well as object motion, offering a simple way to evaluate multi-view feature consistency.

## C. Additional Results

Here we provide additional qualitative and quantitative results, extending our results reported in the main paper.

Qualitative results. In Fig. 6, we present additional qualitative results of SceneDINO using our 3D feature distilativ-

Figure 7. Failure cases of SceneDINO on KITTI-360. We provide failure cases of SceneDINO. We visualize the input image, the predicted feature field using the first three principal components, the SSC prediction, and the SSC ground truth. We observe that our semantic predictions struggle in shaded regions. We only visualize surface voxels within the field of view for the sake of clarity.

Figure 8. 2D SceneDINO features on KITTI-360. We visualize our 2D rendered features and DINO features for a given input image ( left ). We use the first three principal components for feature visualization. Notably, SceneDINO's features ( middle ) are smoother and capture finer structures than DINO ( right ). Additionally, SceneDINO's features are high-resolution, while DINO generates features with a stride of 8.

tion approach on unsupervised semantic scene completion. We also provide visualizations of our unsupervised SSC baseline, S4C + STEGO. Qualitatively, our approach obtains more accurate SSC results and is able to segment faraway objects, such as cars, better than the S4C + STEGO baseline. This observation aligns with the quantitative results presented in Tab. 1 of the main paper.

Figure 8 qualitatively analyzes our 2D rendered fea-

tures against DINO. Our features exhibit a smooth appearance for uniform regions, such as sidewalks. Additionally, SceneDINO's features better capture fine structures like poles than DINO features. 2D rendered SceneDINO features are also high resolution in contrast to DINO features that exhibit a lower resolution.

Failure cases. In Fig. 7, we provide failure cases of SceneDINO's SSC predictions. Our predictions exhibit two common failure cases. First, shadowed regions often lead to wrong semantic predictions. Regions affected by significant brightness changes are breaking the brightness consistency, subsequently offering a poor learning signal during training, thus impeding accurate predictions of shadowed regions. Second, objects such as cars can entail tail-like artifacts, not accurately capturing the geometry. As our multiview image and feature reconstruction training cannot handle dynamic objects, tail-like artifacts could be caused by the poor learning signal for dynamic objects.

Quantitative results. In Tab. 8, we provide additional semantic scene completion results of 3D supervised approaches as an additional point of comparison. In particular, we report official SSCBench [63] results of VoxFormer S [62] and OccFormer [117]. Both utilize 3D supervision, including both semantic and geometric annotations. We also report the results of SSCNet [95]. This approach trains using 3D supervision but utilizes a depth image during inference. While SceneDINO achieves state-of-the-art segmentation accuracy in the unsupervised setting, supervised approaches are significantly more accurate.

(a) SceneDINO

Figure 9. Confusion matrices for 2D unsupervised semantic segmentation on KITTI-360. Rows represent ground-truth class labels (normalized to 1), while columns correspond to predicted class labels. We report results for ( a ) SceneDINO and ( b ) STEGO on the SSCBench-KITTI-360 test split.

generally observe that SceneDINO performs well in segmenting frequent classes, such as "road", "building", and "sky". Less frequent classes, such as "fence" and "pole", are less well segmented. Classes including very small and fine structures (e.g., "pole") are completely missed by SceneDINO. This trend can also be observed for our 3D unsupervised baseline S4C + STEGO and 2D STEGO. We also observe that class-wise metrics strongly correlate between 2D and 3D.

Table 12. Camera pose analysis on SSCBench-KITTI-360. We extend the camera pose analysis in Tab. 5 and report detailed results of SceneDINO with unsupervised camera poses estimated by SOFT2 [121] and ORB-SLAM3 [7]. For reference, we also provide results obtained using the KITTI-360 dataset poses. Semantic results using mloU and class IoU, and geometric results using IoU, Precision, and Recall, and (all in %, ) on SSCBench-KITTI-360 test using three depth ranges.

| Method               | SceneDINO (Ours)   | SOFT2   | ORB-SLAM3   | KITTI-360   |        |        |       |       |       |
|----------------------|--------------------|---------|-------------|-------------|--------|--------|-------|-------|-------|
| Poses                | 12.8 m             | 51.2 m  | 12.8 m      | 25.6 m      | 51.2 m |        |       |       |       |
| Range                | 12.8 m             | 25.6 m  | 51.2 m      | 12.8 m      | 25.6 m | 51.2 m |       |       |       |
| mloU                 | 10.58              | 9.58    | 7.7         | 9.28        | 10.86  | 7.76   |       |       |       |
| car                  | 14.87              | 13.98   | 10.4        | 13.97       | 14.09  | 9.72   |       |       |       |
| bicycle              | 0.04               | 0.03    | 0.03        | 0.03        | 0.03   | 0.03   |       |       |       |
| motorcycle           | 0.00               | 0.00    | 0.00        | 0.00        | 0.00   | 0.00   |       |       |       |
| truck                | 0.00               | 0.00    | 0.00        | 0.00        | 0.00   | 0.00   |       |       |       |
| other-v              | 0.00               | 0.00    | 0.00        | 0.00        | 0.00   | 0.00   |       |       |       |
| person               | 0.02               | 0.01    | 0.01        | 0.00        | 0.00   | 0.00   |       |       |       |
| road                 | 44.48              | 44.50   | 36.01       | 44.07       | 44.88  | 51.19  | 49.28 |       |       |
| sidewalk             | 16.57              | 16.49   | 23.56       | 24.35       | 25.66  | 22.83  | 26.11 |       |       |
| building             | 19.40              | 23.40   | 18.58       | 19.19       | 20.27  | 12.33  | 18.27 |       |       |
| fence                | 1.79               | 1.00    | 1.66        | 1.67        | 1.91   | 1.91   | 0.90  |       |       |
| vegetation           | 32.20              | 25.65   | 20.6        | 32.60       | 24.91  | 31.22  | 25.57 |       |       |
| terrain              | 25.59              | 18.11   | 14.7        | 23.89       | 19.48  | 16.26  | 18.22 |       |       |
| 0.18                 | 0.11               | 0.01    | 0.00        | 0.00        | 0.00   | 0.05   | 0.05  |       |       |
| traffic-sign         | 0.00               | 0.01    | 0.00        | 0.03        | 0.03   | 0.02   | 0.00  |       |       |
| other-obj            | 0.08               | 0.05    | 0.05        | 0.05        | 0.03   | 0.00   | 0.00  |       |       |
| Geometric validation | 37.0               | 34.5    | 45.4        | 42.21       | 36.56  | 49.54  | 42.27 | 37.60 |       |
| Precision            | 49.91              | 41.85   | 36.0        | 45.42       | 40.21  | 36.96  | 49.57 | 41.95 |       |
| Recall               | 54.74              | 45.66   | 40.7        | 54.42       | 45.45  | 50.98  | 43.67 | 41.0  | 41.59 |

trained using unsupervised camera poses estimated by ORB-SLAM3 [7]. Table 12, extends these results and reports detailed SSC results using two different unsupervised stereo visual SLAM approaches-SOFT2 [121] and ORBSLAM3 [7]. Using unsupervised and visually estimated poses leads to a minor drop in both semantic and geometric SSC validation. While ORB-SLAM3 poses lead to slightly better semantic accuracy than SOFT2 poses, SOFT2 estimated poses result in higher geometric accuracy. Still, both SOFT2 and ORB-SLAM3 provide poses accurate enough for train SceneDINO, reaching a similar accuracy to employing KITTI-360 poses.

Out-of-domain results. We illustrate on out-of-domain prediction in Fig. 10. While our SceneDINO model is trained on the KITTI-360 dataset, we still obtain plausible features when inferring 2D features for vastly different scenes. The 2D rendered features still show a strong correlation with semantically uniform regions, showcasing the generalization of our feature field.

## D. Limitations and Future Work

Target features. Our method builds on DINO [11] to obtain target features. While we learn to lift these features into 3D and improve multi-view feature consistency, we cannot improve the discriminative power of the target features per se . However, SceneDINO can be trained using arbitrary 2D target features and can profit from future advances in SSL

Figure 10. 2D SceneDINO features on out-of-domain images.

We visualize our 2D rendered features ( right ) given an out-ofdomain image ( left ) from ADE20K [126]. We use the first three principal components for feature visualization. While not trained on such scenes, SceneDINO still produces plausible feature maps.

representation. Note that training SceneDINO requires only 2 days on a single GPU and our training transfers seamlessly to different target features ( e.g. , DINov2 ), thus, utilizing SceneDINO differently is straightforward.

Dynamic objects. Our loss does not model dynamic objects and relies on a static scene assumption. This can potentially cause inaccurate predictions for dynamic classes such as person in our experiments. Recent works in depth estimation have explicitly modeled the probability of areas being dynamic [125] and even their motion within the scene [123], which might be extended to SceneDINO.

View sampling and camera poses. For sampling views during training, we rely on the sampling scheme of S4C [37]. This is not directly applicable to other nondriving datasets, where the sampling needs to be tuned. In addition, our approach requires accurate camera poses for each view. We demonstrated that these can be obtained in an unsupervised way for KITTI-360 ( cf. Tab. 5 &amp; Tab. 12). However, obtaining unsupervised camera poses in more challenging scenarios and conditions is still challenging [120].

Future work. SceneDINO is only trained using a single dataset to be comparable to existing SSC approaches. However, scaling our approach to multiple datasets of more variable scenes could lead to more general feature representations. Ultimately, scaling SceneDINO to internet-scale videos might enable strong zero-shot and cross-domain 3D scene understanding.

## References

- [120] Lucas R. Agostinho, Nuno M. Ricardo, Maria I. Pereira, Pinto Antoine, and Andry M. Pinto. A practical survey on visual odometry for autonomous driving in challenging

scenarios and conditions. IEEE Access , 10:72182-72205, 2022. vi

[121] Igor Cviˇsi'c, Ivan Markovi'c, and Ivan Petrovi'c. SOFT2: Stereo visual odometry for road vehicles based on a pointto-epipolar-line metric. IEEE Trans. Robot. , 39(1):273-288, 2023. vi

[122] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junji Bae, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In NeurIPS '2019 , pages 8024-8035. i

[123] Yihong Sun and Bharath Hariharan. Dynamo-Depth: Fixing unsupervised depth estimation for dynamical scenes. In NeurIPS '2023 , pages 54987-5500. vi

[124] Narayanan Sundaram, Thomas Brox, and Kurt Keutzer. Dense point trajectories by GPU-accelerated large displacement optical flow. In ECCV , pages 438-451, 2010. ii

[125] Sungmin Woo, Woonje Lee, Woo Woo Jin, Doogyoon Lee, and Sangyoun Lee. Prodepth: Boosting self-supervised multi-frame monocular depth with probabilistic fusion. In ECCV , pages 201-217, 2024. vi

[126] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ADE20K dataset. In CVPR , pages 5122-5130, 2017. vi