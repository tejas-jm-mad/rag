

# [PAGE 1]
2025-9-25
Video models are zero-shot learners and reasoners
Thaddäus Wiedemer*1, Yuxuan Li1, Paul Vicol1, Shixiang Shane Gu1, Nick Matarese1, Kevin Swersky1,
Been Kim1, Priyank Jaini*1 and Robert Geirhos*1
1Google DeepMind
The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language
processing from task-specific models to unified, generalist foundation models. This transformation
emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the
same primitives apply to today’s generative video models. Could video models be on a trajectory
towards general-purpose vision understanding, much like LLMs developed general-purpose language
understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn’t explicitly trained
for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing
object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate
the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo’s emergent
zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision
foundation models.
Project page: https://video-zero-shot.github.io/
1. Introduction
We believe that video models will become unifying, general-purpose foundation models for machine
vision just like large language models (LLMs) have become foundation models for natural language
processing (NLP). Within the last few years, NLP underwent a radical transformation: from task-
specific, bespoke models (e.g., one model for translation, another one for question-answering, yet
another one for summarization) to LLMs as unified foundation models. Today’s LLMs are capable
of general-purpose language understanding, which enables a single model to tackle a wide variety
of tasks including coding [38], math [12], creative writing [74], summarization, translation [89],
and deep research [55, 67]. These abilities started to emerge from simple primitives: training large,
generative models on web-scale datasets [e.g 6, 77]. As a result, LLMs are increasingly able to solve
novel tasks through few-shot in-context learning [6, 17] and zero-shot learning [43]. Zero-shot
learning here means that prompting a model with a task instruction replaces the need for fine-tuning
or adding task-specific inference heads.
Machine vision today in many ways resembles the state of NLP a few years ago: There are excellent
task-specific models like “Segment Anything” [42, 62] for segmentation or YOLO variants for object
detection [41, 64]. While attempts to unify some vision tasks exist [1, 47, 49–51, 58, 63, 76, 80, 81,
86], no existing model can solve any problem just by prompting. However, the exact same primitives
that enabled zero-shot learning in NLP also apply to today’s generative video models—large-scale
training with a generative objective (text/video continuation) on web-scale data [83]. In this article,
we therefore ask: Do video models develop general-purpose vision understanding, similar to how
LLMs developed general-purpose language understanding? We answer this question in the affirmative:
1. Analyzing 18,384 generated videos across 62 qualitative and 7 quantitative tasks, we report
that Veo 3 can solve a wide range of tasks that it was neither trained nor adapted for.
2. Based on its ability to perceive, model, and manipulate the visual world, Veo 3 shows early
forms of “chain-of-frames (CoF)” visual reasoning like maze and symmetry solving.
3. While task-specific bespoke models still outperform a zero-shot video model, we observe a
substantial and consistent performance improvement from Veo 2 to Veo 3, indicating a rapid
advancement in the capabilities of video models.
2. Methods
Approach and motivation
Our method is simple: We prompt Veo. This minimalist strategy is
intentional, as it mirrors the transformation of NLP from task-specific fine-tuning or training to
* Joint leads.
arXiv:2509.20328v1  [cs.LG]  24 Sep 2025
[FIGURE img_p0_000]

[EQ eq_p0_000] -> see eq_p0_000.tex

[EQ eq_p0_001] -> see eq_p0_001.tex


# [PAGE 2]
Video models are zero-shot learners and reasoners
Perception
Modeling
Manipulation
Reasoning
· · ·
· · ·
Blind deblurring
Blind denoising
Dalmation illusion
Shape cue-conflict
Rorschach blots
Edge detection
Low-light enhancing
Super-resolution
Conjunctive search
Keypoint localization
Segmentation
Rigid bodies
Material optics mirror
Memory
Material optics glass
Colors: additive
Buoyancy stone
Object packing
Colors: subtractive
Soft bodies
Buoyancy bottle cap
Visual Jenga
Air resistance: earth
Air resistance: moon
Parsing into parts
Categorizing objects
Character recognition
Flammability
Character generation
Inpainting
Outpainting
Editing with doodles
Manipulation: jar
Manipulation: throw
Novel view synthesis
Background removal
Reposing
Style transfer
Scene composition
Affordance
Professional headshot
Text manipulation
Drawing
Visual instructions
Transfiguration
Colorization
Manipulation: balls
Sequence: arrows
Sequence: squares
Sequence: circles
Sudoku
Robot navigation
Water puzzle
Sequence: dots
Connecting colors
Spatial reasoning
Tree BFS
Water solving
Graph traversal
Sorting numbers
Tool use
Rule extrapolation
0
1
Success rate
Figure 1 | A qualitative overview of Veo 3’s zero-shot abilities. The plot shows Veo 3’s success
rate across 12 samples as a rough estimate of model performance on 62 tasks across the vision stack.
Tasks are described in Sec. 3 and shown in Sec. A. Find videos of all tasks on our project page.
prompting a capable foundation model [53, 78, 88]. Here, we adopt the same philosophy to explore
the capabilities of Veo 3 as a general-purpose vision model.
Takeaway 1
In NLP, prompting replaced task-specific training or adaptation. A similar paradigm
shift is on the horizon in machine vision, facilitated by video models.
Video generation
For each task, we query the publicly available Veo 2 or Veo 3 models via Google
Cloud’s Vertex AI API. We prompt the model with an initial input image (which the model uses as
the first frame) and a text instruction. The models then generate a 16:9 video at 720p resolution,
24 FPS, for a duration of 8s. Veo 3 has model ID veo-3.0-generate-preview and Veo 2 model ID
veo-2.0-generate-001. According to the Vertex documentation [29], the API uses an LLM-based
prompt rewriter. This means that for some tasks, the solution is likely to come from the LLM instead
of the video (e.g., Fig. 55: Sudoku). We treat the system (rewriter and video generator) as a single
black-box entity. However, to isolate the video model’s reasoning abilities, we verified that a standalone
LLM (Gemini 2.5 Pro [12]) could not reliably solve key tasks (Fig. 58: Robot navigation, Sec. 4.5:
Maze solving, Sec. 4.6: Visual symmetry) from the input image alone.
Why Veo?
The core argument of this paper—that video models are zero-shot learners and reasoners—
can be supported by demonstrating success on any sufficiently capable model. We choose Veo because
it has consistently ranked high on text2video and image2video leaderboards [54]. Unless noted
otherwise, our figures are generated with Veo 3. To provide a sense of how rapidly performance is
improving, our quantitative analyses compare Veo 3 with its predecessor, Veo 2, released roughly
within half a year of each other: Veo 2 was announced in December 2024 and released in April
2025 [24, 26], while Veo 3 was announced in May 2025 and released in July 2025 [27, 28].
3. Qualitative results: Sparks of visual intelligence?
We begin with a comprehensive, qualitative investigation across visual tasks to assess the potential of
video models as visual foundation models. We organize our findings into four hierarchical capabilities,
each building on the last (c.f. Fig. 1 and Fig. 2):
1. Perception as a foundational ability to understand visual information.
2. Modeling, which builds on the perception of objects to form a model of the visual world.
3. Manipulation, which meaningfully alters the perceived and modeled world.
4. Reasoning across space and time over a sequence of manipulation steps.
2
[FIGURE img_p1_001]
[FIGURE img_p1_002]
[FIGURE img_p1_003]
[FIGURE img_p1_004]
[FIGURE img_p1_005]
[FIGURE img_p1_006]
[FIGURE img_p1_007]
[FIGURE img_p1_008]


# [PAGE 3]
Video models are zero-shot learners and reasoners
Perception – superresolution
Perception – conjunctive search / binding problem
Modeling – buoyancy
Modeling – memory of world states
Manipulation – 3D-aware reposing
Manipulation – jar opening
Reasoning – robot navigation
Reasoning – rule extrapolation
Figure 2 | Veo 3 zero-shot learning and reasoning examples. From classic perceptual tasks
(superresolution, visual search) to modeling (buoyancy, memory of world states after zooming in),
manipulation (pose editing, simulating dexterous manipulation) and visual reasoning (navigation,
rule extrapolation): Veo 3 can zero-shot solve a host of visual tasks that are specified as an input
image and a text prompt. Examples are shown in Sec. A; videos of all tasks are on our project page.
While capability boundaries often overlap, this hierarchy provides a framework for understanding the
emergent abilities of video models. For example, solving a maze (see Fig. 57 and Sec. 4.5) requires
perceiving the maze, modeling its state (walls vs. floor), and finally manipulating an object (a mouse,
a circle) to move from start to finish.
For each task in this section, we prompt Veo 3 twelve times and report the success rate in the
caption. The success rate is defined as the fraction of generated videos that solved the task (as
determined by the authors). A success rate greater than 0 suggests that the model possesses the ability
to solve the task, while a success rate closer to 1 indicates that the task is solved reliably irrespective
of the random seed. While not a substitute for the systematic quantification we perform in Sec. 4,
this provides a ballpark estimate of the model’s capabilities.
Perception
Computer vision has historically relied on a suite of specialized models for tasks like
segmentation [42, 62], object detection [41, 64], and edge detection [82]. While some backbones
can be adapted or fine-tuned for other tasks, training-free transfer to novel tasks is rare, limiting
generalization. As we show here, this is changing with large video models.
Without any task-specific training, Veo 3 can perform a range of classic computer vision tasks,
including edge detection (Fig. 10), segmentation (Fig. 11), keypoint localization (Fig. 12), super-
resolution (Fig. 13), blind deblurring (Fig. 14), denoising (Fig. 15) and low-light enhancing (Fig. 16).
These emergent abilities extend to more complex perceptual tasks like conjunctive search (Fig. 17)
and interpreting ambiguous images such as the the classic dalmatian illusion (Fig. 18), the cat shape
in a texture-shape cue conflict image (Fig. 19), and colored blots from the Rorschach test (Fig. 20).
Apart from denoising—the classic diffusion objective—none of these tasks are explicitly trained for in
video models.
Takeaway 2
Veo 3 shows emergent zero-shot perceptual abilities well beyond the training task.
Just like LLMs replaced task-specific NLP models, video models will likely replace most bespoke
models in computer vision—once they become sufficiently cheap and reliable.
3
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
[FIGURE img_p2_023]
[FIGURE img_p2_024]
[FIGURE img_p2_025]
[FIGURE img_p2_026]
[FIGURE img_p2_027]
[FIGURE img_p2_028]
[FIGURE img_p2_029]
[FIGURE img_p2_030]
[FIGURE img_p2_031]
[FIGURE img_p2_032]


# [PAGE 4]
Video models are zero-shot learners and reasoners
Modeling: intuitive physics & world models
Based on their perception of the visual world, video
models are starting to model it, too. Modeling the world and the principles that govern it (e.g., laws of
physics) is a critical step toward successful prediction and action. Several works have investigated and
quantified intuitive physics in deep models [e.g., 3, 4, 9, 19, 36, 39, 52, 57, 59, 65, 68, 73, 79, 87].
Here, we investigate an exemplary subset of tasks from these works.
Veo’s grasp of physics is demonstrated by its ability to model rigid and soft body dynamics and their
surface interactions (Fig. 22). It also displays an understanding of various physical properties, like
flammability (Fig. 21), air resistance affecting falling objects (Fig. 23), buoyancy (Fig. 24), and optical
phenomenon like refraction and reflection (Fig. 27) and additive/subtractive color mixing (Fig. 28).
As illustrated by the Visual Jenga task [5], Veo can remove objects from a scene in a physically
plausible order (Fig. 25) and understands which objects fit into a backpack (Fig. 26).
Beyond physical characteristics, Veo understands abstract relationships which is an important
aspect of modeling the world. As a trivial example, Veo can distinguish categories like toys from other
objects like a laptop (Fig. 29). On samples inspired by the Omniglot dataset [45], we demonstrate Veo’s
ability to recognize patterns, generate variations thereof, and parse larger wholes into parts (Fig. 30).
Lastly, Veo maintains a memory of the world state across time and camera movements within the
video context (Fig. 31).
Manipulation: editing & imagination
Based on its ability to perceive objects and model their relation
to each other and the world, Veo can meaningfully manipulate the visual world, too. This enables
Veo 3 to perform a variety of zero-shot image editing tasks like backgorund removal (Fig. 32), style
transfer (Fig. 33), colorization (Fig. 34), inpainting (Fig. 35), and outpainting (Fig. 36). Furthermore,
it can manipulate text elements (Fig. 37), and edit images based on doodle instructions (Fig. 38).
Veo’s understanding of 3D world enables it to compose scenes from individual components (Fig. 39),
generate novel views of objects and characters (Figs. 40 and 41), smoothly transform one object
into another (Fig. 42), or change the perspective, lighting, and appearance to turn a selfie into a
professional photograph (Fig. 43).
This ability to plausibly modify a scene allows it to imagine complex interactions, simulate
dexterous object manipulation (Fig. 44; note that we do not test actual robots as e.g. [2] do),
interpreting object affordances (Fig. 45), demonstrating how to draw a shape (Fig. 46) and roll a
burrito (Fig. 47). Overall, video models can meaningfully manipulate and simulate the (digital)
visual world.
Visual reasoning across time and space
Perception, modeling, and manipulation all integrate to
tackle visual reasoning. While language models manipulate human-invented symbols, video models
can apply changes across the dimensions of the real world: time and space. Since these changes
are applied frame-by-frame in a generated video, this parallels chain-of-thought in LLMs and could
therefore be called chain-of-frames, or CoF for short. In the language domain, chain-of-thought
enabled models to tackle reasoning problems [78]. Similarly, chain-of-frames (a.k.a. video generation)
might enable video models to solve challenging visual problems that require step-by-step reasoning
across time and space.
We see early signs of this ability in tasks such as generating a valid graph traversal (Fig. 48), per-
forming visual breadth-first search on a tree (Fig. 49), completing visual sequences (Fig. 50), connect-
ing matching colors (Fig. 51), fitting shapes into holes (Fig. 52), and sorting numbers (Fig. 53). Fur-
thermore, Veo can use tools to accomplish a visual task (Fig. 54) and solve simple Sudokus (Fig. 55) or
visual puzzles (Fig. 56). Finally, it can solve mazes and navigation tasks (Figs. 57 and 58 and Sec. 4.5)
and extrapolate rules from visual examples (Fig. 59). While not always perfect, the model’s ability to
solve such problems in a zero-shot manner points to exciting possibilities for more advanced visual
reasoning and planning in future, with more capable video models.
Takeaway 3
Frame-by-frame video generation parallels chain-of-thought in language models.
Just like chain-of-thought (CoT) enables language models to reason with symbols, a “chain-of-
frames” (CoF) enables video models to reason across time and space.
4


# [PAGE 5]
Video models are zero-shot learners and reasoners
Original image
Generated edge map (Veo 3)
Ground-truth edge map
1
2
3
4
5
6
7
8
9
10
k
0.4
0.5
0.6
0.7
max OIS@k
0.74
0.57
0.77
Best frame
1
2
3
4
5
6
7
8
9
10
k
0.4
0.5
0.6
0.7
0.74
0.51
0.74
Last frame
Model
Veo 3
Veo 2
Nano Banana
Figure 3 | Edge detection on all 50 test images from BIPEDv2 [71, 72]. We generate 10 videos per
sample and report best performance over 𝑘attempts as a function of 𝑘. Prompt: “All edges in this
image become more salient by transforming into black outlines. Then, all objects fade away [...]” Details
& full prompt: Sec. B.1.
Summary
Taken together, the qualitative examples from this section indicate that a capable video
model like Veo 3 possesses a strong and generalizable zero-shot learning ability. While the results are
not always perfect, the model consistently demonstrates the capacity to solve a wide variety of tasks
for which it was not explicitly trained.
4. Quantitative results
The previous section offered a qualitative exploration of video model capabilities. In this section, we
add a quantitative assessment for seven select tasks. As in Sec. 3, we consider different facets of visual
understanding: For perception, we assess Veo on edge detection, segmentation, and object extraction.
For manipulation, we examine image editing performance. Finally, we evaluate reasoning abilities
through maze solving, visual symmetry, and visual analogies. We do not include specific evaluations
for modeling, since this area is well addressed by recent benchmarks, see Sec. 3.
We evaluate performance separately for the best frame and the last frame (where applicable).
Best frame reports the best performance across any frame in the generated videos. This indicates
the performance ceiling, but the optimal frame is not known a priori. Therefore, we also report
performance on the last frame of each video, which may underestimate a model’s ability but has the
practical advantage that the frame is predetermined. This distinction is important because Veo tends
to continue animating a scene even after task completion, potentially reducing last frame performance.
Where applicable, we use the state-of-the-art image editing model Nano Banana [25] as a reference.
As a general trend, we observe a large performance increase from Veo 2 to Veo 3, often matching
or exceeding Nano Banana performance. Performance tends to improve substantially from 𝑘= 1 to
𝑘= 10 attempts, indicating that a good solution can be found in a reasonable number of tries. While
image models are excellent zero-shot learners, too [7, 11, 34, 40], video models are the more general
framework because of their ability to process both temporal and spatial information.
4.1. Perception: Edge detection
Despite not being trained for it, Veo 3 can be prompted to detect, and therefore perceive, edges.
Fig. 3 details edge detection performance (measured by OIS; details and prompt in Sec. B.1) for
Veo 2 and Veo 3. While Veo 3 (0.77 pass@10) is not on par with task-specific SOTA (0.90) [72],
5
[FIGURE img_p4_033]
[FIGURE img_p4_034]
[FIGURE img_p4_035]


# [PAGE 6]
Video models are zero-shot learners and reasoners
Original image
Generated frame (Veo 3)
Extracted masks
Ground-truth masks
1
2
3
4
5
6
7
8
9
10
k
0.2
0.3
0.4
0.5
0.6
0.7
max mIoU@k
0.73
0.52
0.50
0.66
0.74
Best frame
1
2
3
4
5
6
7
8
9
10
k
0.2
0.3
0.4
0.5
0.6
0.7
0.73
0.40
0.41
0.42
0.56
Last frame
Model
Background
Veo 3
Veo 2
Nano Banana
green
white
Figure 4 | Class-agnostic instance segmentation on a subset of 50 easy images (1-3 large objects)
from LVIS [31]. Prompt: “[...] each distinct entity is overlaid in a different flat color [...] the background
fades to {white, green} [...]” Details & full prompt: Sec. B.2.
Original image
Generated frame (Veo 3)
1
2
3
4
5
6
7
8
9
10
k
0.2
0.4
0.6
0.8
Pass@k
0.93
0.63
Last frame
Model
Veo 3
Veo 2
Chance
Figure 5 | Object extraction on an animal dataset. Prompt: “The background changes to white [...] all
animals line up in a row [...]” Details & full prompt: Sec. B.3.
its performance is remarkable for two reasons: First, it is zero-shot. Second, many of Veo 3’s edge
maps are more detailed than the ground truth. For example, Veo 3 correctly outlines foliage and tire
profiles; this hurts performance on the dataset but seems more indicative of a dataset limitation than
a model limitation (the annotators understandably did not bother to trace each and every edge).
4.2. Perception: Segmentation
Instance segmentation asks to delineate (i.e., perceive) distinct objects in an image. Contrary to
classic instance segmentation or promptable segmentation, we prompt models to segment all objects
in a scene, without specifying an object category or location. We report mean Intersection over Union
(mIoU); experiment details are in Sec. B.2. As shown in Fig. 4, Veo 3 achieves an mIoU of 0.74 (best
frame pass@10), comparable to Nano Banana’s 0.73. Naturally, Veo 3 lacks behind the performance
of a bespoke model like SAMv2 [62], but nevertheless shows remarkable zero-shot segmentation
abilities. Interestingly, the prompt really matters: Veo consistently performs better with a green
background than a white one (0.74 vs. 0.66 best frame pass@10); possibly due to the widespread
use of green screens. See also Sec. C for prompting best practices.
4.3. Manipulation: Object extraction
Can Veo perceive and extract (i.e., manipulate) all objects in a scene? We test this using a simple
dataset depicting one to nine animals (details in Sec. B.3). Veo is asked to extract and line up all
6
[FIGURE img_p5_036]
[FIGURE img_p5_037]
[FIGURE img_p5_038]
[FIGURE img_p5_039]
[FIGURE img_p5_040]
[FIGURE img_p5_041]


# [PAGE 7]
Video models are zero-shot learners and reasoners
Original image
Generated frame (Veo 3)
Generated frame (Veo 3)
Fidelity
Precision
0
20
40
60
80
100
Human Rating %
100
60
20
10
Model
Veo 3
Veo 2
Figure 6 | Image editing on a subset of Emu-edit [69]. Prompt: “Create a smooth, static animation
that slowly changes the color of the fire hydrant to red. [...]” Details & full prompt: Sec. B.4.
Pass@k %
5×5 Grid
7×7 Grid
9×9 Grid
Irregular
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
80
16
92
74
14
78
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
8
72
40
2
38
1
2
3
4
5
6
7
8
9 10
k
0
5
10
15
20
0
8
22
2
12
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
0
0
75
Model
Veo 3
Veo 2
Nano Banana
Gemini 2.5 Pro I2T
Gemini 2.5 Pro T2T
Figure 7 | Maze solving. Mazes of various sizes with start (red) and goal (green) locations. Prompt:
“[...] The red square slides smoothly along the white path, stopping perfectly on the green square [...]”
Details & full prompt: Sec. B.5. Veo 2 struggles to solve even small sizes, mostly due to illegal
moves early in the generation. Veo 3 performs much better and benefits from multiple attempts. For
reference, we evaluate Nano Banana and show Gemini 2.5 Pro’s performance on mazes presented as
images (I2T) or ASCII text (T2T).
animals in a row (in some sense, a visual “tally”), with white space between them. To assess whether
the number of extracted animals is correct by counting connected components in the last frame. Fig. 5
shows a sample and the results. While Veo 2 performs around chance, Veo 3 achieves up to 92 6%
pass@10. Given the simplicity of the task, a perfect model should easily achieve 100 0% accuracy.
4.4. Manipulation: Image editing
Image editing requires manipulating images according to a text instruction (e.g., adding/removing
objects or changing their appearance). We evaluate whether Veo can edit images on a random subset
of 30 samples from Emu-edit [69]. Veo has a strong bias for animated scenes and might introduce
unintended changes (e.g., camera movement, animating people). We therefore ask three human
raters to evaluate fidelity (correct edit) and precision (correct edit without unintended changes). An
example edit and results are shown in Fig. 6. We find that Veo 3 especially excels in preserving
details and textures across edits. With better control to curb unintended changes, video models
could become capable, 3D-aware image editors (and likely video editors, too). If unintended changes
such as camera movement or animating people can be controlled better, video models could become
capable 3D-aware image and video editors (see also Molad et al. [58], Rotstein et al. [66]).
7
[FIGURE img_p6_042]
[FIGURE img_p6_043]
[FIGURE img_p6_044]
[FIGURE img_p6_045]
[FIGURE img_p6_046]
[FIGURE img_p6_047]
[FIGURE img_p6_048]

[EQ eq_p6_000] -> see eq_p6_000.tex

[EQ eq_p6_001] -> see eq_p6_001.tex

[EQ eq_p6_002] -> see eq_p6_002.tex

[EQ eq_p6_003] -> see eq_p6_003.tex


# [PAGE 8]
Video models are zero-shot learners and reasoners
Pass@k %
Original image
Generated frame (Veo 3)
Original image
Generated frame (Veo 3)
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
80
100
37
20
88
Best frame
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
80
100
37
0
44
Last frame
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
80
100
28
0
100
Best frame
1
2
3
4
5
6
7
8
9 10
k
0
20
40
60
80
100
28
0
72
Last frame
Shapes
Random patterns
Model
Veo 3
Veo 2
Nano Banana
Figure 8 | Visual symmetry. Prompt: “Instantly reflect this pattern along the central, vertical axis while
keeping the existing colored pattern without modification. [...]” A model has to color all cells correctly
to pass. Details & full prompt: Sec. B.6.
4.5. Reasoning: Maze solving
Maze solving tests a model’s ability to plan a path in a constrained environment, a key aspect of
reasoning. In our setup, a red circle needs to navigate to a target location (green circle) without
crossing any walls. We automatically verify path correctness (details in Sec. B.5) and present results
for different mazes in Fig. 7. Veo 3 shows zero-shot maze solving abilities, significantly outperforming
Veo 2 which often produces illegal moves. For instance, in the 5×5 grids, Veo 3 achieves a pass@10 rate
of 78% compared to Veo 2’s 14%. The consistent performance gap highlights the advancing reasoning
capabilities of the newer model. While Nano Banana matches or surpasses Veo 3’s performance on
rectangular mazes, it fails to solve irregular mazes entirely. Similarly, Gemini 2.5 Pro outperforms
Veo 3 on small mazes when given an ASCII representation of the maze (T2T), but falls behind on 9×9
mazes, and generally struggles when the maze is represented as image (as opposed to text) input.
Both comparisons highlight the advantages of solving a visual task step-by-step in a visual medium.
4.6. Reasoning: Visual symmetry solving
Pattern completion for symmetry assess the ability to understand and apply spatial reasoning. We
use a custom dataset of shapes (e.g., heart, letters) and random patterns (see Sec. B.6 for details).
Fig. 8 shows that Veo 3 outperforms Veo 2 and Nano Banana by a large margin. We also use this task
to systematically analyze how different prompts affect performance in Sec. C: The pass@1 difference
between best and worst prompt is 40 percentage points on the shape split and 64 percentage points
on the random split.
4.7. Reasoning: Visual analogy completion
Visual analogies test a model’s ability to understand transformations and relationships between
objects, a form of abstract reasoning. Concretely, we prompt the model to fill the missing quadrant
of a 2×2 grid to complete the analogy (e.g., A is to B as C is to ?). We evaluate the correctness of the
generated infill for four transformation types from KiVA [85], see Sec. B.7 for details. The results are
summarized in Fig. 9. While Veo 2 struggles to understand any analogies, Veo 3 correctly completes
examples for color and resize. However, both models perform below chance (0.33) on reflect and
rotate analogies, indicating an erroneous, systematic bias.
8
[FIGURE img_p7_049]
[FIGURE img_p7_050]
[FIGURE img_p7_051]
[FIGURE img_p7_052]

[EQ eq_p7_000] -> see eq_p7_000.tex


# [PAGE 9]
Video models are zero-shot learners and reasoners
Original image
Completion (Veo 3)
Color
Resize
Reflect
Rotate
0
20
40
60
80
100
Pass@1 %
95
67
29
19
68
40
23
22
Model
Veo 3
Veo 2
Chance
Figure 9 | Visual analogy solving on four transformations à 50 samples from KiVA [85]. Prompt:
“[...] generate the missing object in the lower right region and solve the visual analogy. [...]” Pass@1 is
evaluated on the last frame, results up to pass@10 can be found in Sec. B.7.
Takeaway 4
While far from perfect, Veo 3—building on its ability to perceive, model and
manipulate objects—shows emergent visual reasoning abilities.
5. Discussion
Summary
A breakthrough in machine vision started the deep learning revolution in 2012 [44], but
in recent years, natural language processing has seen the most rapid progress. This was driven by
the rise of general-purpose LLMs, whose ability to solve novel tasks in a zero-shot fashion has led
them to replace most task-specific models in NLP. We here make the case that machine vision is on
the cusp of a similar paradigm shift, enabled by emergent abilities of large-scale video models. Our
core finding is that Veo 3 can solve a wide range of tasks in a zero-shot manner, spanning the full
vision stack from perception to modeling, manipulation and even early forms of visual reasoning.
While its performance is not yet perfect, the massive and consistent improvement from Veo 2 to Veo 3
indicates that video models will become general-purpose foundation models for vision, just as LLMs
have for language.
Performance is a lower bound
Tasks can be represented in a myriad of ways; a maze, for example,
can be presented as a black-and-white grid, a video game, or a photorealistic scene, with the prompt
requesting a solution in the form of a line, a moving object, or a glowing path. Moreover, visually, a
maze could be represented as a black-and-white grid, a Pac-Man game, or a photorealistic top-down
view of an apartment. This has three implications: First, prompt engineering—including the visual
prompt a.k.a. starting frame—is as important for visual tasks as it is for LLMs (see also Sec. C and
[46] for a discussion). Second, we must distinguish between a model’s task performance and its
underlying ability (i.e., competence) to solve that task [18, 21]. Third, as a consequence, the model
performance reported here with a given visual and textual prompt should be considered a lower
bound on the model’s true capabilities. This also holds for the tasks that we report as failure cases
in Sec. D, such as providing visual instructions to fold laundry (Fig. 76), planning to move a sofa
between rooms separated by a small door (Fig. 77), or certain visual puzzles (Fig. 70).
Video generation is expensive, but costs tend to fall
While generating a video is currently more
expensive than running a bespoke, task-specific model, the economics of general-purpose models
are on a predictable trajectory. Epoch AI [13] estimates that LLM inference costs are falling by a
factor of 9× to 900× per year for a given performance level. In NLP, early generalist models were
also considered prohibitively expensive (“GPT-3’s size makes it challenging to deploy” [6, p. 8]).
Nevertheless, rapidly falling inference costs, combined with the appeal of generalist models, have
replaced most task-specific language models. If NLP is a guide, the same trend will play out in vision.
Jack of many trades, master of few?
For many tasks, Veo 3’s performance is below state of the art
of specialized models. This mirrors the early days of LLMs; GPT-3 reported performance well below
fine-tuned models on many tasks [6, cf. Tables 3.1, 3.3, 3.4, 3.5]). This did not stop language models
from becoming foundation models, and we don’t believe it will stop video models from becoming
vision foundation models for two reasons. First, the step-change in performance from Veo 2 to Veo 3
9
[FIGURE img_p8_053]
[FIGURE img_p8_054]

[EQ eq_p8_000] -> see eq_p8_000.tex


# [PAGE 10]
Video models are zero-shot learners and reasoners
is evidence of rapid progress over time. Second, our scaling results from Sec. 4 show pass@10 to
be consistently higher than pass@1 with no signs of a plateau. Therefore, inference-time scaling
methods [e.g. 33, 56, 70, 75] in combination with the standard optimization toolkit like post-training
with automatic verifiers are likely to boost performance. For the tasks we test here, Veo 3 is akin to a
pre-trained language model that has yet to undergo instruction tuning or RLHF [60, 61].
Outlook
This is an exciting time for vision. Seeing NLP’s recent transformation from task-specific
to generalist models, it is conceivable that the same transformation will happen in machine vision
through video models (a “GPT-3 moment for vision”), enabled by their emergent ability to perform a
broad variety of tasks in a zero-shot fashion, from perception to visual reasoning.
Acknowledgements
We would like to thank Oyvind Tafjord, Mike Mozer, Katherine Hermann, Andrew Lampinen, Viorica
Patraucean, Shiry Ginosar, Ross Goroshin, Abhijit Ogale, Claire Cui, Kun Zhang for helpful feed-
back/discussions, Anish Nangia for Vertex, Tuan Anh Le for suggesting the Omniglot evaluation,
Medhini Narasimhan and Pieter-Jan Kindermans for Veo insights and discussion, Micromelon Robotics
for permission to use the image in Fig. 58, Shelly Sheynin for permission to use the EMU Edit dataset,
David Fleet and Jon Shlens for their support, and the Veo team for developing an incredible model.
Author contributions
Leads: TW, PJ, RG. Project idea: RG. Core contributors: YL, PV. Partial contributor: SG. Fig. 38: NM.
Advisors: KS, BK.
References
[1] P. Acuaviva, A. Davtyan, M. Hassan, S. Stapf, A. Rahimi, A. Alahi, and P. Favaro.
From
generation to generalization: Emergent few-shot learning in video diffusion models. arXiv
preprint arXiv:2506.07280, 2025.
[2] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha,
A. Zholus, et al. V-JEPA 2: Self-supervised video models enable understanding, prediction and
planning. arXiv preprint arXiv:2506.09985, 2025.
[3] H. Bansal, Z. Lin, T. Xie, Z. Zong, M. Yarom, Y. Bitton, C. Jiang, Y. Sun, K.-W. Chang, and
A. Grover. Videophy: Evaluating physical commonsense for video generation, 2024.
[4] D. M. Bear, E. Wang, D. Mrowca, F. J. Binder, H.-Y. F. Tung, R. T. Pramod, C. Holdaway, S. Tao,
K. Smith, F.-Y. Sun, L. Fei-Fei, N. Kanwisher, J. B. Tenenbaum, D. L. K. Yamins, and J. E. Fan.
Physion: Evaluating physical prediction from vision in humans and machines, 2021.
[5] A. Bhattad, K. Preechakul, and A. A. Efros. Visual jenga: Discovering object dependencies via
counterfactual inpainting. arXiv preprint arXiv:2503.21770, 2025.
[6] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,
G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information
processing systems, 33:1877–1901, 2020.
[7] R. Burgert, K. Ranasinghe, X. Li, and M. S. Ryoo. Peekaboo: Text to image diffusion models are
zero-shot segmentors. arXiv preprint arXiv:2211.13224, 2022.
[8] D. Campbell, S. Rane, T. Giallanza, C. N. De Sabbata, K. Ghods, A. Joshi, A. Ku, S. Frankland,
T. Griffiths, J. D. Cohen, et al. Understanding the limits of vision language models through the
lens of the binding problem. Advances in Neural Information Processing Systems, 37:113436–
113460, 2024.
10


# [PAGE 11]
Video models are zero-shot learners and reasoners
[9] A. Cherian, R. Corcodel, S. Jain, and D. Romeres. LLMPhy: Complex physical reasoning using
large language models and world models. arXiv preprint arXiv:2411.08027, 2024.
[10] F. Chollet. On the measure of intelligence. arXiv preprint arXiv:1911.01547, 2019.
[11] K. Clark and P. Jaini. Text-to-image diffusion models are zero shot classifiers. Advances in Neural
Information Processing Systems, 36:58921–58937, 2023.
[12] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram,
D. Zhang, E. Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodal-
ity, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261,
2025.
[13] B. Cottier, B. Snodin, D. Owen, and T. Adamczewski. LLM inference prices have fallen rapidly
but unequally across tasks, march 2025. URL https://epoch.ai/data-insights/llm-i
nference-price-trends. Accessed: 2025-09-12.
[14] M. Deitke, R. Liu, M. Wallingford, H. Ngo, O. Michel, A. Kusupati, A. Fan, C. Laforte, V. Voleti,
S. Y. Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information
Processing Systems, 36:35799–35813, 2023.
[15] P. Dollár and C. L. Zitnick. Structured forests for fast edge detection. In ICCV, 2013.
[16] P. Dollár and C. L. Zitnick. Fast edge detection using structured forests. ArXiv, 2014.
[17] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, T. Liu, et al. A survey on
in-context learning. arXiv preprint arXiv:2301.00234, 2022.
[18] C. Firestone. Performance vs. competence in human–machine comparisons. Proceedings of the
National Academy of Sciences, 117(43):26562–26571, 2020.
[19] Q. Garrido, N. Ballas, M. Assran, A. Bardes, L. Najman, M. Rabbat, E. Dupoux, and Y. LeCun.
Intuitive physics understanding emerges from self-supervised pretraining on natural videos.
arXiv preprint arXiv:2502.11831, 2025.
[20] R. Geirhos, P. Rubisch, C. Michaelis, M. Bethge, F. A. Wichmann, and W. Brendel. ImageNet-
trained CNNs are biased towards texture; increasing shape bias improves accuracy and robust-
ness. In International conference on learning representations, 2019.
[21] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann.
Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.
[22] D. Geng, C. Herrmann, J. Hur, F. Cole, S. Zhang, T. Pfaff, T. Lopez-Guevara, C. Doersch, Y. Aytar,
M. Rubinstein, C. Sun, O. Wang, A. Owens, and D. Sun. Motion prompting: Controlling video
generation with motion trajectories. arXiv preprint arXiv:2412.02700, 2024.
[23] N. Gillman, C. Herrmann, M. Freeman, D. Aggarwal, E. Luo, D. Sun, and C. Sun. Force
prompting: Video generation models can learn and generalize physics-based control signals,
2025. URL https://arxiv.org/abs/2505.19386.
[24] Google. Veo 2 announcement. https://blog.google/technology/google-labs/vide
o-image-generation-update-december-2024/, 2024. Accessed: September 22, 2025.
[25] Google. Nano Banana: Gemini Image Generation Overview. https://gemini.google/ov
erview/image-generation/, 2025. Accessed: September 22, 2025.
[26] Google. Veo 2 launch. https://developers.googleblog.com/en/veo-2-video-gen
eration-now-generally-available/, 2025. Accessed: September 22, 2025.
11


# [PAGE 12]
Video models are zero-shot learners and reasoners
[27] Google. Veo 3 announcement. https://blog.google/technology/ai/generative-m
edia-models-io-2025/, 2025. Accessed: September 22, 2025.
[28] Google. Veo 3 launch. https://cloud.google.com/blog/products/ai-machine-l
earning/veo-3-fast-available-for-everyone-on-vertex-ai, 2025. Accessed:
September 22, 2025.
[29] Google Cloud. Vertex AI Veo Prompt Rewriter. https://cloud.google.com/vertex-ai/
generative-ai/docs/video/turn-the-prompt-rewriter-off#prompt-rewriter,
2025. Accessed: September 22, 2025.
[30] R. L. Gregory. The intelligent eye, 1970.
[31] A. Gupta, P. Dollar, and R. Girshick. LVIS: A dataset for large vocabulary instance segmentation.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.
[32] M. Ivanitskiy. Maze dataset. https://pypi.org/project/maze-dataset/0.3.4/, 2025.
Accessed: June 31, 2025.
[33] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel,
A. Carney, et al. OpenAI o1 system card. arXiv preprint arXiv:2412.16720, 2024.
[34] P. Jaini, K. Clark, and R. Geirhos. Intriguing properties of generative classifiers. In The Twelfth
International Conference on Learning Representations, 2023.
[35] R. C. James. Sight for sharp eyes. LIFE, 58(7):120, 1965.
[36] S. Jassim, M. Holubar, A. Richter, C. Wolff, X. Ohmer, and E. Bruni. GRASP: A novel benchmark
for evaluating language grounding and situated physics understanding in multimodal language
models. arXiv preprint arXiv:2311.09048, 2023.
[37] Y. Jeong, A. Uselis, S. J. Oh, and A. Rohrbach. Diffusion classifiers understand compositionality,
but conditions apply. arXiv preprint arXiv:2505.17955, 2, 2025.
[38] J. Jiang, F. Wang, J. Shen, S. Kim, and S. Kim. A survey on large language models for code
generation. arXiv preprint arXiv:2406.00515, 2024.
[39] B. Kang, Y. Yue, R. Lu, Z. Lin, Y. Zhao, K. Wang, G. Huang, and J. Feng. How far is video
generation from world model: A physical law perspective. arXiv preprint arXiv:2411.02385,
2024.
[40] L. Khachatryan, A. Movsisyan, V. Tadevosyan, R. Henschel, Z. Wang, S. Navasardyan, and H. Shi.
Text2video-zero: Text-to-image diffusion models are zero-shot video generators. In Proceedings
of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023.
[41] R. Khanam and M. Hussain. Yolov11: An overview of the key architectural enhancements. arXiv
preprint arXiv:2410.17725, 2024.
[42] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C.
Berg, W.-Y. Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference
on computer vision, pages 4015–4026, 2023.
[43] T. Kojima, S. S. Gu, M. Reid, Y. Matsuo, and Y. Iwasawa. Large language models are zero-shot
reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.
[44] A. Krizhevsky, I. Sutskever, and G. E. Hinton. ImageNet classification with deep convolutional
neural networks. Advances in neural information processing systems, 25, 2012.
[45] B. M. Lake, R. Salakhutdinov, and J. B. Tenenbaum. Human-level concept learning through
probabilistic program induction. Science, 350(6266):1332–1338, 2015.
12

[EQ eq_p11_000] -> see eq_p11_000.tex

[EQ eq_p11_001] -> see eq_p11_001.tex


# [PAGE 13]
Video models are zero-shot learners and reasoners
[46] A. K. Lampinen, S. C. Chan, A. K. Singh, and M. Shanahan. The broader spectrum of in-context
learning. arXiv preprint arXiv:2412.03782, 2024.
[47] D. H. Le, T. Pham, S. Lee, C. Clark, A. Kembhavi, S. Mandt, R. Krishna, and J. Lu. One diffusion
to generate them all. In Proceedings of the Computer Vision and Pattern Recognition Conference,
pages 2671–2682, 2025.
[48] K. Leng, Z. Zhang, J. Liu, Z. Boukhers, W. Sui, C. Yang, and Z. Li. Superedge: Towards a
generalization model for self-supervised edge detection. CoRR, 2024.
[49] Z.-Y. Li, R. Du, J. Yan, L. Zhuo, Z. Li, P. Gao, Z. Ma, and M.-M. Cheng. Visualcloze: A universal
image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960,
2025.
[50] W. Lin, X. Wei, R. Zhang, L. Zhuo, S. Zhao, S. Huang, H. Teng, J. Xie, Y. Qiao, P. Gao, et al.
Pixwizard: Versatile image-to-image visual assistant with open-language instructions. arXiv
preprint arXiv:2409.15278, 2024.
[51] Y. Lin, M. Huang, S. Zhuang, and Z. Mao. Realgeneral: Unifying visual generation via temporal
in-context learning with video models. arXiv preprint arXiv:2503.10406, 2025.
[52] D. Liu, J. Zhang, A.-D. Dinh, E. Park, S. Zhang, and C. Xu. Generative physical AI in vision: A
survey. arXiv preprint arXiv:2501.10928, 2025.
[53] P. Liu, W. Yuan, J. Fu, Z. Jiang, H. Hayashi, and G. Neubig. Pre-train, prompt, and predict:
A systematic survey of prompting methods in natural language processing. ACM computing
surveys, 55(9):1–35, 2023.
[54] LMSYS ORG. Lmsys org text-to-video leaderboard. https://lmarena.ai/leaderboard/t
ext-to-video, September 2025. Accessed: 2025-09-23.
[55] C. Lu, C. Lu, R. T. Lange, J. Foerster, J. Clune, and D. Ha. The AI Scientist: Towards fully
automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024.
[56] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhu-
moye, Y. Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural
Information Processing Systems, 36:46534–46594, 2023.
[57] F. Meng, J. Liao, X. Tan, W. Shao, Q. Lu, K. Zhang, Y. Cheng, D. Li, Y. Qiao, and P. Luo. Towards
world simulator: Crafting physical commonsense-based benchmark for video generation, 2024.
[58] E. Molad, E. Horwitz, D. Valevski, A. R. Acha, Y. Matias, Y. Pritch, Y. Leviathan, and Y. Hoshen.
Dreamix: Video diffusion models are general video editors. arXiv preprint arXiv:2302.01329,
2023.
[59] S. Motamed, L. Culp, K. Swersky, P. Jaini, and R. Geirhos. Do generative video models understand
physical principles? arXiv preprint arXiv:2501.09038, 2025.
[60] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal,
K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback.
Advances in neural information processing systems, 35:27730–27744, 2022.
[61] B. Peng, C. Li, P. He, M. Galley, and J. Gao. Instruction tuning with GPT-4. arXiv preprint
arXiv:2304.03277, 2023.
[62] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson,
et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
13


# [PAGE 14]
Video models are zero-shot learners and reasoners
[63] R. Ravishankar, Z. Patel, J. Rajasegaran, and J. Malik. Scaling properties of diffusion models
for perceptual tasks. In Proceedings of the Computer Vision and Pattern Recognition Conference,
pages 12945–12954, 2025.
[64] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object
detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages
779–788, 2016.
[65] R. Riochet, M. Y. Castro, M. Bernard, A. Lerer, R. Fergus, V. Izard, and E. Dupoux.
Int-
Phys: A framework and benchmark for visual intuitive physics reasoning.
arXiv preprint
arXiv:1803.07616, 2018.
[66] N. Rotstein, G. Yona, D. Silver, R. Velich, D. Bensaid, and R. Kimmel. Pathways on the image
manifold: Image editing via video generation. In Proceedings of the Computer Vision and Pattern
Recognition Conference, pages 7857–7866, 2025.
[67] S. Schmidgall, Y. Su, Z. Wang, X. Sun, J. Wu, X. Yu, J. Liu, M. Moor, Z. Liu, and E. Barsoum.
Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227,
2025.
[68] L. M. Schulze Buschoff, E. Akata, M. Bethge, and E. Schulz. Visual cognition in multimodal
large language models. Nature Machine Intelligence, pages 1–11, 2025.
[69] S. Sheynin, A. Polyak, U. Singer, Y. Kirstain, A. Zohar, O. Ashual, D. Parikh, and Y. Taigman.
Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.
[70] C. Snell, J. Lee, K. Xu, and A. Kumar. Scaling LLM test-time compute optimally can be more
effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
[71] X. Soria, E. Riba, and A. Sappa. Dense extreme inception network: Towards a robust CNN
model for edge detection. In The IEEE Winter Conference on Applications of Computer Vision
(WACV ’20), 2020.
[72] X. Soria, A. Sappa, P. Humanante, and A. Akbarinia. Dense extreme inception network for edge
detection. Pattern Recognition, 139:109461, 2023. ISSN 0031-3203. doi: https://doi.org/10.1
016/j.patcog.2023.109461. URL https://www.sciencedirect.com/science/article/
pii/S0031320323001619.
[73] H.-Y. Tung, M. Ding, Z. Chen, D. Bear, C. Gan, J. Tenenbaum, D. Yamins, J. Fan, and K. Smith.
Physion++: Evaluating physical scene understanding that requires online inference of different
physical properties. Advances in Neural Information Processing Systems, 36, 2024.
[74] T. Wang, J. Chen, Q. Jia, S. Wang, R. Fang, H. Wang, Z. Gao, C. Xie, C. Xu, J. Dai, et al. Weaver:
Foundation models for creative writing. arXiv preprint arXiv:2401.17268, 2024.
[75] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou.
Self-consistency improves chain of thought reasoning in language models.
arXiv preprint
arXiv:2203.11171, 2022.
[76] X. Wang, W. Wang, Y. Cao, C. Shen, and T. Huang. Images speak in images: A generalist painter
for in-context visual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition, pages 6830–6839, 2023.
[77] J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou,
D. Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682,
2022.
14


# [PAGE 15]
Video models are zero-shot learners and reasoners
[78] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-
thought prompting elicits reasoning in large language models. Advances in neural information
processing systems, 35:24824–24837, 2022.
[79] L. Weihs, A. Yuile, R. Baillargeon, C. Fisher, G. Marcus, R. Mottaghi, and A. Kembhavi. Bench-
marking progress to infant-level physical reasoning in ai. Transactions on Machine Learning
Research, 2022.
[80] S. Xiao, Y. Wang, J. Zhou, H. Yuan, X. Xing, R. Yan, C. Li, S. Wang, T. Huang, and Z. Liu.
Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition
Conference, pages 13294–13304, 2025.
[81] J. Xie, A. Tonioni, N. Rauschmayr, F. Tombari, and B. Schiele. Test-time visual in-context tuning.
In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19996–20005,
2025.
[82] S. Xie and Z. Tu. Holistically-nested edge detection. In Proceedings of the IEEE international
conference on computer vision, pages 1395–1403, 2015.
[83] S. Yang, J. Walker, J. Parker-Holder, Y. Du, J. Bruce, A. Barreto, P. Abbeel, and D. Schuurmans.
Video as the new language for real-world decision making. arXiv preprint arXiv:2402.17139,
2024.
[84] W. Yang, W. Wang, H. Huang, S. Wang, and J. Liu. Sparse gradient regularized deep retinex
network for robust low-light image enhancement. IEEE Transactions on Image Processing, 30:
2072–2086, 2021.
[85] E. Yiu, M. Qraitem, A. N. Majhi, C. Wong, Y. Bai, S. Ginosar, A. Gopnik, and K. Saenko. Kiva: Kid-
inspired visual analogies for testing large multimodal models. arXiv preprint arXiv:2407.17773,
2024.
[86] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese. Taskonomy: Disentangling
task transfer learning. In Proceedings of the IEEE conference on computer vision and pattern
recognition, pages 3712–3722, 2018.
[87] C. Zhang, D. Cherniavskii, A. Zadaianchuk, A. Tragoudaras, A. Vozikis, T. Nijdam, D. W.
Prinzhorn, M. Bodracska, N. Sebe, and E. Gavves. Morpheus: Benchmarking physical reasoning
of video generative models with real physical experiments. arXiv preprint arXiv:2504.02918,
2025.
[88] Y. Zhou, A. I. Muresanu, Z. Han, K. Paster, S. Pitis, H. Chan, and J. Ba.
Large language
models are human-level prompt engineers. In The eleventh international conference on learning
representations, 2022.
[89] W. Zhu, H. Liu, Q. Dong, J. Xu, S. Huang, L. Kong, J. Chen, and L. Li. Multilingual ma-
chine translation with large language models: Empirical results and analysis. arXiv preprint
arXiv:2304.04675, 2023.
[90] C. L. Zitnick and P. Dollár. Edge boxes: Locating object proposals from edges. In ECCV, 2014.
15


# [PAGE 16]
Video models are zero-shot learners and reasoners
Appendix
A. Qualitative results:
Perception, Modeling, Manipulation, Reasoning
A.1. Perception
Figure 10 | Edge detection. Prompt: “All edges in this image become more salient by transforming into
black outlines. Then, all objects fade away, with just the edges remaining on a white background. Static
camera perspective, no zoom or pan.” Success rate: 0.92.
Figure 11 | Segmentation. Prompt: “Create an animation of instance segmentation being performed on
this photograph: each distinct entity is overlaid in a different flat color [...]” (full prompt: Sec. B.2).
Success rate: 0.33.
Figure 12 | Keypoint localization. Prompt: “Add a bright blue dot at the tip of the branch on which
the macaw is sitting. The macaw’s eye turns bright red. Everything else turns pitch black. Static camera
perspective, no zoom or pan.” Success rate: 0.58.
Figure 13 | Super-resolution. Prompt: “Perform superresolution on this image. Static camera perspective,
no zoom or pan.” Success rate: 0.75.
16
[FIGURE img_p15_055]
[FIGURE img_p15_056]
[FIGURE img_p15_057]
[FIGURE img_p15_058]
[FIGURE img_p15_059]
[FIGURE img_p15_060]
[FIGURE img_p15_061]
[FIGURE img_p15_062]
[FIGURE img_p15_063]
[FIGURE img_p15_064]
[FIGURE img_p15_065]
[FIGURE img_p15_066]


# [PAGE 17]
Video models are zero-shot learners and reasoners
Figure 14 | Blind deblurring. Prompt: “Unblur image including background. Static camera perspective,
no zoom or pan.” Success rate: 1.0.
Figure 15 | Blind denoising. Each quadrant was corrupted with a different type of noise. Clockwise
from top left: Gaussian noise, salt-and-pepper noise, speckle noise, shot noise. Prompt: “Remove the
noise from this image. Static camera perspective, no zoom or pan.” Success rate: 1.0.
Original low-light image
Veo 3-generated lit image
Ground-truth lit image
Figure 16 | Low-light enhancing. Prompt: “Fully restore the light in this image. Static camera
perspective, no zoom or pan.” Success rate: 0.92. Image and ground-truth source: LOLv2 dataset [84].
Figure 17 | Conjunctive search / binding problem. Prompt: “The blue ball instantly begins to glow.
Static camera perspective, no zoom no pan no movement no dolly no rotation.” Success rate: 0.75.
Inspiration: [8].
17
[FIGURE img_p16_067]
[FIGURE img_p16_068]
[FIGURE img_p16_069]
[FIGURE img_p16_070]
[FIGURE img_p16_071]
[FIGURE img_p16_072]
[FIGURE img_p16_073]
[FIGURE img_p16_074]
[FIGURE img_p16_075]
[FIGURE img_p16_076]
[FIGURE img_p16_077]
[FIGURE img_p16_078]


# [PAGE 18]
Video models are zero-shot learners and reasoners
Figure 18 | Dalmatian illusion understanding. Prompt: “Static camera perspective.” Success rate: 1.0.
Image credit: [30, 35].
Figure 19 | Shape (cue-conflict) understanding. Prompt: “Transform the animal in this image into a
sketch of the animal surrounded by its family.” Success rate: 1.0. Image credit: [20].
Figure 20 | Rorschach blot interpretation. Prompt: “The patterns transform into objects.” Success
rate: undefined (1.0 for prompt following). Image credit: H. Rorschach, public domain via wikipedia.
A.2. Modeling
Figure 21 | Material properties. Prompt: “The bunsen burner at the bottom turns on. Sped up time
lapse. Static camera, no pan, no zoom, no dolly.” Success rate: 0.25.
18
[FIGURE img_p17_079]
[FIGURE img_p17_080]
[FIGURE img_p17_081]
[FIGURE img_p17_082]
[FIGURE img_p17_083]
[FIGURE img_p17_084]
[FIGURE img_p17_085]
[FIGURE img_p17_086]
[FIGURE img_p17_087]
[FIGURE img_p17_088]
[FIGURE img_p17_089]
[FIGURE img_p17_090]


# [PAGE 19]
Video models are zero-shot learners and reasoners
Figure 22 | Physics body transform. Rigid body (top). Prompt: “A person picks up the vase and
puts it back on the table in a sideways orientation. Static camera, no pan, no zoom, no dolly.” Success
rate: 1.0. Soft body (bottom). Prompt: “A person drapes a thin silk scarf over the vase. Static camera,
no pan, no zoom, no dolly.” Success rate: 0.67.
Figure 23 | Gravity and air resistance. On earth (top). Prompt: “The objects fall due to gravity. Static
camera, no pan, no zoom, no dolly.” Success rate: 0.5. On the moon (bottom). Prompt: “The objects
fall down on the moon due to gravity. Static camera, no pan, no zoom, no dolly.” Success rate: 0.5.
19
[FIGURE img_p18_091]
[FIGURE img_p18_092]
[FIGURE img_p18_093]
[FIGURE img_p18_094]
[FIGURE img_p18_095]
[FIGURE img_p18_096]
[FIGURE img_p18_097]
[FIGURE img_p18_098]
[FIGURE img_p18_099]
[FIGURE img_p18_100]
[FIGURE img_p18_101]
[FIGURE img_p18_102]


# [PAGE 20]
Video models are zero-shot learners and reasoners
Figure 24 | Buoyancy. Prompt: “The hand lets go of the object. Static camera, no pan, no zoom, no
dolly.” Success rate (bottle cap): 0.58; success rate (rock): 0.83.
Figure 25 | Visual Jenga, inspired by [5]. Prompt: “A hand quickly removes each of the items in this
image, one at a time.” Success rate, based on removal of at least three objects: 0.5.
Figure 26 | Object packing. Prompt: “A person puts all the objects that can fit in the backpack inside of
it. Static camera, no pan, no zoom, no dolly.” Success rate: 0.75.
20
[FIGURE img_p19_103]
[FIGURE img_p19_104]
[FIGURE img_p19_105]
[FIGURE img_p19_106]
[FIGURE img_p19_107]
[FIGURE img_p19_108]
[FIGURE img_p19_109]
[FIGURE img_p19_110]
[FIGURE img_p19_111]
[FIGURE img_p19_112]
[FIGURE img_p19_113]
[FIGURE img_p19_114]
[FIGURE img_p19_115]
[FIGURE img_p19_116]


# [PAGE 21]
Video models are zero-shot learners and reasoners
Figure 27 | Material optics. Glass (top). Prompt: “A giant glass sphere rolls through the room. Static
camera, no pan, no zoom, no dolly.” Note that the image through the glass sphere is inverted. Success
rate: 0.92. Mirror (bottom). Prompt: “A giant mirror-polish metal sphere rolls through the room. Static
camera, no pan, no zoom, no dolly.” Note that the image reflected off the sphere is not inverted).
Success rate: 1.0.
Figure 28 | Color mixing. Additive (lights, top). Prompt: “The spotlight on the left changes color to
green, and the spotlight on the right changes color to blue.” Success rate: 0.92. Subtractive (paints,
bottom). Prompt: “A paintbrush mixes these colors together thoroughly until they blend completely.
Static camera, no pan, no zoom.” Success rate: 0.75.
Figure 29 | Categorizing objects. Prompt: “A person puts all the kids toys in the bucket. Static camera,
no pan, no zoom, no dolly.” Success rate: 0.33.
21
[FIGURE img_p20_117]
[FIGURE img_p20_118]
[FIGURE img_p20_119]
[FIGURE img_p20_120]
[FIGURE img_p20_121]
[FIGURE img_p20_122]
[FIGURE img_p20_123]
[FIGURE img_p20_124]
[FIGURE img_p20_125]
[FIGURE img_p20_126]
[FIGURE img_p20_127]
[FIGURE img_p20_128]
[FIGURE img_p20_129]
[FIGURE img_p20_130]
[FIGURE img_p20_131]


# [PAGE 22]
Video models are zero-shot learners and reasoners
Figure 30 | Character recognition, generation, and parsing, inspired by the Omniglot dataset [45].
Recognition (top). Prompt: “The background of the grid cell with the same symbol as the one indicated
on the right turns red. All other grid cells remain unchanged. After that, a spinning color wheel appears
in the top right corner.” (Note: Veo 3 has a prior to keep things moving, which is detrimental for tasks
where the solution is obtained in an early frame. We observe that a ‘motion outlet’, such as a color
wheel, can indicate task completion and ‘freeze’ the solution.) Success rate: 0.33. Generation of
variations (middle). Prompt: “The page is filled line-by-line with hand-written practice variations of
the symbol.” Success rate: 0.25. Parsing into parts (bottom). Color and numbers in final frame are
added post-hoc to show stroke order. Prompt: “Stroke-by-stroke, a replica of the symbol is drawn on
the right.” Success rate: 0.5.
Figure 31 | Memory of world states. Prompt: “The camera zooms in to give a close up of the person
looking out the window, then zooms back out to return to the original view.” Success rate: 1.0.
A.3. Manipulation
Figure 32 | Background removal. Prompt: “The background changes to white. Static camera perspective,
no zoom or pan.” Success rate: 0.83.
22
[FIGURE img_p21_132]
[FIGURE img_p21_133]
[FIGURE img_p21_134]
[FIGURE img_p21_135]
[FIGURE img_p21_136]
[FIGURE img_p21_137]
[FIGURE img_p21_138]
[FIGURE img_p21_139]
[FIGURE img_p21_140]
[FIGURE img_p21_141]
[FIGURE img_p21_142]
[FIGURE img_p21_143]
[FIGURE img_p21_144]
[FIGURE img_p21_145]
[FIGURE img_p21_146]


# [PAGE 23]
Video models are zero-shot learners and reasoners
Figure 33 | Style transfer. Prompt: “The scene transforms into the style of a Hundertwasser painting,
without changing perspective or orientation; the macaw does not move. Static camera perspective, no
zoom or pan.” Success rate: 0.75.
Figure 34 | Colorization. Prompt: “Perform colorization on this image. Static camera perspective, no
zoom or pan.” Success rate: 0.08.
Figure 35 | Inpainting. Prompt: “The white triangles become smaller and smaller, then disappear
altogether. Static camera perspective, no zoom or pan.” Success rate: 1.0.
Figure 36 | Outpainting. Prompt: “Rapidly zoom out of this static image, revealing what’s around it.
The camera just zooms back, while the scene itself and everything in it does not move or change at all, it’s
a static image.” Success rate: 1.0.
23
[FIGURE img_p22_147]
[FIGURE img_p22_148]
[FIGURE img_p22_149]
[FIGURE img_p22_150]
[FIGURE img_p22_151]
[FIGURE img_p22_152]
[FIGURE img_p22_153]
[FIGURE img_p22_154]
[FIGURE img_p22_155]
[FIGURE img_p22_156]
[FIGURE img_p22_157]
[FIGURE img_p22_158]


# [PAGE 24]
Video models are zero-shot learners and reasoners
Figure 37 | Text manipulation. Prompt: “Animation of the text rapidly changing so that it is made out
of different types of candy (top left text) and pretzel sticks (bottom right text). Static camera perspective,
no zoom or pan.” Success rate: 0.33.
Figure 38 | Image editing with doodles. Prompt: “Changes happen instantly.” Success rate: 1.0.
Figure 39 | Scene composition. Prompt: “A smooth animation blends the zebra naturally into the scene,
removing the background of the zebra image, so that the angle, lighting, and shading look realistic. The
final scene perfectly incorporates the zebra into the scene.” Success rate: 0.75.
Figure 40 | Single-image novel view synthesis. Prompt: “Create a smooth, realistic animation where
the camera seems to rotate around the object showing the object from all the sides. Do not change
anything else. No zoom. No pan.” Success rate: 0.92. Image source: [14].
24
[FIGURE img_p23_159]
[FIGURE img_p23_160]
[FIGURE img_p23_161]
[FIGURE img_p23_162]
[FIGURE img_p23_163]
[FIGURE img_p23_164]
[FIGURE img_p23_165]
[FIGURE img_p23_166]
[FIGURE img_p23_167]
[FIGURE img_p23_168]
[FIGURE img_p23_169]
[FIGURE img_p23_170]


# [PAGE 25]
Video models are zero-shot learners and reasoners
Figure 41 | 3D-aware reposing. Prompt: “The knight turns to face to the right and drops on one knee,
lifting the shield above his head to protect himself and resting the hilt of his weapon on the ground.”
Success rate: 0.83.
Figure 42 | Transfiguration. Prompt: “A magical spell smoothly transforms the structure of the teacup
into a mouse.” Success rate: 0.17.
Figure 43 | Professional headshot generation. Prompt: “Turn this selfie into a professional headshot
for LinkedIn.” Success rate: 0.42. Image credit: photo by George Pisarevsky on Unsplash.
25
[FIGURE img_p24_171]
[FIGURE img_p24_172]
[FIGURE img_p24_173]
[FIGURE img_p24_174]
[FIGURE img_p24_175]
[FIGURE img_p24_176]
[FIGURE img_p24_177]
[FIGURE img_p24_178]
[FIGURE img_p24_179]


# [PAGE 26]
Video models are zero-shot learners and reasoners
Figure 44 | Dexterous manipulation. Jar opening (top). Prompt: “Use common sense and have the
two robot hands attached to robot arms open the jar, like how a human would.” Success rate: 1.0.
Throwing and catching (middle). Prompt: “Use common sense and have the two robot hands attached
to robot arms throw the ball in the air, the ball goes up off the screen, hands move to positions to catch
the ball, and catch the falling ball, like how a human would.” Success rate: 1.0. Rotating Baoding
balls (bottom). Prompt: “A human hand holds two metal Baoding balls. The fingers, including the
thumb, index, and middle finger, skillfully manipulate the balls, causing them to rotate smoothly like two
planets orbiting around each other and continuously in the palm, one ball circling the other in a fluid
motion.” Success rate: 0.08.
Figure 45 | Affordance recognition. Prompt: “The robot hands mounted on robot arms pick up the
hammer, naturally like how a human would.” Success rate: 0.5.
Figure 46 | Drawing. Prompt: “A person draws a square. Static camera, no pan, no zoom, no dolly.”
Success rate: 0.33.
26
[FIGURE img_p25_180]
[FIGURE img_p25_181]
[FIGURE img_p25_182]
[FIGURE img_p25_183]
[FIGURE img_p25_184]
[FIGURE img_p25_185]
[FIGURE img_p25_186]
[FIGURE img_p25_187]
[FIGURE img_p25_188]
[FIGURE img_p25_189]
[FIGURE img_p25_190]
[FIGURE img_p25_191]
[FIGURE img_p25_192]
[FIGURE img_p25_193]
[FIGURE img_p25_194]
[FIGURE img_p25_195]
[FIGURE img_p25_196]


# [PAGE 27]
Video models are zero-shot learners and reasoners
Figure 47 | Visual instruction generation. Prompt: “A montage clearly showing each step to roll a
burrito.” Success rate: 0.25. Inspiration: [83] and Reddit.
A.4. Reasoning
Figure 48 | Graph traversal. Prompt: “Starting from the blue well, an unlimited supply of blue water
moves through the connected channel system without spilling into the black area.” Success rate: 0.08.
Figure 49 | Tree BFS. Prompt: “From the blue water basin, an unlimited supply of water flows at
constant speed into the cave system until all caves are filled. Static camera perspective, no zoom or pan.”
Success rate: 0.17.
Figure 50 | Sequence completion inspired by Raven’s progressive matrices. Each of the four pairs
shows input (left) and generated output (right). Prompt: “Draw the figure that completes the pattern
in the rightmost box. The images in the boxes are static. Do not modify the existing images, only draw in
the empty box. Static camera, no zoom, no pan, no dolly.” Success rate: 0.33 for dots, 1.0 for arrows,
0.75 for shrinking circles, 0.83 for growing squares.
27
[FIGURE img_p26_197]
[FIGURE img_p26_198]
[FIGURE img_p26_199]
[FIGURE img_p26_200]
[FIGURE img_p26_201]
[FIGURE img_p26_202]
[FIGURE img_p26_203]
[FIGURE img_p26_204]
[FIGURE img_p26_205]
[FIGURE img_p26_206]
[FIGURE img_p26_207]
[FIGURE img_p26_208]
[FIGURE img_p26_209]
[FIGURE img_p26_210]
[FIGURE img_p26_211]
[FIGURE img_p26_212]
[FIGURE img_p26_213]


# [PAGE 28]
Video models are zero-shot learners and reasoners
Figure 51 | Connecting colors. Prompt: “Draw three curves, one connecting each pair of circles of the
same color.” Success rate: 0.25.
Figure 52 | Shape fitting. Prompt: “The scene shows three colored pieces, and a wooden panel with
three holes. Each colored piece fits into one and only one hole. A hand grabs each colored piece and puts
it into an empty hole that has the exact same shape - if it doesn’t fit, the hand tries another hole. All the
objects must be placed in their respective holes.” Success rate: 0.25.
Figure 53 | Sorting numbers. Prompt: “The video starts with some numbered bubbles. The bubbles pop
and disappear one at a time, in numeric order, starting from the one with the smallest number.” Success
rate: 0.08.
Figure 54 | Tool use. Prompt: “A person retrieves the walnut from the aquarium.” Success rate: 0.92
(retrieval via tool) and 0.08 (retrieval via tool without intersecting the glass).
Figure 55 | Simple Sudoku completion. Prompt: “Create a static, smooth, animation that solves the
given 4x4 sudoku. Enter the missing numbers one by one. Do not change anything else in the picture.
Only fill the numbers in the empty cells so the sudoku is solved properly. A cursor moves and fills the
correct number in the empty boxes.” Success rate: 0.67.
28
[FIGURE img_p27_214]
[FIGURE img_p27_215]
[FIGURE img_p27_216]
[FIGURE img_p27_217]
[FIGURE img_p27_218]
[FIGURE img_p27_219]
[FIGURE img_p27_220]
[FIGURE img_p27_221]
[FIGURE img_p27_222]
[FIGURE img_p27_223]
[FIGURE img_p27_224]
[FIGURE img_p27_225]
[FIGURE img_p27_226]
[FIGURE img_p27_227]
[FIGURE img_p27_228]
[FIGURE img_p27_229]
[FIGURE img_p27_230]
[FIGURE img_p27_231]
[FIGURE img_p27_232]
[FIGURE img_p27_233]
[FIGURE img_p27_234]


# [PAGE 29]
Video models are zero-shot learners and reasoners
Figure 56 | Water puzzle solving. Prompt: “The tap is turned on and water starts flowing rapidly
filling the containers. Create a smooth, static animation showing the containers getting filled with water
in the correct order.” (note: not all containers can be filled since some pipes are closed off—Veo fills
the correct containers, in the right order.) Success rate: 0.5.
Figure 57 | Maze solving. Prompt: “Without crossing any black boundary, the grey mouse from the
corner skillfully navigates the maze by walking around until it finds the yellow cheese.” Success rate: 0.17.
Figure 58 | Robot navigation. Prompt: “The robot drives to the blue area. Static camera perspective, no
movement no zoom no scan no pan.” Success rate: 0.58. Image credit: Micromelon Robotics website
with permission from Tim Hadwen.
Figure 59 | Rule extrapolation inspired by ARC-AGI [10]. Prompt: “Modify the lower-right grid to
adhere to the rule established by the other grids. You can fill cells, clear cells, or change a cell’s color. Only
modify the lower-right grid, don’t modify any of the other grids. Static scene, no zoom, no pan, no dolly.”
Success rate: 0.08. While Veo 3 doesn’t follow the prompt perfectly, the output grid (bottom right) is
completed correctly.
29
[FIGURE img_p28_235]
[FIGURE img_p28_236]
[FIGURE img_p28_237]
[FIGURE img_p28_238]
[FIGURE img_p28_239]
[FIGURE img_p28_240]
[FIGURE img_p28_241]
[FIGURE img_p28_242]
[FIGURE img_p28_243]
[FIGURE img_p28_244]
[FIGURE img_p28_245]
[FIGURE img_p28_246]
[FIGURE img_p28_247]
[FIGURE img_p28_248]


# [PAGE 30]
Video models are zero-shot learners and reasoners
True Positive
False Positive
False Negative
True Negative
Figure 60 | Graded Veo 3 edge map. While false negatives reflect genuine oversights of Veo 3 (e.g.,
cracks in the road, lettering on the car), many false positives correspond to actual image details that
seem to be erroneously excluded from the ground truth (e.g., the outline of the trees, the reflection in
the car window, and the tire profiles).
B. Quantitative results: experimental details
Table 1 | Video count breakdown for quantitative tasks. For segmentation, 2 × 1 splits indicate one
test set with two different background color prompts (white/green). For the prompt sensitivity study
on the symmetry task (Sec. C), 2 × 10 splits indicate 2 splits (random/shape) across 10 tested prompt
variations. For the qualitative tasks, we additionally generated 744 videos (62 tasks × 12 samples).
Task
Splits
Imgs/split
Pass@
Video models
Total videos
Edge
1
50
10
2
1000
Segmentation
2 × 1
50
10
2
2000
Object extraction
1
54
10
2
1080
Editing
1
30
1
2
60
Maze
4
50
10
2
8000
Symmetry
2
25
10
2
1000
Symmetry prompt analysis
2 × 10
25
1
1
500
Analogy
4
50
10
2
4000
Total
17640
B.1. Perception: Edge detection
We provide details for the image editing task in Sec. 4.1.
Evaluation
As is standard in the literature, we refine and binarize predicted edges and allow for
small local shifts compared to the ground truth [15, 16, 48, 90]. Concretely, we use non-maximum
suppression, then binarize with one of 16 evenly-spaced thresholds, then thin the binary edge map.
At each threshold, we find the optimal mapping between predicted and ground-truth edge pixels
within a radius of 0.75 % of the image diagonal (around 11 pixels). Fig. 60 shows an example rating
of a Veo 3-generated edge map. We report the best OIS over 𝑘attempts (optimal image scale; the
maximum 𝐹1-score over all thresholds) for the best/last frame.
Dataset
We used all 50 test images from BIPEDv2 [71, 72].
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API. We also tested Nano Banana gemini-2.5-flash-image-
preview through Google AI Studio.
30
[FIGURE img_p29_249]
[FIGURE img_p29_250]

[EQ eq_p29_000] -> see eq_p29_000.tex

[EQ eq_p29_001] -> see eq_p29_001.tex


# [PAGE 31]
Video models are zero-shot learners and reasoners
Veo
All edges in this image become more salient by transforming into black outlines. Then, all objects
fade away, with just the edges remaining on a white background. Static camera perspective, no
zoom or pan.
Nano Banana
Outline all edges in the image in black, make everything else white.
Sampling
We generated 10 videos per sample with a fixed prompt.
B.2. Perception: Segmentation
We provide details for the image editing task in Sec. 4.2.
Evaluation
Since the model is free to choose any colors for segmentation masks, we first determine
the number and hue of each mask by considering the hue-difference histogram between the original im-
age and the extracted frame. We smooth the histogram with scipy.ndimage.gaussian_filter1d
with a standard deviation of 2. Peaks with a minimum height of 10% of the maximum and at least 10
hue steps apart are considered to correspond to predicted segmentation masks. We then map each
pixel to the mask with the closest hue.
Contrary to classic instance segmentation [31] or promptable segmentation [42, 62], our prompts
do not specify a class or list of possible classes, a location prior (e.g., point or bounding box), or the
number of instances in the image. This also means that mapping between predictions and annotated
instances is established. Instead, we pair each ground-truth mask (including the background) with
the predicted mask with the highest IoU (intersection over union), if any. We report mIoU as the
average IoU over all pairs (excluding the background).
Dataset
We evaluated on 50 randomly chosen test images from LVIS [31] that contain one to three
objects, each with at least 5000 pixels.
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API. We also tested Nano Banana gemini-2.5-flash-image-
preview through Google AI Studio.
31

[EQ eq_p30_000] -> see eq_p30_000.tex


# [PAGE 32]
Video models are zero-shot learners and reasoners
Veo
Create an animation of instance segmentation being performed on this photograph: each distinct
entity is overlaid in a different flat color.
Scene:
• The animation starts from the provided, unaltered photograph.
• The scene in the photograph is static and doesn’t move.
• First, the background fades to {white, green}.
• Then, the first entity is covered by a flat color, perfectly preserving its silhouette.
• Then the second entity, too, is covered by a different flat color, perfectly preserving its
silhouette.
• One by one, each entity is covered by a different flat color.
• Finally, all entities are covered with different colors.
Camera:
• Static shot without camera movement.
• No pan.
• No rotation.
• No zoom.
• No glitches or artifacts.
Nano Banana
Perform instance segmentation on this image: Mask each distinct entity in a different opaque
flat color that only preserves the silhouette and turn the background green.
Sampling
We generated 10 videos per sample and prompt.
B.3. Manipulation: Object extraction
We provide details for the image editing task in Sec. 4.3.
Evaluation
We extract the last frame from each generated video; the resulting image is converted to
greyscale, a binary mask with threshold 200 is applied, and the number of connected components is
extracted using scipy.ndimage.label, resulting in the count estimate. We also report the chance
baseline which can be calculated as: random −chance = 1 −(1 −𝑝)𝑘where 𝑝is the probability to get
the count correct via guessing (here: 𝑝= 1
9) and 𝑘∈[1, 10].
Dataset
We generated an animal counting dataset using Nano Banana. Starting from a white 16:9
image, we used the following prompt, where number is in [1, 9] and animal is in [‘dog’, ‘elephant’,
‘cat’, ‘brown bear’, ‘horse’, ‘rabbit’, ‘raccoon’]. We manually evaluated the generated dataset for
correctness; the resulting dataset has 54 images (exactly 6 per count).
Nano Banana
Exchange the white space with a realistic photograph of: exactly {number} {animal}, outside, not
overlapping, in a natural landscape.
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API.
32

[EQ eq_p31_000] -> see eq_p31_000.tex

[EQ eq_p31_001] -> see eq_p31_001.tex

[EQ eq_p31_002] -> see eq_p31_002.tex


# [PAGE 33]
Video models are zero-shot learners and reasoners
Veo
The background changes to white. Then:
• If there is just a single animal: the animal sits in the middle of the image, looking straight at
the camera.
• If there are multiple animals: all animals line up in a row, with ample white space between
them.
Sampling
We generated 10 videos per sample with a fixed prompt.
B.4. Manipulation: Image editing
We provide details for the image editing task in Sec. 4.4.
Evaluation
We perform a human study with three human raters to evaluate fidelity (correct edit)
and precision (correct edit with no unintended changes like zooming).
Dataset
We used a random sample of 30 images from the test set of the Emu-edit dataset [69].
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API.
Veo
Create a smooth, static animation that slowly {image specific edit direction}. Do not change
anything else. No zoom, no pan, no dolly.
Sampling
For each image, we generated two samples and use the first sample for human rating.
B.5. Reasoning: Maze solving
We provide details for the image editing task in Sec. 4.5.
Evaluation
Our evaluation process is tailored to the model type. For Veo, we analyze the generated
video frame-by-frame, extracting the path taken by the agent (red circle). We check for any invalid
moves, such as jumping over walls, clipping through boundaries, or any alteration of the goal’s position.
We report the success rate as the fraction of 𝑘attempts where the agent successfully reaches the goal
(green circle) without any illegal moves.
For Nano Banana, which generates the full path in one edit, we assess whether the drawn path
connects the start and end points (allowing for minor discontinuities) and crucially, whether it
intersects with any maze walls or goes off the valid path.
For Gemini 2.5 Pro with a maze input as an image (I2T) or as ASCII (T2T), we check whether the
series of grid positions represents an uninterrupted path from the start position to the goal.
Dataset
For rectangular mazes, we generated 50 random mazes per size using maze-dataset [32],
but replacing the square start and end with circles and swapping their colors. We also drew 10
irregular mazes by hand and flipped/rotated them to obtain 40 unique samples.
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API. We also tested Nano Banana gemini-2.5-flash-image-
preview and Gemini 2.5 Pro gemini-2.5-pro through Google AI Studio.
33

[EQ eq_p32_000] -> see eq_p32_000.tex

[EQ eq_p32_001] -> see eq_p32_001.tex


# [PAGE 34]
Video models are zero-shot learners and reasoners
Veo
Create a 2D animation based on the provided image of a maze. The red square slides smoothly
along the white path, stopping perfectly on the green square. The red square never slides or crosses
into the black areas of the maze. The camera is a static, top-down view showing the entire maze.
Maze:
• The maze paths are white, the walls are black.
• The red square moves to the goal position, represented by a green square.
• The red square slides smoothly along the white path.
• The red square never slides or crosses into the black areas of the maze.
• The red square stops perfectly on the green square.
Scene:
• No change in scene composition.
• No change in the layout of the maze.
• The red square travels along the white path without speeding up or slowing down.
Camera:
• Static camera.
• No zoom.
• No pan.
• No glitches, noise, or artifacts.
34

[EQ eq_p33_000] -> see eq_p33_000.tex

[EQ eq_p33_001] -> see eq_p33_001.tex


# [PAGE 35]
Video models are zero-shot learners and reasoners
Gemini 2.5 Pro I2T
system
Think step by step as needed and output in xml format:
<think>thinking process</think>
<final_answer>final answer</final_answer>
user
The following image shows a maze, represented by colored squares:
• Black squares represent walls and cannot be passed through.
• White squares are empty and can be passed through.
• The red square is the starting point.
• The green square is the end point.
Please solve the maze by providing a path from the starting point to the end point. The path should
be provided as a list of coordinates of each step, where each coordinate is a (row, col) tuple, and
row, col are 0-based indices. Consider the origin (0, 0) to be the top-left corner. Overall, the path
should be provided in the format of [(row1, col1), (row2, col2), ...].
A valid path must:
• Start at the starting point (the red square).
• End at the end point (the green square).
• Avoid the walls (the black squares).
• Pass only through empty space (the white squares).
• Move one square at a time.
• Only move up, down, left, and right, not diagonally.
Correct your answer if you spot any errors.
Here is the maze image: {image}
Nano Banana
Mark the correct path from the red to the green circle through the maze in blue.
35

[EQ eq_p34_000] -> see eq_p34_000.tex

[EQ eq_p34_001] -> see eq_p34_001.tex


# [PAGE 36]
Video models are zero-shot learners and reasoners
Gemini 2.5 Pro T2T
system
Think step by step as needed and output in xml format:
<think>thinking process</think>
<final_answer>final answer</final_answer>
user
The following is an ASCII-representation of a maze:
• ‘#’ represents walls which cannot be passed through.
• ‘ ’ represents empty spaces that can be passed through.
• ‘S’ is the starting point.
• ‘E’ is the end point.
Please solve the maze by providing a path from the starting point to the end point. The path should
be provided as a list of coordinates of each step, where each coordinate is a (row, col) tuple, and
row, col are 0-based indices. Consider the origin (0, 0) to be the top-left corner. Overall, the path
should be provided in the format of [(row1, col1), (row2, col2), ...].
A valid path must:
• Start at the starting point ‘S’.
• End at the end point ‘E’.
• Avoid the walls ‘#’.
• Pass only through empty space ‘ ’.
• Move one square at a time.
• Only move up, down, left, and right, not diagonally.
Correct your answer if you spot any errors.
Here is the maze in ASCII format: {maze}
Sampling
We generated 10 videos per sample with a fixed prompt.
B.6. Reasoning: Visual symmetry solving
We provide details for the visual symmetry task in Sec. 4.6.
Evaluation
We prompt Veo with input images containing a 10×16 grid where a pattern is drawn on
the left half. The goal is to complete the pattern on the empty right half so that the final pattern is
symmetrical along the central vertical axis.
We compare Veo’s best-frame and last-frame solutions with the ground-truth symmetrical grid
and compute the number of incorrectly-colored cells. A cell is determined as incorrectly-colored if
the average color across pixels in the cell is perceptually distinct from the ground-truth average color
in the matching cell. We compute perceptual color differences of the average cell color in the CIELAB
color space, with a difference threshold of 15.0. In Fig. 8, we report the percentage of attempts in
which the best or last frame solution has zero incorrect cells for 𝑘= 1.
Dataset
We created a synthetic grid coloring image dataset to evaluate visual symmetry. We
generated 25 samples using common symmetrical symbols, objects and shapes such as english letters
(e.g., A, H, M, X), geometric shapes (e.g., square, triangle), symmetrical objects (e.g., wineglass,
balloon; together, the shape condition). We also generated 25 samples consisting of randomly-colored
cells (the random condition).
36

[EQ eq_p35_000] -> see eq_p35_000.tex

[EQ eq_p35_001] -> see eq_p35_001.tex

[EQ eq_p35_002] -> see eq_p35_002.tex

[EQ eq_p35_003] -> see eq_p35_003.tex


# [PAGE 37]
Video models are zero-shot learners and reasoners
Majority vote pass@k %
Colour
Resize
Reflect
Rotate
1
2
3
4
5
6
7
8
9 10
k
40
60
80
100
98
85
1
2
3
4
5
6
7
8
9 10
k
40
50
60
70
68
50
1
2
3
4
5
6
7
8
9 10
k
20
25
30
22
17
1
2
3
4
5
6
7
8
9 10
k
10
15
20
25
30
14
7
Model
Veo 3
Veo 2
Chance
Figure 61 | Visual analogy performance over 10 attempts. In contrast to other plots in this paper,
we here report not the best performance over 𝑘attempts, but instead the performance when choosing
the majority vote from 𝑘attempts. As a result, performance is not necessarily monotonic in 𝑘. In
fact, for reflect and rotate, performance decreases with 𝑘, indicating that both models have systematic,
erroneous biases. In the case of Veo 3, the model tends to perform reflections and rotations, but not
along the same axis as shown in the image. Veo 2 simply tends to copy the object without applying
any transformation.
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API. We also tested Nano Banana gemini-2.5-flash-image-
preview through Google AI Studio.
Veo
Instantly reflect this pattern along the central, vertical axis while keeping the existing colored
pattern without modification. Static camera perspective, no zoom or pan.
Sampling
We generated 10 videos per sample with a fixed prompt.
B.7. Reasoning: Visual analogy completion
We provide details for the visual analogy task in Sec. 4.7.
Evaluation
We prompt Veo to solve visual analogies with an input image showing a reference
object pair and a test object. The object images are sourced from the Kid-inspired Visual Analogies
benchmark [KiVA, 85]. Consistent with the multi-choice format in the KiVA benchmark, we evaluated
Veo’s generation by cropping out the generated target object in the lower-right region of the last
frame and compare Veo’s generated object with three candidate object choices using an autorater
(see details below).
In Fig. 9, we report the pass@1 accuracy across different conditions for both Veo 2 and Veo 3 for
𝑘= 1. Fig. 61 shows performance for 𝑘= 10.
Dataset
We used the test trials and choice images from the KiVA benchmark [85].
Models & prompts
We tested Veo 3 veo-3.0-generate-preview and Veo 2 veo-2.0-generate-
preview-001 through the Vertex AI API.
37
[FIGURE img_p36_251]
[FIGURE img_p36_252]
[FIGURE img_p36_253]
[FIGURE img_p36_254]

[EQ eq_p36_000] -> see eq_p36_000.tex

[EQ eq_p36_001] -> see eq_p36_001.tex


# [PAGE 38]
Video models are zero-shot learners and reasoners
We used Gemini 2.5 Pro gemini-2.5-pro through Google AI Studio to identify which image
choice Veo’s generation is most similar with. To enhance the autorater’s image comparison accuracy
for this task, Gemini is prompted with privileged information about the values in the dataset conditions
(see below for the full autorater prompt). If no object is visible in the lower-right region of Veo’s
generated last frame or if the generated object is of a different object type, we randomly sampled one
of three choices as Veo’s choice. In pilot experiments, we found that the Gemini-assisted autorater’s
ratings achieve above 88% agreement with expert human ratings by the authors on 25 samples within
each conditions.
Note that in the prompt, words in { } are updated based on the test condition of the current
generation (one of color, resize, reflect, and rotate) to provide more information of the feature name
and values to direct the image comparison. Image choice orders are shuffled for each prompt.
Veo
Create a smooth animation to generate the missing object in the lower right region and solve the
visual analogy. The original three objects must remain still. Static shot, no zoom no pan no dolly.
Gemini 2.5 Pro autorater
system
You are an expert visual judge. You will be presented with a "target image" and three "choice
images" labeled A, B, and C. Your goal is to identify the choice image that is most visually similar
to the target image.
Follow these steps:
1. Analyze each provided image and describe the objects shown. Focus on the object {color}.
That is, if the objects appear {green}, {blue}, or {red}.
2. Determine if the primary object in the target image is of the same general category or type
as the objects in the choice images. For example, if the target image shows a dog, and the
choices show a cat, the object types are considered different. If no object is visible in the target
image, the object type is considered to be mismatched.
3. If the object type matches between the target image and the choice images, identify the choice
that is most visually similar to the target image in terms of the object {color}.
Provide a brief justification for your choice, explaining why it is the best match and why the others
are less suitable. Conclude your response with the final answer on a new line in the format:
“Final Answer: [answer]”
where “answer” is one of (“A”, “B”, “C”, or “different object type”). Do not use markdown format
for the final answer line.
user
Please evaluate the following images.
— TARGET IMAGE —
{target object image}
— CHOICE IMAGES —
CHOICE A: {image choice} CHOICE B: {image choice} CHOICE C: {image choice}
Which choice image is most similar to the target image?
Sampling
We generated 10 videos per sample with a fixed prompt.
38

[EQ eq_p37_000] -> see eq_p37_000.tex

[EQ eq_p37_001] -> see eq_p37_001.tex


# [PAGE 39]
Video models are zero-shot learners and reasoners
C. Prompting best practices
Table 2 | Prompt sensitivity study on the visual symmetry task. We report best frame pass@1 %
and the average number of incorrectly-colored cells across 25 samples on each split (shape/random).
Pass@1
Avg incorrect cells
No.
Prompt
Shape
Random
Shape
Random
1
Instantly reflect this pattern along the central, vertical
axis while keeping the existing colored pattern without
modification.
48
68
4.16
7.00
2
Instantly reflect this pattern along the central, vertical
axis while keeping the existing colored pattern without
modification. Static camera perspective, no zoom or pan.
42
65
5.00
3.52
3
Instantly reflect this pattern along the central, vertical
axis while keeping the existing colored pattern without
modification. The result needs to be mirror-symmetrical
along the vertical axis. Static camera perspective, no
zoom or pan.
36
52
6.28
9.04
4
One by one, cells in the right half of the grid are filled in
to complete the pattern. The pattern is
mirror-symmetrical along the central vertical line. Static
shot; no zoom, no pan, no dolly.
32
12
10.76
14.08
5
Reflect this pattern along the central, vertical axis.
28
40
9.76
4.52
6
An animation showing the left half of the grid being
mirrored onto the right half to create a symmetrical
pattern. Static shot; no zoom, no pan, no dolly.
24
12
10.96
16.32
7
You’re a master symmetry solver. Your task is to fill the
cells on the right side of the grid to mirror the pattern on
the left, such that it’s symmetrical along the vertical axis.
24
8
9.20
17.72
8
Fill color in the appropriate cells on the right side of the
grid to complete the pattern. The final image should be
symmetrical along the central vertical line. Static shot, no
zoom no pan no dolly.
13
9
10.30
14.74
9
Create a static, smooth, realistic animation completing
the pattern in the image by filling the grid on the right
hand side. Do not change anything else. No zoom, no
pan.
12
4
14.88
21.00
10
A timelapse of a professional pixel artist drawing a
symmetrical pattern onto a white canvas. Static shot; no
zoom, no pan, no dolly.
8
20
14.20
12.64
The results in Secs. 3 and 4 are best-effort estimates of Veo’s performance using carefully chosen
prompts. Generally, performance varies greatly with the exact task description provided in the prompt,
as illustrated by a prompt sensitivity study on the visual symmetry task in Table 2. Here are best
practices from this sensitivity analysis and our other experiments:
• Remove ambiguity. Tasks can be solved in a variety of ways, and natural language descriptions
tend to leave a lot of room for interpretation. The goal should be formulated clearly, e.g., saying
“symmetrical along the central, vertical axis”, rather than just “symmetrical”.
• Specify what shouldn’t change. Veo has a tendency to change any part of the input to create
interesting, dynamic scenes. Including not only a positive task description, but also specifying
what not to change can help mitigate this, e.g., “keep the existing colored pattern without
modification”.
• Providing an outlet. As mentioned above, Veo has a strong prior to keep things moving.
Providing a “motion outlet” in the form of, e.g., a spinning ball can help keep the rest of the
scene static.
• Let the model decide when its done. The motion prior also means that Veo often keeps
39


# [PAGE 40]
Video models are zero-shot learners and reasoners
modifying the scene, even after solving the task. Providing a visual indicator, e.g., “add a
glowing red dot once the goal is reached” allows for easy extraction of the solution from the
generated video.
• Scene and camera controls. Phrases like “static camera, no zoom, no pan, no dolly” can help
keeping the scene static, e.g., for image-to-image tasks.
• Speed control. Some tasks like maze solving benefit from being solved step-by-step. For other
tasks, especially image-to-image tasks, specifying instant changes can help avoid artifacts.
• Realism. Veo was trained to generate plausible, realistic-looking videos. Translating an abstract
task into a realistic setting (including, but not limited to editing the original image to depict
realistic, 3D scenes rather than abstract shapes) can improve generation results. A similar effect
was observed in [37], and we expect visual prompt engineering to emerge as a powerful tool for
video models.
40


# [PAGE 41]
Video models are zero-shot learners and reasoners
D. Failure cases
Figure 62 | Monocular depth estimation. Prompt: “The image transitions to a depth-map of the
scene: Darker colors represent pixels further from the camera, lighter colors represent pixels closer to
the camera. The exact color map to use is provided on the right side of the image. Static scene, no pan,
no zoom, no dolly.” Failure: Veo 3 seems generally unable to color pixels by depth beyond a binary
foreground/background mapping and specifically struggles with using a provided color map.
Figure 63 | Monocular surface normal estimation. Prompt: “The image transitions to a surface-
normal map of the scene: the red/green/blue color channel specify the direction of the surface-normal at
each point, as illustrated on the right side of the image on a sphere. Static scene, no pan, no zoom, no
dolly.” Failure: While Veo 3 shows some promise in coloring surfaces according to their orientation
(e.g., the cube in the front), coloration is inconsistent (compare the two cubes) and doesn’t correctly
interpolate colors (e.g., for the slope on the triangle).
Figure 64 | Force & motion prompting, inspired by [22, 23]. Force prompting (top). Prompt: “The
balls move in the direction indicated by the arrows. Balls without an arrow don’t move. Static scene, no
pan, no zoom, no dolly.” Motion trajectory prompting (bottom). Prompt: “Each car drives out of
the frame following the indicated trajectory. Static camera, no zoom, no pan, no dolly.” Failure: Veo 3
seems unable to follow force/motion annotations with any consistency. Providing annotations for the
first frame and letting the model remove them before generating the scene in motion does not work,
either.
41
[FIGURE img_p40_255]
[FIGURE img_p40_256]
[FIGURE img_p40_257]
[FIGURE img_p40_258]
[FIGURE img_p40_259]
[FIGURE img_p40_260]
[FIGURE img_p40_261]
[FIGURE img_p40_262]
[FIGURE img_p40_263]
[FIGURE img_p40_264]
[FIGURE img_p40_265]
[FIGURE img_p40_266]


# [PAGE 42]
Video models are zero-shot learners and reasoners
Figure 65 | Tying the knot. Prompt: “A knot is tied connecting these two rope ends.” Failure: physics
violation, impossible rope movement.
Figure 66 | Connect the path puzzle.. Prompt: “The path connecting the boy to the object starts
glowing slowly. Nothing else changes. No zoom, no pan, no dolly." Failure: hallucinations, lighting up of
all paths.
Figure 67 | Only five letter word search.. Prompt: “Generate a static video animation using the
provided letter grid. The task is to highlight the only 5-letter English word CHEAT, which may be
oriented in any direction (horizontally, vertically, or diagonally). The animation should consist of a
semi-transparent red rectangle with rounded corners smoothly fading into view, perfectly encapsulating
the five letters of the word. The rectangle should have a subtle, soft glow. Do not change anything else in
the image. The camera must remain locked in place with no movement. No zoom, no pan, no dolly."
Failure: does not recognize words; highlights individual letters randomly.
Figure 68 | Eulerian path.. Prompt: “Create a smooth animation where a red pen traces all existing
edges in a continuous path without lifting the pen. All edges need to be traced. Do not visit any edge twice
and do not lift the pen. No zoom, no pan." Failure: does not trace the edges exactly, traces non-existent
edges.
42
[FIGURE img_p41_267]
[FIGURE img_p41_268]
[FIGURE img_p41_269]
[FIGURE img_p41_270]
[FIGURE img_p41_271]
[FIGURE img_p41_272]
[FIGURE img_p41_273]
[FIGURE img_p41_274]
[FIGURE img_p41_275]
[FIGURE img_p41_276]
[FIGURE img_p41_277]
[FIGURE img_p41_278]


# [PAGE 43]
Video models are zero-shot learners and reasoners
Figure 69 | Solving system of linear equation.. Prompt: “A hand appears and solves the set of linear
equations. It replaces the x, y, z matrix with their correct values that solves the equation. Do not change
anything else." Failure: hallucinations with text on the blackboard.
Figure 70 | Spot the difference.. Prompt: “There are two images. The left image is different from the
right image in 5 spots. Create a static, realistic, smooth animation where a cursor appears and points at
each place where the left image is different from the right image. The cursor points one by one and only
on the left image. Do not change anything in the right image. No pan. No zoom. No movement. Keep the
image static." Failure: does not identify all the differences. Hallucinates differences.
Figure 71 | Perception IQ test.. Prompt: “Create a static, smooth, animation that solves the puzzle
in the given image. The correct pattern should appear at the bottom right to solve the puzzle. Do not
change anything else in the picture. No zoom, no pan, no dolly" Failure: incorrect figure pattern.
Figure 72 | Glass falling. Prompt: “The object falls. Static camera, no pan, no zoom, no dolly.” Failure:
physics violation, glass does not break, and orients itself to be vertical after landing on the floor.
Figure 73 | Collisions. Prompt: “The two objects collide in slow motion. Static camera, no pan, no zoom,
no dolly.” Failure: not physically plausible, the objects pause at the moment of impact and then are
pushed together by an invisible force.
43
[FIGURE img_p42_279]
[FIGURE img_p42_280]
[FIGURE img_p42_281]
[FIGURE img_p42_282]
[FIGURE img_p42_283]
[FIGURE img_p42_284]
[FIGURE img_p42_285]
[FIGURE img_p42_286]
[FIGURE img_p42_287]
[FIGURE img_p42_288]
[FIGURE img_p42_289]
[FIGURE img_p42_290]
[FIGURE img_p42_291]
[FIGURE img_p42_292]
[FIGURE img_p42_293]
[FIGURE img_p42_294]
[FIGURE img_p42_295]
[FIGURE img_p42_296]


# [PAGE 44]
Video models are zero-shot learners and reasoners
Figure 74 | Tiling puzzles. Jigsaw puzzle (top). Prompt: “A hand takes the fitting puzzle piece from
the right, rotates it to be in the correct orientation, then puts it into the hole, completing the puzzle.
Static scene, no pan, no zoom, no dolly.” Failure: wrong piece orientation. Sliding puzzle (middle).
Prompt: “Slide the pieces of this sliding puzzle around one-at-a-time until all edges align.” Failure:
doesn’t maintain piece integrity while sliding, hallucinates new pieces. Scrambled puzzle (bottom).
Prompt: “Unscramble this image.” Failure: image details are inconsistent with original pieces.
Figure 75 | Bottleneck. Prompt: “A person tries to put the golf ball in the vase. Static camera, no pan, no
zoom, no dolly..” Failure: not physically plausible, golf ball is too large to pass through the bottleneck
of the vase.
Figure 76 | Laundry folding. Prompt: “Generate a video of two metal robotic arms properly folding the
t-shirt on the table. Failure: physics violation, implausible folding movements.
44
[FIGURE img_p43_297]
[FIGURE img_p43_298]
[FIGURE img_p43_299]
[FIGURE img_p43_300]
[FIGURE img_p43_301]
[FIGURE img_p43_302]
[FIGURE img_p43_303]
[FIGURE img_p43_304]
[FIGURE img_p43_305]
[FIGURE img_p43_306]
[FIGURE img_p43_307]
[FIGURE img_p43_308]
[FIGURE img_p43_309]
[FIGURE img_p43_310]
[FIGURE img_p43_311]
[FIGURE img_p43_312]
[FIGURE img_p43_313]


# [PAGE 45]
Video models are zero-shot learners and reasoners
Figure 77 | Motion planning; an instance of the piano mover’s problem. Prompt: “The red couch
slides from the left room over into the right room, skillfully maneuvering to fit through the doorways
without bumping into the walls. The walls are fixed: they don’t shift or disappear, and no new walls
are introduced. Static camera, no pan, no zoom, no dolly.” Failure: violating rigid-body integrity, not
keeping to permissible transformations (rotation, translation).
E. LLM use.
Gemini 2.5 Flash and Gemini 2.5 Pro [12] were used for brainstorming task ideas, suggesting related
work that we might have otherwise missed, coding support, and to polish human writing.
F. Image sources.
Where not stated in the figure caption, images were obtained as follows.
• Figs. 10 to 15, 32 to 38 and 74: The original macaw image was generated with Gemini and,
depending on the figure, subsequently modified by the authors (e.g., conversion to grayscale,
adding noise, adding the monkey with Nano Banana).
• Fig. 16: The input image was obtained from here (Apache 2.0 license) based on the LOLv2
dataset [84] and randomly selected. The image was slightly cropped to fit a 16:9 aspect ratio.
• Figs. 17, 21 to 24, 26 to 29, 31, 39, 41, 42, 46, 47, 52, 54, 65, 69, 72, 73 and 75 to 77:
generated with Gemini.
• Fig. 25: The input image was obtained from here (CC0 license).
• Fig. 30: hand drawn by us, inspired by Fig. 1 of the Omniglot paper [45].
• Fig. 40: sample from Objaverse [14]
• Figs. 48 to 51, 53, 55 and 57: created by us.
• Figs. 56, 66, 67 and 70: original image from Reddit.
• Fig. 59: hand drawn by us, inspired by ARC-AGI [10].
• Fig. 60: sample from BIPEDv2 [71, 72].
• Figs. 62 to 64: generated with Gemini, then annotated by us.
• Figs. 68 and 71: hand drawn by us. Inspired by original images from Reddit.
• Figs. 44 and 45: The robot hands are extracted from a frame in this video and were subsequently
adapted with Nano Banana. The hands holding Baoding balls were obtained from here.
45
[FIGURE img_p44_314]
[FIGURE img_p44_315]
[FIGURE img_p44_316]
