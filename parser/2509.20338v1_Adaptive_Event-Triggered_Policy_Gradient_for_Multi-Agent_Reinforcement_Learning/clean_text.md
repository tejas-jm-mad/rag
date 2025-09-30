

# [PAGE 1]
Adaptive Event-Triggered Policy Gradient for
Multi-Agent Reinforcement Learning
Umer Siddique, Abhinav Sinha, Senior Member, IEEE, and Yongcan Cao, Senior Member, IEEE
Abstract—Conventional multi-agent reinforcement learning
(MARL) methods rely on time-triggered execution, where agents
sample and communicate actions at fixed intervals. This ap-
proach is often computationally expensive and communication-
intensive. To address this limitation, we propose ET-MAPG
(Event-Triggered Multi-Agent Policy Gradient reinforcement
learning), a framework that jointly learns an agent’s control
policy and its event-triggering policy. Unlike prior work that
decouples these mechanisms, ET-MAPG integrates them into
a unified learning process, enabling agents to learn not only
what action to take but also when to execute it. For scenarios
with inter-agent communication, we introduce AET-MAPG, an
attention-based variant that leverages a self-attention mecha-
nism to learn selective communication patterns. AET-MAPG em-
powers agents to determine not only when to trigger an action
but also with whom to communicate and what information to ex-
change, thereby optimizing coordination. Both methods can be
integrated with any policy gradient MARL algorithm. Extensive
experiments across diverse MARL benchmarks demonstrate
that our approaches achieve performance comparable to state-
of-the-art, time-triggered baselines while significantly reducing
both computational load and communication overhead.
Index Terms—Multi-agent reinforcement learning, event-
triggered learning and control, self-attention, data-driven con-
trol.
I. INTRODUCTION
Event-triggered control (ETC) [1] is an approach in which
signals are updated or exchanged only when a certain state
or output condition is met, rather than at fixed, periodic
intervals. The main goal of ETC is to reduce communication
and computation while maintaining closed-loop performance,
in contrast to time-triggered control (TTC). ETC has been
widely studied [2]–[5], where most of these methods assume
access to an accurate system dynamics or model. Although
this may be possible in small-scale simulated environments,
it’s often unrealistic or nearly impossible in complex real-
world applications.
To mitigate this model dependence, data-driven ETC meth-
ods have been recently proposed. These methods include
discrete-time formulations [3], [5]–[8] but continuous-time
approaches are scarce, e.g., [4]. For linear time-invariant
systems, several works simplify learning by ignoring dis-
turbances during offline data collection [3], [5], [6]. In
U. Siddique and Y. Cao are with the Unmanned Systems Lab, Depart-
ment of Electrical and Computer Engineering, The University of Texas
at San Antonio, San Antonio, TX 78249, USA. (e-mails: muhammad-
umer.siddique@my.utsa.edu, yongcan.cao@utsa.edu). A. Sinha is with the
GALACxIS Lab, Department of Aerospace Engineering and Engineer-
ing Mechanics, University of Cincinnati, OH 45221, USA. (email: abhi-
nav.sinha@uc.edu).
contrast, the work in [8] proposes a more realistic method that
includes designing the controller and trigger from a single
batch of noisy data, thereby accounting for disturbances and
measurement errors. Moreover, the work in [4] incorporates
disturbances both during learning and in closed-loop oper-
ation via a dynamic triggering strategy that guarantees L2
stability.
Reinforcement learning (RL) has achieved strong empirical
results in sequential decision-making and control, including
robotics [9]–[12]. Yet, most of the RL works focus on
designing a time-triggered control policy, often overlooking
communication cost. While a few model-free RL-based ETC
methods exist [7], [13]–[16], they are developed for a single
agent. In practice, many systems are inherently multi-agent
systems with tight bandwidth constraints. In MARL, multiple
agents are interacting, learning, and coordinating with each
other to solve a shared task, making event-triggered learning
even more important, where multiple agents should act and
communicate only when necessary.
Communication in MARL is essential for coordination
and efficient problem-solving, especially under partial ob-
servability. Early work in MARL introduced deep distributed
recurrent Q-networks (DDRQN), e.g., [17], which demon-
strate that agents can learn communication protocols for
coordination. Building on this, the work in [18] proposed
Reinforced Inter-Agent Learning (RIAL) and Differentiable
Inter-Agent Learning (DIAL) to learn communication end
to end, and the authors in [19] investigated communication
scheduling with relational inductive biases. However, these
approaches often rely on specialized communication net-
works or extensive parameter sharing, which can be costly
at scale. Inspired by the success of self-attention [20], [21],
we instead learn compact, attention-based messages where
agents, during the learning phase, compute attention scores
over shared representations and exchange only the relevant
information.
ETC has also been explored in MARL to some extent.
ETCNet [22] reduces bandwidth by sending messages only
when necessary, but all agents still interact with the envi-
ronment at every time step, and triggering applies only to
inter-agent communication. ETMAPPO [23] integrates ETC
with multi-agent proximal policy optimization algorithms
via a Beta strategy to compress transmitted information
and accelerate convergence in specific UAV environments.
Although their model-free multi-agent PPO method performs
well in the anti-UAV jamming scenario, their approach also
arXiv:2509.20338v1  [eess.SY]  24 Sep 2025


# [PAGE 2]
applies ETC only to communication among agents.
To address these limitations, we propose Event-Triggered
Multi-Agent Policy Gradient reinforcement learning (ET-
MAPG), which jointly learns both the control action head
and an event-trigger head for each agent and decides when
to update the action and/or communicate. Unlike approaches
that learn triggering conditions and control actions with
separate policies [14], [24], [25], ET-MAPG employs a single
shared network with two heads. Due to this, ET-MAPG
reduces the policy network parameters, latency, and improves
efficiency, which is crucial when scaling many agents in
MARL. When inter-agent communication is allowed, we
further introduce AET-MAPG, an attention-based variant of
ET-MAPG that leverages self-attention to facilitate selective,
learned message passing during training. As a consequence
of the proposed approach, the communication graph in AET-
MAPG is inherently sparse since an agent resamples an action
or transmits messages only when its triggering condition
is satisfied. Otherwise, it reuses its previous action and
suppresses messaging. This design reduces communication
and computation while supporting stable and efficient deploy-
ment. Our main contributions are summarized as follows:
• We propose ET-MAPG, a method that jointly learns a
control action head and an event-trigger head for each
agent. The trigger head determines when a new action
should be sampled, thereby providing an improvement
over approaches that learn triggering and control with
separate policies.
• We further propose AET-MAPG, an attention-based vari-
ant of ET-MAPG that uses self-attention as a commu-
nication mechanism during training, which can improve
the coordination efficiency.
• We demonstrate the generality of ET-MAPG and AET-
MAPG by integrating them with three state-of-the-
art multi-agent policy gradient algorithms, including
IPPO [26], MAPPO [27], and IA2C [28].
• Through our extensive experiments, we show that ET-
MAPG and AET-MAPG match the performance of stan-
dard MARL algorithms while reducing computation cost
by up to 50%.
II. BACKGROUND AND PRELIMINARIES
We consider a multi-agent system of 𝑁agents, indexed by
𝑖∈I = {1, . . . , 𝑁}, communicating over a fully connected
undirected graph G = (I, E). The dynamics of each agent
are governed by a discrete-time nonlinear equation given by
x𝑖,𝑘+1 = f𝑖(x𝑖,𝑘, u𝑖,𝑘, {x𝑗,𝑘} 𝑗∈N𝑖),
(1)
where x𝑖,𝑘is the state of agent 𝑖, u𝑖,𝑘is its control input,
and N𝑖is the set of its neighbors. We model the problem
of multi-agent event-triggered learning as a decentralized
partially observable Markov decision process (Dec-POMDP)
which is defined by a tuple ⟨I, X, {U𝑖}, P, 𝑟, {O𝑖}, Z, 𝛾⟩,
where I = {1, . . . , 𝑁} is the set of 𝑁agents. The true state
of the environment is x ∈X. At each timestep 𝑘, every
agent 𝑖∈I selects an action u𝑖,𝑘∈U𝑖from its individual
action set. This forms a joint action u𝑘= (u1,𝑘, . . . , u𝑁,𝑘) ∈
U, where U = ×𝑖∈IU𝑖is the joint action space. The
joint action governs the state transition according to the
probability function P(x𝑘+1 | x𝑘, u𝑘). In this cooperative
setting, the team of agents receives a single shared reward,
r𝑘= r(x𝑘, u𝑘). The agents, however, do not observe the
true state x𝑘. Instead, after the transition to state x𝑘+1, each
agent 𝑖receives a private observation o𝑖,𝑘+1 ∈O𝑖. The joint
observation o𝑘+1 = (o1,𝑘+1, . . . , o𝑁,𝑘+1) is determined by the
observation function Z(o𝑘+1 | x𝑘+1, u𝑘). Finally, 𝛾∈[0, 1)
is the discount factor.
Each agent maintains a local action-observation history,
𝜏𝑖,𝑘= (o𝑖,0, u𝑖,0, . . . , u𝑖,𝑘−1, o𝑖,𝑘). Actions are chosen accord-
ing to a local, stochastic policy, u𝑖,𝑘∼𝜋𝑖(· | 𝜏𝑖,𝑘). The team
of agents aims to learn a joint policy 𝝅that factorizes into
the local policies
𝝅(u𝑘| 𝝉𝑘) =
𝑁
Ö
𝑖=1
𝜋𝑖(u𝑖,𝑘| 𝜏𝑖,𝑘),
(2)
where 𝝉𝑘= (𝜏1,𝑘, . . . , 𝜏𝑁,𝑘) is the joint history. The objec-
tive is to find a joint policy that maximizes the expected
discounted return
𝑱(𝝅) = E𝝅,P,Z
" ∞
∑︁
𝑘=0
𝛾𝑘r(x𝑘, u𝑘)
#
.
(3)
Remark 1. In this model, we adopt the centralized training
with decentralized execution (CTDE) paradigm (see [17],
[18], [26]–[28]), which combines the benefits of having
global information during training and decentralized scalabil-
ity at execution. In CTDE, agents are assumed to have access
to the full state of the system during training, which helps
in mitigating non-stationarity in dynamic environments (a
common challenge that arises when multiple agents interact
with a shared environment, causing the dynamics to shift
from the perspective of each agent). However, during execu-
tion, agents’ actions only depend on their local observations,
which is crucial for real-world deployment where global state
information is usually unavailable or may not be possible to
get due to bandwidth, latency, or privacy constraints.
To conserve computational and communication resources,
we depart from the standard time-triggered paradigm. In-
stead, we employ an event-triggered scheme where each
agent decides independently when to compute and broadcast
a new action. Each agent 𝑖maintains its own sequence of
event times {𝑡𝑖
𝑗} 𝑗∈N. At an event time 𝑡𝑖
𝑗, it updates its history
𝜏𝑖,𝑡𝑖
𝑗and computes a new action by sampling from its policy,
u𝑖,𝑡𝑖
𝑗∼𝜋𝑖(· | 𝜏𝑖,𝑡𝑖
𝑗). This action is then held constant until the
next event
u𝑖,𝑘= u𝑖,𝑡𝑖
𝑗,
∀𝑘∈[𝑡𝑖
𝑗, 𝑡𝑖
𝑗+1).
(4)
The next event time, 𝑡𝑖
𝑗+1, is determined by a local triggering
rule based on an error signal, e𝑖,𝑘= x𝑖,𝑘−x𝑖,𝑡𝑖
𝑗, such that
𝑡𝑖
𝑗+1 = inf{𝑘> 𝑡𝑖
𝑗| T𝑖(x𝑖,𝑘, e𝑖,𝑘) ≥0}.
(5)

[EQ eq_p1_000] -> see eq_p1_000.tex

[EQ eq_p1_001] -> see eq_p1_001.tex

[EQ eq_p1_002] -> see eq_p1_002.tex

[EQ eq_p1_003] -> see eq_p1_003.tex


# [PAGE 3]
The goal is to co-design the policies {𝜋𝑖} and triggering
functions {T𝑖} to maximize the expected return 𝐽(𝝅) while
significantly reducing the frequency of policy evaluations and
network communication.
Previously, the approaches often relied on parameter shar-
ing (e.g., latent information) or specialized architectures for
message exchange, both of which typically assume access to
full state information or sufficient bandwidth (see e.g., [17],
[18] and references therein). To address these limitations, we
leverage attention mechanisms as a communication tool. Self-
attention computes queries Q, keys K, and values V, and
evaluates attention
𝒜(Q, K, V) = softmax
 QK⊤
√𝑑𝑘

V,
(6)
where 𝑑𝑘is the dimensionality of K. Multi-head self-
attention extends this by projecting Q, K, V into multiple
subspaces, applying attention in parallel, and concatenating
the outputs. This allows agents to selectively focus on rel-
evant information and capture diverse interaction patterns.
By integrating this multi-head attention as a communication
mechanism with event-triggered learning, we let MARL
agents decide what, how, and when to communicate. Note
that we assume that the agents communicate over a complete
undirected graph at the triggering instants, which indicates
a bidirectional messaging sharing at the triggering instants
only.
III. PROPOSED APPROACH
We now present the proposed event-triggered framework
in multi-agent settings. Since our objective is to address
large-scale multi-agent problems with high-dimensional (or
possibly even continuous state-action spaces in general), we
focus on multi-agent deep policy gradient methods such as
IPPO [26], MAPPO [27], and IA2C [28]. Among these
methods, IPPO and IA2C learn independent actors and critics
for each agent, whereas MAPPO employs a centralized critic
with decentralized actors. Since all these methods share the
same actor-critic architecture, the proposed framework can
be easily extended to any of them.
In a cooperative setting, the goal of agents is to learn a joint
policy 𝝅𝜽= (𝜋1,𝜃, . . . , 𝜋𝑛,𝜃) that maximizes the discounted
sum of rewards given by
max
𝝅𝜽𝑱(𝝅𝜽) = max
𝝅𝜽
 
E𝝅,P,Z
" ∞
∑︁
𝑘=0
𝛾𝑘r(x𝑘, u𝑘)
#!
,
(7)
where P denotes the environment transition dynamics.
Remark 2. In independent learning settings, where agents are
learning independently by observing their local observations
and performing actions individually, this decomposes into
maximizing each agent’s expected return in the form of
𝐽𝑖(𝜋𝑖,𝜃) = E𝜋𝑖,𝜃
h ∞
∑︁
𝑘=0
𝛾𝑘
𝑖𝑟𝑖,𝑘
i
,
∀𝑖= 1, . . . , 𝑛.
(8)
To maximize (7), standard MARL algorithms update poli-
cies at every time step, which is equivalent to TTC. Tradi-
tional ETC designs have attempted to subdue the limitations
of time-triggered execution, although they design triggering
conditions manually (e.g., based on state deviations) and treat
them separately from the control policy [7], [14], [24]. To
address this issue, we propose ET-MAPG, an independent
learning MARL algorithm in which each agent 𝑖jointly learns
both its control action u𝑖,𝑘and its triggering condition T𝑖such
that
 T𝑖, u𝑖,𝑘
 = 𝜋𝑖,𝜃
 z𝑖,𝑘, 𝜏𝑖,𝑘
 , ∀z𝑖,𝑘∈Z.
(9)
Proposition 1. Let each agent 𝑖in a multi-agent system, at
a given timestep 𝑘, maintain a local observation z𝑖,𝑘and a
state-action history 𝜏𝑖,𝑘. Consider a parametric policy 𝜋𝑖,𝜃
that jointly outputs the control action u𝑖,𝑘and the triggering
decision T𝑖as given in (9). If agent 𝑖maximizes the expected
cumulative reward with a triggering regularization
𝐽𝑖(𝜋𝑖,𝜃) = E𝜋𝑖,𝜃
" ∞
∑︁
𝑘=0
𝛾𝑘
𝑖𝑟𝑖(z𝑖,𝑘, u𝑖,𝑘)
#
−Ψ · I(T𝑖= 1), (10)
where Ψ penalizes frequent triggering, and I is the indicator
function, then the optimal policy 𝜋∗
𝑖,𝜃= arg max𝜋𝑖,𝜃𝐽𝑖(𝜋𝑖,𝜃)
simultaneously learns what action to take and when to
update.
Remark 3. This unified approach reduces the model com-
plexity by avoiding the need for hand-crafted triggers and
improves sample efficiency by allowing agents to dynami-
cally decide both what action to take and when to update.
In fact, if T𝑖= 1 at every timestep, then a higher penalty
discourages frequent policy updates and incentivizes efficient
communication and computation by trading off performance
with triggering frequency.
Following the policy gradient theorem [29], the gradient
for each agent 𝑖is computed as
∇𝑖,𝜃𝐽(𝜋𝑖,𝜃) =E𝜋𝑖,𝜃

A𝑖,𝑘(z𝑖,𝑘, u𝑖,𝑘)∇𝑖,𝜃log 𝜋𝑖,𝜃(u𝑖,𝑘| z𝑖,𝑘)

−Ψ · I(T𝑖= 1),
(11)
where A𝑖,𝑘is the vector advantage function operated compo-
nentwise at a given timestep. Positive components indicate
improvement for that particular agent, and negative compo-
nents indicate a worse policy than the baseline.
Remark 4. Our proposed method ET-MAPG is model-
agnostic and is sufficiently general to be extended to a large
class of MARL algorithms.
As an illustrative example, we first extend IPPO [26]
to incorporate the proposed framework by employing the
clipped surrogate PPO objective augmented with the trig-
gering penalty
E𝜋𝑖,𝜃
h
min

𝜌𝑖,𝜃AIPPO
𝑖,𝑘, ¯𝜌𝑖,𝜃AIPPO
𝑖,𝑘
 i
−Ψ · I(T𝑖= 1),
(12)

[EQ eq_p2_000] -> see eq_p2_000.tex

[EQ eq_p2_001] -> see eq_p2_001.tex

[EQ eq_p2_002] -> see eq_p2_002.tex

[EQ eq_p2_003] -> see eq_p2_003.tex

[EQ eq_p2_004] -> see eq_p2_004.tex

[EQ eq_p2_005] -> see eq_p2_005.tex


# [PAGE 4]
0.0
2.5
5.0
7.5
10.0
xi
IPPO
Et-Mapg
Aet-Mapg
0
2
4
6
8
10
Time (Seconds)
0
20
40
1
2 x2
i
(a) State and Lyapunov.
0.0
0.5
1.0
0
2
4
6
8
10
Time (Seconds)
0.0
0.5
1.0
T i
(b) Triggering instants and moving avg.
100
101
0
2
4
6
8
10
Time (Seconds)
101
log(ti
k+1 −ti
k)
(c) Inter-event time.
Fig. 1: Performance comparison between IPPO and our proposed methods in the perturbed multi-agent single integrators.
Subfigures (b) and (c) show ET-MAPG (top) and AET-MAPG (bottom) results, respectively.
where 𝜌𝑖,𝜃and ¯𝜌𝑖,𝜃denote PPO’s ratio and clipping terms
for agent 𝑖. The advantage AIPPO
𝑖,𝑘
is estimated using TD(𝜆)
given by
AIPPO
𝑖,𝑘
=
∑︁
𝑘
(𝛾𝑖𝜆)𝑘−1  𝑟𝑖(z𝑖,𝑘, u𝑖,𝑘)
+𝛾V𝑖,𝜃(z𝑖,𝑘+1) −V𝑖,𝜃(z𝑖,𝑘) ,
where V𝑖,𝜃is the vector value function for agent 𝑖. As
a consequence of this modular design, ET-MAPG can be
extended to other policy gradient MARL algorithms.
For instance, with MAPPO [27], the centralized critic
enables advantage estimation as
AMAPPO
𝑖,𝑘
=
∑︁
𝑘
(𝛾𝑖𝜆)𝑘−1  𝑟𝑖(z𝑖,𝑘, u𝑖,𝑘)
+𝛾V𝑖,𝜃(x𝑘+1) −V𝑖,𝜃(x𝑘) ,
where x𝑘denotes the global state. Similarly, in IA2C, the
advantage reduces to
AIA2C
𝑖,𝑘
= 𝑟𝑖(z𝑖,𝑘, u𝑖,𝑘−V𝑖,𝜃(z𝑖,𝑘).
Therefore, our approach provides a general event-triggered
extension to any policy gradient MARL algorithms.
While ET-MAPG efficiently learn both the control ac-
tion and policy and triggering condition, it assumes agents
act independently without explicit communication. However,
many cooperative tasks require coordination, where agents
can communicate to reach a consensus or mitigate non-
stationarity [30]. To this end, we propose AET-MAPG, a
variant of ET-MAPG which integrates event-triggered com-
munication with self-attention.
Proposition 2. Let ET-MAPG be an independent learning
framework in which each agent 𝑖jointly learns its control
action u𝑖,𝑘and the triggering decision T𝑖. AET-MAPG is a
variant of ET-MAPG that integrates event-triggered commu-
nication with self-attention, such that when T𝑖= 1, agent
𝑖broadcasts its learned message to all other agents over
G. Each agent aggregates received messages via multi-head
self-attention
𝑏𝑖=
∑︁
𝑗
𝛼𝑖𝑗𝑣𝑗;
𝛼𝑖𝑗= softmax
 [Q]𝑖[K𝑗]⊤
√𝑑𝑘

,
(13)
where 𝛼𝑖𝑗denotes the attention weight of agent 𝑖attending
to agent 𝑗. This mechanism allows agents to selectively
exchange information only when triggering conditions are
met, ensuring both coordination and efficiency.
The aggregated message 𝑏𝑖is then fused into the policy
network. With multiple attention heads (ℎ= 4 in our experi-
ments), agents capture diverse interaction patterns, improving
robustness in cooperative settings.
IV. EXPERIMENTAL RESULTS
To demonstrate the effectiveness of our proposed methods,
we evaluate them across diverse environments with different
levels of complexity and communication demands. Specifi-
cally, we consider (i) perturbed chain of single integrators, (ii)
the repeated penalty matrix game [31], and (iii) multi-agent
particle environments (MPE) [32], which are a collection of
2D simulated physics environments that require cooperation
or competition among agents. Via these experiments, we eval-
uate both control and coordination abilities of our methods.
For all experiments, we perform hyperparameter optimization
and report results for the best-performing configurations.
Furthermore, for the sake of reproducibility, we run each
experiment with five different seeds and report the mean
performance. Unless otherwise specified, we benchmark ET-
MAPG and AET-MAPG against IPPO [26] in the main
experiments and show the generality of our methods with
other policy gradient methods in ablation studies.
In the first case, the agents seek stabilization of the origin
from different initial conditions. At each time step 𝑘, agent
𝑖updates according to x𝑖,𝑘+1 = x𝑖,𝑘+ u𝑖,𝑘𝑇𝑠, where 𝑇𝑠is the
sampling time. The reward function for agent 𝑖is given by the
improvement in state convergence, which means encouraging
an agent to reduce the absolute value of the state (i.e., getting
closer to the equilibrium point) while penalizing large or

[EQ eq_p3_000] -> see eq_p3_000.tex

[EQ eq_p3_001] -> see eq_p3_001.tex

[EQ eq_p3_002] -> see eq_p3_002.tex


# [PAGE 5]
0.0
0.2
0.4
0.6
0.8
1.0
Number of steps
×105
−30
−20
−10
0
10
Rewards
IPPO
Et-Mapg
Aet-Mapg
(a) Rewards.
0.0
0.5
1.0
0
2
4
6
8
10
Time (Seconds)
0.0
0.5
1.0
T i
(b) Triggering instants and moving avg.
100
101
0
2
4
6
8
10
Time (Seconds)
100
101
log(ti
k+1 −ti
k)
(c) Inter event time.
Fig. 2: Performance comparison between IPPO and our proposed methods in the repeated matrix game. Subfigures (b) and
(c) show TIGER (top) and ATT-TIGER (bottom) results, respectively.
unnecessary control actions. In a cooperative setting, the goal
of all agents is to reach the equilibrium point.
The results in Fig. 1 show that both proposed methods
successfully stabilize the system while requiring substantially
fewer action updates. The initial state value, which is 10, is
driven to equilibrium in each case. However, ET-MAPG and
AET-MAPG achieve this while reducing triggering events by
at least 60%. Fig. 1a (top) illustrates the state trajectories
of the system, while the bottom plot shows the decay of
the standard quadratic Lyapunov function over time for a
representative agent 𝑖. In both cases, ET-MAPG and AET-
MAPG match IPPO performance . Fig. 1b demonstrates
the communication frequency with the policy, where the
triggering frequencies for ET-MAPG (top) and AET-MAPG
(bottom) are significantly lower than those of the IPPO
baseline, which triggers policy interaction at each time step.
Here, blue and red triggering instants and moving average
curves correspond to the two agents in the environment.
Finally, Fig. 1c shows the inter-event times across agents,
demonstrating that they remain strictly positive and adaptive,
avoiding Zeno behavior while ensuring stable convergence.
We now test the proposed algorithms on the repeated
penalty matrix game by Claus and Boutilier [31], which is a
two-agent cooperative setting defined by the payoff matrix

ℓ
0
10
0
2
0
10
0
ℓ

,
with ℓ≤0 (set to ℓ= −100 for more complexity). At each
timestep, agents observe their local state, perform their own
actions, and receive rewards. Specifically, agent 1 selects a
row as its action and agent 2 selects a column, producing a
joint payoff from the corresponding matrix entry (the entry
where their choices intersect). To earn the highest reward
of 10, the agents must select the correct combination of
actions simultaneously. Alternatively, certain actions are con-
sidered safe, which guarantee a smaller but reliable reward
of 2, regardless of what the other agent chooses. At the
same time, failure to cooperate may incur the penalty of
ℓ= −100. Each episode length is 25, where each position
in the matrix remains the same and agents only rely on a
constant observation, which makes the environment stateless
aside from episode progression. As the penalty is strongly
negative, even small deviations from cooperative behavior
lead to catastrophic consequences, and agents fall into the
local Nash equilibria (i.e., agents keep choosing the safe
actions that yield rewards equal to 2).
Fig. 2a demonstrates the rewards achieved by our proposed
methods compared to the IPPO, indicating that our methods
perform comparably to the IPPO. Although our methods
have slightly lower rewards, their triggering frequency is
significantly lower than the standard IPPO, which triggers at
every time step (see Fig. 2b). The inter-event time in Fig. 2c
confirms that our methods maintain strictly positive intervals,
adapting the triggering schedule dynamically and avoiding
Zeno behavior. These results show that both ET-MAPG and
AET-MAPG preserve performance while reducing communi-
cation overhead, even in sparse environments with high-risk
coordination.
Our third evaluation domain is the Multi-Agent Particle
Environments (MPE), a widely used suite of continuous
2D tasks where agents with simple dynamics must solve
cooperative or competitive problems under partial observabil-
ity. Since our focus is on cooperative MARL, we consider
two tasks: Simple Reference and Simple Spread. These tasks
require agents to balance control efficiency with inter-agent
communication and make them suitable for testing event-
triggered methods. In these environments, each agent ob-
serves its local position and velocity, relative positions of
landmarks and other agents, and optional communication
inputs. The action space of each agent includes no-action,
move-left, move-right, move-down, and move-up. In Simple
Reference, the environment consists of two agents and three
landmarks. Each landmark is a fixed circular location, and
each agent is assigned a private target landmark known only
to the other agent. In this environment, agents act both as
speakers and listeners, and their goal is to navigate to their
assigned targets. The reward function in this environment

[EQ eq_p4_000] -> see eq_p4_000.tex


# [PAGE 6]
0.0
0.2
0.4
0.6
0.8
1.0
Number of steps
×105
−2.4
−2.2
−2.0
−1.8
−1.6
−1.4
−1.2
−1.0
Reward
IPPO
Et-Mapg
Aet-Mapg
(a) Rewards.
0.0
0.5
1.0
0
2
4
6
8
10
Time (Seconds)
0.0
0.5
1.0
T i
(b) Triggering instants and moving avg.
100
101
0
2
4
6
8
10
Time (Seconds)
100
101
log(ti
k+1 −ti
k)
(c) Inter event time.
Fig. 3: Performance comparison between IPPO and our proposed methods in the Simple Reference MPE. Subfigures (b)
and (c) show TIGER (top) and ATT-TIGER (bottom) results, respectively.
0.0
0.2
0.4
0.6
0.8
1.0
Number of steps
×105
−3.2
−3.0
−2.8
−2.6
−2.4
−2.2
−2.0
Reward
IPPO
Et-Mapg
Aet-Mapg
(a) Rewards.
0.0
0.5
1.0
0
2
4
6
8
10
Time (Seconds)
0.0
0.5
1.0
T i
(b) Triggering instants and moving avg.
100
101
0
2
4
6
8
10
Time (Seconds)
100
101
log(ti
k+1 −ti
k)
(c) Inter event time.
Fig. 4: Performance comparison between IPPO and our proposed methods in the Simple Spread MPE. Subfigures (b) and
(c) show TIGER (top) and ATT-TIGER (bottom) results, respectively.
consists of local and global rewards, where a local reward for
each agent is the negative distance to its own target, and the
global reward is defined as the average distance of all agents
to their respective targets. In Simple Spread, there are three
agents and three landmarks. The goal in this environment is
to cover all landmarks while avoiding collisions. Again, the
reward function consists of both local and global components,
where a local reward is a penalty of −1 for each collision and
a global reward is the negative sum of the minimum distances
from each landmark to the closest agent, encouraging agents
to spread out and efficiently cover all landmarks.
Figs. 3a and 4a present the learning curves in both environ-
ments and show that while ET-MAPG significantly reduces
communication frequency, it may converge to slightly lower
rewards compared to the baseline. This is because, in an
event-triggered setting, and especially in environments where
communication is necessary, the proposed methods allow
an efficient communication mechanism. Instead of constant
updates, agents learn to share information only at the most
critical moments. This targeted communication helps them
build a better model of the environment’s dynamics and
achieve tighter coordination as a team. This is evident as
AET-MAPG leverages selective communication to match
the reward performance of IPPO, while also achieving the
most resource-efficient behavior. Communication frequencies
shown in Figs. 3b and 4b demonstrate that both methods
substantially reduce communication frequencies, where AET-
MAPG performs the best. Figs. 3c and 4c present the inter-
event time. Once again, these results show that triggering in-
tervals remain strictly positive and dynamically adjusted over
time, which avoids Zeno behavior. These results demonstrate
that our proposed event-triggered methods also generalize to
high-dimensional, partially observable MPE domains, achiev-
ing both performance and resource-efficiency.
To further evaluate the generality of our proposed frame-
work, we conduct ablation studies by integrating the joint
learning of event-triggered policies with control policies in
IA2C [28] and MAPPO [27], which are two other state-of-
the-art MARL algorithms. We evaluate these methods in the
multi-agent single integrator environment first, owing to the
fact that greater insights could be obtained from a simple
environment. Since the objective in this environment is to
drive the system toward the origin (the equilibrium point),
event-triggered methods achieve this efficiently by reusing
previously sampled actions rather than resampling at every
timestep, as required in standard MARL algorithms. Fig. 5

[EQ eq_p5_000] -> see eq_p5_000.tex

[EQ eq_p5_001] -> see eq_p5_001.tex


# [PAGE 7]
0.0
0.5
1.0
Et-Mapg (MAPPO)
Aet-Mapg (MAPPO)
0
2
4
6
8
10
0.0
0.5
1.0
Et-Mapg (IA2C)
0
2
4
6
8
10
Aet-Mapg (IA2C)
Time (s)
T i
Fig.
5:
Communication
frequencies
of
event-triggered
MAPPO and IA2C, along with their attention-based variants
in multi-agent single-integrator environment.
presents the communication frequencies of event-triggered
MAPPO and IA2C, along with their attention-based coun-
terparts. The results show that both event-triggered MAPPO
and IA2C significantly reduce the number of action samples
compared to their standard MARL baselines that are time-
triggered. Although IA2C demonstrates weaker performance
relative to MAPPO and IPPO, it still achieves over 50% re-
duction in communication. Fig. 6 shows the inter-event times,
verifying that triggering intervals remain strictly positive and
adapt dynamically, thereby avoiding Zeno behavior. Overall,
these results demonstrate that our framework is both effec-
tive and generalizable across different MARL paradigms. In
particular, while MAPPO employs a centralized critic with
decentralized actors and IPPO/IA2C adopts a fully indepen-
dent actor–critic architecture, our event-triggered methods
integrate seamlessly with these architectures while retaining
their performance.
100
101
Et-Mapg (MAPPO)
100
101
Aet-Mapg (MAPPO)
0
2
4
6
8
10
100
101
Et-Mapg (IA2C)
0
2
4
6
8
10
100
101
Aet-Mapg (IA2C)
Time (s)
log(ti
k+1 −ti
k)
Fig. 6: Inter event time of event-triggered MAPPO and
IA2C, along with their attention-based variants in multi-agent
single-integrator environment.
V. CONCLUSIONS AND FUTURE WORK
In this paper, we introduced a novel framework for event-
triggered MARL. We proposed ET-MAPG, a method that
jointly learns a control action head and an event-trigger head
for each agent within a single policy network, where the
triggering head dynamically determines when a new action
should be sampled from the action head, in contrast to prior
approaches that rely on separate policies for control and
triggering. Building on this, we further proposed AET-MAPG,
an attention-based variant of ET-MAPG that incorporates self-
attention as a communication mechanism during training.
By enabling agents to share messages only when triggering
conditions are satisfied selectively, AET-MAPG achieves effi-
cient coordination while maintaining a sparse communication
graph. Through extensive experiments across both control
and standard MARL benchmarks, we demonstrated that ET-
MAPG and AET-MAPG achieve performance comparable to
MARL baselines while reducing communication and compu-
tation costs by up to 50%. Finally, we show that our proposed
methods are general and can be seamlessly integrated with
any multi-agent policy gradient methods, including IPPO,
MAPPO, and IA2C.
Our work also has some limitations. First, our current
framework is restricted to discrete action spaces. Second,
communication in AET-MAPG is assumed to occur over a
complete undirected graph, where all agents share messages
bidirectionally. Third, our framework is currently suitable
only for policy gradient MARL methods. As future work, we
plan to address these limitations by extending our methods
to continuous action spaces and more complex nonlinear
systems, developing techniques for communication over dy-
namic graphs, and extending our framework to value-based
MARL approaches.
REFERENCES
[1] M. Miskowicz, Event-Based Control and Signal Processing, 1st ed.
CRC Press, 2015.
[2] A. Selivanov and E. Fridman, “Event-triggered ℎ∞control: A switch-
ing approach,” IEEE Transactions on Automatic Control, vol. 61,
no. 10, pp. 3221–3226, 2015.
[3] V. Digge and R. Pasumarthy, “Data-driven event-triggered control
for discrete-time lti systems,” in 2022 European Control Conference
(ECC), 2022, pp. 1355–1360.
[4] W.-L. Qi, K.-Z. Liu, R. Wang, and X.-M. Sun, “Data-driven L2-
stability analysis for dynamic event-triggered networked control sys-
tems: A hybrid system approach,” IEEE Transactions on Industrial
Electronics, vol. 70, no. 6, pp. 6151–6158, 2023.
[5] L. A. Q. Cordovil Jr, P. H. S. Coutinho, I. Bessa, M. L. C. Peixoto, and
R. M. Palhares, “Learning event-triggered control based on evolving
data-driven fuzzy granular models,” International Journal of Robust
and Nonlinear Control, vol. 32, no. 5, pp. 2805–2827, 2022.
[6] W. Liu, J. Sun, G. Wang, F. Bullo, and J. Chen, “Data-driven self-
triggered control via trajectory prediction,” IEEE Transactions on
Automatic Control, vol. 68, no. 11, pp. 6951–6958, 2023.
[7] D. Baumann, J.-J. Zhu, G. Martius, and S. Trimpe, “Deep reinforce-
ment learning for event-triggered control,” in 2018 IEEE Conference
on Decision and Control (CDC), 2018, pp. 943–950.
[8] X. Wang, J. Berberich, J. Sun, G. Wang, F. Allg¨ower, and J. Chen,
“Model-based and data-driven control of event- and self-triggered
discrete-time linear systems,” IEEE Transactions on Cybernetics,
vol. 53, no. 9, pp. 6066–6079, 2023.


# [PAGE 8]
[9] L. Bus¸oniu, T. De Bruin, D. Toli´c, J. Kober, and I. Palunko, “Re-
inforcement learning for control: Performance, stability, and deep
approximators,” Annual Reviews in Control, vol. 46, pp. 8–28, 2018.
[10] D. Bertsekas, Reinforcement learning and optimal control.
Athena
Scientific, 2019, vol. 1.
[11] J. Ibarz, J. Tan, C. Finn, M. Kalakrishnan, P. Pastor, and S. Levine,
“How to train your robot with deep reinforcement learning: lessons we
have learned,” The International Journal of Robotics Research, vol. 40,
no. 4-5, pp. 698–721, 2021.
[12] U. Siddique, A. Sinha, and Y. Cao, “On deep reinforcement learning
for target capture autonomous guidance,” in AIAA SCITECH, 2024.
[13] K. G. Vamvoudakis and H. Ferraz, “Model-free event-triggered control
algorithm for continuous-time linear systems with optimal perfor-
mance,” Automatica, vol. 87, pp. 412–420, 2018.
[14] X. Zhong, Z. Ni, H. He, X. Xu, and D. Zhao, “Event-triggered rein-
forcement learning approach for unknown nonlinear continuous-time
system,” in 2014 International Joint Conference on Neural Networks
(IJCNN), 2014, pp. 3677–3684.
[15] X. Yang, H. He, and D. Liu, “Event-triggered optimal neuro-controller
design with reinforcement learning for unknown nonlinear systems,”
IEEE Transactions on Systems, Man, and Cybernetics: Systems,
vol. 49, no. 9, pp. 1866–1878, 2019.
[16] U. Siddique, A. Sinha, and Y. Cao, “Adaptive event-triggered rein-
forcement learning control for complex nonlinear systems,” in 2025
American Control Conference (ACC).
IEEE, 2025, pp. 212–217.
[17] J. N. Foerster, Y. M. Assael, N. de Freitas, and S. Whiteson, “Learning
to communicate to solve riddles with deep distributed recurrent q-
networks,” arXiv preprint arXiv:1602.02672, 2016.
[18] J. Foerster, I. A. Assael, N. De Freitas, and S. Whiteson, “Learning to
communicate with deep multi-agent reinforcement learning,” Advances
in neural information processing systems, vol. 29, 2016.
[19] D. Kim, S. Moon, D. Hostallero, W. J. Kang, T. Lee, K. Son, and Y. Yi,
“Learning to schedule communication in multi-agent reinforcement
learning,” arXiv preprint arXiv:1902.01554, 2019.
[20] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by
jointly learning to align and translate,” arXiv preprint arXiv:1409.0473,
2014.
[21] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,”
Advances in neural information processing systems, vol. 30, 2017.
[22] G. Hu, Y. Zhu, D. Zhao, M. Zhao, and J. Hao, “Event-triggered
communication network with limited-bandwidth constraint for multi-
agent reinforcement learning,” IEEE Transactions on Neural Networks
and Learning Systems, vol. 34, no. 8, pp. 3966–3978, 2021.
[23] Z. Feng, M. Huang, Y. Wu, D. Wu, J. Cao, I. Korovin, S. Gorbachev,
and N. Gorbacheva, “Approximating nash equilibrium for anti-uav
jamming markov game using a novel event-triggered multi-agent
reinforcement learning,” Neural Networks, vol. 161, pp. 330–342,
2023.
[24] J. Lu, L. Han, Q. Wei, X. Wang, X. Dai, and F.-Y. Wang, “Event-
triggered deep reinforcement learning using parallel control: A case
study in autonomous driving,” IEEE Transactions on Intelligent Vehi-
cles, vol. 8, no. 4, pp. 2821–2831, 2023.
[25] J. Chen, X. Meng, and Z. Li, “Reinforcement learning-based event-
triggered model predictive control for autonomous vehicle path fol-
lowing,” in ACC.
IEEE, 2022, pp. 3342–3347.
[26] C. S. De Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. Torr,
M. Sun, and S. Whiteson, “Is independent learning all you need in
the starcraft multi-agent challenge?” arXiv preprint arXiv:2011.09533,
2020.
[27] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu,
“The surprising effectiveness of ppo in cooperative multi-agent games,”
Advances in neural information processing systems, vol. 35, pp.
24 611–24 624, 2022.
[28] G. Papoudakis, F. Christianos, L. Sch¨afer, and S. V. Albrecht,
“Benchmarking multi-agent deep reinforcement learning algorithms
in cooperative tasks,” in Proceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks (NeurIPS),
2021. [Online]. Available: http://arxiv.org/abs/2006.07869
[29] R. S. Sutton, D. McAllester, S. Singh, and Y. Mansour, “Policy gradient
methods for reinforcement learning with function approximation,”
Advances in neural information processing systems, vol. 12, 1999.
[30] G. Papoudakis, F. Christianos, A. Rahman, and S. V. Albrecht, “Deal-
ing with non-stationarity in multi-agent deep reinforcement learning,”
arXiv preprint arXiv:1906.04737, 2019.
[31] C. Claus and C. Boutilier, “The dynamics of reinforcement learning in
cooperative multiagent systems,” AAAI/IAAI, vol. 1998, no. 746-752,
p. 2, 1998.
[32] I. Mordatch and P. Abbeel, “Emergence of grounded compositional lan-
guage in multi-agent populations,” arXiv preprint arXiv:1703.04908,
2017.
