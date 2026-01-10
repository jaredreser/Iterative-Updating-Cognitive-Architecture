# Iterative-Updating-Cognitive-Architecture

**Iterative-Updating-Cognitive-Architecture** is a reference implementation of the *iterative updating* cognitive architecture described in **Reser (2022, arXiv:2203.17255)**. The core idea is that thought proceeds by **partial updates** to a capacity-limited working set: each iteration **retains** a fraction of the current contents and **recruits** new contents via **pooled multi-cue (multiassociative) search** over a candidate pool.

This repository provides:
- a transparent **NumPy simulator** for inspecting dynamics (overlap, resets, lure trapping),
- a trainable **PyTorch module** (retain–drop–add + differentiable Top-K selection),
- optional **iterative inhibition** (“reject-and-research”) for steering away from lures,
- toy benchmarks and **ablations** to test the role of overlap, pooled search, inhibition, and curriculum scheduling.

The goal is to make the architecture **runnable, measurable, and comparable**—as a cognitive model and as an AI building block.

---

## Quickstart

### NumPy simulator

```python
from iuca.sim import IterativeWorkingMemorySim
from iuca.utils import set_seed

set_seed(0)
sim = IterativeWorkingMemorySim(N=300, K=16, r=0.8, tau=0.8, seed=0)
overlap, W_next = sim.step()
print(f"overlap={overlap:.3f}", W_next[:5])
```

### PyTorch module

```python
import torch
from iuca.torch_module import IterativeUpdatingWorkingMemory
from iuca.utils import set_seed

set_seed(0)
B, K, N, d = 2, 8, 128, 64
W = torch.randn(B, K, d)
C = torch.randn(B, N, d)
model = IterativeUpdatingWorkingMemory(d=d, K=K)
candidate_mask = torch.ones(B, N, dtype=torch.bool)

W_next, keep_probs, select_probs, aux = model(W, C, candidate_mask=candidate_mask)
print(W_next.shape, aux["selection_entropy"])
```

---

## Background and reference

This repo is inspired by:

> Reser, J. E. (2022). *A Cognitive Architecture for Machine Consciousness and Artificial Superintelligence: Thought Is Structured by the Iterative Updating of Working Memory.* arXiv:2203.17255.

If you use this repo in academic work, please cite the arXiv paper (and feel free to cite this software repo as well).


In “A Cognitive Architecture for Machine Consciousness and Artificial Superintelligence: Thought Is Structured by the Iterative Updating of Working Memory” (arXiv:2203.17255), I, Jared Reser lay out a proposal for what a thought process would look like if we tried to engineer it directly into AI, rather than treating intelligence as something that falls out of ever-larger pattern recognizers.

Reser, J. 2022. A Cognitive Architecture for Machine Consciousness and Artificial Superintelligence: Updating Working Memory Iteratively. arXiv: 2203.17255 

You can also see this article at aithought.com with videos.


The paper’s central claim is simple to state: the workflow of thought is iterative. Instead of one working-memory state being replaced wholesale by the next, each new state should preserve some proportion of the previous state while adding and subtracting other elements. This “partial updating” causes successive states to overlap, so a train of thought becomes a chain of intermediate states that remain causally and semantically linked over time.

I argue that this overlap is not just a philosophical gloss, it’s grounded in the biology of persistent activity. Mammalian working memory is framed as having two key persistence mechanisms operating on different time scales: sustained firing (seconds) supporting the focus of attention, and synaptic potentiation (minutes to hours) supporting a broader short-term store. In this view, as some items drop out and others enter, the remaining coactive subset “stitches” the stream together, making continuity and multi-step reasoning possible.

Crucially, the paper doesn’t stop at saying states overlap. It proposes a mechanism for how the next update is chosen: the currently coactive working-memory contents jointly “cospread” activation across the network, performing a multiassociative search over long-term memory for the most context-relevant next addition(s). This repeated “search → update → search again (with modified context)” is presented as a compounding process that can build structured inferences, predictions, and plans across multiple steps.

Because the manuscript is meant to be both explanatory and constructive, it also explicitly positions iterative updating as an engineering blueprint: a way to implement a global-workspace-like working set that is updated continuously, supports long-range dependencies, and can be trained developmentally by expanding persistence/overlap over time. The paper even provides a glossary of introduced terms (e.g., iterative updating, cospreading, multiassociative search, SSC/icSSC, iterative compounding, iterative thread) intended to carve the system into reusable conceptual parts.


Here is a list of concrete claims and “working insights” extracted from the paper, phrased as testable or at least operationally meaningful statements.

A) Core computational principle
Thought is organized by continuous partial updating: each new working-memory state preserves a proportion of the prior state (not complete replacement), making the stream of thought a chain of overlapping iterations.
Iterative overlap is the mechanism of continuity: overlap between successive working-memory states creates “recursive nesting” so each state is embedded in the one before it, enabling stateful cognition rather than stateless reactions.
Iterative updating is simultaneously (i) an information-processing strategy, (ii) a model of working memory, (iii) a theory of consciousness, and (iv) an AI programming principle. Cognitive Architecture Iterativ…

B) Working memory structure: two persistence tiers + iteration in both
Working memory has two key persistence mechanisms with different timescales: sustained firing maintains the FoA (seconds), while synaptic potentiation maintains a broader short-term store (minutes to hours).
Both stores iterate: the FoA iterates via sustained firing; the short-term store iterates as a pool of synaptically potentiated units that is continuously added to and subtracted from, yielding isomorphic “incremental updating” across neural and psychological levels. Cognitive Architecture Iterativ…
The persisting “topic” of thought corresponds to the longest-lasting active units, while other contextual features come and go around it. Cognitive Architecture Iterativ…

C) Control variables and “modes” of thought
Rate of updating is a control parameter (how much of the set changes per step) that tunes looseness vs tightness of coupling—superficial/distractible vs concentrated/systematic processing.
Implicit vs explicit processing is framed as different overlap regimes (system-1-like = faster updating / less overlap; system-2-like = slower updating / more overlap and longer maintenance of intermediates). Cognitive Architecture Iterativ…
Dopamine is proposed to reduce the rate of updating (stabilize the set), mediating a shift toward explicit/effortful processing under novelty/surprise/reward/error. Cognitive Architecture Iterativ…
Boundaries between “thoughts” are marked by intermittent non-iterative updates (high-percentage replacement events), while within-thought processing shows sustained low-percentage turnover. Cognitive Architecture Iterativ…

D) How new content is selected: pooled search (multiassociative search)
Selection of the next update is a pooled spreading-activation search: the currently coactive set combines (“cospreads”) activation energy through the global network to converge on the most context-relevant next item(s).
Multiassociative search is described as an explicit stepwise algorithm (items maintained vs dropped vs newly activated; plus mechanisms where the newest addition redistributes activation weights and can contextually alter the “fuzzy” composition/meaning of items). Cognitive Architecture Iterativ…
The search contributors are not just FoA items: potentiated short-term-store units plus active sensory/motor cortex, hippocampus, basal ganglia, and other systems all contribute to the pooled search that selects the next update.
Multiassociative search produces novel inference as a standard case: even when the set of assemblies is unprecedented, the system can converge on either recall (same result as last time) or a new item (novel inference) depending on current coactivity.
Multiassociative search implies multiassociative learning: each search event can retune associative strengths (Hebbian-style), so search doesn’t just use memory—it updates semantic/procedural structure over time. Cognitive Architecture Iterativ…

E) Prediction and inference as the product of iteration
Updates generated by search are predictions: iterative updating + pooled search is framed as a brain-level autoregressive mechanism that captures conditional dependencies across sequences of events.
Iterative compounding: the product of one search becomes part of the next state’s cue-set, so search is repeatedly modified by its own outputs, compounding inferences/predictions across steps.

F) Reasoning patterns as working-memory dynamics (figures → mechanisms)
Iterative inhibition: when the newest update is judged unhelpful/prepotent, it is inhibited so the remaining set must converge on the next-most-pertinent item; repeated inhibition rounds progressively restrict the search tree. Cognitive Architecture Iterativ…
Planning = dense iteration: planning is characterized as (i) lower update rate, (ii) fewer full “jumps,” and (iii) more intermediate iterations before action—explicitly mapping planning to “chain-of-thought-like” intermediate steps.
Attractor states as beliefs/truths: iterative updating tends to converge toward stable item-sets (attractors) interpreted as beliefs; thinking is framed as progressive narrowing/compression toward generalizable statements. Cognitive Architecture Iterativ…

G) Threading, subproblems, and compositional problem solving
Iterative thread: a line of thought is a chain of iteratively updated states that can be reiterated or “picked up where it left off.”
Subproblem decomposition via store cooperation: the FoA iterates on a subproblem while the short-term store holds the broader objective; interim results can be suspended and later reactivated.
Merging of subsolutions: outputs from separate iterative episodes can be coactivated in a new state and used together for multiassociative search to yield a hybrid/final solution.
Backward reference / conditional branching emerges when prior threads/subsolutions are stored and later reconverged upon, allowing departures from the default forward-iterative flow.
Schemas as dynamic packets that can be recalled and co-iterated: a previously learned multi-item schema can be pulled in midstream and used as an organizing heuristic/script that iterates with the current line of thought.
Transfer learning as “recognize partial overlap → import prior thread content”: encountering a later situation that shares items with an earlier episode triggers reuse of prior iterative structure to generalize toward a similar conclusion. Cognitive Architecture Iterativ…

H) AI training/development implications (as stated)
Maturational training schedule for AI: start with minimal working-memory span/overlap and gradually expand toward superhuman span as experience accumulates. Cognitive Architecture Iterativ…
Long-horizon inference depends on persistence preventing “cache misses”: prolonging persistence makes each search more specific (more constraints) and preserves intermediate results long enough to compound into higher-order inferences. Cognitive Architecture Iterativ…
Mathematical Formalization of Iterative Updating Working Memory
This section provides a minimal mathematical formalization of an iterative working-memory architecture in which (i) a limited-capacity focus-of-attention working set is updated incrementally over discrete cognitive iterations, (ii) the next working-memory content is selected by pooled multi-cue (multiassociative) search, and (iii) an inhibitory steering mechanism can suppress unhelpful candidates to force exploration of alternatives. The same formalization can be interpreted as a cognitive/neural process model or as an implementable AI module.


At discrete iterations \(t = 0,1,2,\dots\), the system maintains a capacity-limited working set \(W_t\) (the **focus of attention**) of size \(K\). Each update produces a new working set \(W_{t+1}\) by **keeping** \(rK\) slots (retention fraction \(r\in[0,1]\)) and **recruiting** \((1-r)K\) new items from a candidate pool \(C_t\). New recruits are selected by **pooled multiassociative search**: the current working set is pooled into a query \(q_t\), candidates are scored by similarity to \(q_t\), and a Top-K (or differentiable approximation) selects the recruits. Optionally, an inhibition trace \(h_t\) suppresses recently selected/rejected candidates to encourage escape from lures and local attractors.
