# Awesome-World-Models

[![arXiv](https://img.shields.io/badge/arXiv-Slow_Reason_System-b31b1b.svg)](http://arxiv.org/abs/$$$$$$) 
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/JiahuaDong/Awesome-World-Models)
[![Last Commit](https://img.shields.io/github/last-commit/JiahuaDong/Awesome-World-Models)](https://github.com/JiahuaDong/Awesome-World-Models)
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)]()

<!-- omit in toc -->
## 📢 Updates

- **2026.01**: 

<!-- omit in toc -->
## 👀 Introduction

Welcome to the repository for our survey paper, "Learning to Model the World: A Survey of World Models in Artificial Intelligence". This repository provides resources and updates related to our research. For a detailed introduction, please refer to [our survey paper]($$$$$$).

World models (WMs) provide a unified approach for modeling how environments evolve over time by learning predictive representations of states and observations. Recent advances in large-scale generative modeling and multimodal foundation models have substantially broadened their applicability across a wide range of interactive and multimodal domains; however, existing research remains fragmented across modeling paradigms, application domains, and evaluation protocols. This survey provides a systematic and in-depth review of WMs in artificial intelligence. Based on the world modeling paradigms of existing methods, we first categorize WMs into four major branches with formal mathematical formulations: reinforcement learning-based, observation-level generative, latent space, and object-centric world models. We further review a broad range of WM applications spanning robotics, autonomous driving, scientific discovery, virtual game simulation, GUI-based agents, as well as interpretability and trustworthiness, and summarize benchmark datasets, evaluation metrics, simulation platforms, and comparative results across WMs. Finally, we discuss key challenges, including long-horizon consistency, controllability, robustness, evaluation limitations, and generalization, and outline promising directions for future research. This survey aims to offer a unified reference for understanding, comparing, and advancing WMs.

![image](./assets/develop.png)

The recent timeline of world models, covering core methods and the release of open-source and closed-source reproduction projects.

<!-- omit in toc -->
## 📒 Table of Contents

- [Awesome-World-Models](#awesome-world-models)
  - [Part 1: Reinforcement Learning-Based World Models](#part-1-reinforcement-learning-based-world-models)
  - [Part 2: Observation-Level Generative World Models](#part-2-observation-level-generative-world-models)
    - [Language Observations](#language-observations)
    - [Visual Observations](#visual-observations)
    - [3D and 4D Observations](#3d-and-4d-observations)
  - [Part 3: Latent Space World Models](#part-3-latent-space-world-models)
  - [Part 4: Object-centric World Models](#part-4-object-centric-world-models)
  - [Part 5: World Models for Robotics](#part-5-world-models-for-robotics)
    - [Manipulation](#manipulation)
      - [Visual Futrue Prediction](#visual-futrue-prediction)
      - [Latent Action State Imagination](#latent-action-state-imagination)
      - [Control-Oriented Planning](#control-oriented-planning)
      - [World Action Modeling](#world-action-modeling)
    - [Navigation](#navigation)
    - [Policy Learning](#policy-learning)
    - [Locomotion](#locomotion)
  - [Part 6: World Models for Autonomous Driving](#part-6-world-models-for-autonomous-driving)
    - [Predictive Modeling](#predictive-modeling)
    - [Action-Conditioned Imagination](#action-conditioned-imagination)
    - [Decision-Centric Integration](#decision-centric-integration)
  - [Part 7: World Models for Science](#part-7-world-models-for-science)
    - [Social Science and Socioeconomic Systems](#social-science-and-socioeconomic-systems)
    - [Physical and Natural Sciences](#physical-and-natural-sciences)
  - [Part 8: World Models for Virtual Game Simulation](#part-8-world-models-for-virtual-game-simulation)
    - [Pixel-Level Observation Prediction](#pixel-level-observation-prediction)
    - [3D Mesh-Level Observation Prediction](#3d-mesh-level-observation-prediction)
  - [Part 9: World Models for GUI-Based Agents](#part-9-world-models-for-gui-based-agents)
  - [Part 10: Interpretable and Trustworthy World Models](#part-10-interpretable-and-trustworthy-world-models)
  - [Part 11: Benchmark of World Models](#part-11-benchmark-of-world-models)
    - [Benchmark Datasets \& Evaluation Metrics](#benchmark-datasets--evaluation-metrics)
    - [Physics Engines \& Simulation Platforms](#physics-engines--simulation-platforms)
  - [Part 12: Performance Comparison](#part-12-performance-comparison)
  - [Citation](#citation)

## Part 1: Reinforcement Learning-Based World Models

* DreamerV3: Mastering diverse control tasks through world models [[Paper]](https://www.nature.com/articles/s41586-025-08744-2.pdf)   ![](https://img.shields.io/badge/Nature-2025-0077c2)
*  Dreamer: Dream to control: Learning behaviors by latent imagination [[Paper]](https://iclr.cc/virtual_2020/poster_S1lOTC4tDS.html) ![](https://img.shields.io/badge/ICLR-2020-0077c2)
* DreamSmooth: Improving model-based reinforcement learning via reward smoothing [[Paper]](https://iclr.cc/virtual/2024/poster/19014) ![](https://img.shields.io/badge/ICLR-2024-0077c2)
* PlaNet: Learning latent dynamics for planning from pixels [[Paper]](https://icml.cc/virtual/2019/poster/3764) ![](https://img.shields.io/badge/ICML-2019-0077c2)
* DreamerV2: Mastering atari with discrete world models [[Paper]](https://iclr.cc/virtual/2021/poster/2742) ![](https://img.shields.io/badge/ICLR-2021-0077c2)
* PIGDreamer: Privileged information guided world models for safe partially [[Paper]](https://icml.cc/virtual/2025/poster/44134) ![](https://img.shields.io/badge/ICML-2025-0077c2)
* HarmonyDream: Task Harmonization Inside World Models [[Paper]](https://icml.cc/virtual/2024/poster/32730) ![](https://img.shields.io/badge/ICML-2024-0077c2)
* DyMoDreamer: World modeling with dynamic modulation [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/117925) ![](https://img.shields.io/badge/NeurIPS-2025-0077c2)
* TD-MPC2: Scalable, robust world models for continuous control  [[Paper]](https://iclr.cc/virtual/2024/poster/18722) ![](https://img.shields.io/badge/ICLR-2024-0077c2)
* Hieros: Hierarchical imagination on structured state space sequence world models [[Paper]](https://icml.cc/virtual/2024/poster/34428) ![](https://img.shields.io/badge/ICML-2024-0077c2)
* THICK: Learning hierarchical world models with adaptive temporal abstractions from discrete latent dynamics [[Paper]](https://iclr.cc/virtual/2024/poster/18558) ![](https://img.shields.io/badge/ICLR-2024-0077c2)
* MoSim: Neural motion simulator pushing the limit of world models in reinforcement learning [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/34450) ![](https://img.shields.io/badge/CVPR-2025-0077c2)
* R2I: Mastering memory tasks with world models [[Paper]](https://iclr.cc/virtual/2024/poster/19565) 
![](https://img.shields.io/badge/ICLR-2024-0077c2)
* LEQ: Model-based offline reinforcement learning with lower expectile q-learning [[Paper]](https://iclr.cc/virtual/2025/37540) ![](https://img.shields.io/badge/ICLR-2025-0077c2)
* DIMA: Revisiting multi-agent world modeling from a diffusion-inspired perspective [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/115775) ![](https://img.shields.io/badge/NeurIPS-2025-0077c2)
* PCM: Policy-conditioned environment models are more generalizable [[Paper]](https://icml.cc/virtual/2024/poster/33439) ![](https://img.shields.io/badge/ICML-2024-0077c2)
* CoWorld: Making offline RL online: Collaborative world models for offline visual reinforcement learning [[Paper]](https://neurips.cc/virtual/2024/poster/93263) ![](https://img.shields.io/badge/NeurIPS-2024-0077c2)
* IQ-MPC: Reward-free world models for online imitation learning [[Paper]](https://icml.cc/virtual/2025/poster/44035) ![](https://img.shields.io/badge/ICML-2025-0077c2)
* WAKER: Reward-free curricula for training robust world models [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2024/hash/0a2b3e9107efed3d361b29f300a903ff-Abstract-Conference.html) ![](https://img.shields.io/badge/ICLR-2024-0077c2)
* REM: Improving token-based world models with parallel observation prediction [[Paper]](https://icml.cc/virtual/2024/poster/34279) ![](https://img.shields.io/badge/ICML-2024-0077c2)
* cRSSM: Dreaming of many worlds: Learning contextual world models aids zero-shot generalization [[Paper]](https://openreview.net/pdf?id=zHt4K5zX4P) ![](https://img.shields.io/badge/EWRL-2024-0077c2)
* Adaptive world models: Learning behaviors by latent imagination under non-stationarity $$$ [[Paper]](https://neurips.cc/virtual/2024/104950) ![](https://img.shields.io/badge/NeurIPS-2024-0077c2)
* PWM: Policy learning with multi-task world models [[Paper]](https://iclr.cc/virtual/2025/poster/28766) ![](https://img.shields.io/badge/ICLR-2025-0077c2)

## Part 2: Observation-Level Generative World Models

### Language Observations
*ByteSized32-State-Prediction: Can language models serve as text-based world simulators? $$$[[Paper]](https://aclanthology.org/2024.acl-short.1/) ![](https://img.shields.io/badge/ACL-2024-0077c2)
* GPT-4: Gpt-4 technical report [[Paper]](https://arxiv.org/pdf/2303.08774) ![](https://img.shields.io/badge/arXiv-2023.03-B31B1B)
* Llama 3: The llama 3 herd of models [[Paper]](https://arxiv.org/pdf/2407.21783) ![](https://img.shields.io/badge/arXiv-2024.07-B31B1B)
* LLMCWM: Language agents meet causality – bridging LLMs and causal world models [[Paper]](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5c5bc3553815adb4d1a8a5b8701e41a9-Abstract-Conference.html) ![](https://img.shields.io/badge/ICLR-2025-0077c2)
* Surge: On the potential of large language mode [[Paper]](https://aclanthology.org/2025.emnlp-main.162/) ![](https://img.shields.io/badge/EMNLP-2025-0077c2)
* RAP: Reasoning with language model is planning with world model [[Paper]](https://aclanthology.org/2023.emnlp-main.507/) ![](https://img.shields.io/badge/EMNLP-2023-0077c2)
* Making large language models into world models with precondition and effect knowledge [[Paper]](https://aclanthology.org/2025.coling-main.503.pdf) ![](https://img.shields.io/badge/COLING-2025-0077c2)
* LWM: World model on million-length video and language with blockwise ringattention [[Paper]](https://iclr.cc/virtual/2025/poster/30229) ![](https://img.shields.io/badge/ICRL-2025-0077c2)
### Visual Observations
* Sora: Video generation models as world simulators [[Paper]](https://openai.com/index/video-generation-models-as-world-simulators/) ![](https://img.shields.io/badge/OpenAI-2024-0077c2)
* TeleWorld: Macro-from-Micro Planning for High-Quality and Parallelized Autoregressive Long Video Generation [[Paper]](https://arxiv.org/abs/2508.03334) ![](https://img.shields.io/badge/arXiv-2025.08-red)
* Gen-3 [[Blog]](https://runwayml.com/research/introducing-gen-3-alpha) ![](https://img.shields.io/badge/blog-2024.06-red)
* T2v-turbo: Breaking the quality bottleneck of video consistency model with mixed reward feedback [[Paper]](https://papers.nips.cc/paper_files/paper/2024/file/8a57aa8e8b57e64a42e95f7dceb0adb9-Paper-Conference.pdf) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
* Emu3: Next-token prediction is all you need [[Paper]](https://arxiv.org/pdf/2409.18869) ![](https://img.shields.io/badge/arXiv-2024.09-B31B1B)
* SPMEM: Video world models with long-term spatial memory [[Paper]](https://neurips.cc/virtual/2025/poster/118886) ![](https://img.shields.io/badge/NeurIPS-2025-0077c2)
* Wan: Open and Advanced Large-Scale Video Generative Models [[Paper]](https://arxiv.org/pdf/2503.20314) ![](https://img.shields.io/badge/arXiv-2025.03-B31B1B)
* LLaVA: Visual instruction tuning [[Paper]](https://papers.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-0077c2)
* Vid2world: Crafting video diffusion models to interactive world models [[Paper]](https://arxiv.org/pdf/2505.14357) ![](https://img.shields.io/badge/arXiv-2025.05-B31B1B)
* VideoCrafter2: Overcoming data limitations for high-quality video diffusion models [[Paper]](https://cvpr.thecvf.com/virtual/2024/poster/29306) ![](https://img.shields.io/badge/CVPR-2024-0077c2)
* IRASim: A Fine-Grained World Model for Robot Manipulation [[Paper]](https://iccv.thecvf.com/virtual/2025/poster/2655) ![](https://img.shields.io/badge/ICCV-2025-0077c2)
* WISA: World simulator assistant for physics-aware text-to-video generation [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/119925) ![](https://img.shields.io/badge/NeurIPS-2025-0077c2)
* CoLA-World: Co-Evolving Latent Action World Models [[Paper]](https://arxiv.org/pdf/2510.26433) ![](https://img.shields.io/badge/arXiv-2025.10-B31B1B) 
### 3D and 4D Observations
* 4D-fy: Text-to-4d generation using hybrid score distillation sampling [[Paper]](https://cvpr.thecvf.com/virtual/2024/poster/29662) ![](https://img.shields.io/badge/CVPR-2024-0077c2)
* WonderJourney: Going from anywhere to everywhere [[Paper]](https://cvpr.thecvf.com/virtual/2024/poster/31689) ![](https://img.shields.io/badge/CVPR-2023-0077c2)
* SceneScape: Text-driven consistent scene generation [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d62a85ebfed2f680eb5544beae93191-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2023-0077c2)
* LiDARCrafter: Dynamic 4d world modeling from lidar sequences [[Paper]](https://arxiv.org/pdf/2508.03692v2) ![](https://img.shields.io/badge/arXiv-2025.08-B31B1B)
* Text2room: Extracting textured 3d meshes from 2d text-to-image models [[Paper]](https://openaccess.thecvf.com/content/ICCV2023/html/Hollein_Text2Room_Extracting_Textured_3D_Meshes_from_2D_Text-to-Image_Models_ICCV_2023_paper.html) ![](https://img.shields.io/badge/ICCV-2023-0077c2)
* WonderWorld: Interactive 3d scene generation from a single image [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/33364) ![](https://img.shields.io/badge/CVPR-2025-0077c2)
* Invisible Stitch: Generating smooth 3d scenes with depth inpainting [[Paper]](https://openreview.net/attachment?id=grImaQQkFx&name=pdf) ![](https://img.shields.io/badge/3DV-2024-0077c2)

## Part 3: Latent Space World Models
* I-JEPA: Self-supervised learning from images with a joint-embedding predictive architecture [[Paper]](https://cvpr.thecvf.com/virtual/2023/poster/21019) ![](https://img.shields.io/badge/CVPR-2023-blue)
* V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video [[Paper]](https://openreview.net/forum?id=L2pdpUFUaX) ![](https://img.shields.io/badge/TMLR-2024-blue)
* V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning [[Paper]](https://neurips.cc/virtual/2024/104556) ![](https://img.shields.io/badge/arXiv-2025-red)
* seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models [[Paper]](https://neurips.cc/virtual/2024/104556) ![](https://img.shields.io/badge/NeurIPS-2025-red)
* MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features [[Paper]](https://arxiv.org/abs/2307.12698) ![](https://img.shields.io/badge/arXiv-2023-red)
* DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning [[Paper]](https://proceedings.mlr.press/v267/zhou25t.html) ![](https://img.shields.io/badge/ICML-2025-blue)
* DINO-world: Back to the Features: DINO as a Foundation for Video World Models [[Paper]](https://arxiv.org/abs/2507.19468) ![](https://img.shields.io/badge/arXiv-2025-red)
* DINO-Foresight: Looking into the Future with DINO [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/116713) ![](https://img.shields.io/badge/NeurIPS-2025-blue)
* World Models Group Latents: Learning Abstract World Models with a Group-Structured Latent Space [[Paper]](https://arxiv.org/abs/2506.01529) ![](https://img.shields.io/badge/arXiv-2025-red)

## Part 4: Object-centric World Models
* Object-centric Learning with Slot Attention [[Paper]](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html) ![](https://img.shields.io/badge/NeurIPS-2020-blue)
* SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models [[Paper]](https://iclr.cc/virtual/2023/poster/11779) ![](https://img.shields.io/badge/ICLR-2023-blue)
* FIOC-WM: Learning Interactive World Model for Object-Centric Reinforcement Learning [[Paper]](https://arxiv.org/abs/2511.02225) ![](https://img.shields.io/badge/arXiv-2025-red)
* G-SWM: Improving Generative Imagination in Object-Centric World Models [[Paper]](https://icml.cc/virtual/2020/poster/6582) ![](https://img.shields.io/badge/ICML-2020-blue)
* When Object-Centric World Models Meet Policy Learning: From Pixels to Policies, and Where It Breaks [[Paper]](https://arxiv.org/abs/2511.06136) ![](https://img.shields.io/badge/arXiv-2025-red)
* Objects matter: object-centric world models improve reinforcement learning in visually complex environments [[Paper]](https://arxiv.org/abs/2501.16443) ![](https://img.shields.io/badge/arXiv-2025-red)
* LSlotFormer: Object-Centric World Model for Language-Guided Manipulation [[Paper]](https://openreview.net/forum?id=29p13QihRM) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* Compositional OCL: Unifying Causal and Object-centric Representation Learning allows Causal Composition [[Paper]](https://iclr.cc/virtual/2025/37564) ![](https://img.shields.io/badge/ICLRW-2024-blue)
* Dyn-O: Building Structured World Models with Object-Centric Representations [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/117215) ![](https://img.shields.io/badge/NeurIPS-2025-blue)
* MEAD: Efficient Exploration and Discriminative World Model Learning with an Object-Centric Abstraction [[Paper]](https://iclr.cc/virtual/2025/poster/28750) ![](https://img.shields.io/badge/ICLR-2025-blue)
* Object-Centric Latent Action Learning [[Paper]](https://iclr.cc/virtual/2025/37546) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* Object-Centric Representations Generalize Better Compositionally with Less Compute [[Paper]](https://iclr.cc/virtual/2025/37565) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* CarFormer: Self-driving with Learned Object-Centric Representations [[Paper]](https://eccv.ecva.net/virtual/2024/poster/2110) ![](https://img.shields.io/badge/ECCV-2024-blue)

## Part 5: World Models for Robotics

### Manipulation
#### Visual Futrue Prediction
* RoboDreamer: Learning Compositional World Models for Robot Imagination [[Paper]](https://proceedings.mlr.press/v235/zhou24f.html) ![](https://img.shields.io/badge/ICML-2024-blue)
* Grounding Video Models to Actions through Goal Conditioned Exploration [[Paper]](https://arxiv.org/abs/2411.07223) ![](https://img.shields.io/badge/ICLR-2025-blue)
* ViPRA: Video Prediction for Robot Actions [[Paper]](https://arxiv.org/abs/2511.07732) ![](https://img.shields.io/badge/arXiv-2025.11-red)
* FlowDreamer: A RGB-D World Model with Flow-based Motion Representations for Robot Manipulation [[Paper]](https://arxiv.org/abs/2505.10075) ![](https://img.shields.io/badge/arXiv-2025.05-red)
* TesserAct: Learning 4D Embodied World Models [[Paper]](https://arxiv.org/abs/2504.20995) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* ORV: 4D Occupancy-centric Robot Video Generation [[Paper]](https://arxiv.org/abs/2506.03079) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2510.07313) ![](https://img.shields.io/badge/arXiv-2025.10-red)
* Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control [[Paper]](https://arxiv.org/abs/2506.01943) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation [[Paper]](https://arxiv.org/abs/2507.12898) ![](https://img.shields.io/badge/arXiv-2025.07-red)
* LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation [[Paper]](https://arxiv.org/abs/2505.11528) ![](https://img.shields.io/badge/arXiv-2025.05-red)
* KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models [[Paper]](https://arxiv.org/abs/2509.21027) ![](https://img.shields.io/badge/arXiv-2025.09-red)

#### Latent Action State Imagination
* Learning Latent Dynamics for Planning from Pixels [[Paper]](https://arxiv.org/abs/1811.04551) ![](https://img.shields.io/badge/ICML-2019-blue)
* FLARE: Robot Learning with Implicit World Modeling [[Paper]](https://arxiv.org/abs/2505.15659) ![](https://img.shields.io/badge/arXiv-2025.05-red)
* AdaWorld: Learning Adaptable World Models with Latent Actions [[Paper]](https://arxiv.org/abs/2503.18938) ![](https://img.shields.io/badge/ICML-2025-blue)
* DyWA: Dynamic World Adaptation for Generalizable World Models [[Paper]](https://arxiv.org/abs/2503.16806) ![](https://img.shields.io/badge/arXiv-2025.03-red)
* OSVI-WM: One-Shot Visual Imitation via World Models [[Paper]](https://arxiv.org/abs/2505.20425) ![](https://img.shields.io/badge/arXiv-2025.05-red)
* DEMO$^3$: Multi-Stage Manipulation with Demonstration-Augmented Reward, Policy, and World Model Learning [[Paper]](https://arxiv.org/abs/2503.01837) ![](https://img.shields.io/badge/ICML-2025-blue)
* Strengthening Generative Robot Policies through Predictive World Modeling [[Paper]](https://arxiv.org/abs/2502.00622) ![](https://img.shields.io/badge/arXiv-2025.02-red)
* In-Context Policy Improvement for Contact-Rich Manipulation with Pretrained Generative Models [[Paper]](https://arxiv.org/abs/2508.15021) ![](https://img.shields.io/badge/CoRL-2025-blue)




#### Control-Oriented Planning
* Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS) [[Paper]](https://arxiv.org/abs/1805.12114) ![](https://img.shields.io/badge/NeurIPS-2018-blue)
* TD-MPC: Temporal Difference Learning for Model Predictive Control [[Paper]](https://arxiv.org/abs/2203.04955) ![](https://img.shields.io/badge/ICML-2022-blue)
* TD-MPC2: Scalable, Robust World Models for Continuous Control [[Paper]](https://arxiv.org/abs/2310.16828) ![](https://img.shields.io/badge/ICLR-2024-blue)
* Time-Aware World Model for Adaptive Prediction and Control (TAWM) [[Paper]](https://arxiv.org/abs/2506.08441) ![](https://img.shields.io/badge/ICML-2025-blue)
* Planning with Diffusion for Flexible Behavior Synthesis [[Paper]](https://arxiv.org/abs/2205.09991) ![](https://img.shields.io/badge/ICML-2022-blue)
* Decision Diffuser: Is Conditional Generative Modeling All You Need for Decision-Making? [[Paper]](https://arxiv.org/abs/2211.15657) ![](https://img.shields.io/badge/ICLR-2023-blue)
* Potential-Based Diffusion Motion Planning [[Paper]](https://arxiv.org/abs/2407.06169) ![](https://img.shields.io/badge/ICML-2024-blue)
* HiP: Compositional Foundation Models for Hierarchical Planning [[Paper]](https://arxiv.org/abs/2309.08587) ![](https://img.shields.io/badge/NeurIPS-2023-blue)
* PIVOT-R: Primitive-Driven Waypoint-Aware World Model for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2410.10394) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
* Hdflow: Hierarchical diffusion-flow planning for long-horizon robotic assembl [[Paper]](https://openreview.net/forum?id=nokbt6AbcM) ![](https://img.shields.io/badge/NeurIPS__WS-2025-blue)
* ManipDreamer: Learning Manipulation World Models with Action-Tree Supervisions [[Paper]](https://arxiv.org/abs/2504.16464) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* RoboHorizon: An LLM-Assisted Multi-View World Model for Long-Horizon Robotic Manipulation [[Paper]](https://arxiv.org/abs/2501.06605) ![](https://img.shields.io/badge/arXiv-2025.01-red)
* Mobile Manipulation with Active Inference [[Paper]](https://arxiv.org/abs/2507.17338) ![](https://img.shields.io/badge/arXiv-2025.07-red)

#### World Action Modeling

* Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2508.05635) ![](https://img.shields.io/badge/arXiv-2025.08-red) 
* WoW: Towards a World Omniscient World Model through Embodied Interaction [[Paper]](https://arxiv.org/abs/2509.22642) ![](https://img.shields.io/badge/arXiv-2025.09-red) 
* PAR: Physical Autoregressive Model for Robotic Manipulation without Action Pretraining [[Paper]](https://arxiv.org/abs/2508.09822) ![](https://img.shields.io/badge/arXiv-2025.08-red) 
* iMoWM: Taming Interactive Multi-Modal World Model for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2510.09036) ![](https://img.shields.io/badge/arXiv-2025.10-red) 
* PhysicalAgent: Towards General Cognitive Robotics with Foundation World Models [[Paper]](https://arxiv.org/abs/2509.13903) ![](https://img.shields.io/badge/arXiv-2025.09-red) 
* World4Omni: A Zero-Shot Framework from Image Generation World Model to Robotic Manipulation [[Paper]](https://arxiv.org/abs/2506.23919v1) ![](https://img.shields.io/badge/arXiv-2025.06-red) 
### Navigation
* NWM: Navigation World Models [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Bar_Navigation_World_Models_CVPR_2025_paper.pdf) ![会议徽章](https://img.shields.io/badge/CVPR-2025-blue)
* Kinodynamic: Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent World Models [[Paper]](https://arxiv.org/abs/2509.26339) ![会议徽章](https://img.shields.io/badge/RSS%20Workshop25-blue)
* Neuro-Symbolic: Perspective-Shifted Neuro-Symbolic World Models for Socially-Aware Robot Navigation [[Paper]](https://arxiv.org/abs/2503.20425) ![会议徽章](https://img.shields.io/badge/arXiv-2025.03-red)
* Abs-Sim2Rea: Abstract Sim2Real through Approximate Information States [[Paper]](https://openreview.net/pdf?id=FoDmB0ixdt) ![会议徽章](https://img.shields.io/badge/NeurIPS%20Workshop-2023-blue)
* Unified WM: Unified world models: Memory-augmented planning and foresight for visual navigation [[Paper]](https://arxiv.org/pdf/2510.08713v1) ![会议徽章](https://img.shields.io/badge/arXiv-2025.10-red)
* Scene Graph World: Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation [[Paper]](https://arxiv.org/abs/2508.06990) ![会议徽章](https://img.shields.io/badge/arXiv-2025.08-red)
* Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension [[Paper]](https://arxiv.org/abs/2412.03704) ![会议徽章](https://img.shields.io/badge/arXiv-2024.12-red)
* X-Mobility: End-To-End Generalizable Navigation via World Modeling [[Paper]](https://arxiv.org/pdf/2410.17491v3) ![会议徽章](https://img.shields.io/badge/arXiv-2024.10-red)
* WMNav: Integrating Vision-Language Models into World Models for Object Goal Navigation [[Paper]](https://arxiv.org/abs/2503.02247) ![会议徽章](https://img.shields.io/badge/IROS-2025-blue)
* RECON: Rapid Exploration for Open-World Navigation with Latent Goal Models [[Paper]](https://arxiv.org/abs/2104.05859) ![会议徽章](https://img.shields.io/badge/arXiv-2021.04-red)
* NaVi-WM: Deductive Chain-of-Thought Augmented Socially-aware Robot Navigation World Model [[Paper]](https://arxiv.org/abs/2510.23509) ![会议徽章](https://img.shields.io/badge/arXiv-2025.10-red)
* MindJourney: Test-Time Scaling with World Models for Spatial Reasoning [[Paper]](https://arxiv.org/abs/2507.12508) ![会议徽章](https://img.shields.io/badge/arXiv-2025.07-red)
* NavMorph: A Self-Evolving World Model for Vision-and-Language Navigation in Continuous Environments [[Paper]](https://arxiv.org/abs/2506.23468) ![会议徽章](https://img.shields.io/badge/arXiv-2025.06-red)
* Deep Active Inference with Diffusion Policy and Multiple Timescale World Model for Real-World Exploration and Navigation [[Paper]](https://arxiv.org/pdf/2510.23258v1) ![会议徽章](https://img.shields.io/badge/arXiv-2025.10-red)
* World Model Implanting for Test-time Adaptation of Embodied Agents [[Paper]](https://arxiv.org/abs/2509.03956) ![会议徽章](https://img.shields.io/badge/arXiv-2025.09-red)
* After Persistent Embodied WM: Learning 3D Persistent Embodied World Models [[Paper]](https://arxiv.org/abs/2505.05495) ![会议徽章](https://img.shields.io/badge/arXiv-2025.05-red)

### Policy Learning
* WMPO: World Model-based Policy Optimization for Vision-Language-Action Models [[Paper]](https://arxiv.org/abs/2511.09515) ![](https://img.shields.io/badge/arXiv-2025.11-red)
* World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation [[Paper]](https://arxiv.org/abs/2509.19080) ![](https://img.shields.io/badge/arXiv-2025.09-red)
* DAWM: Diffusion Action World Models for Offline Reinforcement Learning via Action-Inferred Transitions [[Paper]](https://arxiv.org/abs/2509.19538) ![](https://img.shields.io/badge/arXiv-2025.09-red)
* Latent Action World Models for Control with Unlabeled Trajectories [[Paper]](https://arxiv.org/abs/2512.10016) ![](https://img.shields.io/badge/arXiv-2025.12-red)
* Prelar: World model pre-training with learnable action representation [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-73337-6_11) ![](https://img.shields.io/badge/ECCV-2024-blue)
* World models can leverage human videos for dexterous manipulation [[Paper]](https://arxiv.org/abs/2512.13644) ![](https://img.shields.io/badge/arXiv-2025.12-red)
* TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos [[Paper]](https://arxiv.org/abs/2511.21690) ![](https://img.shields.io/badge/arXiv-2025.11-red)
* 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model [[Paper]](https://arxiv.org/abs/2506.06199) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* Ctrl-World: A Controllable Generative World Model for Robot Manipulation [[Paper]](https://arxiv.org/abs/2510.10125) ![](https://img.shields.io/badge/arXiv-2025.10-red)
* Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics [[Paper]](https://arxiv.org/abs/2501.10100) ![](https://img.shields.io/badge/arXiv-2025.01-red)
* WorldGym: World Model as An Environment for Policy Evaluation [[Paper]](https://arxiv.org/abs/2506.00613) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* Worldeval: World model as realworld robot policies evaluator [[Paper]](https://arxiv.org/abs/2505.19017) ![](https://img.shields.io/badge/arXiv-2025.05-red)
### Locomotion
* WMP: World Model-based Perception for Visual Legged Locomotion [[Paper]](https://wmp-loco.github.io/) ![会议徽章](https://img.shields.io/badge/ICRA-2025-blue)  
* WMR: Learning Humanoid Locomotion with World Model Reconstruction [[Paper]](https://arxiv.org/abs/2502.16230) ![会议徽章](https://img.shields.io/badge/arXiv-2025.02-red)
* World Model Predictive Control for Robust Locomotion [[Paper]](https://arxiv.org/abs/2408.14472) ![会议徽章](https://img.shields.io/badge/RSS-2024-blue)
* Ego: Ego-Vision World Model for Humanoid Contact Planning [[Paper]](https://ego-vcp.github.io/) ![会议徽章](https://img.shields.io/badge/arXiv-2025.10-red)

## Part 6: World Models for Autonomous Driving

### Predictive Modeling
* UniWorld: Autonomous Driving Pre-training via World Models [[Paper]](https://arxiv.org/abs/2308.07234) ![](https://img.shields.io/badge/arXiv-2023.8-red)
* HoloDrive: Holistic 2D-3D Multi-Modal Street Scene Generation for Autonomous Driving [[Paper]](https://arxiv.org/abs/2412.01407) ![](https://img.shields.io/badge/arXiv-2024.12-red)
* WoVoGen: World Volume-Aware Diffusion for Controllable Multi-camera Driving Scene Generation [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72989-8_19/) ![](https://img.shields.io/badge/ECCV-2024-blue)
* NeMo: Neural Volumetric World Models for Autonomous Driving [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72643-9_12) ![](https://img.shields.io/badge/ECCV-2024-blue)
* UnO: Unsupervised Occupancy Fields for Perception and Forecasting [[Paper]](https://ieeexplore.ieee.org/document/10658015) ![](https://img.shields.io/badge/CVPR-2024-blue)
* Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion [[Paper]](https://openreview.net/forum?id=Psl75UCoZM) ![](https://img.shields.io/badge/ICLR-2024-blue)
* Renderworld: World Model with Self-Supervised 3D Label [[Paper]](https://ieeexplore.ieee.org/abstract/document/11127609) ![](https://img.shields.io/badge/ICRA-2025-blue)
* DriveDreamer4D: World Models Are Effective Data Machines for 4D Driving Scene Representation [[Paper]](https://ieeexplore.ieee.org/abstract/document/11094017) ![](https://img.shields.io/badge/CVPR-2025-blue)

### Action-Conditioned Imagination
* GAIA-1: A Generative World Model for Autonomous Driving [[Paper]](https://arxiv.org/abs/2309.17080) ![](https://img.shields.io/badge/arXiv-2023.09-red)
* WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens [[Paper]](https://arxiv.org/abs/2401.09985) ![](https://img.shields.io/badge/arXiv-2024.01-red)
* OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72624-8_4) ![](https://img.shields.io/badge/ECCV-2024-blue)
* DriveDreamer: Towards Real-World-Drive World Models for Autonomous Driving [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-73195-2_4) ![](https://img.shields.io/badge/ECCV-2024-blue)
* Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2024/hash/a6a066fb44f2fe0d36cf740c873b8890-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
* Drive-WM: Driving Into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving [[Paper]](https://ieeexplore.ieee.org/abstract/document/10655190) ![](https://img.shields.io/badge/CVPR-2024-blue)
* DriveWorld: 4D Pre-Trained Scene Understanding via World Models for Autonomous Driving [[Paper]](https://ieeexplore.ieee.org/document/10655336) ![](https://img.shields.io/badge/CVPR-2024-blue)
* InfinityDrive: Breaking Time Limits in Driving World Models [[Paper]](https://arxiv.org/abs/2412.01522) ![](https://img.shields.io/badge/arXiv-2024.12-red)
* DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/33130) ![](https://img.shields.io/badge/AAAI-2025-blue)


### Decision-Centric Integration
* Think2Drive: Efficient Reinforcement Learning by Thinking with Latent World Model for Autonomous Driving (in CARLA-V2) [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-72995-9_9) ![](https://img.shields.io/badge/ECCV-2024-blue)
* Doe-1: Closed-Loop Autonomous Driving with Large World Model [[Paper]](https://arxiv.org/abs/2412.09627) ![](https://img.shields.io/badge/arXiv-2024.12-red)
* AdaWM: Adaptive World Model based Planning for Autonomous Driving [[Paper]](https://openreview.net/forum?id=NEu8wgPctU) ![](https://img.shields.io/badge/ICLR-2025-blue)
* DriveVLA-W0: World Models Amplify Data Scaling Law in Autonomous Driving [[Paper]](https://arxiv.org/abs/2510.12796) ![](https://img.shields.io/badge/arXiv-2025.10-red)

## Part 7: World Models for Science

### Social Science and Socioeconomic Systems
* SWM: Social World Models [[Paper]](https://arxiv.org/abs/2509.00559) ![](https://img.shields.io/badge/arXiv-2025.09-red)
* SWM-AP: Social World Model-Augmented Mechanism Design Policy Learning [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/115517) ![](https://img.shields.io/badge/NeurIPS-2025-blue)
* SocioVerse: A World Model for Social Simulation Powered by LLM Agents and a Pool of 10 Million Real-World Users [[Paper]](https://arxiv.org/abs/2504.10157) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* TwinMarket: A Scalable Behavioral and Social Simulation for Financial Markets [[Paper]](https://neurips.cc/virtual/2025/loc/san-diego/poster/116676) ![](https://img.shields.io/badge/NeurIPS-2025-blue)
* A Virtual Reality-Integrated System for Behavioral Analysis in Neurological Decline [[Paper]](https://iclr.cc/virtual/2025/37544) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* Bidding for Influence: Auction-Driven Diffusion Image Generation [[Paper]](https://icml.cc/virtual/2025/49321) ![](https://img.shields.io/badge/ICMLW-2025-blue)
* Typhon: Effectively Designing 2-Dimensional Sequence Models for Multivariate Time Series [[Paper]](https://iclr.cc/virtual/2025/37514) ![](https://img.shields.io/badge/ICLRW-2025-blue)

### Physical and Natural Sciences
* CellFlux: Simulating Cellular Morphology Changes via Flow Matching [[Paper]](https://icml.cc/virtual/2025/poster/46535) ![](https://img.shields.io/badge/ICML-2025-blue)
* ODesign: A World Model for Biomolecular Interaction Design [[Paper]](https://arxiv.org/abs/2510.22304) ![](https://img.shields.io/badge/arXiv-2025.10-red)
* Medical world model: Generative simulation of tumor evolution for treatment planning [[Paper]](https://arxiv.org/abs/2506.02327) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* CheXWorld: Exploring Image World Modeling for Radiograph Representation Learning [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/34038) ![](https://img.shields.io/badge/CVPR-2025-blue)
* Xray2Xray: World Model from Chest X-rays with Volumetric Context [[Paper]](https://arxiv.org/abs/2506.03079) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* EchoWorld: Learning Motion-Aware World Models for Echocardiography Probe Guidance [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/34389) ![](https://img.shields.io/badge/CVPR-2025-blue)
* PINT: Physics-Informed Neural Time Series Models with Applications to Long-term Inference on WeatherBench 2m-Temperature Data [[Paper]](https://iclr.cc/virtual/2025/37537) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* Reconstructing Dynamics from Steady Spatial Patterns with Partial Observations [[Paper]](https://iclr.cc/virtual/2025/35783) ![](https://img.shields.io/badge/ICLRW-2025-blue)
* HEP-JEPA: A foundation model for collider physics [[Paper]](https://iclr.cc/virtual/2025/33136) ![](https://img.shields.io/badge/ICLRW-2025-blue)

## Part 8: World Models for Virtual Game Simulation

### Pixel-Level Observation Prediction
* Diffusion For World Modeling: Visual Details Matter In Atari [[Paper]](https://neurips.cc/virtual/2024/poster/95428) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
* GameNGen: Diffusion Models Are Real-Time Game Engines [[Paper]](https://iclr.cc/virtual/2025/poster/29770) ![](https://img.shields.io/badge/ICLR-2025-blue)
* Oasis: A Universe In A Transformer [[Paper]](https://oasis-model.github.io/) ![](https://img.shields.io/badge/github-2025.10-red)
* AnimeGamer: Infinite Anime Life Simulation with Next Game State Prediction [[Paper]](https://iccv.thecvf.com/virtual/2025/poster/1518) ![](https://img.shields.io/badge/ICCV-2025-blue)
* Matrix-Game 2.0: An Open-Source, Real-Time, And Streaming Interactive World Model [[Paper]](https://arxiv.org/abs/2508.13009) ![](https://img.shields.io/badge/arXiv-2025.08-red)
* Matrix-Game: Interactive World Foundation Model [[Paper]](https://arxiv.org/abs/2506.18701) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* RealPlay: From Virtual Games To Real-World Play [[Paper]](https://arxiv.org/abs/2506.18901) ![](https://img.shields.io/badge/arXiv-2025.06-red)
* GameFactory: Creating New Games With Generative Interactive Videos [[Paper]](https://arxiv.org/abs/2501.08325) ![](https://img.shields.io/badge/arXiv-2025.01-red)

### 3D Mesh-Level Observation Prediction
* HunyuanWorld 1.0: Generating Immersive, Explorable, And Interactive 3D Worlds From Words Or Pixels [[Paper]](https://arxiv.org/abs/2507.21809) ![](https://img.shields.io/badge/arXiv-2025.08-red)
* Matrix-3D: Omnidirectional Explorable 3D World Generation [[Paper]](https://arxiv.org/abs/2508.08086) ![](https://img.shields.io/badge/arXiv-2025.08-red)

## Part 9: World Models for GUI-Based Agents
* NeuralOS: Towards Simulating Operating Systems via Neural Generative Models [[Paper]](https://arxiv.org/abs/2507.08800) ![](https://img.shields.io/badge/arXiv-2025.07-red)
* ViMo: A Generative Visual GUI World Model for App Agents [[Paper]](https://arxiv.org/abs/2504.13936) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* Unlocking Smarter Device Control: Foresighted Planning With A World Model-Driven Code Execution Approach [[Paper]](https://arxiv.org/abs/2505.16422) ![](https://img.shields.io/badge/arXiv-2025.05-red)
* R-WoM: Retrieval-augmented World Model For Computer-use Agents [[Paper]](https://arxiv.org/abs/2510.11892) ![](https://img.shields.io/badge/arXiv-2025.10-red)
* WKM: Agent Planning With World Knowledge Model [[Paper]](https://neurips.cc/virtual/2024/poster/93977) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
* WebDreamer: Is Your LLM Secretly A World Model Of The Internet? Model-Based Planning For Web Agents [[Paper]](https://openreview.net/forum?id=c6l7yA0HSq) ![](https://img.shields.io/badge/TMLR-2025-blue)
* WMA: Web Agents With World Models: Learning And Leveraging Environment Dynamics In Web Navigation [[Paper]](https://iclr.cc/virtual/2025/poster/28448) ![](https://img.shields.io/badge/ICLR-2025-blue)
* WebSynthesis: World-Model-Guided MCTS for Efficient WebUI-Trajectory Synthesis [[Paper]](https://arxiv.org/abs/2507.04370) ![](https://img.shields.io/badge/arXiv-2025.07-red)
* WebEvolver: Enhancing Web Agent Self-Improvement with Coevolving World Model [[Paper]](https://arxiv.org/abs/2504.21024) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents [[Paper]](https://arxiv.org/abs/2504.15785) ![](https://img.shields.io/badge/arXiv-2025.04-red)
* SimuRA: A World-Model-Driven Simulative Reasoning Architecture for General Goal-Oriented Agents [[Paper]](https://arxiv.org/abs/2507.23773) ![](https://img.shields.io/badge/arXiv-2025.07-red)
* Dyna-Think: Synergizing Reasoning, Acting, and World Model Simulation in AI Agents [[Paper]](https://arxiv.org/abs/2506.00320) ![](https://img.shields.io/badge/arXiv-2025.06-red)

## Part 10: Interpretable and Trustworthy World Models
* General agents need world models [[Paper]](https://icml.cc/virtual/2025/poster/44620) ![会议徽章](https://img.shields.io/badge/ICML-2025-blue)
* GPT：A Causal World Model Underlying Next Token Prediction: Exploring GPT in a Controlled Environment [[Paper]](https://arxiv.org/abs/2412.07446) ![会议徽章](https://img.shields.io/badge/arXiv-2024.12-red)
* Stan&Terry：Transformers Use Causal World Models in Maze-Solving Tasks [[Paper]](https://iclr.cc/virtual/2025/37549) ![会议徽章](https://img.shields.io/badge/ICLR-2025-blue)
* MLP：When Do Neural Networks Learn World Models? [[Paper]](https://arxiv.org/abs/2502.09297) ![会议徽章](https://img.shields.io/badge/arXiv-2025.02-red)
* LLMs：Linear spatial world models emerge in large language models. [[Paper]](https://arxiv.org/abs/2506.02996) ![会议徽章](https://img.shields.io/badge/arXiv-2025.06-red)
* Mistral：Revisiting the othello world model hypothesis. [[Paper]](https://arxiv.org/abs/2503.04421) ![会议徽章](https://img.shields.io/badge/arXiv-2025.03-red)
* GPT-2-style transformer：Scaling laws for pre-training agents and world models. [[Paper]](https://arxiv.org/abs/2411.04434) ![会议徽章](https://img.shields.io/badge/arXiv-2024.11-red)
* VAE：How hard is it to confuse a world model? [[Paper]](https://arxiv.org/abs/2510.21232) ![会议徽章](https://img.shields.io/badge/arXiv-2025.10-red)
* DeepSeek：Utilizing world models for adaptively covariate acquisition under limited budget for causal decision making. [[Paper]](https://iclr.cc/virtual/2025/37518) ![会议徽章](https://img.shields.io/badge/ICLR-2025-blue)
## Part 11: Benchmark of World Models

### Benchmark Datasets \& Evaluation Metrics
* Frozen in time: A joint video and image encoder for end-to-end retrieval. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/html/Bain_Frozen_in_Time_A_Joint_Video_and_Image_Encoder_for_ICCV_2021_paper.html) ![会议徽章](https://img.shields.io/badge/ICCV-2021-blue)
* Panda-70m: Captioning 70m videos with multiple cross-modality teachers. [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Panda-70M_Captioning_70M_Videos_with_Multiple_Cross-Modality_Teachers_CVPR_2024_paper.html) ![会议徽章](https://img.shields.io/badge/CVPR-2024-blue)
* Ego4d: Around the world in 3,000 hours of egocentric video. [[Paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Grauman_Ego4D_Around_the_World_in_3000_Hours_of_Egocentric_Video_CVPR_2022_paper.html) ![会议徽章](https://img.shields.io/badge/CVPR-2022-blue)
* Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. [[Paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Miech_HowTo100M_Learning_a_Text-Video_Embedding_by_Watching_Hundred_Million_Narrated_ICCV_2019_paper.html) ![会议徽章](https://img.shields.io/badge/ICCV-2019-blue)
* Worldscore:A unified evaluation benchmark for world generation. [[Paper]](https://arxiv.org/abs/2504.00983) ![会议徽章](https://img.shields.io/badge/arXiv-2025.04-red)
* Open x-embodiment: Robotic Learning Datasets and RT-X Models : Open X-Embodiment Collaboration0. [[Paper]](https://ieeexplore.ieee.org/document/10611477) ![会议徽章](https://img.shields.io/badge/ICRA-2024-blue)
* Room-Across-Room: Multilingual vision-and-language navigation with dense spatiotemporal grounding. [[Paper]](https://aclanthology.org/2020.emnlp-main.356/) ![会议徽章](https://img.shields.io/badge/EMNLP-2020-blue)
* Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. [[Paper]](https://arxiv.org/abs/2505.09694) ![会议徽章](https://img.shields.io/badge/arXiv-2025.05-red)
* Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. [[Paper]](https://arxiv.org/abs/1910.10897) ![会议徽章](https://img.shields.io/badge/arXiv-2019.10-red)
* nuscenes: A multimodal dataset for autonomous driving. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/html/Caesar_nuScenes_A_Multimodal_Dataset_for_Autonomous_Driving_CVPR_2020_paper.html) ![会议徽章](https://img.shields.io/badge/CVPR-2020-blue)
* ACT-Bench: Towards Action Controllable World Models for Autonomous Driving. [[Paper]](https://arxiv.org/abs/2412.05337) ![会议徽章](https://img.shields.io/badge/arXiv-2024.12-red)
* Jump cell painting dataset: morphological impact of 136,000 chemical and genetic perturbations. [[Paper]](https://www.biorxiv.org/content/10.1101/2023.03.23.534023v1) ![会议徽章](https://img.shields.io/badge/BioRxiv-2023-blue)
* Random Forest Classifier：A machine learning modelto predict hepatocellular carcinoma response to transcatheter arterial chemoembolization.  [[Paper]](https://pubs.rsna.org/doi/full/10.1148/ryai.2019180021) ![会议徽章](https://img.shields.io/badge/Radiology-2019-purple)
* The arcade learning environment: An evaluation platform for general agents. [[Paper]](https://arxiv.org/abs/1207.4708) ![会议徽章](https://img.shields.io/badge/arXiv-2012-red)
* Minerl: A large-scale dataset of minecraft demonstrations. [[Paper]](https://www.ijcai.org/proceedings/2019/339) ![会议徽章](https://img.shields.io/badge/IJCAI-2019-blue)
* OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. [[Paper]](https://neurips.cc/virtual/2024/poster/97468) ![会议徽章](https://img.shields.io/badge/NeurIPS-2024-blue)
* Windows agent arena: Evaluating multi-modal OS agents at scale.  [[Paper]](https://icml.cc/virtual/2025/poster/45035) ![会议徽章](https://img.shields.io/badge/ICML-2025-blue)
### Physics Engines \& Simulation Platforms
![image](./assets/sim_list.png)
* Gazebo: Design and use paradigms for Gazebo, an open-source multi-robot simulator [[Paper]](https://ieeexplore.ieee.org/document/1389727) ![](https://img.shields.io/badge/IROS-2004.09-blue)
* WebotsTM: Professional Mobile Robot Simulation [[Paper]](https://arxiv.org/abs/cs/0412052) ![](https://img.shields.io/badge/arXiv-2004.12-red)
* Bullet Physics SDK [[github]](https://github.com/bulletphysics/bullet3) ![](https://img.shields.io/badge/github-2015-red) [[Blog]](https://pybullet.org/wordpress/) ![](https://img.shields.io/badge/blog-2021.04-red)
* PyBullet Quickstart Guide [[SDK]](https://raw.githubusercontent.com/bulletphysics/bullet3/master/docs/pybullet_quickstartguide.pdf) ![](https://img.shields.io/badge/SDK-2016-red) [[Blog]](https://pybullet.org/wordpress/) ![](https://img.shields.io/badge/blog-2021.04-red)
* MuJoCo [[github]](https://github.com/google-deepmind/mujoco) ![](https://img.shields.io/badge/github-2021-red) [[Blog]](https://mujoco.readthedocs.io/en/stable/overview.html) ![](https://img.shields.io/badge/blog-2021-red)
* Simbody: multibody dynamics for biomedical research [[Paper]](https://www.sciencedirect.com/science/article/pii/S2210983811000241) ![](https://img.shields.io/badge/IUTAM-2011-blue)
* NVIDIA PhysX SDK [[SDK]](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/) ![](https://img.shields.io/badge/SDK-2025.07-red)
* Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning [[Paper]](https://arxiv.org/abs/2108.10470) ![](https://img.shields.io/badge/arXiv-2021.08-red) [[Blog]](https://sites.google.com/view/isaacgym-nvidia) ![](https://img.shields.io/badge/blog-2021.08-red)
* Omniverse [[Notion]](https://docs.omniverse.nvidia.com/dev-overview/latest/index.html) ![](https://img.shields.io/badge/Notion-Docs-red) [[SDK]](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html) ![](https://img.shields.io/badge/SDK-Docs-red)
* Genesis: A Generative and Universal Physics Engine for Robotics and Beyond [[github]](https://genesis-embodied-ai.github.io/) ![](https://img.shields.io/badge/github-2024-red) [[SDK]](https://genesis-world.readthedocs.io/en/latest/) ![](https://img.shields.io/badge/SDK-2024-red)
## Part 12: Performance Comparison
![image](./assets/world_score.png)
## Citation
If you find this work useful, welcome to cite us.
```bib


```

<!-- omit in toc -->
## ⭐ Star History

<a href="https://github.com/JiahuaDong/Awesome-World-Models">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=JiahuaDong/Awesome-World-Models&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=JiahuaDong/Awesome-World-Models&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=JiahuaDong/Awesome-World-Models&type=Date" />
 </picture>
</a>
