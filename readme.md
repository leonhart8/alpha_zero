# Alpha Zero 

## Authors :
<ul>
    <li>Zhengyang LAN</li>
    <li>Luqman FERDJANI</li>
</ul>

## Goal of the repository

This repository was created in the context of the "Sequential Decision Making" class of the University of Lille's
Master's in Data Science. The final examination consisted in choosing among a selection of research papers in order to :

<ul>
<li>Understand the selected paper</li>
<li>Draw comparisons between the algorithms presented in the paper and what was seen in class</li>
<li>Implement the algorithms, comment them and benchmark</li>
</ul>

## Game choices

We chose a paper exposing the AlphaZero algorithm, a novel reinforcement learning algorithm first applied in the game
of go that provided state of the art performances using only self play.

However we decided not to apply the paper's algorithm to go, as the game's complicated nature requires a very big amount
of computations. Indeed the original implementation used several GPUs and TPUs in parallel in order to speed up the 
learning, and still took several weeks of training to reach satisfactory results.

Also we are not masters of Deep Learning, and the proposed architecture required knowing how to program asynchronous Deep Learning
networks of 20 residual blocks and convolutions.

As such we decided to keep things simple at start and focus on easier games with simple deep learning
architectures while keeping true to the core philosophy of Alpha Zero : using neural networks to learn both
value function and policy, using MCTS in order to both improve and evaluate the policy, all the while using only self
play without any handcrafted features and expert domain knowledge

## Roadmap

We conducted the project in several steps :

<ol>
<li>Reading the paper thoroughly and looking up resources on model based RL and MCTS</li>
<li>Freshen up on deep learning</li>
<li>Applying AlphaZero to a very simple game first : tic tac toe</li>
<li>Applying AlphaZero to harder games</li>
<li>Finding ways to evaluate our implementation of AlphaZero</li>
</ol>

The last step is not so obvious. One could first think of benchmarking against previous iterations of the algorithm such
as AlphaGo Fan and AlphaGo Lee. However these implementations required domain expert knowledge and are not so obvious to
implement. Indeed one of the advantages of AlphaZero is that its non reliance on domain expert knowledge makes its implementation
easy.

## Implemented games

### Tic Tac Toe

TO DO

## TO DO

<ul>
<li>Implement AlphaZero for Tic Tac Toe</li>
<li>Implement AlphaZero for a harder game</li>
<li>Find a way to benchmark our implementation of Alpha Zero</li>
</ul>

