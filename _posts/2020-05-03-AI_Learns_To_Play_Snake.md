---
layout: post
title: "AI Learns To Play Snake"
date: 2020-05-03
display_banner: true
image: "/assets/blogs/2020-05-03/imgs/Snake-AI.png"
reading_time: false
github:
    has_github_link: true
    repository_url: https://github.com/Chrispresso/SnakeAI
---

This is a re-upload from my old site. But for anyone who has not seen it, or if you want to see it again, here you go!

In the [last post](https://chrispresso.github.io/Lets_Make_A_Genetic_Algorithm) I discussed what a Genetic Algorithm is, how to implement one, and how it can be used to find the minimum of a function. Of course, it can be a little boring to just watch a function converge to a global minimum. Because of this I decided to show how you can use a Genetic Algorithm to create an AI that learns to play Snake! If you just want a video, skip to the bottom. If you want to play around with this code, check out my [repo on github](https://github.com/Chrispresso/SnakeAI). 

- [What is Snake?](#what-is-snake)
- [How is this Accomplished?](#how-is-this-accomplished)
- [The Neural Network](#the-neural-network)
    - [What is the Purpose of the NN?](#what-is-the-purpose-of-the-nn)
    - [How Does the NN Work?](#how-does-the-nn-work)
    - [What Do Input Nodes Receive?](#what-do-input-nodes-receive)
    - [Vision](#vision)
    - [What are Hidden Layers for?](#what-are-hidden-layers-for)
    - [Bringing it all together](#bringing-it-all-together)
- [The Genetic Algorithm](#the-genetic-algorithm)
    - [What is the Purpose of the GA?](#what-is-the-purpose-of-the-ga)
    - [How Does the GA Work?](#how-does-the-ga-work)
- [Confused?](#confused)
- [Seeing it in Action](#seeing-it-in-action)
- [Conclusion](#conclusion)

## What is Snake?
For anyone unfamiliar with the game of Snake you really have one objective: eat apples. Anytime you eat an apple, you grow in length. If you run into yourself or the wall, you lose. Basic directions from the classic game consist of: up, down, left, right.

## How is this Accomplished?
Using a Neural Network and a Genetic Algorithm!

## The Neural Network
### What is the Purpose of the NN?
The Neural Network serves a main purpose of deciding what a Snake will do. By taking in portions of the environment as input, the Snake will be choose a direction to go.

### How Does the NN Work?
Below is a Neural Network for a snake. Each snake will receive a Neural Network of the same architecture (same nodes in input layer, hidden layers and output layer). In this case we have two hidden layers. The input nodes are the "sensor" inputs that the snake receives from the surroundings. These nodes feed through the Network to the output layer where one decision is made: the direction to move.

![SnakeAI neural network visualization](/assets/blogs/2020-05-03/imgs/snake_nn_viz.png){: .center-image}
*<center>Figure 1. Snake Neural Network</center>*

Red lines connecting two nodes mean there is a negative, or inhibiting, connection between them. Blue lines, however, mean there is a positive connection. Input and output layer nodes can be either green or white. If the input nodes are green it means there is a non-zero value associated with one of the sensors. Only one node in the output layer can be green at a time. Hidden layers vary from dark gray to bright blue depending on how much activation is seen in that node.

### What Do Input Nodes Receive?
Below is a breakout of three types of inputs that the Network can receive: vision, direction and tail direction.

![SnakeAI inputs](/assets/blogs/2020-05-03/imgs/snake_inputs.png){: .center-image}
*<center>Figure 2. Snake Neural Network with Vision</center>*

Direction and tail direction are both one-hot encoded variables. This means that although they can take on multiple values, in this case four, they can only have one node active at a time. This allows for up, down, left and right to have their own weights. The same encoding is used for the output layer and possible values and their meanings can be seen below, where each column would be a different possible setup:

![one-hot encoded directions](/assets/blogs/2020-05-03/imgs/one-hot_encoding.png){: .center-image}
*<center>Figure 3. One-hot Encoded Directions and<br>Their Meaning</center>*

Based on Figure 2, you can determine that the <b>direction is left</b>, <b>tail direction is up</b> and the snake is <b>deciding to go left</b>.

### Vision
I've talked about direction and tail direction, but there is still one more type of input: vision. So what is it? Well I've allowed the snake to see in either 4, 8 or 16 directions. An illustration of this can be seen below:

![16-direction](/assets/blogs/2020-05-03/imgs/direction16.png){: .center-image}
*<center>Figure 4. Depiction of 4, 8, and 16 Direction Vision</center>*

Basically this means is the snake is part of Star Wars Resistance. Alright, this might seem complicated at first, but it's not too bad. If the snake sees is 4 directions, then it sees only where the blue lines are (up, down, left, right). If the snake sees in 8 directions, then it sees where green and blue lines are. If the snake sees in 16 directions then it simply sees in all yellow, green and blue lines.

Each vision line can see one of three things:
1. Distance to wall
2. Does it see an apple?\*
3. Does it see itself?\*

An important observation is that $ V_{0} $ is always line pointing upwards and each vision line moves in a clockwise manner until it reaches the beginning. In Figure 2 there are 24 vision nodes, meaning that snake sees in 8 directions. You could decode it to figure out which vision lines see what, but I wouldn't bother. The main take away here is to understand where the inputs are coming from

*denotes it can be either binary (yes/no) or distance (set in `settings.py`)

### What are Hidden Layers for?
While I do think this is an extremely important part to understand, I am not going to get too in depth in this post. I will dedicate an entire post to understanding Neural Networks in the future but for now you can think of hidden layers as learning layers. If you were to not have any hidden layers, you would just have input nodes connected to output nodes, which wouldn't result in anything. Generally the more hidden layers you have, the more complex things you can learn.

### Bringing it all together
Here is a screenshot of a single individual playing:

![Snake](/assets/blogs/2020-05-03/imgs/snake.png){: .center-image}

You can see that the snake is moving left and the tail direction is up. The inputs from the surroundings are then fed through the Neural Network and a decision is made for the snake to go left in this case.

## The Genetic Algorithm
### What is the Purpose of the GA?
Neural Networks are often used in supervised learning. You have some training data, toss in a Neural Network and have it learn the weights to help with unseen data. With Snake, however, there is no training data. Training data could be used but would require a ton of work playing Snake and labeling what the correct decision is. By using a Genetic Algorithm you can treat the weights as chromosomes of individuals and slowly evolve the population.

### How Does the GA Work?
At first the Genetic Algorithm will start with a population of randomly initialized snakes. These snakes take in their surroundings as input, feed it through their Neural Network and finally make a decision. After all snakes have played, parent snakes are randomly selected for reproduction (crossover). In this case I use roulette wheel selection based on their fitness to determine which snakes are selected for being parents.

![Roulette wheel fitness](assets/blogs/2020-05-03/imgs/equal_vs_unequal_roulette_wheel_fitness.png){: .center-image}
*<center>Figure 5. Equal vs. Unequal Fitness in Roulette Wheel</center>*

You can see how in the equal fitness roulette wheel, each snake would have the same probability to be selected as a parent. In the unequal fitness roulette wheel you see Snake A has a much higher probability to be selected as a parent. Does this mean that Snake A will slowly dominate the population and all offspring will essentially be clones? No! Because I'm using simulated binary crossover, the offspring are created as a distribution around the parents. This means that even though Snake A may be a parent in many cases, the offspring will not be an exact clone, which helps in exploration!

Let's quickly look at simulated binary crossover with <b>eta=1</b>, <b>eta=10</b> and <b>eta=100</b>:

![Simulated Binary Crossover with eta=1](assets/blogs/2020-05-03/imgs/SBX_eta1.png){: .center-image}
*<center>Figure 6. Simulated Binary Crossover, eta=1</center>*

![Simulated Binary Crossover with eta=10](assets/blogs/2020-05-03/imgs/SBX_eta10.png){: .center-image}
*<center>Figure 7. Simulated Binary Crossover, eta=10</center>*

![Simulated Binary Crossover with eta=100](assets/blogs/2020-05-03/imgs/SBX_eta100.png){: .center-image}
*<center>Figure 8. Simulated Binary Crossover, eta=100</center>*

You can see that as eta increases, the distribution of children are formed more densely around the parents! Because of this I will be using <b>eta=100</b>. Since the weights of the Neural Network are to be confined within `[-1, 1]`, having a larger value of eta will help ensure offspring have reasonable values for weights. One last type of crossover I want to quickly show is single point binary crossover (SPBX). You can think of SPBX as taking a point in the parent chromosomes and swapping data. Below is an example of SPBX where the point is denoted in red:

![Single Point Binary Crossover](assets/blogs/2020-05-03/imgs/SPBX.gif){: .center-image}
*<center>Figure 9. Single Point Binary Crossover</center>*

Notice what data remains the same between <b>parent 1</b> and <b>offspring 1</b> as opposed to <b>parent 2</b> and <b>offspring 2</b>. In this case I represent each box as a gene. All of this can be changed within `settings.py` where you can choose the probability that SPBX occurs vs. SBX.

Once offspring have been created, each gene within each chromosome of each offspring has a small chance of mutation. Sound confusing? Don't worry. It just means that even once crossover occurs there is still a chance for the genes of offspring to change. The next population can then consist of either only offspring, known as the comma method, or offspring and parents, known as the plus method. These can be changed in `settings.py` if you want to mess with them. Comma method can be good if you have an extremely large search space and are trying to focus on exploration. Another notation for this is $ (\mu, k, \lambda) $, or as a normal person would call it, the `(num_parents, lifespan, num_offspring)` notation. This means you have some number of parents that are allowed to be in the population for a given lifespan and create some number of offspring. At the end of the next generation, you then select the top `num_parents` using the roulette wheel for creating offspring. This is then repeated until your computer dies or until you're satisfied with the snakes. Note that `(num_parents, 1, num_offspring)` is the same as using the comma method and that `(num_parents, k>1, num_offspring)` can only be used in the plus method.

So you might be asking, "what is the fitness of the snakes, then?". That is a great question! Thank you for asking. You may remember from my previous post I talked about how the fitness function of each Genetic Algorithm problem will be different. For this one, I want to:
    1. Reward snakes early on for exploration + finding a couple apples.
    2. Have an increasing reward for snakes as they find more apples.
    3. Penalize snakes for taking a lot of steps.
Putting those rules into code looks something like this:<br>
<div align="center">
$ f(steps, apples) = steps + (2^{apples} + 500*apples^{2.1}) - (0.25*steps^{1.3} * apples^{1.2}) $
</div>

## Confused?
I know that Genetic Algorithms and Neural Networks can be confusing the first, second, even 50th time. I wanted to make a special section to talk about how this works with no math involved. You can think of the Neural Network of each snake as the brain. The weights between nodes is roughly "how important something is".  As the inputs connect to other nodes, the weights may change, but you may slowly begin to see what each snake prioritizes. Some snakes may have a lot of blue lines (positive connections), while other snakes may have many red lines (negative/inhibiting connections) and some may have a good mixture! Ultimately the Neural Network decides what a snake will do based off the surroundings.

Because Genetic Algorithms are modeled loosely after organisms, they can help slowly evolve a population of individuals. In this case the population consists of many snakes. At first the snakes will know nothing - simply running into walls, accidentally killing themselves, etc. Eventually a snake will find an apple and get rewarded. Lets' say two snakes found apples in a generation. If one of the snakes found the apple by going up, and the other snake found the apple by going left, there is a chance that if they reproduce, the offspring will know how to go up <b>and</b> left for an apple! This then happens for many generations!

## Seeing it in Action
If you want you can skip to 3:00 into the video as the stuff prior is covered up above.

<iframe width="560" height="315" src="https://www.youtube.com/embed/vhiO4WsHA6c" class="center-image" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Conclusion
In this post we took a look at Snake, Neural Networks and Genetic Algorithms. I've already made a post on Genetic Algorithms but this time we got to see how they can be applied to things such as Neural Networks for finding optimal weights. Using the Neural Network in conjunction with a Genetic Algorithm allowed the creation of generalized learning for a population of snakes to learn the game! You also saw how you can use the generalized learning to take a snake trained in a 10x10 grid and have it perform in a 50x50 grid!

I hope you enjoyed this post as it took quite a lot of work to create and video edit everything along with graphics. As always, feedback is welcomed! If you did enjoy this and know anyone who would be interested, feel free to share! Let me know if you would like to see more stuff like this!