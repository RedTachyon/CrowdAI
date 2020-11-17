# Physics model

This document will attempt to outline the physics used in the simulation for future reference.
All descriptions apply only to the current version of the simulation - I'll try to keep this updated if I change anything.

I'm trying to keep as much of the physics manually controlled, and not handled by Unity automatically, as long as that's reasonable performance-wise.

An agent is a cylinder of radius 0.25 on a 20x20 plane (range [-10, 10]).
The plane is surrounded by obstacles that cover up ranges [-10, -9.5] and [9.5, 10] at each side.

The simulation evolves in timesteps. Every 5 timesteps, there's a decision step that decides the action taken for the following 5 steps.

Actions have two axes: linear acceleration (modeled as a force internally, handled by Unity), and angular rotation.

Linear acceleration action has a range of [-0.3, 1.0], allowing limited backwards motion and favoring forward motion.

The public parameter moveSpeed determines the scale of the force added. 

Additionally, a drag force is added, antiparallel to the current velocity, proportional to its magnitude and to the dragFactor parameter, according to the equation:

<img src="https://render.githubusercontent.com/render/math?math=F_d = - d\  || v ||  \frac{v}{|| v ||} = -d\  v">

(I tested quadratic drag, but the agents kept drifting for a relatively long time; The best realism would be obtained
using both a linear and a quadratic term, but at the moment, the linear version seems to be working well)


The angular speed is about as simple as it gets - the rotation of the agent is statically changed to a new position, without 
any angular momentum. Also, the angular drag of the rigidbody is set to a very high value to make sure collisions don't get it spinning.
