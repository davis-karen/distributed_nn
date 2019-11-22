 Implementation Notes (For future Karen)
---

The goal is to implement a version of Hogwild! in pytorch as per the [paper](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf).
Read the paper - Argggh maths :) 


#### My understanding of what to  implement

From what I can see it seems like the only difference between Hogwild! (asynchronous sgd) and normal sgd is that we have a shared model. Each process uses this shared model and then trains as usual.
##### Steps
- Put the model in shared memory 
- Spin up separate processes 
- Each process creates a randomised batch sampling from the dataset
- Each process trains as normal using shared model
- When all of the processes have finished training test the model

##### Outstanding Questions
###### Question 1
The paper talks about sparse models and collisions of gradient updates being rare. How can I test this? Can I visualise the updates?
The first example I'm planning on running is a small MLP - I'm not expecting it to be sparse - interested to see the result


###### Question 2
Is there a basic unit test I can write to test the model? And/Or gates maybe??
###### I tried AND gates but there was not enough data (funnily enough) to test the model. <br>
###### I was able to write a test to predict all 0s and then all 1s with very little data



###### Question 3
It seems like having a shared model living somewhere in memory is the key to  this algorithm.
Does that mean RAM for both cpu and gpu?
How does it work for very large models? Do we ever reach a point where we see pagination issues?

###### Question 4
All of the above are assuming multi cpus/gpus on the same machine. How to introduce distribution across machines? Is there something hidden in the torch.multiprocessing module or will I have to write some custom code?

###### https://pytorch.org/docs/stable/distributed.html Seems like there a pytorch module called distributed that can help with this

#### Pytorch implementation details

###### Pytorch default initialization stuff:

- Pytorch.nn.Linear defaults to kaiming uniform for weights initialization and expects leaky relu activation function
- The Loss object is instantiated with θ (parameters) from the model
- We call loss.backward to calculate the gradients
- The Optimiser object is instantiated with θ (parameters) from the model
- We call optimiser.step to apply the gradients to the weights and take a step forward

###### Pytorch multiprocessing stuff
 - Host to GPU copies are much faster when they originate from pinned (page-locked) memory. CPU tensors and storages expose a pin_memory() method, that returns a copy of the object, with data put in a pinned region.
 - Once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional non_blocking=True argument to a to() or a cuda() call. This can be used to overlap data transfers with computation.
 - You can make the DataLoader return batches placed in pinned memory by passing pin_memory=True to its constructor.
 - to() member function. It's job is to put the tensor on which it's called to a certain device whether it be the CPU or a certain GPU.
