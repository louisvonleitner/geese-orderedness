This project tries to understand the predictability of bird movement to conclude about its complexity.

Specifically, we consider the "order" inside the system of a flock of geese. Order is hard to define, it is the opposite of chaos. In similar studies, it has been defined by the normalized velocity alignment, which is also called phase. (Vicseck et al. 1995) However, this parameter does not work very well for flocks of migrating geese, because deviation in this metric of order is minimal.

The underlying data is location data of white fronted geese and has been gathered by Professor Hayakawa at Tohoku University. This type of bird is a migrating bird, flying in the characteristic V-shape formation. Therefore, one should only with limitation transfer conclusions about this system to other bird species systems.

We first develop a suitable metric for orderedness. Entropy is a concept that describes the "likelihood" of a system being in a state. Order is not likely and requires active effort.
Our metric calculates entropy in the space of velocities of the birds. Specifically, we use Principal Component Analysis to gain information about the directionality of the "disorder".
Now that this metric is developed, we can make predictions about it.

The basic thought behind the method is:
"If we can predict the change in order over time very well, then geese flock systems are less complex"
This is because if we can predict the change in order of the system, then there must be some set of rules which guide geese movement in the flock formation. If we can predict accurately with a less complex ML model, that means the set of rules is simpler.

First though, it is helpful to make a quick small example of predicting structure:
Bird flocks vary between 4-100 individuals per flock. Given the positions, velocities and accelerations of 3 birds inside a flock, can we predict how many birds are in the flock?
Somewhat, but not excessively. A naive predictor estimating the mode of number of birds every time has a validation MAE of 0.9 x standard deviations.
The best gradient boosted tree prediction model I could come up with only achieved 0.45 x standard deviations. Transformers are around the same mark.

As standard deviation is around 16, that means that the predictions are off by about 8 birds, so by almost 10% of the range of values. 
So, one could say that this prediction task is somewhat solveable, but I have not managed to solve it to a sufficiently good extent.

A pattern that happened with Transformers and Gradient Boosted Trees is that when choosing architecture too big, the model would overfit on the train loss without impacting validation loss. 
This seems to be because the model learns the general trends, but when well generalizable patterns are not exploitable any more, it starts overfitting.

It seems to be that this prediction task is very complex and only knowing the positions of 3 geese is not enough. (Flock sizes vary between 5-100 birds)
Having tried with 4 and 5 visible geese to the model, results are not much different, only improving slightly. None of the architectures used so far were able 
to reliably forecast the number of geese. 
Even when increasing the number of datapoints by generating more different samples through masking from a single frame (point in time), the performance did not change at all. 
This leads me to believe that this prediction task is just not perfectly solveable because of a high ratio of noise to signal.

Next up:
Trying Transformers at predicting changes in order over time...

