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
Yes, with mean error of 1,8 birds and 95% explained variance using gradient boosted trees. This points in the direction of predictability of structure in geese flocks.

Next up:
Trying Transformers at predicting changes in order...

