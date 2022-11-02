# Experimental study of Neural ODE training with adaptive solver for dynamical systems modeling

This work has started during the supervised project at [ESPCI](https://www.espci.psl.eu/) (*PSE* aka Team Scientific Project), after some preliminary work on Neural ODE. The paper is accepted at the 2nd NeurIPS [workshop on Deep Learning and Differential Equations](https://dlde-2022.github.io/).



## Abstract 
Neural Ordinary Differential Equations (ODEs) was recently introduced as a new family of neural network models, which relies on black-box ODE solvers for inference and training. Some ODE solvers  called adaptive can adapt their evaluation strategy depending on the
complexity of the problem at hand, opening great perspectives in machine learning. However, this paper describes a simple set of experiments to show why adaptive solvers cannot be seamlessly leveraged as a black-box for dynamical systems modelling. By taking
the Lorenz'63 system as a showcase, we show that a naive application of the Fehlberg's method does not yield the expected results. Moreover, a simple workaround is proposed that assumes a tighter interaction between the solver and the training strategy.


