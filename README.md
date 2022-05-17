# AIMMS-Modelling-Contest-22

This repository houses the implementations of the solutions developed by the TUD team as part of the [https://iccopt2022.lehigh.edu/competition-and-prizes/aimms-mopta-competition/](AIMMS-MOPTA 2022 optimization modeling competition) [1].

The folder optimisation-model contains the implementation of our deterministic and stochastic optimisation approach while the sim-heuristic folder contains the implementation of our sim-heuristic approach. To run a solution, call 

   ``` python3 main.py <solution type> [arguments pertaining to that solution type] ```

where the choices for solution type are **optimise** -- which runs our optimisation models, **alns** -- which runs the SH and **gui** - which launches the GUI. To find out all arguments for a solution type, run
    ``` python3 main.py <solution type> -h ```

Some examples are as follows:

``` python3 main.py optimise -a S4 -i 70 -s 50```
runs the 4 stage stochastic optimisation model (**-a S4**) for an instance size of 70 (**-i 70**) with samples from 50 scenarios (**-s 50**).The choices for algorithm type are D, S, S2, S3 ans S4 where D represents the deterministic model and S is the stochastic model where the numbers represent the number of stages the chosen stochastic model has. For instance size, we stick to the sizes of 70, 100, 140 and 200 as noted in the contest. We also have scenarios saved for the values 50, 100, 200 and 1000.


``` python3 main.py gui ```
launches the GUI. The options on the toolbar allow users to either run the algorithm to produce results or load an already saved result. After running or loading a solution, the initial solution is shown as a plot. The actual start times and assignments can also be downloaded as csv via the **save schedule** button. Sequel to this, we can then simulate the arrival of emergencies via the **simulate next emergency** button which randomly draws emergencies and applies out policy. 

``` python3 main.py alns -i 70 -s 50 -b 100 ```
runs the SH for an instance size of 70 (\textit{-i 70}) with samples from 50 scenarios (\textit{-s 50}) and a budget of 100 iterations (\textit{-b 100}). The other hyperparameters used for the SH are set to tuned values by default and options to change these settings can be accessed via 

``` python3 main.py alns -h ```

Note that the first time running the SH it can take up to 15 minutes to compile.


[1] Karmel S. Shehadeh and Luis F. Zuluaga (2022). "14th AIMMS-MOPTA Optimization Modeling Competition. Surgery Scheduling in Flexible Operating Rooms Under Uncertainty", Modeling and Optimization: Theory and Application (MOPTA), website accessed on May 16, 2022. Available: https://iccopt2022.lehigh.edu/competition-and-prizes/aimms-mopta-competition/
