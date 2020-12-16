# FAQ about calibration

## Collect field/measured data

## Parameter screening/sensitivity analysis

- How are we going to select the parameters to calibrate?

Do (before-calibration) sensitivity analysis to understand the inputs to which the output is sensitive and identify calibration parameters $t_1$, \ldots, $t_q$.

- Can we treat the known parameter (if we believe that the true physical value is known) as unknown and include into the calibration?

Yes, especially if a parameter is an influential parameter, allowing it to deviate from the true physical value may produce an empirically better computer model of reality. In that case, its prior distribution would be centered at the true physical value, reflecting an expectation of the model's accuracy, but with a non-zero variance.

- How can we select the correct number of calibration parameters?

Selecting the correct number of model parameters for the calibration is important because if the amount of measured data is insufficient to identify the calibration parameters, the resulting posterior distribution may be subjected to an unacceptable degree of uncertainty. Additionally, using fewer parameters would also reduce computation cost given that Bayesian calibration is computationally prohibitive in a high-dimensional parameter space.
 
## Design the experiment to obtain the computer model output

- How can we decide the initial computer model design (before calibration) to obtain the output of the computer model?

To create the calibration input data file $D_c$, i.e. the training dataset for the Gaussian process emulator, $m$ simulations are run at different parameter values. Parameter values are determined (e.g., using Latin hypercube sampling (LHS)) so that we try to cover as much space as possible in the multi-dimensional space of the calibration parameters. A rule of thumb for training GP models is to have 10 LHS samples per parameter. (Choosing the sample size of computer experiment)
 
## Selecting appropriate priors

- How are we going to select the prior distribution for the unknown parameters?

Known information or restrictions on unknown parameters can be specified through prior. The prior distribution should cover the range that is plausible for the true value of parameters. If little prior information is available on unknown parameter, a prior with mass somewhat more concentrated on a default or midway value can be used. Gamma and log-normal distributions are commonly used priors in the literature to reflect the real life. If there is no information, a uniform prior can be used. 

## Combining the prior and the likelihood

- How are we going to estimate the calibration parameters?

The calibration parameter can be estimated through its posterior distribution. \tcb{Also, another question comes to my mind---How do we decide which sampling method to use? Not sure if it is a valid question} \tcb{Need to read about sequential design approach (this might go to initial computer model design part--Not sure the next sentences are correct} The calibration input coordinates should cover the range that is plausible for the true value of parameters. This suggests a sequential design approach, beginning with values spanning the prior distribution then adding more points over the range covered by its posterior distribution. 

- Is it OK to interpret the estimates of parameters  obtained from calibration as true physical values?

It is dangerous to interpret the estimates of parameters that are obtained by calibration as estimates of the true physical values of those parameters. Sometimes, you can get a worse fit and less accurate predictions with the physically true parameters of the parameters.

## Evaluating performance of the calibrated model

- How can we know whether the modeling choices that we made (e.g., covariance functions) are appropriate?

\tcb{Do after-calibration sensitivity analysis--Need to explain more}

- What are the strategies that we use to understand the predictive accuracy and uncertainty on the simulation outputs after calibration?


