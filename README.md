- agent.py:
Defines what the methods an agent (a policy) should define

- agents: 
This folder contains several agents (policies). The "random" agent is currently the only one that fully works.

- baselines: 
The baselines written by OpenAI.

- controller.py: 
Takes an agent (policy) and an environment, and runs the policy on the environment.

- definitions.py: 
Defines all the variables that can be used with main.py: the enviromnents, the agent(s), the predictors, the default values.

- evaluator.py: 
Takes a ReplayBuffer file (from replays/) and a predictor, and evaluates it. Saves the result in a file (in results/).

- main.py: 
"Demo" file which implements the 3 main steps that can be done.
1. Collect data from an environment and save it in a ReplayBuffer file.
2. Evaluate a predictor on a ReplayBuffer and save it in a Result file.
3. Visualize the results of the predictors.

- modelWrapper.py: 
A meta-agent (policy), which takes another agent (policy) and runs it, while evaluating a predictor or saving a ReplayBuffer.

- models: 
This folder will contains in the future the DeepQ policies learned by the DeepQ baseline.

- policyExamples.py: 
Obsolete demo file for RL examples (LQR, discrete Q-Learning, ...)

- predictor.py: 
Defines the methods a Predictor should define.

- predictors: 
This folder contains several "Predictors", ie. model that can be trained and used to predict the dynamics.
Currently contains:
	- A "full" neural net
	- A "linear" neural net (Xt+1 = A(Xt) + W(Xt) * Ut)
	- A Gaussian Process
	- An "identity" predictor (ie. Xt+1 is predicted to be Xt)

- replayBuffer.py: 
The class that handles the replay files to record episodes and load them.

- result.py: 
The class that handles the result files to store predictions made by predictors, and to visualize errors.

- results: 
This folder contains the results generated by evaluator.py.

- simulation.py: 
Obsolete file.

- tools.py: 
A bunch of useful classes and functions are coded here (LQR, normalizer, Gaussian Processes).

- visualisator.py: 
The class that handles the visualisation of the results.
