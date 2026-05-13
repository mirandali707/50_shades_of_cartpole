# 50_shades_of_cartpole
NMM 2026 final project: RL vs classical control methods for cartpole

# Classical control
## [PID](pid/PID.ipynb)
### P (proportional) on pole angle
Take the error of the pole angle (relative to 0, the goal), multiply that by the gain (hand-tuned), and output the sign of the controller output value as the action.

Incredibly simple to write, no learning, no model state... dead simple. 
Doesn't do very well though, average performance is around 40-45 steps per trial.

### PD (proportional + derivative) on pole angle
Do the same thing as the P (proportional) controller except add another term that multiplies a gain by the change in error from the previous iteration to this one. Same as before, the action is the sign of the controller output.

Not much harder than the P controller, no learning, only needs to keep track of the last error. And, it SOLVES CARTPOLE! 500 steps (the max for this Cartpole env) straight out of the box (notably I used someone else's hand-tuned gain).