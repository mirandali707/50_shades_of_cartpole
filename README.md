# 50_shades_of_cartpole
NMM 2026 final project: RL vs classical control methods for cartpole
[This spreadsheet](https://docs.google.com/spreadsheets/d/1qUzuzJ2d2wACyRdAANu_JexWEW6xBzRqvHZk63TVAFM/edit?gid=184273463#gid=184273463) has an overview of the results!

# Classical control
## [PID](pid/PID.ipynb)
### P (proportional) on pole angle
Take the error of the pole angle (relative to 0, the goal), multiply that by the gain (hand-tuned), and output the sign of the controller output value as the action.

Incredibly simple to write, no learning, no model state... dead simple. 
Doesn't do very well though, average performance is around 40-45 steps per trial.

### PD (proportional + derivative) on pole angle
Do the same thing as the P (proportional) controller except add another term that multiplies a gain by the change in error from the previous iteration to this one. Same as before, the action is the sign of the controller output.

Not much more code than the P controller, no learning, only needs to keep track of the last error. And, it SOLVES CARTPOLE! 500 steps (the max for this Cartpole env) straight out of the box (notably I used someone else's hand-tuned gain).
I didn't even need to add a separate PD controller for the cart position, or the I term!