# Offline RL Progress Logging Design

## Goal

Make the second-stage offline RL scripts print useful training progress to the terminal so long-running runs no longer appear frozen.

## Recommended Approach

Use a shared progress formatting helper inside the offline RL utilities and call it from `train_td3bc.py`, `train_iql.py`, and `train_cql.py`.

This keeps the three scripts consistent, avoids copy-pasted print logic, and lets us align the console output with the existing step-based training loop. We will continue saving structured metrics to files and optionally to Weights & Biases, but we will also print:

- periodic training updates every `log_every` steps
- periodic evaluation updates every `eval_freq` steps
- normalized D4RL score during evaluation when a real D4RL env is available

## Scope

- Add shared terminal progress formatting helpers.
- Print train-step progress for TD3BC, IQL, and CQL.
- Print eval progress for TD3BC, IQL, and CQL.
- Keep current file logging behavior unchanged.
- Turn off generation-stage `wandb` by default in the shipped config so network retries do not obscure console feedback.

## Error Handling

- Console logging must not assume every metric key exists.
- Evaluation logging should gracefully print whatever keys are available from `evaluate_policy`.
- Logging should work whether the run uses a real D4RL env or the toy eval env.

## Testing

- Add tests that validate the shared formatting helpers.
- Add CLI-level tests that confirm training scripts emit progress text to stdout on short runs.
- Run the full unit test suite after implementation.
