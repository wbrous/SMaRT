import dataclasses

from smrt.train import train_loop


# If this fails: resuming from a checkpoint loses optimizer momentum/step
# count continuity (a discontinuity between the resumed run's loss and a
# straight-through run's loss at the same step), even though dropout is 0
# so data order is the only remaining source of nondeterminism, controlled
# here by re-seeding identically before both runs.
def test_resume_matches_straight_through_run(tiny_config, tmp_path):
    ckpt_dir_a = tmp_path / "ckpt_a"
    ckpt_dir_b = tmp_path / "ckpt_b"

    cfg_a = dataclasses.replace(
        tiny_config,
        train=dataclasses.replace(tiny_config.train, max_steps=11, checkpoint_every=10, checkpoint_dir=str(ckpt_dir_a)),
    )
    train_loop(cfg_a, stop_step=10)  # steps 0..9, checkpoint written at step_10.pt

    result_resumed = train_loop(cfg_a, resume_path=str(ckpt_dir_a / "step_10.pt"))  # runs step 10 (1 more step)

    cfg_straight = dataclasses.replace(
        tiny_config,
        train=dataclasses.replace(tiny_config.train, max_steps=11, checkpoint_every=11, checkpoint_dir=str(ckpt_dir_b)),
    )
    result_straight = train_loop(cfg_straight)  # runs steps 0..10 straight through

    assert abs(result_resumed["final_loss"] - result_straight["final_loss"]) < 1e-4
