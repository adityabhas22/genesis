import argparse
import os
import pickle
from importlib import metadata

import torch

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    step_idx = 0
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)
            step_idx += 1

            # Track commanded vs. actual velocities during evaluation
            cmd = env.commands[0].detach().cpu().numpy()
            lin_vel = env.base_lin_vel[0].detach().cpu().numpy()
            ang_vel = env.base_ang_vel[0].detach().cpu().numpy()
            print(
                f"[step {step_idx}] cmd_lin(x,y)={cmd[0]:.2f},{cmd[1]:.2f} "
                f"cmd_yaw={cmd[2]:.2f} | lin_vel(x,y,z)={lin_vel[0]:.2f},{lin_vel[1]:.2f},{lin_vel[2]:.2f} "
                f"ang_vel(x,y,z)={ang_vel[0]:.2f},{ang_vel[1]:.2f},{ang_vel[2]:.2f} "
                f"rew={rews.item():.3f}"
            )


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
