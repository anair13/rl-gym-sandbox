import argparse
import logging
import sys
from gym import wrappers

import gym

import numpy as np
from discretizing_q_agent import DiscretizingQAgent
from linear_q_agent import LinearQAgent

alg = DiscretizingQAgent

if __name__ == '__main__':
    # Call `undo_logger_setup` if you want to undo Gym's logger setup
    # and configure things manually. (The default should be fine most
    # of the time.)
    gym.undo_logger_setup()
    logger = logging.getLogger()
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # You can set the level to logging.DEBUG or logging.WARN if you
    # want to change the amount of output.

    # env = gym.make('Copy-v0')
    # env = gym.make('SpaceInvaders-v0')
    env = gym.make('CartPole-v0')
    # env = gym.make('Humanoid-v1')
    # env = gym.make('Hopper-v1')
    # env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(directory=outdir, force=True, video_callable=False)(env)
    agent = alg(env.action_space, logger)

    reward = 0
    done = False
    k = 100
    last_k = np.array([0 for _ in range(k)])

    episode_count = 50000
    # env.seed(0)
    # random.seed(0)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.INFO)
    for i in range(episode_count):
        ob = env.reset()
        reward = 0.0
        s = 0
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            s += reward
            if done:
                reward = -200
                agent.act(ob, reward, done)
                done = False
                logger.debug("Episode complete, reward = %d" % s)
                last_k[i % k] = s
                break
        if i % k == 0:
            m = np.mean(last_k)
            logger.info("Step %d average reward %f" % (i, m))
            if m > 196:
                print "DONE!", m
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    print agent.minimums
    print agent.maximums

    # Close the env and write monitor result info to disk
    env.close()

    # Upload to the scoreboard. We could also do this from another
    # process if we wanted.
    # logger.info("Successfully ran RandomAgent. Now trying to upload results to the scoreboard. If it breaks, you can always just try re-uploading the same results.")
    # gym.upload(outdir)