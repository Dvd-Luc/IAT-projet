from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from epsilon_profile import EpsilonProfile
from controller import AgentInterface
import sys
import argparse
import pickle

parser = argparse.ArgumentParser(description='Run game')
parser.add_argument('--play', help='Run the program without trainning')
args = parser.parse_args()

def main():
    
    game = SpaceInvaders(display=True)
    n_episodes = 1200
    max_steps = 1000
    gamma = 0.5
    alpha = 1
    eps_profile = EpsilonProfile(1.0,0.1)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    controller = QAgent(game, eps_profile, gamma, alpha)
    if not args.play :
        controller.learn(game,n_episodes,max_steps)
        test_spaceInvader(game,controller,max_steps, speed=0.1, display=True)
    else :
        state = game.reset()
        with open('Qmatrix_0.5', 'rb') as read_file:
            controller.Q = pickle.load(read_file)
        while True:
            action = controller.select_action(state)
            state, reward, is_done = game.step(action)
            sleep(0.0001)

def test_spaceInvader(env: SpaceInvaders, agent: AgentInterface, max_steps: int, nepisodes : int = 1, speed: float = 0., same = True, display: bool = False):
    n_steps = max_steps
    sum_rewards = 0.
    for _ in range(nepisodes):
        state = env.reset()
        if display:
            env.render()

        for step in range(max_steps):
            action = agent.select_greedy_action(state)
            next_state, reward, terminal = env.step(action)

            if display:
                #sleep(speed)
                env.render()

            sum_rewards += reward
            if terminal:
                n_steps = step+1  # number of steps taken
                break
            state = next_state
    return n_steps, sum_rewards

if __name__ == '__main__' :
    main()
