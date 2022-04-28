from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from controller.qagent import QAgent
from epsilon_profile import EpsilonProfile
from controller import AgentInterface
import numpy


def main():

    game = SpaceInvaders(display=True)
    n_episodes = 3
    max_steps = 1000
    # gamma = 0.9
    alpha = 1
    eps_profile = EpsilonProfile(1.0,0.1)
    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    episodes = numpy.arange(50, 350, 50, dtype=int)
    gammas = numpy.arange(0.1, 1.1,0.1);
    for gamma in gammas:
        for episode in episodes:
            controller = QAgent(game,eps_profile, gamma, alpha)
            controller.learn(game,episode,max_steps)
    # controller = QAgent(game, eps_profile, gamma, alpha)
    # controller.learn(game,n_episodes,max_steps)
    # test_spaceInvader(game,controller,max_steps, speed=0.1, display=True)
 
    state = game.reset()
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
