import numpy as np
from controller import AgentInterface
from game.SpaceInvaders import SpaceInvaders
from epsilon_profile import EpsilonProfile
import pandas as pd
import pickle

class QAgent(AgentInterface):
    """ 
    Cette classe d'agent représente un agent utilisant la méthode du Q-learning 
    pour mettre à jour sa politique d'action.
    """

    def __init__(self, spaceInvader :SpaceInvaders, eps_profile: EpsilonProfile, gamma: float, alpha: float):
        """A LIRE
        Ce constructeur initialise une nouvelle instance de la classe QAgent.
        Il doit stocker les différents paramètres nécessaires au fonctionnement de l'algorithme et initialiser la 
        fonction de valeur d'action, notée Q.
        :param maze: Le labyrinthe à résoudre 
        :type maze: Maze
        :param eps_profile: Le profil du paramètre d'exploration epsilon 
        :type eps_profile: EpsilonProfile
        
        :param gamma: Le discount factor 
        :type gamma: float
        
        :param alpha: Le learning rate 
        :type alpha: float
        - Visualisation des données
        :attribut mazeValues: la fonction de valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        :penser à bien stocker aussi la taille du labyrinthe (nx,ny)
        :attribut qvalues: la Q-valeur stockée qui sera écrite dans un fichier de log après la résolution complète
        :type mazeValues: data frame pandas
        """
        # Initialise la fonction de valeur Q
        self.Q = np.zeros([16, 10, 2, 2, 4]) #ecartX; ecartY; DirectionAlien; Etat bullet; actions possibles

        self.spaceInvader = spaceInvader
        self.na = spaceInvader.na

        # Paramètres de l'algorithme
        self.gamma = gamma
        self.alpha = alpha

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        # Visualisation des données (vous n'avez pas besoin de comprendre cette partie)
        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'ecartX': [16], 'ecartY': [10], 'directionAlien': [2], 'BulletState': [2]})

    def learn(self, env, n_episodes, max_steps):
        """Cette méthode exécute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parallèle avec le cours.
        :param env: L'environnement 
        :type env: gym.Envselect_action
        :param num_episodes: Le nombre d'épisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'étape par épisode
        :type max_num_steps: int
        # Visualisation des données
        Elle doit proposer l'option de stockage de (i) la fonction de valeur & (ii) la Q-valeur 
        dans un fichier de log
        """
        n_steps = np.zeros(n_episodes) + max_steps
        
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise le jeu
            state = env.reset()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, is_done = env.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                
                if is_done:
                    n_steps[episode] = step + 1  
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)

            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                state = env.reset()
            #    #print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[state][self.select_greedy_action(state)]), end =" ")
                self.save_log(env, episode)

        #self.values.to_csv('visualisation/logV.csv')
        print(self.qvalues)
        self.qvalues.to_csv('logQ.csv')

    def updateQ(self, state, action, reward, next_state):
        """À COMPLÉTER!
        Cette méthode utilise une transition pour mettre à jour la fonction de valeur Q de l'agent. 
        Une transition est définie comme un tuple (état, action récompense, état suivant).
        :param state: L'état origine
        :param action: L'action
        :param reward: La récompense perçue
        :param next_state: L'état suivant
        """

        #print(self.Q[state].shape)
    
        self.Q[state[0]][state[1]][state[2]][state[3]][action] = (1. - self.alpha) * self.Q[state[0]][state[1]][state[2]][state[3]][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state[0]][next_state[1]][next_state[2]][next_state[3]]))

        if (reward ==1):
            print(self.Q[state[0]][state[1]][state[2]][state[3]][action])
            #print("reward {}, max selfQ {}, alpha {}, Q_state{}".format(reward, np.max(self.Q[next_state]), self.alpha, self.Q[state] ))
            #print (self.Q[state][action])

    def select_action(self, state : 'Tuple[int, int]'):
        """À COMPLÉTER!
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).
        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            a = np.random.randint(self.na)      # random action
        else:
            a = self.select_greedy_action(state)
        return a

    def select_greedy_action(self, state : 'Tuple[int, int]'):
        """
        Cette méthode retourne l'action gourmande.
        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[state])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[state] == mx)[0])

    def save_log(self, env, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """
        state = env.reset()
        # Construit la fonction de valeur d'état associée à Q
        # V = np.zeros([40,10,2,2])
        # for state in self.spaceInvader.get_state():
        #     val = self.Q[state][self.select_action(state)]
        #     V[state] = val

        with open('Qmatrix_{}'.format(self.gamma), 'wb') as write_file:
            pickle.dump(self.Q, write_file)

        # with open('Qmatrix_{}'.format(self.gamma), 'rb') as read_file:
        #     loadedQ = pickle.load(read_file)
        # print(self.Q == loadedQ)
        # print('selfQ{}'.format(self.Q))
        # print('loadedQ{}'.format(loadedQ))


        self.qvalues = self.qvalues.append({'episode': episode, 'value': self.Q[state[0]][state[1]][state[2]][state[3]][self.select_greedy_action(state)]}, ignore_index=True)
        # self.values = self.values.append({'episode': episode, 'value': np.reshape(V,(1, 40*10*2*2))[0]},ignore_index=True)