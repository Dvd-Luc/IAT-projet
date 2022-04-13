class AgentInterface:
    """ 
    L'interface requise par tous les agents.
    """

    def select_action(self, state):
        """ 
        Select an action given the current policy and a state
        """
        pass

    def select_greedy_action(self, state):
        return self.select_action(state)