# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions, Actions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        kInf=1e100
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        new_pos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        # new_ghost_states = successorGameState.getGhostStates()
        new_ghost_positions = successorGameState.getGhostPositions()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if new_pos in new_ghost_positions:
            return -kInf
        capsules_position = currentGameState.getCapsules()
        food_positions = currentGameState.getFood().asList()
        action_vec=Actions.directionToVector(action)
        if action=="Stop":
            return random.uniform(-0.2,0.2)
        # print(f"action:{action}, action_vec:{action_vec}")
        current_ghost_positions = currentGameState.getGhostPositions()
        current_ghost_scared_times = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
        not_scared_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]==0]
        scared_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]>0 and current_ghost_scared_times[i]<=1.2*util.manhattanDistance(current_ghost_positions[i],new_pos)+2]
        edible_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]>1.2*util.manhattanDistance(current_ghost_positions[i],new_pos)+2]
        current_self_position = currentGameState.getPacmanPosition()
        def DotProduct(a,b):
            return a[0]*b[0]+a[1]*b[1]
        def CrossProduct(a,b):
            return a[0]*b[1]-a[1]*b[0]
        def EuclideanDistance(a,b):
            return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        def DistanceAnalysis(current_self_position,object_postion_list,flag="None"):
            if len(object_postion_list)==0:
                return 0
            if current_self_position in object_postion_list:
                return kInf
            res=0
            for obj_pos in object_postion_list:
                if flag=="Ghost" and util.manhattanDistance(current_self_position,obj_pos)>=6:
                    continue
                vec_to_obj=(obj_pos[0]-current_self_position[0],obj_pos[1]-current_self_position[1])
                # print(f"vec_to_obj:{vec_to_obj}")
                # print(f"action:{action}, action_vec:{action_vec}")
                # print(f"EuclideanDistance(action_vec,(0,0)):{EuclideanDistance(action_vec,(0,0))}")
                # print(f"EuclideanDistance(vec_to_obj,(0,0)): {EuclideanDistance(vec_to_obj,(0,0))}")
                cos_theta=DotProduct(action_vec,vec_to_obj)/(EuclideanDistance(action_vec,(0,0))*EuclideanDistance(vec_to_obj,(0,0)))
                distance_to_obj=EuclideanDistance(current_self_position,obj_pos)
                res+=(cos_theta+1)/distance_to_obj
            return res

        da_for_foods=DistanceAnalysis(current_self_position,food_positions)
        if new_pos in currentGameState.getFood().asList():
            da_for_foods=200
        da_for_unscared_ghosts=DistanceAnalysis(current_self_position,not_scared_ghosts_positions,"Ghost")
        da_for_scared_ghosts=DistanceAnalysis(current_self_position,scared_ghosts_positions)
        da_for_capsules=DistanceAnalysis(current_self_position,capsules_position)
        da_for_edible_ghosts=DistanceAnalysis(current_self_position,edible_ghosts_positions)
        res=da_for_capsules*2-da_for_unscared_ghosts*2-da_for_scared_ghosts*0.2+da_for_foods*0.2+da_for_edible_ghosts
        res*=random.uniform(0.9, 1.1)
        # print(f"res:{res}")
        return res

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def MinMaxSearch(self,gameState: GameState,depth_remain:int,agentIndex:int) -> tuple[int, list[Actions]]:
        if depth_remain==0:
            # print(f"depth_remain:{depth_remain}")
            # print(f"returning leaf {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions)==0:
            # print(f"depth_remain:{depth_remain}")
            # print(f"returning leaf {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        kInf=1e100
        res_action=[]
        res_val=0
        if agentIndex==0:
            # Max
            res_val = -kInf
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.MinMaxSearch(successorGameState,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                if val>res_val:
                    res_val=val
                    # print(f"action:{action}, action_list:{action_list}")
                    res_action=[action]+action_list
        else:
            # Mins
            res_val = kInf
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.MinMaxSearch(successorGameState,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                if val<res_val:
                    res_val=val
                    res_action=[action]+action_list
        # print(f"depth_remain:{depth_remain}")
        # print(f"returning {res_val}, {res_action}")
        return res_val,res_action

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        stat = self.MinMaxSearch(gameState,self.depth,0)
        # print(f"stat:{stat}")
        return stat[1][0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def AlphaBetaSearch(self,gameState: GameState,alpha,beta,depth_remain:int,agentIndex:int) -> tuple[int, list[Actions]]:
        if alpha>beta:
            raise ValueError("alpha should be less than beta")
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions)==0:
            # print(f"depth_remain: leaf {depth_remain}, alpha:{alpha}, beta:{beta}")
            # print(f"returning {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        if depth_remain==0:
            # print(f"depth_remain: leaf {depth_remain}, alpha:{alpha}, beta:{beta}")
            # print(f"returning {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        res_action=[]
        res_val=0
        kInf=1e100
        if agentIndex==0:
            # Max
            res_val = -kInf
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.AlphaBetaSearch(successorGameState,alpha,beta,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                alpha=max(alpha,val)
                # print(f"now alpha:{alpha}, beta:{beta}")
                if val>res_val:
                    res_val=val
                    # print(f"action:{action}, action_list:{action_list}")
                    res_action=[action]+action_list
                if val>beta:
                    break
        else:
            # Mins
            res_val = kInf
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.AlphaBetaSearch(successorGameState,alpha,beta,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                beta=min(beta,val)
                # print(f"now alpha:{alpha}, beta:{beta}")
                if val<res_val:
                    res_val=val
                    res_action=[action]+action_list
                if alpha>val:
                    break
        # print(f"depth_remain:{depth_remain}, alpha:{alpha}, beta:{beta}")
        # print(f"returning {res_val}, {res_action}")
        return res_val,res_action

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        kInf=1e100
        stat = self.AlphaBetaSearch(gameState,-kInf,kInf,self.depth,0)
        # print(f"stat:{stat}")
        return stat[1][0]

last_score=0
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def ExpectMaxSearch(self,gameState: GameState,depth_remain:int,agentIndex:int) -> tuple[int, list[Actions]]:
        if depth_remain==0:
            # print(f"depth_remain:{depth_remain}")
            # print(f"returning leaf {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        legal_actions = gameState.getLegalActions(agentIndex)
        if len(legal_actions)==0:
            # print(f"depth_remain:{depth_remain}")
            # print(f"returning leaf {self.evaluationFunction(gameState)}, {[]}")
            return self.evaluationFunction(gameState),[]
        kInf=1e100
        res_action=[]
        res_val=0
        if agentIndex==0:
            # Max
            res_val = -kInf
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.ExpectMaxSearch(successorGameState,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                if action=="Stop":
                    val-=100
                if val>res_val:
                    res_val=val
                    # print(f"action:{action}, action_list:{action_list}")
                    res_action=[action]+action_list
        else:
            # Mins
            res_val = kInf
            val_list=[]
            for action in legal_actions:
                successorGameState = gameState.generateSuccessor(agentIndex,action)
                nxt_depth=depth_remain-1 if agentIndex==gameState.getNumAgents()-1 else depth_remain
                val,action_list=self.ExpectMaxSearch(successorGameState,nxt_depth,(agentIndex+1)%gameState.getNumAgents())
                val_list.append(val)
                if val<res_val:
                    res_val=val
                    res_action=[action]+action_list
            res_val=sum(val_list)/len(val_list)
        # print(f"depth_remain:{depth_remain}")
        # print(f"returning {res_val}, {res_action}")
        return res_val,res_action

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        global last_score
        last_score=gameState.getScore()
        stat = self.ExpectMaxSearch(gameState,self.depth,0)
        # print(f"stat:{stat}")
        return stat[1][0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    if currentGameState.isLose():
        return -500
    if currentGameState.isWin():
        return 500
    kInf=1e100
    capsules_position = currentGameState.getCapsules()
    current_pos = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()
    # print(f"action:{action}, action_vec:{action_vec}")
    current_ghost_positions = currentGameState.getGhostPositions()
    current_ghost_scared_times = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]
    not_scared_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]==0]
    scared_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]>0 and current_ghost_scared_times[i]<=1.2*util.manhattanDistance(current_ghost_positions[i],current_pos)+2]
    edible_ghosts_positions = [current_ghost_positions[i] for i in range(len(current_ghost_positions)) if current_ghost_scared_times[i]>1.2*util.manhattanDistance(current_ghost_positions[i],current_pos)+2]
    current_self_position = currentGameState.getPacmanPosition()
    def DotProduct(a,b):
        return a[0]*b[0]+a[1]*b[1]
    def CrossProduct(a,b):
        return a[0]*b[1]-a[1]*b[0]
    def EuclideanDistance(a,b):
        return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
    def DistanceAnalysis(current_self_position,object_postion_list,flag="None"):
        if len(object_postion_list)==0:
            return 0
        if current_self_position in object_postion_list:
            return kInf
        res=0
        for obj_pos in object_postion_list:
            if flag=="Ghost" and util.manhattanDistance(current_self_position,obj_pos)>=6:
                continue
            distance_to_obj=util.manhattanDistance(current_self_position,obj_pos)
            res=max(res,1/distance_to_obj)
        return res

    da_for_foods=DistanceAnalysis(current_self_position,food_positions)
    da_for_unscared_ghosts=DistanceAnalysis(current_self_position,not_scared_ghosts_positions,"Ghost")
    da_for_scared_ghosts=DistanceAnalysis(current_self_position,scared_ghosts_positions)
    da_for_capsules=DistanceAnalysis(current_self_position,capsules_position)
    da_for_edible_ghosts=DistanceAnalysis(current_self_position,edible_ghosts_positions)
    res=da_for_capsules*2-da_for_unscared_ghosts*2-da_for_scared_ghosts*0.2+da_for_foods*0.2+da_for_edible_ghosts*1
    if da_for_unscared_ghosts<1/6:
        res+=(da_for_foods*0.2+da_for_edible_ghosts*1)*5
    # res*=random.uniform(0.9, 1.1)
    res*=100
    global last_score
    res+=(currentGameState.getScore()-last_score)*10
    # print(f"res:{res}")
    return res

# Abbreviation
better = betterEvaluationFunction
