# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    action_list=[]
    # action_list=["South", "South", "West", "South", "West", "West", "South", "West"]
    vis_stk=util.Stack()
    has_visited={}
    vis_stk.push(problem.getStartState())
    # print(f'push {problem.getStartState()}')
    has_visited[problem.getStartState()]=True
    nex_idx={}
    nex_idx[problem.getStartState()]=0
    best_actions={}
    best_actions[problem.getStartState()]=[]
    stored_successors={}
    def SafelyFetchSuccessors(problem,stored_successors,state):
        if state in stored_successors:
            return stored_successors[state]
        else:
            stored_successors[state]=problem.getSuccessors(state)
            return stored_successors[state]
    while True:
        if vis_stk.isEmpty():
            break
        cur_state=vis_stk.pop()
        # print(f'pop {cur_state}')
        cur_actions=best_actions[cur_state]
        if problem.isGoalState(cur_state):
            action_list=cur_actions
            break
        vis_stk.push(cur_state)
        # print(f'push {cur_state}')
        # successors=problem.getSuccessors(cur_state)
        successors=SafelyFetchSuccessors(problem,stored_successors,cur_state)
        # print(f"getting successors of {cur_state} with {successors}")
        while nex_idx[cur_state]>=len(successors):
            tmp=vis_stk.pop()
            # print(f'pop {tmp}')
            if vis_stk.isEmpty():
                break
            cur_state=vis_stk.pop()
            # print(f'pop {cur_state}')
            cur_actions=best_actions[cur_state]
            vis_stk.push(cur_state)
            # print(f'push {cur_state}')
            # successors=problem.getSuccessors(cur_state)
            successors=SafelyFetchSuccessors(problem,stored_successors,cur_state)
            # print(f'getting successors of {cur_state} with {successors}')
        if vis_stk.isEmpty():
            break
        next_state,action,_=successors[nex_idx[cur_state]]
        nex_idx[cur_state]+=1
        if next_state not in has_visited:
            vis_stk.push(next_state)
            # print(f'push {next_state}')
            best_actions[next_state]=cur_actions+[action]
            has_visited[next_state]=True
            nex_idx[next_state]=0
    return action_list

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    action_list=[]
    vis_que=util.Queue()
    has_visited={}
    vis_que.push((problem.getStartState(),[]))
    has_visited[problem.getStartState()]=True
    while True:
        if vis_que.isEmpty():
            break
        cur_state, cur_actions=vis_que.pop()
        if problem.isGoalState(cur_state):
            action_list=cur_actions
            break
        for next_state,action,_ in problem.getSuccessors(cur_state):
            if next_state not in has_visited:
                vis_que.push((next_state,cur_actions+[action]))
                has_visited[next_state]=True
    return action_list

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    action_list=[]
    vis_que=util.PriorityQueue()
    has_visited={}
    vis_que.push(problem.getStartState(),0)
    dis={}
    dis[problem.getStartState()]=0
    best_actions={}
    best_actions[problem.getStartState()]=[]
    while True:
        if vis_que.isEmpty():
            break
        cur_state=vis_que.pop()
        cur_actions=best_actions[cur_state]
        # print("cur_state:",cur_state, "cur cost=",dis[cur_state])
        if problem.isGoalState(cur_state):
            action_list=cur_actions
            # print("minimal cost:",dis[cur_state])
            break
        if not cur_state in has_visited:
            for next_state,action,cost in problem.getSuccessors(cur_state):
                # print(f"next_state={next_state}")
                # print(f"try update {next_state} with cost={dis[cur_state]+cost}")
                if dis[cur_state]+cost<dis.get(next_state,1e20):
                    vis_que.update(next_state,dis[cur_state]+cost)
                    dis[next_state]=min(dis.get(next_state,1e20),dis[cur_state]+cost)
                    best_actions[next_state]=cur_actions+[action]
        has_visited[cur_state]=True
    return action_list

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    kInf=1e100
    action_list=[]
    vis_que=util.PriorityQueue()
    has_visited={}
    vis_que.push(problem.getStartState(),heuristic(problem.getStartState(),problem))
    dis={}
    dis[problem.getStartState()]=0
    best_actions={}
    best_actions[problem.getStartState()]=[]
    while True:
        if vis_que.isEmpty():
            break
        cur_state=vis_que.pop()
        cur_actions=best_actions[cur_state]
        # print("cur_state:",cur_state, "cur cost=",dis[cur_state])
        if problem.isGoalState(cur_state):
            action_list=cur_actions
            # print("minimal cost:",dis[cur_state])
            break
        if not cur_state in has_visited:
            for next_state,action,cost in problem.getSuccessors(cur_state):
                # print(f"next_state={next_state}")
                # print(f"try update {next_state} with cost={dis[cur_state]+cost}")
                if dis[cur_state]+cost<dis.get(next_state,kInf):
                    vis_que.update(next_state,dis[cur_state]+cost+heuristic(next_state,problem))
                    dis[next_state]=min(dis.get(next_state,kInf),dis[cur_state]+cost)
                    best_actions[next_state]=cur_actions+[action]
        has_visited[cur_state]=True
    return action_list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
