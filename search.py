import util

class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    fringe = util.Stack()
    start_state = problem.getStartState()
    fringe.push((start_state, []))
    visited = set()

    while not fringe.isEmpty():
        current_state, actions = fringe.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited:
            visited.add(current_state)
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    fringe.push((successor, actions + [action]))

    return []

def breadthFirstSearch(problem: SearchProblem):
    fringe = util.Queue()
    start_state = problem.getStartState()
    fringe.push((start_state, []))
    visited = set()

    while not fringe.isEmpty():
        current_state, actions = fringe.pop()

        if problem.isGoalState(current_state):
            return actions

        if current_state not in visited:
            visited.add(current_state)
            for successor, action, step_cost in problem.getSuccessors(current_state):
                if successor not in visited:
                    fringe.push((successor, actions + [action]))

    return []

def uniformCostSearch(problem: SearchProblem):
    fringe = util.PriorityQueue()
    start_state = problem.getStartState()
    fringe.push((start_state, [], 0), 0)
    visited = {}

    while not fringe.isEmpty():
        current_state, actions, current_cost = fringe.pop()

        if problem.isGoalState(current_state):
            return actions

        if (current_state not in visited) or (current_cost < visited[current_state]):
            visited[current_state] = current_cost
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                if (successor not in visited) or (new_cost < visited.get(successor, float('inf'))):
                    fringe.push((successor, actions + [action], new_cost), new_cost)

    return []

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    fringe = util.PriorityQueue()
    start_state = problem.getStartState()
    h_start = heuristic(start_state, problem)
    fringe.push((start_state, [], 0), h_start)
    visited = {}

    while not fringe.isEmpty():
        current_state, actions, current_cost = fringe.pop()

        if problem.isGoalState(current_state):
            return actions

        if (current_state not in visited) or (current_cost < visited[current_state]):
            visited[current_state] = current_cost
            for successor, action, step_cost in problem.getSuccessors(current_state):
                new_cost = current_cost + step_cost
                if (successor not in visited) or (new_cost < visited.get(successor, float('inf'))):
                    priority = new_cost + heuristic(successor, problem)
                    fringe.push((successor, actions + [action], new_cost), priority)

    return []

bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch