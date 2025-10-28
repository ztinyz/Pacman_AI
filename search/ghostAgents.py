# ghostAgents.py
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


from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared: speed = 0.5

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class CoffeeGhostAgent(Agent):
    def __init__(self, index):
        Agent.__init__(self, index)
        self.index = index
        self.state = "SEEKING_COFFEE"  # States: SEEKING_COFFEE, SEEKING_GHOST
        self.coffeeTarget = None
        self.ghostTarget = None
        self.ghostTargetIndex = None 
        self.currentPath = []  # Store the computed path
        
    def getAction(self, state):
        """
        Main decision logic - loops between coffee and ghosts
        """
        legalActions = state.getLegalActions(self.index)
        
        if Directions.STOP in legalActions:
            legalActions.remove(Directions.STOP)
            
        if not legalActions:
            return Directions.STOP
        
        # Get current position
        myPos = state.getGhostPosition(self.index)
        
        # Check state transitions
        if self.state == "SEEKING_COFFEE":
            coffeeList = self.getCoffeePositions(state)
            if myPos in coffeeList:
                print(f"Ghost {self.index} reached coffee at {myPos}! Now seeking ghost.")
                self.state = "SEEKING_GHOST"
                self.ghostTarget = None  # Reset ghost target
                self.currentPath = []  # Reset path
                
        elif self.state == "SEEKING_GHOST":
            # Check if we reached the ghost target
            if self.ghostTarget and util.manhattanDistance(myPos, self.ghostTarget) < 2:
                print(f"Ghost {self.index} reached ghost at {self.ghostTarget}! Now seeking coffee.")
                self.state = "SEEKING_COFFEE"
                self.coffeeTarget = None
                self.ghostTarget = None
                self.ghostTargetIndex = None
                self.currentPath = []
        
        # Execute action based on current state
        if self.state == "SEEKING_COFFEE":
            return self.goToCoffee(state, legalActions)
        else:  # SEEKING_GHOST
            return self.goToRandomGhost(state, legalActions)
    
    def getCoffeePositions(self, state):
        """Get all coffee machine positions"""
        coffeeGrid = state.data.layout.coffee
        coffeeList = []
        for x in range(coffeeGrid.width):
            for y in range(coffeeGrid.height):
                if coffeeGrid[x][y]:
                    coffeeList.append((x, y))
        return coffeeList
    
    def bellmanFord(self, state, start, goal):
        """
        Bellman-Ford algorithm to find shortest path from start to goal.
        Returns the next action to take, or None if no path exists.
        """
        # Convert positions to integers
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Initialize distances
        distance = {}
        predecessor = {}
        
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    distance[(x, y)] = float('inf')
                    predecessor[(x, y)] = None
        
        # If start or goal is invalid (e.g., inside a wall), abort
        if start not in distance or goal not in distance:
            return None

        distance[start] = 0
        
        # Get all valid positions (nodes)
        nodes = [pos for pos in distance.keys()]
        
        # Relax edges |V|-1 times
        for _ in range(len(nodes) - 1):
            updated = False
            for node in nodes:
                if distance[node] == float('inf'):
                    continue
                    
                # Check all neighbors (up, down, left, right)
                x, y = node
                neighbors = [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1)
                ]
                
                for neighbor in neighbors:
                    nx, ny = neighbor
                    # Check if neighbor is valid (in bounds and not a wall)
                    if (0 <= nx < width and 0 <= ny < height and 
                        not walls[nx][ny]):
                        # Edge weight is 1 for all moves
                        if distance[node] + 1 < distance[neighbor]:
                            distance[neighbor] = distance[node] + 1
                            predecessor[neighbor] = node
                            updated = True
            
            # Early termination if no updates
            if not updated:
                break
        
        # Reconstruct path from goal to start
        if goal not in distance or distance[goal] == float('inf'):
            return None  # No path exists
        
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = predecessor[current]
            if current is None:
                return None
        path.reverse()
        
        return path
    
    def getActionFromPath(self, state, path):
        """Convert the next position in path to an action"""
        if not path:
            return None
            
        myPos = state.getGhostPosition(self.index)
        myPos = (int(myPos[0]), int(myPos[1]))
        nextPos = path[0]
        
        dx = nextPos[0] - myPos[0]
        dy = nextPos[1] - myPos[1]
        
        if dx == 1:
            return Directions.EAST
        elif dx == -1:
            return Directions.WEST
        elif dy == 1:
            return Directions.NORTH
        elif dy == -1:
            return Directions.SOUTH
        else:
            return Directions.STOP
    
    def goToCoffee(self, state, legalActions):
        """Move towards nearest coffee machine using Bellman-Ford"""
        myPos = state.getGhostPosition(self.index)
        myPos = (int(myPos[0]), int(myPos[1]))
        coffeeList = self.getCoffeePositions(state)
        
        if not coffeeList:
            return random.choice(legalActions)
        
        # Find nearest coffee by actual path distance
        minDistance = float('inf')
        bestPath = None
        
        for coffee in coffeeList:
            path = self.bellmanFord(state, myPos, coffee)
            if path and len(path) < minDistance:
                minDistance = len(path)
                bestPath = path
                self.coffeeTarget = coffee
        
        if bestPath:
            action = self.getActionFromPath(state, bestPath)
            if action and action in legalActions:
                return action
        
        # Fallback to random if Bellman-Ford fails
        return random.choice(legalActions)
    
    def goToRandomGhost(self, state, legalActions):
        """Move towards a random other ghost using Bellman-Ford"""
        myPos = state.getGhostPosition(self.index)
        myPos = (int(myPos[0]), int(myPos[1]))
        
        # Get other ghost positions
        otherGhosts = []
        for i in range(1, state.getNumAgents()):
            if i != self.index:
                ghostState = state.getGhostState(i)
                if not ghostState.isPacman:
                    ghostPos = state.getGhostPosition(i)
                    otherGhosts.append((int(ghostPos[0]), int(ghostPos[1])))
        
        if not otherGhosts:
            return random.choice(legalActions)
        
        # Pick a new random ghost target if we don't have one OR update existing target's position
        if self.ghostTarget is None:
            # First time: pick a random ghost index to follow
            self.ghostTargetIndex = random.choice([i for i in range(1, state.getNumAgents()) if i != self.index])
            print(f"Ghost {self.index} now targeting ghost {self.ghostTargetIndex}")
        
        # Always get the current position of the target ghost
        targetGhostPos = state.getGhostPosition(self.ghostTargetIndex)
        self.ghostTarget = (int(targetGhostPos[0]), int(targetGhostPos[1]))
        
        # Recalculate path every turn to track the moving ghost
        path = self.bellmanFord(state, myPos, self.ghostTarget)
        
        if path:
            action = self.getActionFromPath(state, path)
            if action and action in legalActions:
                return action
        
        # Fallback to random if Bellman-Ford fails
        return random.choice(legalActions)
    
    def getSuccessorPosition(self, state, action):
        """Get the position after taking an action"""
        myPos = state.getGhostPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        return (int(myPos[0] + dx), int(myPos[1] + dy))