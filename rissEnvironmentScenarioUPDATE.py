'''
 This is an environment that takes into account the following:
 - 2 agents
 - one is a predator, the other is the prey
 - the prey could either be an enemy or an ally
 - the ally's goal is green
 - the enemy's goal is red
 - Reward:
 - for prey: go to the goal and record the negative distance
 - predator:
 -    ally: negative magnitude of action
 -    enemy: negative distance of the relative position of predator and enemy
 - Observation function takes agent as an input
 - check the action space of each environment
 - print action and observation space
​
author: Beverley-Claire Okogwu (CMU RISS 2020 Scholar)
mentors: Dr.Ding Zhao & Mengdi Xu
email: beverley.okogwu@gmail.com, okogwub@dickinson.edu
'''
#import statements:
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        #setting the world properties...
        world.dim_c=2
        num_agents=2
        world.num_agents = num_agents
        num_adversaries=1
        num_landmarks=2
        #adding the agents...
        #probability that agent is an enemy/ally
        prob_enemy=np.random.uniform(0,1) # TODO: use uniform 0-1 and pick a threshold between 0, 1 (done & confirmed)
        threshold = 0.5      
        world.agents = [Agent() for i in range(num_agents)]
        for i,agent in enumerate(world.agents):
            agent.name='agent %d' % i
            agent.collide = False # TODO: 
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3            
            # agent either an ally or an enemy           
            agent.ally= True if prob_enemy<threshold and i<num_agents else False          
            agent.enemy=True if prob_enemy>=threshold and i<num_agents else False
        #adding the landmarks...
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2 #make landmark very big
            landmark.boundary = False
        #making the initial conditions...
        self.reset_world(world)
        return world
    
    def reset_world(self, world):        
        #agents' random properties...
        #reset probabilities of agents: there is a 50% chance of the agent being an ally and a 50% chance of the agent being an enemy if not an adversary
        prob_enemy=np.random.uniform(0,1)
        threshold = 0.5        
        for i,agent in enumerate(world.agents):
            if agent.ally:                
               agent.color= np.array([0,128,0]) if not agent.adversary else np.array([0,0,255])#green
               agent.goal_a = world.landmarks[1]                
            elif agent.enemy:
               agent.color=  np.array([255,0,0])if not agent.adversary else np.array([0,0,255])#red
               agent.goal_a = world.landmarks[0]
        # TODO: if the color is defined before assigning the type, there will be mismatch between the color and the true type, which may be misleading when debugging (done)
        #set one landmark to red (enemy goal)
        world.landmarks[0].color = np.array([255, 0, 0])        
        #set the other landmark to green (ally goal)
        world.landmarks[1].color = np.array([0, 128, 0])
        #setting the goal landmark...
        # TODO: the agent object may raise bug since it is defined in the previous forloop, try to define it clearly (done)
        # TODO: the goal_a is a property of the agent, and hence should be agent.goal_a (done)
        #setting the random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
    #adapted from simple_adversary.py
    
    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for l in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
            dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
            return tuple(dists)

    def good_agents(self,world):
        return [agent for agent in world.agents if not agent.adversary]
    def enemy_agents(self, world):
        return [agent for agent in world.agents if  agent.enemy]    
    def ally_agents(self, world):
        return [agent for agent in world.agents if  agent.ally]
    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]
    def reward(self, agent, world):
        # if agent is an enemy, rewarded based on negative distance to each landmark
        if agent.enemy:
            rwd = self.enemy_reward(agent, world)
        elif agent.ally:
            rwd = self.ally_reward(agent, world)
        else:
            rwd = self.adversary_reward(agent, world)
        return rwd
        
    def enemy_reward(self, agent, world):
        # Rewarded based on how close the enemy is to the goal landmark, and how far the adversary is from it
        shaped_reward = True
        shaped_adv_reward = True
        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5
        # Calculate positive reward for agents
        enemy_agents = self.enemy_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in enemy_agents]) # TODO: good agents not defined
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in enemy_agents]) \
                    < 2 * agent.goal_a.size:
                pos_rew += 5
            pos_rew -= min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in enemy_agents])
        return pos_rew + adv_rew
    def ally_reward(self, agent, world):
        # Reward is the negative magnitude of the action OR simply distance from landmark
        # the distance to the goal
        return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark
        shaped_reward = True
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 2 * agent.goal_a.size:
                adv_rew += 5
            return adv_rew
    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)
