#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, time

from keras.layers import Input
#from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from keras.optimizers import Adam
import math

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        #pass
        self.env = gym.make(environment_name)
        self.Num_input = len(self.env.observation_space.high)
        self.Num_output = self.env.action_space.__dict__['n']#get the number of action
        
        if environment_name=='CartPole-v0':
            self.learning_rate = 1e-4 # for CartPole lr = 0.25e-3, need 2020 episodes; for MountainCar
        else:
            self.learning_rate = 1e-3
        
        print(self.learning_rate)
        
        self.inputs = Input(shape = (self.Num_input,))
        self.outputs = Dense(self.Num_output, activation='linear')(self.inputs)
        
        self.model = Model(inputs = self.inputs, outputs = self.outputs)
        
        
        self.model.compile(optimizer=Adam(lr=self.learning_rate),loss='mean_squared_error')
        
        print("gooooooooood")
        #plot_model(self.model, to_file='model.png')
        
        self.path_model = 'model_linear.h5'
        self.path_weight = 'weight_linear.h5'
        

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save(self.path_model)
        self.model.save_weights(self.path_weight)

    def load_model(self, model_file):
        # Helper function to load an existing model.
        self.model=load_model(model_file)
        

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(self.path_weight)



class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        pass

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        pass

    def append(self, transition):
        # Appends transition to the memory. 	
        pass




class DQN_Agent():
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 

        #pass
        self.env = gym.make(environment_name)
        self.Num_input = len(self.env.observation_space.high)
        self.Num_output = self.env.action_space.__dict__['n']#get the number of action
        #print(environment_name,self.Num_input,self.Num_output)
        self.if_render = render
        self.environment_name = environment_name
        if environment_name=='CartPole-v0':
            self.iteration = 1000000
            self.episode = float("inf")
            
        if environment_name == 'MountainCar-v0':
            self.iteration = float("inf")
            self.episode = 3000
        
        self.QNet = QNetwork(environment_name)
        

    def epsilon_greedy_policy(self, q_values, epsilon = 0.05):
        # Creating epsilon greedy probabilities to sample from.             
        #pass
        non_max = epsilon/self.Num_output
        max_greedy = 1 - epsilon + epsilon/self.Num_output
        max_inx = np.argmax(q_values)
        
        elements = list(range(self.Num_output))
        probabilities = [non_max] * self.Num_output
        probabilities[max_inx] = max_greedy
        action = np.random.choice(elements, 1, p=probabilities) # sample from the distribution
        return action[0]
        

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        #pass
        return np.argmax(q_values)

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        #pass
        itera = 0
        episode = 0
        decay = (0.5-0.05)/100000
        epsilon=1#0.5
        average_reward_episode = []
        while True:
            
            episode +=1 
            initial_state = self.env.reset()
            
            
            average_step = []
            average_loss = []
            
            
            if self.if_render:
                self.env.render()

            total_reward = 0
            num_steps = 0
            discount = 1
            nextstate = initial_state# initial state
            #print(nextstate.shape)
            input_state = np.reshape(nextstate, (1, self.Num_input))
            #print(input_state)
            q_values = self.QNet.model.predict(x = input_state)
            
            action = self.epsilon_greedy_policy(q_values[0], epsilon = max(epsilon,0.05))# initial action
            #print(q_values,self.epsilon_greedy_policy(q_values[0], epsilon = 0.05),self.epsilon_greedy_policy(q_values, epsilon = 0.05))
            cur_state = nextstate
            
            #print(q_values)
            #print(action)
            
            if self.environment_name=='CartPole-v0' or self.environment_name =='SpaceInvaders-v0':
                gamma =0.99
            
            if self.environment_name == 'MountainCar-v0':
                gamma =0.1#1
            
            if itera > self.iteration or episode > self.episode:
                break
            
            #self.QNet.model.fit(x=input_state, y=q_values)
            #break
            while True:
                epsilon-=decay
                itera += 1
                input_state_cur = np.reshape(cur_state, (1, self.Num_input))#S
                nextstate, reward, is_terminal, debug_info = self.env.step(action)#S'
                input_state_next = np.reshape(nextstate, (1, self.Num_input))#S'
                if self.if_render:
                    self.env.render()
                
                
                
                if is_terminal:
                    q_values = self.QNet.model.predict(x = input_state_cur)
                    q_values[0][action] = reward
                    self.QNet.model.fit(x=input_state_cur, y=q_values,verbose=0,epochs=1)
                    break
                
                q_values_next = self.QNet.model.predict(x = input_state_next)#A'
                nextaction = self.epsilon_greedy_policy(q_values_next[0], epsilon = max(epsilon,0.05))#A'
                
                ##########
                q_values = self.QNet.model.predict(x = input_state_cur)#***********important!!!---Q(S', )
                #print(q_values,action)
                #q_values[0][action] = reward + gamma * q_values_next[0][nextaction]
                q_values[0][action] = reward + gamma * max(q_values_next[0])
                
                hist = self.QNet.model.fit(x = input_state_cur, y = q_values,verbose=0,epochs=1)
                
                #print(self.QNet.model.predict(x = input_state_cur)[0]-q_values[0])
                
                #########
                
                cur_state = nextstate
                action = nextaction
                
                average_loss.append(hist.history['loss'][0])
                
                
                if itera % 199 ==0:
                    
                    print("Average loss:",sum(average_loss) / float(len(average_loss)))
                    average_loss=[]
                
                total_reward += discount * reward
                discount *= gamma
                num_steps += 1
            
            average_reward_episode.append(total_reward)
            average_step.append(num_steps)
            
            if episode % 20==0:
                self.QNet.model.save("save/"+self.environment_name+"LQN.h5")
            
            if episode % 20==0:
                kk = self.test_in_train()
                
                
                self.QNet.model.save("save/"+self.environment_name+".h5")
                if episode % 20==0:
                    print("     Average reward:",kk,"epi:",episode,"itera:",itera,epsilon)
                    #print("Average reward:",sum(average_reward_episode) / float(len(average_reward_episode)),
                    #    "Average step:",sum(average_step) / float(len(average_step)),"epi:",episode,"itera:",itera,epsilon)
                average_reward_episode=[]
                average_step=[]
            

    def test_in_train(self):
        episode2=0
        average_reward_episode2 = []
        while True:

            episode2 +=1 
            initial_state2 = self.env.reset()
            
            average_step2 = []
            average_loss2 = []

            total_reward2 = 0
            num_steps2 = 0
            discount2 = 1
            nextstate2 = initial_state2# initial state

            input_state2 = np.reshape(nextstate2, (1, self.Num_input))

            q_values2 = self.QNet.model.predict(x = input_state2)
            action2 = self.greedy_policy(q_values2[0])# initial action
            cur_state2 = nextstate2


            while True:
                #itera += 1
                input_state_cur2 = np.reshape(cur_state2, (1, self.Num_input))#S
                nextstate2, reward2, is_terminal2, debug_info2 = self.env.step(action2)#S'
                input_state_next2 = np.reshape(nextstate2, (1, self.Num_input))#S'
                            
                if is_terminal2:
        
                    total_reward2 += discount2 * reward2
                    num_steps2 += 1
                    break
    
                q_values_next2 = self.QNet.model.predict(x = input_state_next2)#A'
                nextaction2 = self.greedy_policy(q_values_next2[0])#A'
    
    
                cur_state2 = nextstate2
                action2 = nextaction2
                total_reward2 += discount2 * reward2
                #discount *= gamma
                num_steps2 += 1

            #print(total_reward2)
            average_reward_episode2.append(total_reward2)
            average_step2.append(num_steps2)
        


            if episode2 >= 20:
                kk = sum(average_reward_episode2) / float(len(average_reward_episode2))
                print("Average reward:",sum(average_reward_episode2) / float(len(average_reward_episode2)),
                    "Average step:",sum(average_step2) / float(len(average_step2)),"epi:",episode2)
                average_reward_episode2=[]
                average_step2=[]
                break
        return kk
    
    
    
    
    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory. 
        #pass
        self.QNet.load_model(model_file)
        episode=0
        average_reward_episode = []
        while True:
            
            episode +=1 
            initial_state = self.env.reset()
            
            
            average_step = []
            average_loss = []
            
            
            if True:#self.if_render:
                self.env.render()

            total_reward = 0
            num_steps = 0
            discount = 1
            nextstate = initial_state# initial state
            
            input_state = np.reshape(nextstate, (1, self.Num_input))
            
            q_values = self.QNet.model.predict(x = input_state)
            action = self.greedy_policy(q_values[0])# initial action
            cur_state = nextstate
            
            
            if self.environment_name=='CartPole-v0' or self.environment_name =='SpaceInvaders-v0':
                gamma =0.99
            
            if self.environment_name == 'MountainCar-v0':
                gamma =1
            
            
            while True:
                
                #itera += 1
                input_state_cur = np.reshape(cur_state, (1, self.Num_input))#S
                nextstate, reward, is_terminal, debug_info = self.env.step(action)#S'
                input_state_next = np.reshape(nextstate, (1, self.Num_input))#S'
                if True:#self.if_render:
                    self.env.render()
                
                
                
                
                if is_terminal:
                    
                    total_reward += discount * reward
                    num_steps += 1
                    break
                
                q_values_next = self.QNet.model.predict(x = input_state_next)#A'
                nextaction = self.greedy_policy(q_values_next[0])#A'
                
                
                cur_state = nextstate
                action = nextaction
                
                
                
                total_reward += discount * reward
                discount *= gamma
                num_steps += 1
            
            average_reward_episode.append(total_reward)
            average_step.append(num_steps)
            
        
            if episode >= 100:
                print("Average reward:",sum(average_reward_episode) / float(len(average_reward_episode)),
                    "Average step:",sum(average_step) / float(len(average_step)),"epi:",episode)
                average_reward_episode=[]
                average_step=[]
                break
        
    
    def burn_in_memory():
        # Initialize your replay memory with a burn_in number of episodes / transitions. 

        pass
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str)
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    return parser.parse_args()





def run_random_policy(env):
    
    initial_state = env.reset()
    #env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        #env.render()
        
        print(debug_info)

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        #time.sleep(1)

    return total_reward, num_steps






def main(args):

    args = parse_arguments()
    environment_name = args.env

    #print(args.env, args.render, args.train, args.model_file)
    

    # Setting the session to allow growth, so it doesn't allocate all GPU memory. 
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)
    
    agent = DQN_Agent(environment_name,render=False)
    
    if args.train>0:
        agent.train()
    
    if args.model_file != None:
        agent.test(args.model_file)
    
    """
    env = gym.make(environment_name)
    print("state:",env.observation_space)
    print("state:",env.observation_space.high)
    print("state:",env.observation_space.low)
    print("action",env.action_space)
    
    epsilon =0.05
    Num_output = 5
    q_values = [0, 5 ,14, 1, 6]
    q_values = np.array([q_values])
    
    non_max = epsilon/Num_output
    max_greedy = 1 - epsilon + epsilon/Num_output
    max_inx = np.argmax(q_values)
    
    elements = list(range(Num_output))
    probabilities = [non_max] * Num_output
    probabilities[max_inx] = max_greedy
    action = np.random.choice(elements, 1, p=probabilities)
    print(probabilities)
    print(elements)
    print(action[0])
    """
    
    #run_random_policy(env)

    # You want to create an instance of the DQN_Agent class here, and then train / test it. 

if __name__ == '__main__':
    main(sys.argv)

