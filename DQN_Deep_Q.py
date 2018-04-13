#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse, time

from keras.layers import Input
#from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Model
from keras.utils import plot_model
from keras.models import load_model
from keras.optimizers import Adam
import random

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
            self.learning_rate = 0.35e-3 # for CartPole lr = 0.25e-3, need 2020 episodes; for MountainCar
        else:
            self.learning_rate = 1e-3
        
        print(self.learning_rate)
        self.inputs = Input(shape = (self.Num_input,))#input layer
        net = Dense(15, activation='relu')(self.inputs)#hidden 1
        net = Dense(20, activation='relu')(net)#hidden 2
        net = Dense(10, activation='relu')(net)#hidden3
        self.outputs =Dense(self.Num_output, activation='linear')(net)#output layer
        
        self.model = Model(inputs = self.inputs, outputs = self.outputs)
        
        
        #optmiz = keras.optimizers.RMSprop(lr=self.learning_rate)
        self.model.compile(optimizer=Adam(lr=self.learning_rate),loss='mean_squared_error')
        #self.model.compile(optimizer=optmiz,loss='mean_squared_error')
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
        #self.model.load_weights(self.path_weight)
        pass


        


class Replay_Memory():
    
    def __init__(self, memory_size=100000, burn_in=50000):
    #def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.memory_size = memory_size
        self.memory = []
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        minibatch = random.sample(self.memory, batch_size)
        return minibatch

    def append(self, transition):
        # Appends transition to the memory. 	
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory=self.memory[1:]




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
            np.random.seed(42)
            
        if environment_name == 'MountainCar-v0':
            self.iteration = float("inf")
            self.episode = 3000
        
        self.QNet = QNetwork(environment_name)
        
        if self.environment_name=='CartPole-v0':
            self.replay_memory = Replay_Memory()
        else:
            self.replay_memory = Replay_Memory(memory_size=100000, burn_in=50000)
        
        
        
        

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
        
        #self.QNet.load_model('save/MountainCar-v0DRM.h5')
        itera = 0
        episode = 0
        decay = (0.5-0.05)/100000
        epsilon= 1#0.5
        flag = 0
        perform_cross_time=[]
        old_Model = QNetwork(self.environment_name)#keras.models.clone_model(self.QNet.model)
        model_list = [self.QNet.model.get_weights(), self.QNet.model.get_weights()]#[old_model, curr_model]
        #model_list = [self.QNet.model, self.QNet.model]#[old_model, curr_model]
        average_reward_episode = []
        average_step = []
        average_loss = []
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
            
            
            
                                    
            input_state = np.reshape(initial_state, (1, self.Num_input))
            cur_state = initial_state
            
            
            if self.environment_name=='CartPole-v0' or self.environment_name =='SpaceInvaders-v0':
                gamma =0.99
                
            
            if self.environment_name == 'MountainCar-v0':
                gamma =1
            
            if itera > self.iteration or episode > self.episode:
                break
            
            
            while True:
                epsilon -= decay
                itera += 1
                #old_Model = model_list[0]
                old_Model.model.set_weights(model_list[0])
                
                q_values = self.QNet.model.predict(x = input_state)
                action = self.epsilon_greedy_policy(q_values[0], epsilon = max(epsilon,0.05))# initial action
                
                input_state_cur = np.reshape(cur_state, (1, self.Num_input))#S
                nextstate, reward, is_terminal, debug_info = self.env.step(action)#S'
                input_state_next = np.reshape(nextstate, (1, self.Num_input))#S'
                
                self.replay_memory.append([cur_state, action, reward, nextstate, is_terminal])#store transition into Memory
                
                
                if self.if_render:
                    self.env.render()
                
                ####################
                BATCH_num = 32
                minibatch = self.replay_memory.sample_batch(BATCH_num)
                
                inputs = np.zeros((BATCH_num, self.Num_input)) # matrix of (batch, state_dim)
                targets = np.zeros((inputs.shape[0], self.Num_output)) #matrix of (batch, action_dim)
                
                for i in list(range(0,len(minibatch))):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]
                    reward_t = minibatch[i][2]
                    nextstate_t = minibatch[i][3]
                    terminal_t = minibatch[i][4]
                    
                    inputs[i] = state_t
                    targets[i] = self.QNet.model.predict(x = np.reshape(state_t, (1, self.Num_input)))[0]
                    #Q_sa = self.QNet.model.predict(x = np.reshape(nextstate_t, (1, self.Num_input)))[0]
                    Q_sa = old_Model.model.predict(x = np.reshape(nextstate_t, (1, self.Num_input)))[0]#use previous weight for stability
                    
                    
                    if terminal_t:
                        targets[i, action_t]= reward_t
                    if not terminal_t:
                        targets[i, action_t]= reward_t + gamma * max(Q_sa)
                    
                    
                
                loss = self.QNet.model.train_on_batch(inputs, targets)
                model_list.append(self.QNet.model.get_weights())
                model_list = model_list[1:]
                ####################
                    
                
                cur_state = nextstate
                
                average_loss.append(loss)
                
                if itera % 10000 ==0:
                    flag=1

                if itera % 199 ==0:
                    cpy_ave_loss = sum(average_loss) / float(len(average_loss))
                    print("Average loss:",cpy_ave_loss)                    
                    average_loss=[]
                    
                
                if self.environment_name=='CartPole-v0' and itera % int(self.iteration/3.0)==0:
                    a = int(self.iteration/itera)
                    self.QNet.model.save("save/"+self.environment_name+"DRM_stab"+str(itera)+".h5")
                
                
                
                total_reward += discount * reward
                #discount *= gamma
                num_steps += 1
                
                if is_terminal:
                    break
            
            average_reward_episode.append(total_reward)
            average_step.append(num_steps)
            
            if self.environment_name == 'MountainCar-v0' and episode % int(self.episode/3.0)==0:
                    a = int(self.episode/episode)
                    self.QNet.model.save("save/"+self.environment_name+"DRM_stab"+str(episode)+".h5")

            
            if episode % 20==0:
                self.QNet.model.save("save/"+self.environment_name+"DRM_stab.h5")
                kk = self.test_in_train()
                if self.environment_name=='CartPole-v0':
                    
                    if episode > 1100 and cpy_ave_loss > 100:
                        print("Do decay!")
                        self.QNet.learning_rate = min(self.QNet.learning_rate/2.0, 0.00001)
                        self.QNet.model.compile(optimizer=Adam(lr=self.QNet.learning_rate),loss='mean_squared_error')
                        
                if self.environment_name=='MountainCar-v0':
                    if episode > 1100 and kk >- 110:
                        print("Do decay!")
                        self.QNet.learning_rate = min(self.QNet.learning_rate/2.0, 0.00001)
                        self.QNet.model.compile(optimizer=Adam(lr=self.QNet.learning_rate),loss='mean_squared_error')
            
            
            
            
            
                if flag ==1:
                	perform_cross_time.append(kk)
                	#perform_cross_time.append(sum(average_reward_episode) / float(len(average_reward_episode)))
                	flag=0
                	print(perform_cross_time)

                if episode % 20==0:
                    print("     Average reward:",kk,"epi:",episode,"itera:",itera,epsilon)
                    
                    
                    
                average_reward_episode=[]
                average_step=[]
        print(perform_cross_time)
            
    
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
        
    
    
    def test(self, model_file=None, numb=100):
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
            
            
            if self.if_render ==1:
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
                if self.if_render ==1:
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
                #discount *= gamma
                num_steps += 1
            
            print(total_reward)
            average_reward_episode.append(total_reward)
            average_step.append(num_steps)
            
        
            if episode >= 100:
                print("Average reward:",sum(average_reward_episode) / float(len(average_reward_episode)),
                    "Average step:",sum(average_step) / float(len(average_step)),"epi:",episode,'std:',np.std(average_reward_episode))
                average_reward_episode=[]
                average_step=[]
                break
    
        
    
    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        
        num_transition = 0
        print(self.replay_memory.burn_in)
        while num_transition < self.replay_memory.burn_in:
            curr_state = self.env.reset()
            
            while True:
                num_transition +=1
                action =self.env.action_space.sample()
                nextstate, reward, is_terminal, debug_info = self.env.step(action)
                
                self.replay_memory.append([curr_state,action,reward,nextstate,is_terminal]) #state, action, reward, next state, terminal flag tuples.
                

                if is_terminal or num_transition >= self.replay_memory.burn_in:
                    break
        #print(len(self.replay_memory.memory))
    



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
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    # Setting this as the default tensorflow session. 
    keras.backend.tensorflow_backend.set_session(sess)
    
    agent = DQN_Agent(environment_name,render=args.render)
    agent.burn_in_memory()
    
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

