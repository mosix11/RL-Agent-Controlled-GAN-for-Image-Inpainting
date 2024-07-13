import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision

# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 400)
# 		self.l2 = nn.Linear(400, 400)
# 		self.l3 = nn.Linear(400, 300)
# 		self.l4 = nn.Linear(300, action_dim)
  
# 		self.max_action = max_action
  
# 		for m in self.modules():
# 			if isinstance(m, nn.Linear):
# 				nn.init.xavier_normal_(m.weight)
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0)

	
# 	def forward(self, x):
# 		x = F.leaky_relu(self.l1(x))
# 		x = F.leaky_relu(self.l2(x))
# 		x = F.tanh(self.l3(x))
# 		# Scale the value of the actions from [-1, 1] which is the output of tanh to the range [-max_action, max_action]
# 		# to cover the range of standard normal distributin
# 		action = self.max_action * F.tanh(self.l4(x)) 
# 		return action 


# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim):
# 		super(Critic, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 400)
# 		self.l2 = nn.Linear(400 + action_dim, 300)
# 		self.l3 = nn.Linear(300, 300)
# 		self.l4 = nn.Linear(300, 1)
  
# 		for m in self.modules():
# 			if isinstance(m, nn.Linear):
# 				nn.init.xavier_normal_(m.weight)
# 				if m.bias is not None:
# 					nn.init.constant_(m.bias, 0)


# 	def forward(self, x, z):
# 		x = F.relu(self.l1(x))
# 		x = F.relu(self.l2(torch.cat([x, z], 1)))
# 		x = self.l3(x)
# 		q_value = self.l4(x)
# 		return q_value 

# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 2048)
# 		self.l2 = nn.Linear(2048, 1024)
# 		self.l3 = nn.Linear(1024, 512)
# 		self.l4 = nn.Linear(512, action_dim)
		
# 		self.max_action = max_action
  



# 	def forward(self, state):
# 		out = F.relu(self.l1(state))
# 		out = F.relu(self.l2(out))
# 		out = F.relu(self.l3(out))
# 		action = self.max_action * torch.tanh(self.l4(out)) 
# 		return action


# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim):
# 		super(Critic, self).__init__()

# 		# Q1 architecture
# 		self.l1 = nn.Linear(state_dim + action_dim, 2048)
# 		self.l2 = nn.Linear(2048, 1024)
# 		self.l3 = nn.Linear(1024, 512)
# 		self.l4 = nn.Linear(512, 1)

# 		# Q2 architecture
# 		self.l5 = nn.Linear(state_dim + action_dim, 2048)
# 		self.l6 = nn.Linear(2048, 1024)
# 		self.l7 = nn.Linear(1024, 512)
# 		self.l8 = nn.Linear(512, 1)
  



# 	def forward(self, state, action):
# 		sa = torch.cat([state, action], 1)

# 		out1 = F.relu(self.l1(sa))
# 		out1 = F.relu(self.l2(out1))
# 		out1 = F.relu(self.l3(out1))
# 		q1 = self.l4(out1)

# 		out2 = F.relu(self.l5(sa))
# 		out2 = F.relu(self.l6(out2))
# 		out2 = F.relu(self.l7(out2))
# 		q2 = self.l8(out2)
# 		return q1, q2


# 	def Q1(self, state, action):
# 		sa = torch.cat([state, action], 1)

# 		out1 = F.relu(self.l1(sa))
# 		out1 = F.relu(self.l2(out1))
# 		out1 = F.relu(self.l3(out1))
# 		q1 = self.l4(out1)
# 		return q1

# class Actor(nn.Module):
# 	def __init__(self, state_dim, action_dim, max_action):
# 		super(Actor, self).__init__()

# 		self.l1 = nn.Linear(state_dim, 400)
# 		self.l2 = nn.Linear(400, 400)
# 		self.l3 = nn.Linear(400, 300)
# 		self.l4 = nn.Linear(300, action_dim)
		
# 		self.max_action = max_action
  


# 	def forward(self, state):
# 		out = F.relu(self.l1(state))
# 		out = F.relu(self.l2(out))
# 		out = F.relu(self.l3(out))
# 		action = self.max_action * torch.tanh(self.l4(out)) 
# 		return action


# class Critic(nn.Module):
# 	def __init__(self, state_dim, action_dim):
# 		super(Critic, self).__init__()

# 		# Q1 architecture
# 		self.l1 = nn.Linear(state_dim + action_dim, 400)
# 		self.l2 = nn.Linear(400, 400)
# 		self.l3 = nn.Linear(400, 300)
# 		self.l4 = nn.Linear(300, 1)

# 		# Q2 architecture
# 		self.l5 = nn.Linear(state_dim + action_dim, 400)
# 		self.l6 = nn.Linear(400, 400)
# 		self.l7 = nn.Linear(400, 300)
# 		self.l8 = nn.Linear(300, 1)
  



# 	def forward(self, state, action):
# 		sa = torch.cat([state, action], 1)

# 		out1 = F.relu(self.l1(sa))
# 		out1 = F.relu(self.l2(out1))
# 		out1 = F.relu(self.l3(out1))
# 		q1 = self.l4(out1)

# 		out2 = F.relu(self.l5(sa))
# 		out2 = F.relu(self.l6(out2))
# 		out2 = F.relu(self.l7(out2))
# 		q2 = self.l8(out2)
# 		return q1, q2


# 	def Q1(self, state, action):
# 		sa = torch.cat([state, action], 1)

# 		out1 = F.relu(self.l1(sa))
# 		out1 = F.relu(self.l2(out1))
# 		out1 = F.relu(self.l3(out1))
# 		q1 = self.l4(out1)
# 		return q1


class Actor(nn.Module):
    def __init__(self, state_dim,  action_dim, max_action):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.state_dim, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, self.action_dim)

        self.max_action = max_action

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        out = F.leaky_relu((self.linear1(x)))
        out = F.leaky_relu((self.linear2(out)))
        out = F.leaky_relu((self.linear3(out)))
        out = F.tanh(self.linear4(out))
        out = self.max_action * F.tanh(self.linear5(out))
        return out
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.num_actions = action_dim
        

        self.linear1 = nn.Linear(self.state_dim, 2048)
        self.linear2 = nn.Linear(2048 + action_dim, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, state, z):
        out = (F.relu(self.linear1(state)))
        out = (F.relu(self.linear2(torch.cat([out, z], dim=1))))
        out = self.linear3(out)
        out = self.linear4(out)
        out = self.linear5(out)

        return out
    
    
class DDPG(nn.Module):
	def __init__(self, state_dim=1024, action_dim=8, action_value_range=2.6):
		super(DDPG, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_value_range = action_value_range

		self.actor = Actor(state_dim, action_dim, action_value_range)
		self.critic = Critic(state_dim, action_dim)

  
  
  
	def forward(self):
		pass
  
  
	def C_training_step(self, experience_batch):
		states, actions, next_states, rewards, dones = experience_batch

		target_Q = rewards.unsqueeze(1)
		current_Q = self.critic(states, actions)
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q) 
		return critic_loss

	def A_training_step(self, experience_batch):
		states, actions, next_states, rewards, dones = experience_batch
		actor_loss = -self.critic(states, self.actor(states)).mean()
		return actor_loss

	def select_action(self, state):
		return self.actor(state)


  

# class DDPG(nn.Module):
# 	def __init__(self, state_dim=1024, action_dim=8, discount=0.9, tau=5e-3, policy_noise=0.2, policy_noise_clip=0.5, action_value_range=2.6):
# 		super(DDPG, self).__init__()
# 		self.state_dim = state_dim
# 		self.action_dim = action_dim
# 		self.discount_factor = discount
# 		self.tau = tau
# 		self.policy_noise = policy_noise
# 		self.policy_noise_clip = policy_noise_clip
# 		self.action_value_range = action_value_range

# 		self.actor = Actor(state_dim, action_dim, action_value_range)
# 		self.actor_target = Actor(state_dim, action_dim, action_value_range)
# 		self.actor_target.load_state_dict(self.actor.state_dict())

# 		self.critic = Critic(state_dim, action_dim)
# 		self.critic_target = Critic(state_dim, action_dim)
# 		self.critic_target.load_state_dict(self.critic.state_dict())	
  
  
  
# 	def forward(self):
# 		pass
  
  
# 	def C_training_step(self, experience_batch):
# 		states, actions, next_states, rewards, dones = experience_batch

# 		with torch.no_grad():
# 			# Select action according to policy and add clipped noise
# 			noise = (
# 				torch.randn_like(actions) * self.policy_noise
# 			).clamp(-self.policy_noise_clip, self.policy_noise_clip)
			
# 			next_action = (
# 				self.actor_target(next_states) + noise
# 			).clamp(-self.action_value_range, self.action_value_range)

# 			# Compute the target Q value
# 			target_Q1, target_Q2 = self.critic_target(next_states, next_action)
# 			target_Q = torch.min(target_Q1, target_Q2)
# 			target_Q = rewards.unsqueeze(1) + dones.unsqueeze(1) * self.discount_factor * target_Q


# 		# Get current Q estimates
# 		current_Q1, current_Q2 = self.critic(states, actions)

# 		# Compute critic loss
# 		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
  
# 		return critic_loss

# 	def A_training_step(self, experience_batch):
# 		states, actions, next_states, rewards, dones = experience_batch
# 		actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
# 		return actor_loss

# 	def select_action(self, state):
# 		return self.actor(state)

# 	def update_target_networks(self):
# 		# Update the frozen target models
# 		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
# 			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
# 			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# class DDPG(nn.Module):
# 	def __init__(self, state_dim=1024, action_dim=8, discount=0.9, tau=5e-3, action_value_range=2.6):
# 		super(DDPG, self).__init__()
# 		self.state_dim = state_dim
# 		self.action_dim = action_dim
# 		self.discount_factor = discount
# 		self.tau = tau
# 		self.action_value_range = action_value_range
# 		self.actor = Actor(state_dim, action_dim, action_value_range)
# 		self.actor_target = Actor(state_dim, action_dim, action_value_range)
# 		self.actor_target.load_state_dict(self.actor.state_dict())

# 		self.critic = Critic(state_dim, action_dim)
# 		self.critic_target = Critic(state_dim, action_dim)
# 		self.critic_target.load_state_dict(self.critic.state_dict())	
  
  
  
# 	def forward(self):
# 		pass
  
  
# 	def C_training_step(self, experience_batch):
# 		states, actions, next_states, rewards, dones = experience_batch
# 		# Compute the target Q value
# 		target_Q = self.critic_target(next_states, self.actor_target(next_states))
# 		target_Q = rewards.unsqueeze(1) + (dones.unsqueeze(1) * self.discount_factor * target_Q).detach()
		
#   		# Get current Q estimate
# 		current_Q = self.critic(states, actions)
  
# 		# Compute critic loss
# 		critic_loss = F.mse_loss(current_Q, target_Q)

# 		return critic_loss

# 	def A_training_step(self, experience_batch):
# 		states, actions, next_states, rewards, dones = experience_batch
# 		actor_loss = -self.critic(states, self.actor(states)).mean()
# 		return actor_loss

# 	def select_action(self, state):
# 		return self.actor(state)

# 	def update_target_networks(self):
# 		# Update the frozen target models
# 		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
# 			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
# 			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



class Environment(nn.Module):
    def __init__(self, GAN, AE, action_dim=8, state_dim=1024, attempts=5):
        super(Environment,self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.attempts = attempts
        
        self.mse_loss = nn.MSELoss()
        
        self.GAN = GAN
        self.AE = AE


    def get_state(self, batch):
        with torch.no_grad():
            GFV = self.AE.get_GFV(batch)
        return GFV
    



    def forward(self, state, action, gt_imgs=None):
        if gt_imgs == None:
            return self.eval_forward(state, action)
        else:
            return self.train_forward(state, action, gt_imgs)
    
    
    def train_forward(self, state, action, gt_img):
        with torch.no_grad():
            
            # The GFV of ground truth images (with no mask)
            GFV_real = self.AE.get_GFV(gt_img)
            
            # The state is the GFV of masked image or in other words noisy GFV
            GFV_noisy_real = state
            # recons_noisy_real = self.AE.decode_GFV(GFV_noisy_real)
            
            # Chosen action by the agent is the latent vector input to the generator
            GFV_clean_fake = self.GAN.generator(action)

            # Discriminator Output
            disc_fake = self.GAN.discriminator(GFV_clean_fake)
            
            # Reconstructing the fake output
            recons_clean_fake = self.AE.decode_GFV(GFV_clean_fake)


        # Discriminator Loss
        loss_D = torch.mean(disc_fake)

        # Loss Between Noisy GFV and Clean GFV
        # Y^ is the GFV generated by the action of RL as the input of GAN 
        # Y is the GFV of ground truth unmasked images generated by AE encoder
        loss_GFV = self.mse_loss(GFV_clean_fake, GFV_real)

        # Norm Loss
        loss_norm = torch.norm(action)

        # Images loss
        # Y^ is the reconstruction of the GFV generated by GAN as the input to AE decoder
        # Y is the ground truth unmasked images
        loss_img_recon = self.AE.loss(recons_clean_fake, gt_img)
        
        
        D_coef = 0.1
        GFV_coef = 0.4
        img_coef = 0.4
        norm_coef = 0.1
        reward_D = loss_D * D_coef
        reward_GFV = -loss_GFV * GFV_coef
        reward_img_recon = -loss_img_recon * img_coef
        reward_norm = -loss_norm * norm_coef
        # Reward Formulation
        reward =  reward_D + reward_GFV  + reward_img_recon  + reward_norm 
        # reward = reward_GFV
        # reward = reward_norm + reward_GFV
        
        losses = [loss_D*D_coef, loss_GFV*GFV_coef, loss_norm*norm_coef, loss_img_recon*img_coef]
        done = True
        next_state = GFV_clean_fake
        return next_state, reward, done , losses
    
    def eval_forward(self, state, action):
        with torch.no_grad():
            GFV_noisy_real = state
            GFV_clean_fake = self.GAN.generator(action)
            
        next_state = GFV_clean_fake
        return next_state, None, True