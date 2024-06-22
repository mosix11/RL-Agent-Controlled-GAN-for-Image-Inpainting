import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 400)
		self.l3 = nn.Linear(400, 300)
		self.l4 = nn.Linear(300, action_dim)
		
		self.max_action = max_action

	
	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = self.max_action * torch.tanh(self.l4(x)) 
		return x 


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3_additional = nn.Linear(300, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, x, u):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(torch.cat([x, u], 1)))
		x = self.l3_additional(x)
		x = self.l3(x)
		return x 



class DDPG(nn.Module):
	def __init__(self, state_dim=1024, action_dim=8, action_value_range=20):
		super(DDPG, self).__init__()
		self.actor = Actor(state_dim, action_dim, action_value_range/2)
		self.actor_target = Actor(state_dim, action_dim, action_value_range/2)
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.critic = Critic(state_dim, action_dim)
		self.critic_target = Critic(state_dim, action_dim)
		self.critic_target.load_state_dict(self.critic.state_dict())	
  



	def select_action(self, state):
		return self.actor(state)


	def train(self, replay_buffer, iterations, batch_size=64, discount=0.99, tau=0.001):

		for it in range(iterations):

			# Sample replay buffer 
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(self.device)
			action = torch.FloatTensor(u).to(self.device)
			next_state = torch.FloatTensor(y).to(self.device)
			done = torch.FloatTensor(1 - d).to(self.device)
			reward = torch.FloatTensor(r).to(self.device)
			
			# Compute the target Q value
			target_Q = self.critic_target(next_state, self.actor_target(next_state))
			target_Q = reward + (done * discount * target_Q).detach()

			# Get current Q estimate
			current_Q = self.critic(state, action)

			# Compute critic loss
			critic_loss = F.mse_loss(current_Q, target_Q)

			# Optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()

			# Compute actor loss
			actor_loss = -self.critic(state, self.actor(state)).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class Environment(nn.Module):
    def __init__(self, GAN, AE, attempts=5):
        super(Environment,self).__init__()
        
        self.nll_loss = nn.NLLLoss()
        self.mse_loss = nn.MSELoss()

        # self.norm = Norm(dims=args.z_dim)
        # self.chamfer = ChamferLoss(args)
        
        self.GAN = GAN
        self.AE = AE

        self.j = 1
        self.attempts = attempts
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.lossess = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
        
    
    def reset(self):
        self.j = 1
        self.i = 0


    def get_state(self, batch):
        with torch.no_grad():
            GFV = self.AE.encoder(batch)
        return GFV
    
    
    def forward(self, input, action):
        with torch.no_grad():
            # Encoder Input
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)

            # Encoder  output
            encoder_out = self.model_encoder(input_var, )

            # D Decoder Output
#            pc_1, pc_2, pc_3 = self.model_decoder(encoder_out)
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = Variable(action, requires_grad=True).cuda()

            # Generator Output
            out_GD, _ = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            out_G = out_G.contiguous().view(-1, args.state_dim)

            # Discriminator Output
            #out_D, _ = self.model_D(encoder_out.view(-1,1,32,32))
        #    out_D, _ = self.model_D(encoder_out.view(-1, 1, 1,args.state_dim)) # TODO Alert major mistake
            out_D, _ = self.model_D(out_GD) # TODO Alert major mistake

            # H Decoder Output
#            pc_1_G, pc_2_G, pc_3_G = self.model_decoder(out_G)
            pc_1_G = self.model_decoder(out_G)


            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input_var, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC


        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(out_G, encoder_out)

        # Norm Loss
        loss_norm = self.norm(z)

        # Chamfer loss
        loss_chamfer = self.chamfer(pc_1_G, pc_1)  # #self.chamfer(pc_1_G, trans_input) instantaneous loss of batch items

        # States Formulation
        state_curr = np.array([loss_D.cpu().data.numpy(), loss_GFV.cpu().data.numpy()
                                  , loss_chamfer.cpu().data.numpy(), loss_norm.cpu().data.numpy()])
      #  state_prev = self.state_prev

        reward_D = state_curr[0]#state_curr[0] - self.state_prev[0]
        reward_GFV =-state_curr[1]# -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2]#-state_curr[2] + self.state_prev[2]
        reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = ( reward_D *0.01 + reward_GFV * 10.0 + reward_chamfer *100.0 + reward_norm*1/10)#( reward_D + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm*0.002) #reward_GFV + reward_chamfer + reward_D * (1/30)  TODO reward_D *0.002 + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm  ( reward_D *0.2 + reward_GFV * 100.0 + reward_chamfer *100 + reward_norm)
      #  reward = reward * 100
     #   self.state_prev = state_curr

        #self.lossess.update(loss_chamfer.item(), input.size(0))  # loss and batch size as input

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

       # if i % args.print_freq == 0 :

      #  if self.j <= 5:
        visuals = OrderedDict(
            [('Input_pc', trans_input_temp.detach().cpu().numpy()),
             ('AE Predicted_pc', pc_1_temp.detach().cpu().numpy()),
             ('GAN Generated_pc', pc_1_G_temp.detach().cpu().numpy())])
        if render==True and self.j <= self.figures:
         vis_Valida[self.j].display_current_results(visuals, self.epoch, self.i)
         self.j += 1

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1


      #  errors = OrderedDict([('loss', loss_chamfer.item())])  # plotting average loss
     #   vis_Valid.plot_current_errors(self.epoch, float(i) / self.epoch_size, args, errors)
        # if self.attempt_id ==self.attempts:
        #     done = True
        # else :
        #     done = False
        done = True
        state = out_G.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.lossess.avg
