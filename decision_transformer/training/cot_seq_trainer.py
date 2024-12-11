import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class CoTSequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, sub_goals = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)


        state_preds, action_preds, reward_preds, plan_preds, plan_encodes = self.model.forward(
            states, actions, rewards, latent_plan, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )


        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss_dt = torch.mean((action_preds - action_target)**2)

        # TODO: contrastive learning
        loss_cl = ...

        # TODO: KL divergence between output of dt and output of encoder
        # TODO: z_prior = output of encoder, z_posterior = output of DT
        # z_posterior = plan_preds
        # z_prior = plan_encodes
        loss_va = z_posterior.kl_divergence(z_prior)

        # TODO: rescale/weight
        loss = loss_dt + loss_va + loss_cl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/va_error'] = loss_va.detach().cpu().item()
            self.diagnostics['training/cl_loss'] = loss_cl.detach().cpu().item()

        return loss.detach().cpu().item()
    
