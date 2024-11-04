import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class InvSequenceTrainer(Trainer):
    """ his is a version for a seperate model to model inverse dynamics """

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(rewards)

        state_preds, _ , _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        loss_dt = self.loss_fn(
            None, None, state_preds,
            None, None, state_target,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        #reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        #reward_target = reward_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

        loss_dt = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss = (1 / 2) * (loss_dt + inv_loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


class InvKSequenceTrainer(Trainer):
    """ his is a version for a seperate model to model inverse dynamics """

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(rewards)

        state_preds, _ , _ = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )

        loss_dt = self.loss_fn(
            None, None, state_preds,
            None, None, state_target,
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        #reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        #reward_target = reward_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

        loss_dt = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss = (1 / 2) * (loss_dt + inv_loss)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()

class UInvSequenceTrainer(Trainer):
    """ his is a version for a seperate model to model inverse dynamics """

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)
        state_target = torch.clone(states)
        state_dim = state_target.shape[2]
        act_dim = action_target.shape[2]
        state_target = torch.cat([torch.zeros((self.batch_size, 1, state_dim), device=state_target.device), state_target[:, 1:, :]], dim=1)

        state_preds, action_preds_dt , _, action_preds_inv = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
        )
        state_preds = torch.cat([torch.zeros((self.batch_size, 1, state_dim), device=state_preds.device), state_preds[:, :-1, :]], dim=1)

        # masking
        state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
        action_preds_dt = action_preds_dt.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_preds_inv = action_preds_inv.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss_pl = torch.mean((action_preds_dt - action_target)**2)

        loss_trans = torch.mean((state_preds - state_target)**2)
        loss_inva = torch.mean((action_preds_inv - action_target)**2)
        transit_weight = 0.5
        loss_inv = transit_weight * loss_trans + (1-transit_weight) * loss_inva
        
        loss = (1 / 2) * (loss_pl + loss_inv)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            # self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/action_dt_error'] = loss_pl.detach().cpu().item()
            self.diagnostics['training/action_inv_error'] = loss_inva.detach().cpu().item()
            self.diagnostics['training/state_error'] =  loss_trans.detach().cpu().item()

        return loss.detach().cpu().item()