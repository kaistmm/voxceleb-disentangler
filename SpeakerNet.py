#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, sys, random
import time, itertools, importlib

from DatasetLoader import test_dataset_loader
from torch.cuda.amp import autocast, GradScaler


class WrappedModel(nn.Module):
    ## The purpose of this wrapper is to make the model structure consistent between single and multi-GPU
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.module = model

    def forward(self, x, label=None, train_E_S=False, spk_indices=None):
        return self.module(x, label, train_E_S, spk_indices)


class SpeakerNet(nn.Module):
    def __init__(self, model, optimizer, trainfunc, nPerSpeaker, **kwargs):
        super(SpeakerNet, self).__init__()
        self.nPerSpeaker = nPerSpeaker

        self.__S__    = importlib.import_module("models." + model).__getattribute__("MainModel")(**kwargs)               # Speaker embedding network
        self.__DE__   = importlib.import_module("models." + 'Autoencoder').__getattribute__("MainModel")(input_dims=self.__S__.out_dims, **kwargs)       # Disentangler based on Auto-encoder
        self.__E_E__  = importlib.import_module("models." + 'ENV_Discriminator').__getattribute__("MainModel")(**kwargs) # Env discriminator for env vector
        self.__E_S__  = importlib.import_module("models." + 'ENV_Discriminator').__getattribute__("MainModel")(**kwargs) # Env discriminator for spk vector

        LossFunction = importlib.import_module("loss." + trainfunc).__getattribute__("LossFunction")
        
        #kwargs['nOut'] = self.__DE__.spk_dims   # 'nOut' should be same as 'spk_dims' value
        self.__L__     = LossFunction(**kwargs) # Speaker recognition loss
        self.__L_C__   = importlib.import_module("loss." + 'MAPC').__getattribute__("LossFunction")(**kwargs) # Correlation loss
        self.__L_AE__  = torch.nn.L1Loss()      # Reconstruction loss

        self.main_net_params = []
        self.disc_spk_params = []

        # Independently collect parameters speaker network (+ Auto-encode) and Discriminator of speaker feature.
        for __nn__ in [self.__S__, self.__DE__, self.__E_E__, self.__L__, self.__L_C__, self.__L_AE__]:
            self.main_net_params += list(__nn__.cuda().parameters())

        for __nn__ in [self.__E_S__]:
            self.disc_spk_params  += list(__nn__.cuda().parameters())

        print("SpeakerNet + Disentangler params size :", sum(p.numel() for p in  list(self.__DE__.parameters()) + list(self.__S__.parameters())))


    def forward(self, data, label=None, train_E_S=False, spk_indices=None):
        """
        A bundle of objective functions
        """
        if not train_E_S:
            data = data.reshape(-1, data.size()[-1]).cuda() # [S, B, T] or [B, T] -> [S*B, T] or [B, T]
            outp = self.__S__.forward(data)                 # [S*B, D]

            if label == None:
                spk_code, env_code = self.__DE__.disentangle_feat(outp)
                return spk_code, env_code
            else:
                outp = outp.reshape(self.nPerSpeaker, -1, outp.size()[-1]) # [S*B,D] -> [S,B,D] #.transpose(1, 0).squeeze(1)
                outp_recon, spk_code, env_code = self.__DE__.forward(outp, feat_swp=random.random() < 0.5)  # [S,B,D], [S,B,D], [S,B,D"], [S,B,D"], swap feature randomly (50%).
                
                # Reconstruction loss of Auto-encoder
                outp = outp.detach()
                nloss_recons = self.__L_AE__(outp_recon.reshape(-1, outp.size()[-1]), outp.reshape(-1, outp.size()[-1]))
                del outp, outp_recon

                # Correlation loss between spk_code and env_code
                nloss_corr = self.__L_C__(spk_code.reshape(-1, spk_code.size()[-1]), env_code.reshape(-1, env_code.size()[-1]))

                # Environment discriminator loss for env_code
                nloss_env_e     = self.__E_E__.forward(env_code.transpose(0,1), use_grl=False) 

                # Environment adversarial loss for speaker network : self.__S__
                nloss_env_s_adv = self.__E_S__.forward(spk_code.transpose(0,1), use_grl=True) # Activate GRL layer (Adversarial loss)
                del env_code

                # For triplet utterances, one utterance was just sampled randomly. 
                # So we exclude it when calculating speaker loss (to avoid duplication). But, this part is not important.
                spk_code     = spk_code.transpose(1, 0).squeeze(1) # [S*B,D"] -> [S,B,D"] -> [B,S,D"]
                B, S_        = spk_indices.size()
                spk_indices  = spk_indices.view(B, S_, 1).expand(B, S_, spk_code.size()[-1])
                spk_code_    = torch.gather(spk_code, 1, spk_indices)

                # Speaker recognition loss
                nloss_spk, prec1 = self.__L__.forward(spk_code_, label) 
                spk_code = spk_code.detach()

                return spk_code, nloss_spk, nloss_corr, nloss_env_e, nloss_env_s_adv, nloss_recons, prec1
        else:
            # Environment discriminator loss for spk_code
            nloss_env_s = self.__E_S__.forward(data, use_grl=False)  # Don't need to activate GRL layer
            return nloss_env_s


class ModelTrainer(object):
    def __init__(self, speaker_model, optimizer, scheduler, gpu, mixedprec, **kwargs):
        self.__model__ = speaker_model

        Optimizer = importlib.import_module("optimizer." + optimizer).__getattribute__("Optimizer")
        self.__optimizer__  = Optimizer(self.__model__.module.main_net_params, **kwargs)
        self.__optimizer2__ = Optimizer(self.__model__.module.disc_spk_params,  **kwargs)

        Scheduler = importlib.import_module("scheduler." + scheduler).__getattribute__("Scheduler")
        self.__scheduler__,  self.lr_step  = Scheduler(self.__optimizer__,  **kwargs)
        self.__scheduler2__, self.lr_step2 = Scheduler(self.__optimizer2__, **kwargs)

        self.scaler = GradScaler()
        self.gpu = gpu
        self.mixedprec = mixedprec

        assert self.lr_step in ["epoch", "iteration"]

    # ## ===== ===== ===== ===== ===== ===== ===== =====
    # ## Train network
    # ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader, verbose, adv_alpha=0, num_D_steps=5, lr_schedule=False):
        self.__model__.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss, loss_spk, loss_corr, loss_ee, loss_es, loss_es_adv, loss_rec = 0, 0, 0, 0, 0, 0, 0
        top1, top2 = 0, 0
        # EER or accuracy

        tstart = time.time()

        for data, data_label, spk_indices in loader:
            data = data.transpose(1, 0) # [B, S, T] -> [S, B, T]

            self.__model__.zero_grad()

            label = torch.LongTensor(data_label).cuda()
            spk_indices = spk_indices.cuda()

            with autocast(enabled=self.mixedprec): 
                spk_code, nloss_spk, nloss_corr, nloss_env_e, nloss_env_s_adv, nloss_recons, prec1 = self.__model__(data, label, spk_indices=spk_indices)

                nloss = nloss_spk + nloss_env_e + nloss_env_s_adv * adv_alpha + nloss_corr + nloss_recons

            if self.mixedprec:
                self.scaler.scale(nloss).backward()
                self.scaler.step(self.__optimizer__)
                self.scaler.update()
            else:
                nloss.backward()
                self.__optimizer__.step()

            self.__model__.zero_grad()
            with autocast(enabled=self.mixedprec): ## Environment Discriminator training for speaker vector
                for ii in range(num_D_steps):
                    nloss_env_s = self.__model__(spk_code, train_E_S=True)

                    if self.mixedprec:
                        self.scaler.scale(nloss_env_s).backward()
                        self.scaler.step(self.__optimizer2__)
                        self.scaler.update()
                    else:
                        nloss_env_s.backward()
                        self.__optimizer2__.step()

                    loss_env_s = nloss_env_s.detach().cpu().item()

                self.__optimizer2__.zero_grad()

            loss        += nloss.detach().cpu().item()
            loss_spk    += nloss_spk.detach().cpu().item()
            loss_corr   += nloss_corr.detach().cpu().item()
            loss_ee     += nloss_env_e.detach().cpu().item()
            loss_es     += nloss_env_s.detach().cpu().item()
            loss_rec    += nloss_recons.detach().cpu().item()
            loss_es_adv += nloss_env_s_adv.detach().cpu().item()
            top1        += prec1.detach().cpu().item()
            counter     += 1
            index       += stepsize
            telapsed    = time.time() - tstart
            tstart      = time.time()

            if verbose:
                sys.stdout.write("\rProcessing {:d} of {:d}:".format(index, loader.__len__() * loader.batch_size))
                sys.stdout.write("LossS {:.2f} LossCrr {:.3f} LossENV_S {:.3f} LossR {:.3f} TEER/TAcc ({:2.3f})% - {:.2f} Hz ".format(
                                  loss_spk / counter, loss_corr / counter, loss_es / counter, loss_rec / counter, top1 / counter, stepsize / telapsed))
                sys.stdout.flush()

            # if you need to schedule learning rate in iteration, you can use below code. But, implementation is not completed.
            """if self.lr_step == "iteration" and lr_schedule: 
                self.__scheduler__.step()
                self.__scheduler2__.step()
            """
        if self.lr_step == "epoch" and lr_schedule:
            self.__scheduler__.step()
            self.__scheduler2__.step()

        return (loss / counter, loss_spk / counter, loss_corr / counter, loss_rec / counter,  top1 / counter)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList(self, test_list, test_path, nDataLoaderThread, distributed, print_interval=100, num_eval=10, **kwargs):

        if distributed:
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        self.__model__.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, **kwargs)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            sampler = None

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=nDataLoaderThread, drop_last=False, sampler=sampler)

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1 = data[0][0].cuda()
            with torch.no_grad():
                ref_feat, _ = self.__model__(inp1)
            feats[data[1][0]] = ref_feat.detach().cpu()
            telapsed = time.time() - tstart

            if idx % print_interval == 0 and rank == 0:
                sys.stdout.write(
                    "\rReading {:d} of {:d}: {:.2f} Hz, embedding size {:d}".format(idx, test_loader.__len__(), idx / telapsed, ref_feat.size()[1])
                )

        all_scores = []
        all_labels = []
        all_trials = []

        if distributed:
            ## Gather features from all GPUs
            feats_all = [None for _ in range(0, torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(feats_all, feats)

        if rank == 0:

            tstart = time.time()
            print("")

            ## Combine gathered features
            if distributed:
                feats = feats_all[0]
                for feats_batch in feats_all[1:]:
                    feats.update(feats_batch)

            ## Read files and compute all scores
            for idx, line in enumerate(lines):
                data = line.split()

                ## Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                ref_feat = feats[data[1]].cuda()
                com_feat = feats[data[2]].cuda()

                if self.__model__.module.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                dist = torch.cdist(ref_feat.reshape(num_eval, -1), com_feat.reshape(num_eval, -1)).detach().cpu().numpy()

                score = -1 * numpy.mean(dist)

                all_scores.append(score)
                all_labels.append(int(data[0]))
                all_trials.append(data[1] + " " + data[2])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing {:d} of {:d}: {:.2f} Hz".format(idx, len(lines), idx / telapsed))
                    sys.stdout.flush()

        return (all_scores, all_labels, all_trials)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):

        torch.save(self.__model__.module.state_dict(), path)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.__model__.module.state_dict()
        loaded_state = torch.load(path, map_location="cuda:%d" % self.gpu)
        if len(loaded_state.keys()) == 1 and "model" in loaded_state:
            loaded_state = loaded_state["model"]
            newdict = {}
            delete_list = []
            for name, param in loaded_state.items():
                new_name = "__S__."+ name
                newdict[new_name] = param
                delete_list.append(name)
            loaded_state.update(newdict)
            for name in delete_list:
                del loaded_state[name]
        for name, param in loaded_state.items():
            origname = name

            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    
                    print("{} is not in the model.".format(origname))
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[name].size(), loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
