#! /user/bin/pthon
#Author : yingwang

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.moses_utils import OneHotVocab
from utils.step_smile_char_dict import SmilesCharDictionary
# define vocabulary
def get_vocabulary(data=None,device=None):
    '''
    build vocabulary optionally from data,defalut is no 
    Return:
    a vocabulary with the methods of char2id and id2char
    '''
    if data:
        vocabulary = OneHotVocab.from_data(data)
    else:
        
        sd = SmilesCharDictionary()
       
        vocabulary = OneHotVocab(sd.idx_char.values())

    if device:
       
        vocabulary.vectors = vocabulary.vectors.to(device)
    return vocabulary


class VAE(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

        self.q_cell = 'gru'
        self.q_bidir = False
        self.q_d_h = 256
        self.q_n_layers = 1 
        self.q_dropout = 0.5
        self.d_cell = 'gru'
        self.d_n_layers = 3
        self.d_z = 128
        self.d_d_h = 512
        self.d_dropout = 0
        self.freeze_embedddings =False
        self.vocabulary = None
        self.device= 'cpu'
        self.__dict__.update(kwargs)

        if self.vocabulary is None:
            self.vocabulary = get_vocabulary() 
        for ss in ('bos','eos','unk','pad'):
            setattr(self,ss,getattr(self.vocabulary,ss))
        
        n_vocab, d_emb = len(self.vocabulary),self.vocabulary.vectors.size(1)
        self.x_emb = nn.Embedding(n_vocab,d_emb,self.pad)

        self.x_emb.weight.data.copy_(self.vocabulary.vectors)

        if self.freeze_embedddings:
            self.x_emb.weight.requires_grad= False

        if self.q_cell == 'gru':
           
            self.encoder_rnn = nn.GRU(
                d_emb,
                self.q_d_h,
                num_layers = self.q_n_layers,
                batch_first= True,
                dropout = self.q_dropout if self.q_n_layers > 1 else 0,
                bidirectional =self.q_bidir

                )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )

        q_d_last = self.q_d_h * (2 if self.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last,self.d_z)
        self.q_logvar = nn.Linear(q_d_last, self.d_z) 

        if self.d_cell == 'gru':
            self.decoder_rnn = nn.GRU(
                d_emb + self.d_z, 
                self.d_d_h,
                num_layers= self.d_n_layers,
                batch_first=True,
                dropout=float(self.d_dropout) if self.d_n_layers > 1 else 0
                )
        else:
            raise ValueError(
                "Invalid q_cell type, should be one of the ('gru',)"
            )
        self.decoder_lat = nn.Linear(self.d_z,self.d_d_h)
        self.decoder_fc = nn.Linear(self.d_d_h,n_vocab)
        self.encoder= nn.ModuleList([self.encoder_rnn,
                                         self.q_mu,
                                         self.q_logvar])
        self.decoder = nn.ModuleList([
            self.decoder_lat,
            self.decoder_rnn,
            self.decoder_fc
            ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder]
        )
    def device(self):
        return next(self.parameters()).device


    def string2tensor(self,string,device='model'):
        '''conver id matrix to tensor'''
        ids = self.vocabulary.string2ids(string,add_bos=True,
                                 add_eos=True)
        tensor = torch.tensor(
        ids, dtype=torch.long,
        device=self.device
        )

        return tensor
    def tensor2string(self,tensor):
        '''conver id tensor to string'''
        ids = tensor.tolist()
        string = self.vocabulary.ids2string(
            ids,rem_bos=True,rem_eos=True)
        return string
    
    def get_collate_device(self):
        return self.device
    
    def get_collate_fn(self):
        device = self.get_collate_device()
        def collate(data):
            data.sort(key=len,reverse=True)
            tensors = [self.string2tensor(string,device=self.device) for 
                       string in data]
            return tensors
        return collate
    
    def forward(self,x):
        """
        param: x, batch ids tensor, tensor format
        return:float,kl term component of loass
        return: float,recon component of loss
        """
        z,kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x,z)

        return kl_loss,recon_loss
    
    def forward_encoder(self,x,return_mu=False):
        """Do encoder forward
      
        x: list of  batch ids tensor, long tensor
        return:(n_batch,d_z) of float,sample of latent vector z
        return: kl term compotent of loss
        
        """

        x = [self.x_emb(i_x) for i_x in x  ]

        x= nn.utils.rnn.pack_sequence(x)
        _,h =self.encoder_rnn(x,None)
        h = h[-(1 + int(self.encoder_rnn.bidirectional)):]
  
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        mu,logvar = self.q_mu(h),self.q_logvar(h)
        eps = torch.randn_like(mu)
        z= mu + (logvar/2).exp()*eps
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        if return_mu:
             return z, kl_loss, mu
        return z, kl_loss
    
    def encode(self,x):
        """
       
        param x:list of tensor of long, input  batch ids tensor
        Return: vector representing encode latent space
        """

        if not isinstance(x,list):
            x = [x]
        z,kl_loss,mu = self.forward_encoder(x,return_mu = True)
        return mu
    
    def forward_decoder(self,x,z,return_y = False):
        """
        :x, list of tensor of long,input id tensor
        :z, latent vector z, (n_batch,d_z)
        return:recon loss
        """
        lengths = [len(i_x) for i_x  in x]
        x= nn.utils.rnn.pad_sequence(
            x,batch_first=True,
            padding_value=self.pad)
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)

        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths,
                                                    batch_first=True)
        
   
        h_0 = self.decoder_lat(z) #self.d_z->self.d_d_h
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(x_input, h_0)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fc(output) # d_d_h -> n_vocab
        recon_loss = F.cross_entropy(
            y[:, :-1].contiguous().view(-1, y.size(-1)),
            x[:, 1:].contiguous().view(-1),
            ignore_index=self.pad
        )
        if return_y:
            return recon_loss,y
        return recon_loss
     
    def decode(self, z, x=None):
        if x is not None:
            rl, y = self.forward_decoder(x,z,return_y=True)
            xr = y.argmax(2)
     
            xrt = []
            for i in xr:
                try:
                    q = i[:(i==self.eos).nonzero()[0]]
                    xrt.append(q)
                except IndexError:
                    xrt.append(i)
            smiles = [self.tensor2string(i_x) for i_x in xrt]
        else:
            smiles = self.sample(z.shape[0], z=z, multinomial=True)
        return smiles

    def get_latent_space_from_smiles(self, smiles):
        """   
        :param x: list of tensors of longs, input sentence x
        :return: vector representing encoded latent space
        """

        x = self.string2tensor(smiles)
        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder([x])       

        return z[0]

    def sample_z_prior(self,n_bath):
        """sample z~p(z) = N(0,1)
       
        
        """
        return torch.randn(n_bath,self.q_mu.out_features,
                           device= self.x_emb.weight.device)
    
    def perturb_z(self,z,noise_norm,constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0,1,size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.uniform(
                    0,noise_norm,size=(z.shape[0],1))
                return z + torch.tensor(noise_amp * noise_vec,dtype=z.type)
            
        else:
            return z
    
    def sample(self,n_batch,max_len=100,z=None,temp=1.0,multinomial=True):
        """
        return: list of tensors of strings, samples sequence x
        """
        with torch.no_grad():
            if z is None:
                z= self.sample_z_prior(n_batch)
            
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)
             # Initial values
            h = self.decoder_lat(z)
           
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch) 
           
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch,
                                                                    max_len) 
            x[:, 0] = self.bos 
            

            end_pads = torch.tensor([max_len], device=self.device).repeat(
                n_batch)

            eos_mask = torch.zeros(n_batch, dtype=torch.bool,
                                   device=self.device)

            # Generating cycle
            for i in range(1, max_len):
       
                x_emb = self.x_emb(w).unsqueeze(1)
                
                x_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(x_input, h)
                y = self.decoder_fc(o.squeeze(1))
                y = F.softmax(y / temp, dim=-1) 
                if multinomial:
                    w = torch.multinomial(y, 1)[:, 0]
                    
                else:
                    w = torch.argmax(y,1)

                
               
                x[~eos_mask, i] = w[~eos_mask]
                i_eos_mask = ~eos_mask & (w == self.eos)
                
                test_condition = torch.zeros((w.shape)).bool().to(self.device)
                test_condition = test_condition | (w==self.eos)
                

                end_pads[i_eos_mask] = i + 1
                eos_mask = eos_mask | i_eos_mask

            # Converting `x` to list of tensors
            new_x = []
            for i in range(x.size(0)):
                new_x.append(x[i, :end_pads[i]])

            return [self.tensor2string(i_x) for i_x in new_x]