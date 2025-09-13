#!/user/bin/python
#Author: yingwang

import os
import time
import re
import sys
import logging
import argparse 
import pandas as pd
import torch
import torch.nn as nn
import pickle

# Model 
from VAE_model import VAE
from VAE_trainer import VAETrainer
from VAE_Generator import SmilesVaeMoleculeGenerator

# utils
from utils.step_utils import build_scoring_function
from utils.step_utils  import set_random_seed
from utils.step_utils  import load_smiles_from_file
from utils.step_utils  import load_model
from utils.step_utils  import torch_device
from utils.step_utils  import pick_diverse_set
from utils.step_utils  import filter_results
from utils.step_utils  import get_raw_scores
from utils.step_utils  import str2bool
from utils.step_utils  import canonicalize_list
from utils.step_train_ligand_binding_model import train_ligand_binding_model
from version import __version__


def get_parser():
    """ Set up parser
    """
    parser = argparse.ArgumentParser(description="DARTING, VAE reinforcement learning",
      formatter_class=CustomFormatter)
    sub_parser = parser.add_subparsers(
      help="Commands Available",
      dest="command")

    # add subparsers
    train_parser(sub_parser) #train
    generate_parser(sub_parser) #generation
    score_parser(sub_parser) # scoring molecules
    sample_parser(sub_parser) # sample funciton
    load_parser(sub_parser) #load parametors
    train_ligand_binding_model_parser(sub_parser) #train ligand model
    return parser


## Define commad line parsing
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """help description"""
    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL,argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    if type(action.default) == type(sys.stdin):
                        print(action.default.name)
                        help += '(default:' + str(action.default.name) + ")"
                    else:
                        help += ' (default: %(default)s)'
        return help


    
## functions that used in several class
def global_arguments(sub_parser):
    """common arguments
    Args:
    sub_parser: argparer.subparser
    Returns:
    None
    """
    # optional runtime behavior
    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")
    opt_runtime.add_argument("--version", action="version", version=__version__)
    opt_runtime.add_argument("--verbose", action="store_true", default=False, help="Verbose output.")
    opt_runtime.add_argument("--quiet", action="store_true", default=False, help="Supress all warnings and info.  Superceeds --verbose.")
    opt_runtime.add_argument("--debug", action="store_true", default=False, help="Debug output. Superceeds --verbose, --debug.")

def scoring_arguments(sub_parser):
    """ Add scoring file path arguments
                 
        Args:   
            sub_parser: argparer.subparser 
        Returns:
            None    
    """
    # Optional runtime behavior
    req_scoring_io = sub_parser.add_argument_group("Files Required for Scoring") 

    #sa model
    req_scoring_io.add_argument('--fscores',
        #required=True,
        default="/cluster/home/ywang/multi_target/code_package/code_to_github/DARTING/data/fpscores.pkl.gz",
        action="store",
        help="Path to fscores.")
    #method for dealing score
    req_scoring_io.add_argument('--opti',
        choices=['gauss','linear'],
        default='gauss',
        help="Reinforcement learner score modifier class."
        )
    #parameters for define total scoring function,the defaults need to modify by yourself
    req_scoring_io.add_argument("--scoring_definition",
        default="/cluster/home/ywang/multi_target/code_package/code_to_github/DARTING/data/data/scoring_definition.csv",
        required=False,
        help='Path to scoring function defintion.')



#### define 6 argument groups

###1. arguments for Generator

def generate_parser(parser):
    """Add arguments for generation"""
    #define sub_parser
    sub_parser = parser.add_parser("generate",
                                   help='Generate SMILES by reinforcement learning')
    
    # required arg-group
    req_io_group = sub_parser.add_argument_group("Required I/O Arugments")
    req_io_group.add_argument("--model_path",
                              default = None,
                              required = True,
                              help = ""
                              )
    #add scoring argumetn group
    scoring_arguments(sub_parser)

    #optinal arg-group
    opt_io_group = sub_parser.add_argument_group("Optional I/O Arguments")
    opt_io_group.add_argument("--model_def",
                              type = str,
                              default = 'VAEmodel',
                              choices=[
                            'VAEmodel',
                            'VAEmodel_moses',
                            ])
    opt_io_group.add_argument("--outF",
                              default = './',
                              help="the outdir for results"
                              )
    opt_io_group.add_argument("--starting_population",
                              default=None,
                              help='Pretrain on SMILES list'
                              )
    
    #optinal runtime
    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")
    ## need modify
    opt_runtime.add_argument('--max_len',
        default=100,
        type=int,
        help='Max length of a SMILES string')
    opt_runtime.add_argument("--seed",
                            default=None,
                            type = int,
                            help= 'Random seed'
                            )
    opt_runtime.add_argument("--keep_top",
                            default= 512,
                            type = int,
                            help= 'Molecules kept each step'
                             )
    opt_runtime.add_argument('--n_epochs',
        default=20,
        type=int,
        help='Epochs to sample')
    opt_runtime.add_argument('--mols_to_sample',
        default=1024,
        type=int,
        help='Molecules sampled at each step')
    opt_runtime.add_argument('--optimize_batch_size',
        default=256,
        type=int,
        help='Batch size for the optimization')
    opt_runtime.add_argument('--optimize_n_epochs',
        default=2,
        type=int,
        help='Number of epochs for the optimization')
    opt_runtime.add_argument('--pretrain_n_epochs',
        default=2,
        type=int,
        help='Number of epochs for training on starting population.')
    opt_runtime.add_argument('--benchmark_num_samples',
        default=4096,
        type=int,
        help='Number of molecules to generate from final model for the benchmark')
    opt_runtime.add_argument('--benchmark_trajectory',
        action='store_true',
        help='Take molecules generated during re-training into account for the benchmark')
    opt_runtime.add_argument("--device",
        type = torch_device,
        default="cpu",
        help='Device to run: "cpu" or "cuda:<device number>"')
    opt_runtime.add_argument('--save_frequency',
        type=int,
        default=5,
        help='How often to save the model')
    opt_runtime.add_argument('--save_payloads',
        action="store_true",
        default=False,
        help='Save the payloads for each epoch.')
    opt_runtime.add_argument('--save_individual_scores',
        default=False,
        action="store_true",
        help='Optionally save individual score values along with payloads. Contingent on --save_payloads.')
    #pretrain on init_population
    train_arg = sub_parser.add_argument_group("Optional Training Parameters")
    train_arg.add_argument('--n_batch',
        type=int, default=512,
        help='Batch size')
    train_arg.add_argument('--clip_grad',
        type=int, default=50,
        help='Clip gradients to this value')
    train_arg.add_argument('--kl_start',
        type=int, default=0,
        help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
        type=float, default=0,
        help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
        type=float, default=1,
        help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
        type=float, default=3 * 1e-4,
        help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
        type=int, default=10,
        help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
        type=int, default=6,
        help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
        type=int, default=1,
        help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
        type=float, default=3 * 1e-4,
        help='Maximum lr weight value')
    train_arg.add_argument('--n_last',
        type=int, default=1000,
        help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs',
        type=int, default=1,
        help='Number of threads')
    train_arg.add_argument('--n_workers',
        type=int, default=0,
        help='Number of workers for DataLoaders')

    # all global parse to this sub_parse, Optional runtime behavior
    global_arguments(sub_parser)

    return sub_parser 
                                
###2. arguments for Trainner

def train_parser(parser):
    """add argument and arg-group for trainning"""
    sub_parser = parser.add_parser("train",help="Train a VAE SMILES model")

    # required arg-group
    req_group = sub_parser.add_argument_group("Required arguments")
    req_group.add_argument('--train_data',
                           type = str,
                           required=True,
                           help='input data to train'
                           )
    
    #optional arg-group
    opt_group = sub_parser.add_argument_group("Optinal I/O Arguments")
    opt_group.add_argument("--model_def",
                           type = str,
                           default = 'VAEmodel',
                           choices=[
                            'VAEmodel',
                            'VAEmodel_moses',
                            ])
    opt_group.add_argument(
        '--validation_data',
                            type=str,
                            required=False,
                            default=None,
                            help="Input data in csv format to validation")
    opt_group.add_argument('--model_save',
                            type=str,
                            default="model_{}.pt".format(int(time.time())),
                            help='Where to save the model')
    opt_group.add_argument('--save_frequency',
                            type=int,
                            default=20,
                            help='How often to save the model')
    opt_group.add_argument('--log_file',
                            type=str,
                            required=False,
                            help='Where to save the log')
    opt_group.add_argument('--vocab_save',
                            type=str,
                            help='Where to save the vocab')
    opt_group.add_argument('--vocab_load',
                            type=str,
                            help='Where to load the vocab; '
                                 'otherwise it will be evaluated')


    train_arg = sub_parser.add_argument_group("Optional Training Parameters")
    train_arg.add_argument("--n_epoch",
                            type=int,
                            default=50,
                            help="Number of training epochs")
    train_arg.add_argument("--device",
                            type = torch_device,
                            default="cpu",
                            help='Device to run: "cpu" or "cuda:<device number>". Use cuda:')
    train_arg.add_argument('--seed',
                            type=int,
                            default=0,
                            help='Seed')
    train_arg.add_argument('--n_batch',
                           type=int, default=512,
                           help='Batch size')
    train_arg.add_argument('--clip_grad',
                           type=int, default=50,
                           help='Clip gradients to this value')
    train_arg.add_argument('--kl_start',
                           type=int, default=0,
                           help='Epoch to start change kl weight from')
    train_arg.add_argument('--kl_w_start',
                           type=float, default=1.0,
                           help='Initial kl weight value')
    train_arg.add_argument('--kl_w_end',
                           type=float, default=1.0,
                           help='Maximum kl weight value')
    train_arg.add_argument('--lr_start',
                           type=float, default=3 * 1e-4,
                           help='Initial lr value')
    train_arg.add_argument('--lr_n_period',
                           type=int, default=10,
                           help='Epochs before first restart in SGDR')
    train_arg.add_argument('--lr_n_restarts',
                           type=int, default=6,
                           help='Number of restarts in SGDR')
    train_arg.add_argument('--lr_n_mult',
                           type=int, default=1,
                           help='Mult coefficient after restart in SGDR')
    train_arg.add_argument('--lr_end',
                           type=float, default=3 * 1e-4,
                           help='Maximum lr weight value')
    train_arg.add_argument('--n_last',
                           type=int, default=1000,
                           help='Number of iters to smooth loss calc')
    train_arg.add_argument('--n_jobs',
                           type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers',
                           type=int, default=0,
                           help='Number of workers for DataLoaders')


    model_arg = sub_parser.add_argument_group('Model Arguments')
    model_arg.add_argument('--q_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Encoder rnn cell type')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--q_d_h',
                           type=int, default=256,
                           help='Encoder hidden dimensionality')
    model_arg.add_argument('--q_n_layers',
                           type=int, default=1,
                           help='Encoder number of layers')
    model_arg.add_argument('--q_dropout',
                           type=float, default=0.5,
                           help='Encoder layers dropout')
    model_arg.add_argument('--d_cell',
                           type=str, default='gru', choices=['gru'],
                           help='Decoder rnn cell type')
    model_arg.add_argument('--d_n_layers',
                           type=int, default=3,
                           help='Decoder number of layers')
    model_arg.add_argument('--d_dropout',
                           type=float, default=0.2,
                           help='Decoder layers dropout')
    model_arg.add_argument('--d_z',
                           type=int, default=128,
                           help='Latent vector dimensionality')
    model_arg.add_argument('--d_d_h',
                           type=int, default=512,
                           help='GRU hidden dimensionality')
    model_arg.add_argument('--m_dropout',
                            type=float,
                            default=0.2,
                            help="Middle layer dropout")
    model_arg.add_argument('--n_mid_layers',
                           type=int, default=1,
                           help='Number of linear middle layers in both decoder and encoder.')  
    model_arg.add_argument('--batchnorm_conv',
                           type=str2bool,
                           default=True,
                           help='Perform batch normalization in convolution layers. Default True. To turn off --batchnorm_conv [false, 0, n, no]')  
    model_arg.add_argument('--batchnorm_mid',
                           type=str2bool,
                           default=True,
                           help='Perform batch normalization in mid layers. Default True. To turn off --batchnorm_conv [false, 0, n, no]')  
    model_arg.add_argument('--lambda_scale',
                           action="store",
                           type=float, default=1,
                           help='Adjusting random noise scale.')   
    model_arg.add_argument('--freeze_embeddings',
                           default=False, action='store_true',
                           help='If to freeze embeddings while training')

    # Optional runtime behavior
    global_arguments(sub_parser)

    return sub_parser

###3. arguments for Sample

def sample_parser(parser):
    """Add subparser for passing filter generation"""
    sub_parser = parser.add_parser("sample",help="Sample SMILES from a model")

    #required parameters
    req_io_group = sub_parser.add_argument_group("Required I/O Arguments")
    req_io_group.add_argument('--model_path',
        default=None,
        required=True,
        help='Path to the pre-trained SMILES VAE model')
    
    #score function
    scoring_arguments(sub_parser)

    #optinal arguments
    opt_io_group = sub_parser.add_argument_group("Optional I/O Arguments")
    opt_io_group.add_argument('--output',
        default="/dev/stdout",
        type=str,
        help="Output file for results")
    #other optinal parameters for running
    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")
    opt_runtime.add_argument("--n_molecules",
        default=10,
        type=int,
        help='Number of molecules to sample from model.')
    opt_runtime.add_argument("--device",
        type = torch_device,
        default="cpu",
        help='Device to run: "cpu" or "cuda:<device number>"')

    opt_runtime.add_argument("--filter",
        default=False,
        action="store_true",
        help="Filter Scores")
    opt_runtime.add_argument("--n_diverse",
        default=None,
        action="store",
        type=int,
        help="Get Diverse Set of n molecules")

    # Optional runtime behavior
    global_arguments(sub_parser)

    return sub_parser 

#
###4. arguments for training ligand-binding model
def train_ligand_binding_model_parser(parser):
    """ Add subparser arguments for passing filter generation """
    sub_parser = parser.add_parser("train_ligand_binding_model", help="Train a Random Forest Regressor Model for Target-Ligand Binding Prediction")

    # required files #
    req_io_group = sub_parser.add_argument_group("Required I/O Arguments")
    req_io_group.add_argument('--uniprot_id',
        default=None,
        required=True,
        help='Target Protein UniProt ID')
    req_io_group.add_argument('--binding_db_path',
        default=None,
        required=True,
        help='Path to the BindingDB data')


    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")
    opt_runtime.add_argument("--output_path",
        default=None,
        type=str,
        help='Path to write pickled sklearn model to.')

    # Optional runtime behavior
    global_arguments(sub_parser)
    return sub_parser


## 5. arguments for scoring smiles individual
def score_parser(parser):
    """ Add subparser arguments for generation """
    sub_parser = parser.add_parser("score", help="Score SMILES strings.")

    # required files
    req_io_smiles_group = sub_parser.add_mutually_exclusive_group(required=True)
    req_io_smiles_group.add_argument('--smi',
        default=None,
        help='Path to list of SMILES to score.')
    req_io_smiles_group.add_argument("--csv",
        default=None,
        help='Path to csv containing SMILES to score. Expects a column with header "smiles".')
    req_io_smiles_group.add_argument('--model_path',
        default=None,
        help='Path to the pre-trained SMILES VAE model')
    scoring_arguments(sub_parser)


    # optional files
    opt_io_group = sub_parser.add_argument_group("Optional I/O Arguments")
    opt_io_group.add_argument('--output',
        default="/.",
        type=str,
        help="Output file for results")

    # optional runtime
    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")

    opt_runtime.add_argument("--n_molecules",
        default=None,
        type=int,
        help='Number of molecules to sample from model.')
    opt_runtime.add_argument("--n_top",
        default=None,
        type=int,
        help='Number of best scoring molecules to keep.')
    opt_runtime.add_argument("--device",
        type = torch_device,
        default="cpu",
        help='Device to run: "cpu" or "cuda:<device number>"')

    opt_runtime.add_argument("--filter", default=False, action="store_true", help="Filter Scores")
    opt_runtime.add_argument("--n_diverse", default=None, action="store", type=int, help="Get Diverse Set of n molecules")

    # Optional runtime behavior
    global_arguments(sub_parser)

    return sub_parser 


#### 6. arguments for load model
def load_parser(parser):
    """ arguments for load model, Add subparser arguments for passing filter generation """
    sub_parser = parser.add_parser("load", help="Load a model for interactive.")

    # required files
    req_io_group = sub_parser.add_argument_group("Required I/O Arguments")
    req_io_group.add_argument('--model_path',
        default=None,
        required=True,
        help='Path to the pre-trained SMILES VAE model')

    opt_runtime = sub_parser.add_argument_group("Optional Runtime Arguments")
    opt_runtime.add_argument("--device",
        type = torch_device,
        default="cpu",
        help='Device to run: "cpu" or "cuda:<device number>"')

    # Optional runtime behavior
    global_arguments(sub_parser)

    return sub_parser 

#### Main function define, 6 functions that respond to 6 group args####

#1. train
def train_main(args):
    """Run Training
    Args:
    args (argparse.Namespace):command line arguments

    Returns:
    None
    
    """
    device = torch.device(args.device)
    logging.debug(f"Devive:{device}")
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    
    #read data for trainning: smiles
    with open(args.train_data,"r") as handle:
        logging.debug("get training data")
        train_data = [line.strip() for line in handle.readlines()]
    
    if args.validation_data:
        logging.debug("get validation data")
        with open(args.validation_data,"r") as handle:
            validation_data = [line.strip() for line in handle.readlines()]
    else:
        validation_data = None
    
    # get the custom vocabulary:
    if args.vocab_load:
        logging.warning("current ignoring custom vocab")
    else:
        pass

    #initial mode
    model = VAE(**vars(args))
    model.to(device)

    #set up trainner
    trainer =  VAETrainer(model,**vars(args))

    #train
    trainer.fit(train_data,validation_data)
    logging.info("Training Complete")
#1. train

#
#2. generator
def generate_main(args):
    """Run generation
     Args:
            args (argparse.Namespace): command line arguments 
        Returns:
            None
    """
    device = torch.device(args.device)
    if device.type.startswith("cuda"):
        torch.cuda.set_device(device.index or 0)
    
    logger = logging.getLogger()

    #set the seed if provided:
    if args.seed:
        set_random_seed(args.seed,args.device)

    # # get reinforcement objective
    logging.debug("reinforcement and making scoring function,")

    # scoring
    if args.save_individual_scores:
        #是否保存单项打分
        print("save individual score")
        scorers,scoring_function = build_scoring_function(
            args.scoring_definition,
            #args.enconding
            #args.cell_lin_model
            args.fscores,
            opti = args.opti,
            return_individual=True,
            vae_model=VAE
            )
    else:
        ## not reruen individual socre, but need to chek the units of build scoring function
        scorers = None
        scoring_function = build_scoring_function(
            args.scoring_definition,
            #args.enconding
            #args.cell_lin_model
            args.fscores,
            opti = args.opti,
            return_individual=False,
            vae_model=VAE
            )
    ## mkdir for save output
    os.system('mkdir -p {}'.format(args.outF))

    #initialize model
    model = load_model(VAE,args.model_path,args.device)

    # fetch innitial population?
    if args.starting_population is None:
        logging.info("Random start is True")
        starting_population = []
    else:
        if os.path.exists(args.starting_population):
            logging.info("taking starting population from file")
            starting_population = load_smiles_from_file(args.starting_population)
        else:
            # is empty or sample in model?
            try:
                num_init_sample = int(args.starting_population)
                logging.info()
                logging.info(f'Sampling starting population size of: {num_init_sample}') 
                starting_population = model.sample(num_init_sample)
            except ValueError:
                logging.error(f'Did not understand starting population: {args.starting_population}\n')
                logging.error(f'Please provide either integer number to sample or path to a population of smiles strings.')
                raise ValueError()     
    logging.info("Objective: ")
    logging.info(scoring_function)       


    # generating smiles
    #define generator
    generator = SmilesVaeMoleculeGenerator(model=model,
                                           max_len = args.max_len,
                                           device = device,
                                           out_dir = args.outF,
                                           lr = args.lr_start
                                           )
    #genrating
    molecules = generator.optimise(objective=scoring_function,
                                   start_population=starting_population,
                                   n_epochs=args.n_epochs,
                                   mols_to_sample=args.mols_to_sample,
                                   keep_top=args.keep_top,
                                   optimize_batch_size=args.optimize_batch_size,
                                   optimize_n_epochs=args.optimize_n_epochs,
                                   pretrain_n_epochs=args.pretrain_n_epochs,
                                   ind_scorers=scorers, #初始分子的打分
                                   save_payloads=args.save_payloads, #保存新分子
                                   save_frequency=args.save_frequency #保存model的频率，默认每10次一保存
                                   )

    #save
    with open(os.path.join(args.outF, "GDM_final_molecules.txt"),'w') as handle:
        for m in molecules:
            handle.write("{}\n".format(m.smiles))

    logger.info("Molecule Generation Complete")

#2. generator
    
# 
##3. score_main
def score_main(args):
    """scoring smiles that load from files"""
    logger = logging.getLogger()
    logger.info("Getting Molecules")

    names = None
    # get molecules
    if args.smi:
        with open(args.smi, 'r') as handle:
            molecules = [line.rstrip() for line in handle.readlines()]
    elif args.csv:
        input_table = pd.read_csv(args.csv, sep=",", header=0)
        molecules = input_table['smiles'].tolist()
    elif args.model_path:
        model = load_model(VAE, args.model_path, args.device)
        molecules = model.sample(args.n_molecules)
    else:
        raise RuntimeError("No molecules or model to sample from provided.")
    
    # santize molecuels 
    #canonicalized_samples = list(set(canonicalize_list(molecules, include_stereocenters=True)) )
    canonicalized_samples = molecules
    scorers, scoring_function = build_scoring_function( 
                            args.scoring_definition,
                            args.fscores,
                            opti=args.opti,
                            return_individual = True,
                            vae_model = VAE)

    logger.info("Scoring")
    df = get_raw_scores(canonicalized_samples, scorers, aggregate_scoring_function=scoring_function)

    if args.filter:
        df = filter_results(df, mean=False)


    # only report the top molecules
    if args.n_top is not None:
        df = df.sort_values('Aggregate', ascending=False)
        df = df.iloc[:args.n_top,:]

    if args.n_diverse is not None:
        diverse_smiles = pick_diverse_set(df['smiles'].tolist(), n_diverse=args.n_diverse)
        df = df[df['smiles'].isin(diverse_smiles)]
    
    # if a table was provided, merge with scores
    if args.csv:
        df = pd.merge(input_table, df, on="smiles")
    df.to_csv(args.output, header=True, index=False, sep=",")
    logger.info("Complete")
# scoring main

# 
#4. samplr molecule from model
def sample_main(args):
    """get nee molecules from model"""

    logger = logging.getLogger()
    logger.info("Getting Molecules")

    model = load_model(VAE, args.model_path, args.device)
    scorers, scoring_function = build_scoring_function(args, return_individual=True)

    c = 0
    passing = []
    while c<args.n_molecules:
        s = model.sample(1)
        s = canonicalize_list(s)
        if len(s)==0:
            continue
        r = get_raw_scores(s,scorers,scoring_function)
        if args.filter:
            f = filter_results(r,mean=False)
            if f.shape[0]==1:
                passing.append(f)
                c+=1
        else:
            passing.append(r)
            c+=1
        logging.debug(f'Current Count: {c}. New molecule: {s[0]}')

    passing = pd.concat(passing)
    passing.to_csv(args.output, header=True, index=False, sep=",")
    logger.info("Complete")
#sample

#
#5. load model
def load_main(args):
    model = load_model(VAE,args.model_path,args.device)
    return model


# 
#6.train_ligand_binding model
def train_ligand_binding_model_main(args):
    train_ligand_binding_model(args.uniprot_id,
                               args.binding_db_path,
                               args.output_path
                               )


# 
## main function
def main():
    """Main"""

    # get all parameters
    parser= get_parser()
    args= parser.parse_args()
    if args.command == None:
        parser.print_help()
        sys.exit(0)

    # set up logger
    level = logging.WARNING
    if args.quiet:
        level = logging.ERROR
    if args.verbose:
        level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(format='%(asctime)s [%(levelname)-8s] %(message)s', level=level) 

    #build command line
    # Run selected command
    r = None # dummy return
    if args.command == "train":
        train_main(args)
    elif args.command == "generate":
        generate_main(args)
    elif args.command == 'score':
        score_main(args)
    elif args.command == 'sample':
        sample_main(args)
    elif args.command == "load":
        r = load_main(args)
    elif args.command == "train_ligand_binding_model":
        r = train_ligand_binding_model_main(args)   
    else:
        logging.error("Did not recognize command.")

    return r



# 
if __name__ == '__main__':
    main()


