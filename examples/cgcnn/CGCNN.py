from os import path as osp
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from omegaconf import DictConfig
import paddle
import ppsci
import ppsci.constraint.supervised_constraint
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import ppsci.optimizer as optim
from ppsci.data.dataset import CGCNNDataset
from ppsci.data.dataset.cgcnn_dataset import collate_pool
from ppsci.arch import CrystalGraphConvNet
import hydra
paddle.device.set_device('cpu')

def evaluate(cfg:DictConfig):
    # load data
    dataset = CGCNNDataset(cfg.EVALUATE_DIR,input_keys='i',label_keys='l',id_keys='c')

    # build model
    structures, _, _ = dataset.raw_data[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=cfg.MODEL.ATOM_FEA_LEN,
                                n_conv=cfg.MODEL.N_CONV,
                                h_fea_len=cfg.MODEL.H_FEA_LEN,
                                n_h=cfg.MODEL.N_H)
        
    cgcnn_evaluate = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
        "sampler": {
            "name": "BatchSampler"
        },
        
        "dataset": {
            "name": "CGCNNDataset",
            "root_dir": cfg.EVALUATE_DIR,
            "input_keys": 'i',
            "label_keys": 'l',
            "id_keys": 'c'
        },
        
        "batch_size": cfg.EVAL.batch_size,         
        "collate_fn": collate_pool},
        loss=ppsci.loss.MAELoss('mean'),
        output_expr= {"l": lambda out: out["out"]},
        metric={"MAE":ppsci.metric.MAE()},
        name="cgcnn_evaluate",
    )
    validator = {cgcnn_evaluate.name: cgcnn_evaluate}
    solver = ppsci.solver.Solver(
        model,
        validator=validator,
        pretrained_model_path=cfg.EVAL.MODEL_PATH,
        cfg=cfg
    )
    
    solver.eval()

def train(cfg:DictConfig):

    # load data
    dataset = CGCNNDataset(cfg.TRAIN_DIR, input_keys='i',label_keys='l',id_keys='c')

    # build model
    structures, _, _ = dataset.raw_data[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=cfg.MODEL.ATOM_FEA_LEN,
                                n_conv=cfg.MODEL.N_CONV,
                                h_fea_len=cfg.MODEL.H_FEA_LEN,
                                n_h=cfg.MODEL.N_H)
    
    cgcnn_constraint = ppsci.constraint.SupervisedConstraint(
        dataloader_cfg={
        "sampler": {
            "name": "BatchSampler"
        },
        
        "dataset": {
            "name": "CGCNNDataset",
            "root_dir": cfg.TRAIN_DIR,
            "input_keys": 'i',
            "label_keys": 'l',
            "id_keys": 'c'
        },
        
        "batch_size": cfg.TRAIN.batch_size,         
        "collate_fn": collate_pool},
        loss=ppsci.loss.MAELoss('mean'),
        output_expr= {"l": lambda out: out["out"]},
        name='cgcnn_constraint',
    )
    
    constraint = {cgcnn_constraint.name: cgcnn_constraint}
    
    cgcnn_valid = ppsci.validate.SupervisedValidator(
        dataloader_cfg={
        "sampler": {
            "name": "BatchSampler"
        },
        
        "dataset": {
            "name": "CGCNNDataset",
            "root_dir": cfg.VALID_DIR,
            "input_keys": 'i',
            "label_keys": 'l',
            "id_keys": 'c'
        },
        
        "batch_size": cfg.TRAIN.batch_size,         
        "collate_fn": collate_pool},
        loss=ppsci.loss.MAELoss('mean'),
        output_expr= {"l": lambda out: out["out"]},
        metric={"MAE":ppsci.metric.MAE()},
        name="cgcnn_valid",
    )
    validator = {cgcnn_valid.name: cgcnn_valid}

    # Learning rate scheduler
    optimizer = optim.Momentum(learning_rate=cfg.TRAIN.lr, momentum=cfg.TRAIN.momentum,
                                   weight_decay=cfg.TRAIN.weight_decay)(model)


    solver = ppsci.solver.Solver(
        model=model,
        constraint=constraint,
        optimizer=optimizer,
        epochs=cfg.TRAIN.epochs,
        eval_during_train=True,
        validator=validator,
        equation=None,
        output_dir=cfg.output_dir,
        cfg=cfg
    )

    # train model
    solver.train()
    
    # evaluate model
    solver.eval()
    
@hydra.main(version_base=None, config_path="./conf", config_name="CGCNN_Demo.yaml")
def main(cfg:DictConfig):
    if cfg.mode == 'train':
        train(cfg)
    elif cfg.mode == 'eval':
        evaluate(cfg)
    else:
        raise ValueError(
            f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")
            
if __name__ == '__main__':
   main()
