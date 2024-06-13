import datetime
from exp_utils.exp_fold import pre_train
import argparse
from Train.Train_HybGNN import train
import logging

def main(args):
    dest=pre_train(args.model_name, args.model_name)
    logging.info("Argument values:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    train(args,dest)
    current_time = datetime.datetime.now()
    logging.info(f'Training end at : {current_time} \n')
    logging.info(f'Info saved in : {dest} \n\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple Deep Learning Model Training')
    parser.add_argument('--model_name', type=str, default='HybGNN', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--dataset_path', type=str, default='', help='')
    parser.add_argument('--flag', type=str, default='IAGNN',
                        help='IAGNN or FGNN, to choose the integration method of GPUM')

    # Model parameter
    parser.add_argument('--inputsp', type=int, default=1000,
                        help='input data samples of raw EEG segment, some para should be revised accoding to unputsp')
    parser.add_argument('--GNN_out_dim', type=int, default=8, help='')
    parser.add_argument('--adj_dim', type=int, default=64, help='dim of QK')
    parser.add_argument('--chan_num', type=int, default=19, help='')
    parser.add_argument('--cls_num', type=int, default=2, help='')
    parser.add_argument('--F1', type=int, default=8, help='for EEGEncoder')
    parser.add_argument('--D', type=int, default=2, help='for EEGEncoder')
    parser.add_argument('--N_next', type=int, default=4, help='for SGP   4 or 2')

    # Training parameter
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--miu', type=float, default=1e-5, help='if miu<0,no entropy regularization')
    parser.add_argument('--total_epoch', type=int, default=100, help='')

    args = parser.parse_args()
    main(args)
