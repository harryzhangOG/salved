from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)
import logging
set_tf_loglevel(logging.FATAL)
import argparse
import warnings
warnings.filterwarnings('ignore')

import pprint

import re
import glob
import imageio

from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.modeling.models import BNN, LNN
from dmbrl.config import create_config
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf

import seaborn as sns
import pickle
from sklearn.neighbors import NearestNeighbors as knn

import ipdb
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def plot_returns(returns, dir=None):
    if type(returns) == str and os.path.exists(returns):
        log_path = returns

        if os.path.exists(os.path.join(log_path, 'logs.mat')):
            logging_data = sio.loadmat(log_path + '/logs.mat')
        else:
            all_logs = glob.glob(os.path.join(log_path, '*.mat'))
            assert all_logs, f'{log_path} must contain at least one .mat'
            print(f"Using {all_logs[0]}")
            logging_data = sio.loadmat(all_logs[0])
        returns = logging_data['returns'][0]

    savefig_name = 'returns.png'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    plt.plot(returns)
    print(len(returns))
    plt.xlabel('Iteration')
    plt.ylabel("Iteration Cost")
    plt.title("Training Curve")
    plt.ylim(0, 110)
    plt.savefig(savefig_path)
    plt.show()
    plt.close()


def plot_trajs(pred_paths, dir=None):
    savefig_name = 'pred_traj.png'

    for p in pred_paths:
        savefig_path = os.path.join(os.path.dirname(p), savefig_name)
        if os.path.exists(savefig_path):
            print(f"Skipping plot since {savefig_path} already exists")
            break
        pred_traj = sio.loadmat(p)['predictions'][:, 0]
        traj = np.array([np.mean(x, axis=1)[0] for x in pred_traj])[:, [0, 2]]
        plt.figure(figsize=(10, 5))
        plt.plot(traj[:, 0], traj[:, 1], marker='*')
        plt.xlim(-150, 50)
        plt.ylim(-50, 50)
        iteration = re.findall(r'train_iter(\d+)', p)[0]
        plt.title(f'Predicted trajectory at train_iter{iteration}')
        plt.gca().set_aspect("equal")
        plt.savefig(savefig_path)
        # plt.show()
        plt.close()

    savefig_name = 'pred_traj.gif'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    if os.path.exists(savefig_path):
        print(f"Skipping plot since {savefig_path} already exists")
        return

    images = []

    for p in pred_paths:
        images.append(imageio.imread(os.path.join(os.path.dirname(p), 'pred_traj.png')))

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved pred_traj to {savefig_path}")
    else:
        print("No pred_traj found.")


def plot_value_heatmap_gif(value_heatmap_paths, dir=None):
    savefig_name = 'value_heatmaps.gif'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    if os.path.exists(savefig_path):
        print(f"Skipping plot since {savefig_path} already exists")
        return

    images = []

    for fp in value_heatmap_paths:
        images.append(imageio.imread(fp))

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved value heatmap to {savefig_path}")
    else:
        print("No value heatmaps found.")


def plot_safe_set_gif(safe_set_paths, dir=None):
    savefig_name = 'safe_set.gif'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    if os.path.exists(savefig_path):
        print(f"Skipping plot since {savefig_path} already exists")
        return

    images = []

    for fp in safe_set_paths:
        images.append(imageio.imread(fp))

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved safe set to {savefig_path}")
    else:
        print("No safe set images found.")


def plot_safe_set_and_traj(train_iter_paths, dir=None):
    image_paths = []
    images = []

    for p in train_iter_paths:

        savefig_name = 'safe_set_and_traj.png'
        savefig_path = os.path.join(p, savefig_name)

        if os.path.exists(savefig_path):
            image_paths.append(savefig_path)
            print(f"Skipping plot since {savefig_path} already exists")
            continue

        stabilized_model_path = os.path.join(p, 'stabilized_model.pkl')

        if not os.path.isfile(stabilized_model_path):
            print(f"Skipping since {stabilized_model_path} not found")
            continue

        predictions_path = os.path.join(p, 'predictions.mat')
        if not os.path.isfile(predictions_path):
            print(f"Skipping since {predictions_path} not found")
            continue

        model = pickle.load(open(stabilized_model_path, 'rb'))
        x_list = np.arange(-150, 50.1, 0.5)
        y_list = np.arange(-50, 50.1, 0.5)[::-1]
        coords = np.transpose([np.tile(x_list, len(y_list)), np.repeat(y_list, len(x_list))])
        x_locs = coords[:, 0].reshape(-1, 1)
        x_vels = np.clip(np.random.normal(0, 0.2, coords.shape[0]), -1, 1).reshape(-1, 1)
        y_locs = coords[:, 1].reshape(-1, 1)
        y_vels = np.clip(np.random.normal(0, 0.2, coords.shape[0]), -1, 1).reshape(-1, 1)
        inputs = np.hstack([x_locs, x_vels, y_locs, y_vels])
        dist_values = np.array([model.kneighbors(x.reshape(1, -1))[0] for x in inputs])
        dist_values_plot = dist_values.reshape(len(y_list), len(x_list))
        plt.figure(figsize=(10 * len(x_list) // len(y_list), 10))
        ax = sns.heatmap(dist_values_plot, xticklabels=5, yticklabels=5)
        ax.set_xticklabels(x_list[::5])
        ax.set_yticklabels(y_list[::5])

        pred_traj = sio.loadmat(predictions_path)['predictions'][:, 0]
        traj = np.array([np.mean(x, axis=1)[1] for x in pred_traj])[:, [0, 2]]
        plt.title(f"Safe set and Predicted trajectory at {os.path.basename(p)}")
        plt.plot(traj[:, 0] + 2 * 150, -traj[:, 1] + 2 * 50, marker='*')
        plt.savefig(savefig_path)
        image_paths.append(savefig_path)
        plt.close()

    savefig_name = 'safe_set_and_traj.gif'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    for p in image_paths:
        if os.path.isfile(p):
            images.append(imageio.imread(p))

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved to {savefig_path}")
    else:
        print("No safe set and trajectories found.")


def gen_safe_set_and_traj_gif(image_paths, dir=None):
    images = []

    savefig_name = 'safe_set_and_traj.gif'
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    for p in image_paths:
        images.append(imageio.imread(p))

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved to {savefig_path}")
    else:
        print("No safe set and trajectories found.")


def combine_to_gif(image_paths, name=None):
    if not image_paths:
        return

    save_dir = os.path.dirname(image_paths[0])
    savefig_path = os.path.join(save_dir, f"{name if name else os.path.basename(image_paths[0]).split('.')[0]}.gif")
    if os.path.exists(savefig_path):
        print(f"Skipping {savefig_path} since already exists")
        return

    images = []
    for p in image_paths:
        try:
            images.append(imageio.imread(p))
            os.remove(p)
        except:
            traceback.print_exc()

    if images:
        imageio.mimsave(savefig_path, images)
        print(f"Saved to {savefig_path}")


def plot_model_error(paths, dir=None):
    if not paths:
        print("Skipping plot_model_error as no paths found")
        return
    is_train = 'train_losses' in paths[0]
    savefig_name = f"model_{'train' if is_train else 'validation'}_error.png"
    savefig_path = os.path.join(dir, savefig_name) if dir is not None else savefig_name

    if os.path.exists(savefig_path):
        print(f"Skipping plot since {savefig_path} already exists")
        return

    if not paths:
        print('Skipping model error plot as no .npy files found')
        return

    losses_all = np.array([])
    num_per_iter = 5

    for fp in paths:
        losses_loaded = np.load(fp)
        num_per_iter = len(losses_loaded)
        losses_all = np.hstack([losses_all, np.mean(losses_loaded, axis=1)])

    np.save(os.path.join(dir, f"{'train' if is_train else 'holdout'}_loss_all.npy"), losses_all)

    plt.plot(np.arange(0, 100, 1.0 / num_per_iter)[:len(losses_all)], losses_all)
    plt.xlabel('Iteration')
    plt.ylabel('Model error')

    plt.title(f"Model {'training' if is_train else 'holdout'} loss")
    plt.savefig(savefig_path)
    plt.close()


def plot_iteration_traj(iteration, log_dir=''):
    save_dir = os.path.join(*[log_dir, "plan_traj", f"train_iter_{iteration}"])
    print(f'Plotting iteration traj for {save_dir}')
    if glob.glob(f"{save_dir}/*.gif"):
        print(f"Skipping {save_dir} since already exists")
        return

    obstacles = []
    if 'pb2' in save_dir:
        obstacles = np.array([[[-30, -20], [-20, 20]]])
    elif 'pb3' in save_dir:
        obstacles = np.array([[[-30, -20], [-20, -10]], [[-30, -20], [0, 20]]])
    elif 'pb4' in save_dir:
        obstacles = np.array([[[-30, -20], [-20, 20]], [[-20, 5], [10, 20]], [[0, 5], [5, 10]], [[-20, 5], [-20, -10]]])

    with tf.variable_scope(f"iter", reuse=tf.AUTO_REUSE):
        value_func = LNN(DotMap({
            'name': 'value',
            'num_nets': 5,
            'load_model': True,
            'model_dir': os.path.join(*[log_dir, f"train_iter{iteration}"])
        }))
        value_func.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001}, suffix = "val")

        predictions_path = os.path.join(*[log_dir, f"train_iter{iteration}", 'predictions.mat'])
        predictions_mat = sio.loadmat(predictions_path)['predictions']

        safe_set_path = os.path.join(*[log_dir, f'train_iter{iteration}', 'stabilized_model.pkl'])
        if os.path.exists(safe_set_path):
            model = pickle.load(open(safe_set_path, 'rb'))
        else:
            model = pickle.load(open(os.path.join(*[log_dir, 'stabilized_model.pkl']), 'rb'))

        # Handle pointbot 1 case
        x_min, x_max = -100, 50
        y_min, y_max = -50, 50
        if predictions_mat[0][0].mean(axis=1)[0][0] < -80:
            x_min = -150

        # Safe set
        x_list = np.arange(x_min, x_max + 0.1, 0.5)
        y_list = np.arange(y_min, y_max + 0.1, 0.5)[::-1]
        coords = np.transpose([np.tile(x_list, len(y_list)), np.repeat(y_list, len(x_list))])
        x_locs = coords[:, 0].reshape(-1, 1)
        x_vels = np.clip(np.random.normal(0, 0.2, coords.shape[0]), -1, 1).reshape(-1, 1)
        y_locs = coords[:, 1].reshape(-1, 1)
        y_vels = np.clip(np.random.normal(0, 0.2, coords.shape[0]), -1, 1).reshape(-1, 1)
        inputs = np.hstack([x_locs, x_vels, y_locs, y_vels])
        dist_values = np.array([model.kneighbors(x.reshape(1, -1))[0] for x in inputs])
        dist_values_plot = dist_values.reshape(len(y_list), len(x_list))

        values_plot = value_func.predict(inputs, factored=True)[0]
        values_list = np.mean(values_plot, axis=0) + (3 * np.std(values_plot, axis=0)) / np.sqrt(len(values_plot))
        values_heatmap = values_list.reshape(len(y_list), len(x_list))

        mask = dist_values_plot <= 2
        dist_values_plot[:, :] = None
        dist_values_plot[mask] = values_heatmap[mask]

        save_dir = os.path.join(*[log_dir, "plan_traj", f"train_iter_{iteration}"])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image_paths = []

        for step in range(len(predictions_mat)):
            savefig_path = os.path.join(*[save_dir, f"time_{str(step).zfill(3)}.png"])
            image_paths.append(savefig_path)
            if os.path.exists(savefig_path):
                continue
            predictions = predictions_mat[step]
            traj = predictions[0].mean(axis=1)
            plan_hor = len(traj) - 1
            pred_cost = predictions[1][0][0]
            plt.figure(figsize=(10 * len(x_list) // len(y_list), 10))

            ax = sns.heatmap(dist_values_plot, xticklabels=5, yticklabels=5)

            for x in obstacles:
                ax.add_patch(patches.Rectangle(((x[0][0] - x_min) / 0.5, (-x[1][1] - y_min) / 0.5), np.diff(x[0]) / 0.5, np.diff(x[1]) / 0.5, fill=False, color='gray'))

            ax.set_xticklabels(x_list[::5])
            ax.set_yticklabels(y_list[::5])
            plt.text((x_max - x_min - 27) / 0.5 + 100, (y_max - y_min - 5) / 0.5, "plan_hor: {}".format(plan_hor), fontsize=12)
            plt.text((x_max - x_min - 20) / 0.5 + 100, (y_max - y_min - 2) / 0.5, "cost: {:.2f}".format(pred_cost), fontsize=12)
            plt.plot((traj[:, 0] - x_min) / 0.5, (-traj[:, 2] - y_min) / 0.5, marker='*', c='cyan')
            plt.plot([(traj[:, 0][0] - x_min) / 0.5], [(-traj[:, 2][0] - y_min) / 0.5], marker='*', c='yellow')
            plt.title(f'Planning Trajectory at time={step} (iteration {iteration})', fontsize=15)
            plt.savefig(savefig_path)
            plt.close()

        combine_to_gif(image_paths, f"{iteration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', '-l', dest='log_path', type=str, required=True)
    parser.add_argument('-p', dest='plot_iter_trajs', action='store_true', default=False)
    args = parser.parse_args()
    log_path = args.log_path

    if os.path.exists(os.path.join(log_path, 'logs.mat')):
        logging_data = sio.loadmat(log_path + '/logs.mat')
    else:
        all_logs = glob.glob(os.path.join(log_path, '*.mat'))
        assert all_logs, f'{log_path} must contain at least one .mat'
        print(f"Using {all_logs[0]}")
        logging_data = sio.loadmat(all_logs[0])

    returns = logging_data['returns'][0]
    print('returns:')
    print(returns, len(returns))

    train_losses_paths = sorted(
        glob.glob(os.path.join(log_path, '**/train_losses.npy'), recursive=True),
        key=lambda x: int(re.findall(r'(\d+)/train.*\.npy', x)[0])
    )

    holdout_losses_paths = sorted(
        glob.glob(os.path.join(log_path, '**/holdout_losses.npy'), recursive=True),
        key=lambda x: int(re.findall(r'(\d+)/holdout.*\.npy', x)[0])
    )

    plot_returns(returns, log_path)

    plot_model_error(train_losses_paths, log_path)
    plot_model_error(holdout_losses_paths, log_path)

    iters_to_plot = np.unique((1 + np.array([x for x in range(2, len(returns) - 1) if
                                             returns[x] - returns[x - 1] > 20 or returns[x] - returns[x + 1] > 20])).flatten())
    print(f"Potential iterations to plot: {iters_to_plot}")
    get_next = True
    iters_to_plot = []
    if args.plot_iter_trajs:
        while get_next:
            iter = input('iteration: ')
            try:
                iters_to_plot.append(int(iter))
            except ValueError:
                get_next = False
        for i in iters_to_plot:
            print(f"plotting planning traj for iteration {i} ...")
            plot_iteration_traj(i, log_path)

