from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def run_policy(lr_list, num_iter_list):
    cmd = 'python cs285/scripts/run_hw1.py \
           --expert_policy_file cs285/policies/experts/{expert_policy_file} \
           --env_name {env_name} \
           --exp_name {exp_name} \
           --n_iter 1 \
           --expert_data cs285/expert_data/expert_data_{expert_data}.pkl \
           --video_log_freq -1 \
           --learning_rate {learning_rate} \
           --num_agent_train_steps_per_iter {num_agent_train_steps_per_iter}'

    for lr in lr_list:
        curr_cmd = cmd.format(expert_policy_file='Ant.pkl', env_name='Ant-v4',
                              exp_name='bc-' + str(lr),
                              expert_data='Ant-v4', learning_rate=lr, num_agent_train_steps_per_iter=1000)
        os.system(curr_cmd)

    for num_iter in num_iter_list:
        curr_cmd = cmd.format(expert_policy_file='Ant.pkl', env_name='Ant-v4',
                              exp_name='bc-' + str(num_iter),
                              expert_data='Ant-v4', learning_rate=0.01, num_agent_train_steps_per_iter=num_iter)
        os.system(curr_cmd)


    # scalars = ["Eval_MeanReturn", "Eval_StdReturn", "Initial_DataCollection_MeanReturn"]

def find_dir(hyperparam, expert_data='Ant-v4'):
    logdir = 'data/q1_bc-{hyperparam}_{expert_data}_*'.format(hyperparam=hyperparam, expert_data=expert_data)
    logfile_path = glob.glob(os.path.join(logdir, 'events*'))[0]
    print("Logging file path: ", os.path.exists(logfile_path), logfile_path)

    return logfile_path

def parse_tensorboard(path):
    event_acc = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0},)
    event_acc.Reload()

    print("Eval_AverageReturn", event_acc.Scalars('Eval_AverageReturn'))
    print()

    eval_wall_time, eval_step, eval_average = event_acc.Scalars('Eval_AverageReturn')[0]
    init_wall_time, init_step, init_average = event_acc.Scalars('Train_AverageReturn')[0]
    return eval_average, init_average

def run_BC():
    # HyperParameters
    lr_list = np.array([0.01, 0.001, 0.0001])
    num_iter_list = np.arange(1000, 6000, 1000)

    # run_policy(lr_list, num_iter_list)

    # Parse Average Reward over tensorboard
    lr_performance = []
    expert_lr_performance = []
    for lr in lr_list:
        logfile_path = find_dir(lr)
        eval_average, init_average = parse_tensorboard(logfile_path)
        lr_performance.append(eval_average)
        expert_lr_performance.append(init_average)
    print("lr_performance: ", lr_performance)

    num_iter_performance = []
    expert_num_iter_performance = []
    for num_iter in num_iter_list:
        logfile_path = find_dir(num_iter)
        eval_average, init_average = parse_tensorboard(logfile_path)
        num_iter_performance.append(eval_average)
        expert_num_iter_performance.append(init_average)
    print("num_iter_performance: ", num_iter_performance)

    # Plot performance vs. learning rate
    plt.plot(lr_list, lr_performance, 'bo-', label="BC-performance")
    plt.plot(lr_list, expert_lr_performance, 'ro-', label="expert-performance")
    plt.title("Result of BC in Ant-v4 over 5 rollouts with different learning rates")
    plt.xlabel("Learning Rate")
    plt.ylabel("Average Reward")
    plt.legend()

    for i in range(len(lr_list)):
        bc_label = '{:.2f}'.format(lr_performance[i])
        expert_label = '{:.2f}'.format(expert_lr_performance[i])
        plt.annotate(bc_label, (lr_list[i], lr_performance[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        plt.annotate(expert_label, (lr_list[i], expert_lr_performance[i]),
                     textcoords="offset points", xytext=(0, 5), ha='center')

    plt.show()

    # Plot performance vs. learning rate
    plt.plot(num_iter_list, num_iter_performance, 'bo-', label="BC-performance")
    plt.plot(num_iter_list, expert_num_iter_performance, 'ro-', label="expert-performance")
    plt.title("Result of BC in Ant-v4 over 5 rollouts with different number of training iterations")
    plt.xlabel("number of training iterations")
    plt.ylabel("Average Reward")
    plt.legend(loc='lower right')

    for i in range(len(num_iter_list)):
        bc_label = '{:.2f}'.format(num_iter_performance[i])
        expert_label = '{:.2f}'.format(expert_num_iter_performance[i])
        plt.annotate(bc_label, (num_iter_list[i], num_iter_performance[i]),
                     textcoords="offset points", xytext=(0, 5), ha='center')
        plt.annotate(expert_label, (num_iter_list[i], expert_num_iter_performance[i]),
                     textcoords="offset points", xytext=(0, 5), ha='center')

    plt.show()


def main():
    run_BC()


if __name__ == main():
    main()


