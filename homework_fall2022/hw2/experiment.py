from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


    # scalars = ["Eval_MeanReturn", "Eval_StdReturn", "Initial_DataCollection_MeanReturn"]

def find_dir(exp_name, expert_data):
    logdir = 'data/q2_pg_{exp_name}_{expert_data}_*'.format(exp_name=exp_name, expert_data=expert_data)
    logfile_path = glob.glob(os.path.join(logdir, 'events*'))[0]
    print("Logging file path: ", os.path.exists(logfile_path), logfile_path)

    return logfile_path

def parse_tensorboard(path):
    event_acc = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0},)
    event_acc.Reload()

    print("Eval_AverageReturn", event_acc.Scalars('Eval_AverageReturn'))
    print()

    eval_average_list = []
    for eval_wall_time, eval_step, eval_average in event_acc.Scalars('Eval_AverageReturn'):
        eval_average_list.append(eval_average)
    return eval_average_list

def run_PG(exp_name_list):
    num_iter_average_return = {}
    num_iter = np.arange(1, 101, 1)

    for exp_name in exp_name_list:
        eval_average_list = parse_tensorboard(find_dir(exp_name, expert_data='CartPole-v0'))
        num_iter_average_return[exp_name] = eval_average_list

        # Plot average_return
        plt.plot(num_iter, eval_average_list, label=exp_name)

    plt.title("Result of PG with different rewards formulations (batch size=1000)")
    plt.xlabel("num_iterations")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()

def main():
    sb_exp_name_list = ['q1_sb_no_rtg_dsa', 'q1_sb_rtg_dsa', 'q1_sb_rtg_na']
    lb_exp_name_list = ['q1_lb_no_rtg_dsa', 'q1_lb_rtg_dsa', 'q1_lb_rtg_na']

    run_PG(sb_exp_name_list)


if __name__ == main():
    main()


