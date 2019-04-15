"""
This script plots train/eval losses of one or many models
"""


import os
import argparse
import matplotlib.pyplot as plt
from metric_store import Metric_Store


def get_metrics(model_name):
    
    # The save directory is one level down from the code directory
    cwd = os.getcwd()
    save_dir = os.path.join(os.path.dirname(cwd), 'saved_models')
    file_path = os.path.join(save_dir, model_name, 'checkpoint_metrics.pickle')

    metric_store = Metric_Store()
    """
    metric_store.log(1,1)
    metric_store.log(0.8,0.9)
    metric_store.log(0.6,0.7)
    metric_store.log(0.3,0.45)
    metric_store.log(0.1,0.14)

    metric_store.save(file_path)
    """

    metric_store = Metric_Store()
    metric_store.load(file_path)
    print(metric_store.test_loss)
    return metric_store
    
    
def plot(metric_store, model_name):
    X = [i+1 for i in range(len(metric_store.test_loss))]

    plt.rc('font', size=11) 
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.title('Model ' + model_name)
    plt.xlabel('Sub-epoch')
    plt.ylabel('Loss')
    tr_loss, = plt.plot(X, metric_store.training_loss)
    ev_loss, = plt.plot(X, metric_store.test_loss)

    plt.legend([tr_loss, ev_loss], ['Training loss', 'Evaluation loss'])
    plt.grid(True)
    plt.xticks(X)

    plt.savefig(model_name)
    plt.show()


def plot_all(model_names, metric_stores):
    
    plt.rc('font', size=10) 
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plot_ids = [i+1 for i in range(len(model_names))]
    for plot_id, model_name, metric_store in zip(plot_ids, model_names, metric_stores):
        X = [i+1 for i in range(len(metric_store.test_loss))]
        plt.subplot(2, 2, plot_id)
        
        plt.title('Model ' + model_name)
        plt.xlabel('Sub-epoch')
        plt.ylabel('Loss')
        tr_loss, = plt.plot(X, metric_store.training_loss)
        ev_loss, = plt.plot(X, metric_store.test_loss)

        plt.legend([tr_loss, ev_loss], ['Training loss', 'Evaluation loss'])
        plt.grid(True)
        #plt.xticks(X)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.savefig(''.join(model_names))
    plt.show()


def main(args):
    metric_store = get_metrics(args.model)
    plot(metric_store, args.model)
    
    models = ['RNNw30', 'RNNw50', 'RNNAw30', 'RNNAw50']
    metric_stores = [get_metrics(model) for model in models]
    plot_all(models, metric_stores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()
    main(args)
    

