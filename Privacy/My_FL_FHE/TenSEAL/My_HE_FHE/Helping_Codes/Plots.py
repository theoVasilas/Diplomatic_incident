import os
import matplotlib.pyplot as plt


import os
import matplotlib.pyplot as plt

def plot_time(timestamps, save=True, save_path="plots"):
    if save:
        save_dir = os.path.join(save_path, "images")
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(timestamps['client'], label='Client Time', marker='o')
    plt.plot(timestamps['server'], label='Server Time', marker='s')

    plt.xlabel('Rounds')
    plt.ylabel('Time (seconds)')
    plt.title('Client vs Server Computation Time')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f"{save_dir}/computation_time.png")
    else:
        plt.show()
    plt.close()



def plot_clients_history(history, save=True, save_path="plots"):
    if save:
        save_dir = os.path.join(save_path, "images")
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for client_id, round_loss in enumerate(zip(*history['val_loss'])):
        plt.plot(round_loss, marker='o', label=f'Client {client_id}')

    plt.xlabel('Rounds')
    plt.ylabel('Validation Loss')
    plt.title('Client Validation Loss Over Rounds')
    plt.legend()
    plt.grid()
    
    if save:
        plt.savefig(f"{save_dir}/client_validation_loss.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    for client_id, round_acc in enumerate(zip(*history['val_accuracy'])):
        plt.plot(round_acc, marker='o', label=f'Client {client_id}')

    plt.xlabel('Rounds')
    plt.ylabel('Validation Accuracy')
    plt.title('Client Validation Accuracy Over Rounds')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f"{save_dir}/client_validation_accuracy.png")
    plt.show()
    plt.close()


def plot_server_history(history, save=True, save_path="plots"):
    
    if save:
        save_dir = os.path.join(save_path, "images")
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(history['Server_val_loss'], label='Server Validation Loss', marker='o', color='red')

    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Server Validation Loss Over Rounds')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f"{save_dir}/server_validation_loss.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(history['Server_val_acc'], label='Server Validation Accuracy', marker='s', color='green')

    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Server Validation Accuracy Over Rounds')
    plt.legend()
    plt.grid()

    if save:
        plt.savefig(f"{save_dir}/server_validation_accuracy.png")
    plt.show()
    plt.close()
