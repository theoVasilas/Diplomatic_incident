from omegaconf import DictConfig, OmegaConf
import hydra
from opacus import PrivacyEngine
from opacus.accountants.analysis import gdp
from opacus.optimizers import DPOptimizer
import torch

from data_utils import load_datasets
from train_utils import test, test_2, Net
# from train_utils import Net, train_DP,train_DP_with_epsilon, train, test
from client_utils import set_parameters, get_parameters

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LEARNING_RATE = cfg.learning_rate
    NOISE_MULTIPLIER = cfg.noise_multiplier
    TARGET_EPSILON = cfg.target_epsilon
    TARGET_DELTA = cfg.target_delta
    LOCAL_EPOCHS = cfg.local_epochs
    SEED = cfg.SEED
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Noise Multiplier: {NOISE_MULTIPLIER}, Target Delta: {TARGET_DELTA}")

    # Load dataset
    trainloader, valloader, _ = load_datasets(cfg, 0)



    # Test different max_grad_norm values
    start_max_grad_norm=2.0
    OmegaConf.set_struct(cfg, False)
    cfg.max_grad_norm = start_max_grad_norm
    print(f"\nTesting start_max_grad_norm = {start_max_grad_norm}")

    # Fresh model & optimizer for each run
    model = Net().to(device)
    parameters = get_parameters(model)
    new_parameters = parameters

    
    for new_max_grad_norm in [0.05 ]: #0.1, 0.2, 0.5, 1.0
        # Initialize Privacy Engine
        #privacy_engine = PrivacyEngine(secure_mode=False, accountant="gdp")
        privacy_engine = PrivacyEngine(secure_mode=False, accountant="rdp")

    
        for turn in range(1):

            local_model = Net().to(device)
            set_parameters(local_model, new_parameters)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=LEARNING_RATE, momentum=0.9)



            privacy_engine.seed = SEED
            # Make model private
            # new_max_grad_norm = cfg.max_grad_norm
            print(f"\nTesting max_grad_norm = {new_max_grad_norm}")
            # new_model, new_optimizer, new_train_loader = privacy_engine.make_private(
            #     module=local_model,
            #     optimizer=optimizer,
            #     data_loader=trainloader,
            #     noise_multiplier=NOISE_MULTIPLIER,
            #     clipping="flat",
            #     max_grad_norm=new_max_grad_norm,
            # )

            new_model, new_optimizer, new_train_loader = privacy_engine.make_private_with_epsilon( 
                module=local_model,
                optimizer=optimizer,
                data_loader=trainloader,

                target_epsilon=TARGET_EPSILON,
                target_delta=TARGET_DELTA,

                epochs=LOCAL_EPOCHS,
                max_grad_norm=new_max_grad_norm,
                clipping="flat", # (flat or per_layer or adaptive)
            )
            print(
                f"After make_private(). "
                f"Model:{type(new_model)}, Optimizer:{type(new_optimizer)}, DataLoader:{type(new_train_loader)}"
            )

            # Train model
            # avg_trainloss, epsilon, max_norm_before_clipping, grand_70_pec = train_DP(
            #     new_model,
            #     new_train_loader,
            #     privacy_engine,
            #     new_optimizer,
            #     TARGET_DELTA,
            #     device, 
            #     LOCAL_EPOCHS
            # )

            sample_rate = min(32 / len(new_train_loader), 1) 

            avg_trainloss, noise_multiplier, max_norm_before_clipping, grand_70_pec = train_DP_with_epsilon(
                        net=new_model,
                        train_loader=new_train_loader,
                        privacy_engine=privacy_engine,
                        target_epsilon=TARGET_EPSILON,
                        target_delta=TARGET_DELTA,
                        sample_rate=sample_rate,
                        optimizer=new_optimizer,
                        device=device,
                        epochs=LOCAL_EPOCHS
                        )

            new_parameters = get_parameters(new_model)

            epsilon = privacy_engine.accountant.get_epsilon(delta=TARGET_DELTA)

            # Output results
            if epsilon is not None:
                print(f"Epsilon = {epsilon:.2f} for (noise_multiplier: {NOISE_MULTIPLIER}, delta: {TARGET_DELTA})")
            else:
                print("Epsilon value not available.")

            print(f"max_norm before clipping: {max_norm_before_clipping}, 70th percentile gradient norm: {grand_70_pec}\n")

            # OmegaConf.set_struct(cfg, False)
            # cfg.max_grad_norm = float(grand_70_pec)

import numpy as np
        
def train_DP(net, 
             train_loader, 
             privacy_engine, 
             new_optimizer, 
             target_delta, 
             device, 
             epochs=1
             ):
    
    net.train()

    criterion = torch.nn.CrossEntropyLoss()

    grad_norms = []

    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["img"], batch["label"]
                new_optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

               # Store individual gradient norms
                print("\n")
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item())
                        grad_norm = p.grad.data.norm(2).item()
                        print(f"Gradient norm before clipping: {grad_norm:.4f}")

                new_optimizer.step()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.03)

                # After optimizer step, you can log the same thing to check post-clipping norms
                for param in net.parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.data.norm(2).item()
                        print(f"Gradient norm after clipping: {grad_norm:.4f}")

                running_loss += loss.item()
    
    avg_trainloss = running_loss / len(train_loader)

    #Opacus
    print("\n\t",type(privacy_engine.accountant))

    # epsilon = privacy_engine.get_epsilon(delta=target_delta)
    epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)

    for param in net.parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"Gradient norm after clipping: {grad_norm:.4f}")


    #Extra information
    max_norm_before_clipping = max(grad_norms) if grad_norms else 0.0
    grand_70_pec = np.percentile(grad_norms, 70)

    return avg_trainloss, epsilon, max_norm_before_clipping, grand_70_pec


from opacus.accountants.utils import get_noise_multiplier

def train_DP_with_epsilon(  net, 
                            train_loader, 
                            privacy_engine,
                            target_epsilon,
                            target_delta,
                            sample_rate,
                            optimizer, 
                            device, 
                            epochs=1
                            ):
    net.to(device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss()

    grad_norms = []

    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Store individual gradient norms
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item()) 

                optimizer.step()
                running_loss += loss.item()

    avg_trainloss = running_loss / len(train_loader)
    
    #Opacus
    # print(f"\n\n sample_rate {sample_rate} \n\n")
    noise_multiplier = get_noise_multiplier(  target_epsilon = target_epsilon,
                                                    target_delta = target_delta,
                                                    sample_rate = sample_rate,
                                                    # epochs = epochs,                 
                                                    steps = 1 ) # epochs or steps ,needs to be revisioned 

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    # print(f"\n\n epsilon {epsilon}")

    #Extra information
    max_norm_before_clipping = max(grad_norms) if grad_norms else 0.0
    grand_70_pec = np.percentile(grad_norms, 70)

    return avg_trainloss ,noise_multiplier, max_norm_before_clipping, grand_70_pec


if __name__ == "__main__":
    main()