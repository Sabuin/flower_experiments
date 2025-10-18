import random
import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flower_recovery_gaussian.task import Net, get_weights, load_data, set_weights, test, train
from flower_recovery_gaussian.rss_utils import add_pairwise_noise

"""Generate additive secret shares for weights."""
def share_weights(weights, num_shares=3):
    shares = []
    for _ in range(num_shares - 1):
        share = [np.random.randn(*w.shape).astype(w.dtype) for w in weights]
        shares.append(share)
    final_share = [w - sum(share[i] for share in shares) for i, w in enumerate(weights)]
    shares.append(final_share)
    return shares


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, noise_std, num_shares, client_id, total_clients):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.noise_std = noise_std
        self.num_shares = num_shares
        self.client_id = client_id
        self.total_clients = total_clients

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
       
       #Helpful print statements ---------------------------------------------------------------
        print(f"[CLIENT {self.client_id}] Starting local train: epochs={self.local_epochs}, "
            f"round={config.get('server_round')}, noise_std={self.noise_std}")
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        print(f"[CLIENT {self.client_id}] Finished local train: loss={train_loss:.6f}")

       # ID's from config
        cid = int(config["true_cid"])
        participants = sorted(int(x) for x in config["sampled_client_ids"].split(","))

        # Setting weights and training
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader, self.local_epochs, self.device)
        updated_weights = get_weights(self.net)

        # Add pairwise noise 
        if self.noise_std > 0:
            local_n = max(1, len(self.trainloader.dataset))
            participants = list(map(int, config["sampled_client_ids"].split(",")))
            
            ##Helpful print statement. 
            print(f"[CLIENT {self.client_id}] Participants this round: {participants} "
                f"(local_n={local_n})")
           
            noisy_weights = add_pairwise_noise(
                updated_weights,
                cid,
                self.client_id,                 
                participants,
                config["server_round"],
                self.noise_std,
                1.0 / local_n,                  # <- for cancelation
            )
        else:
            noisy_weights = updated_weights

        # For debugging only ------------------------------------------------------------#
        mask_layers = [nw - w for nw, w in zip(noisy_weights, updated_weights)]
        mask_lsum = float(np.sum(mask_layers[-1].astype(np.float64)))  # last layer only
        mask_all_sum = float(sum(np.sum(m.astype(np.float64)) for m in mask_layers))
        mask_param_count = int(sum(m.size for m in mask_layers))
        
        print(f"[CLIENT {self.client_id}] Round {config.get('server_round')} "
            f"mask_lsum(last_layer)={mask_lsum:.3e}  "
            f"mask_all_sum(all_layers)={mask_all_sum:.3e}  "
            f"params={mask_param_count}")

        #----------------------------------------------------------------------------------#

        # Return noisy weights
        return noisy_weights, len(self.trainloader.dataset), {
            "train_loss": train_loss,
            "num_examples": len(self.trainloader.dataset),
            "mask_lsum": mask_lsum,
            "mask_all_sum": mask_all_sum,
            "mask_param_count": mask_param_count,
        }


    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {
            "accuracy": accuracy,
            "num_examples": len(self.valloader.dataset),
        }


def client_fn(context: Context):
    dropout_probability = context.run_config.get("dropout-probability", 0.0)
    noise_std = context.run_config["noise-std"]
    partition_id = context.node_config["partition-id"]
    num_shares = context.run_config.get("num-shares", 3)
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    # Dropout simulation
    if random.random() < dropout_probability:
        msg = f"[DROPOUT] Client {partition_id} dropped out"
        print(msg)
        with open("dropouts.txt", "a") as f:
            f.write(msg + "\n")
        return SkippingClient().to_client()

    net = Net()
    trainloader, valloader = load_data(partition_id, num_partitions)

    return FlowerClient(
        net, trainloader, valloader,
        local_epochs, noise_std, num_shares,
        client_id=partition_id,
        total_clients=num_partitions,
    ).to_client()


'''When client drops out, return a "ghost" client to avoid crash. Does not affect procedure'''
class SkippingClient(NumPyClient):
    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        return parameters, 0, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}

# Define Flower ClientApp
app = ClientApp(client_fn)
