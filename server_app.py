from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flower_recovery_gaussian.task import Net, get_weights
from flower_recovery_gaussian.strategy import FedAvgWithDropoutHandling


def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 5)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    min_available_clients = context.run_config.get("min-available-clients", 5)

    # Initialize model
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    #  Use metrics aggregation
    strategy = FedAvgWithDropoutHandling(
        fraction_fit=fraction_fit,
        min_fit_clients=2,
        min_available_clients=min_available_clients,
        fraction_evaluate=0.5,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
