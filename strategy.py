from typing import List, Tuple

import numpy as np
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common import FitIns, parameters_to_ndarrays, ndarrays_to_parameters

def aggregate_train_metrics(results):
    total_n = 0
    total_loss = 0.0

    # For mask diagnostics ---> these are mainly for debugging/understanding what is going on. 
    total_mask_last_w = 0.0
    total_mask_all_w  = 0.0
    total_params_w    = 0

    for _, metrics in results:
        n = int(metrics.get("num_examples", 0))
        total_n += n
        total_loss += float(metrics.get("train_loss", 0.0)) * n

        if "mask_lsum" in metrics:
            total_mask_last_w += float(metrics["mask_lsum"]) * n
        if "mask_all_sum" in metrics and "mask_param_count" in metrics:
            total_mask_all_w  += float(metrics["mask_all_sum"]) * n
            total_params_w    += int(metrics["mask_param_count"]) * n

    if total_n == 0:
        return {}

    out = {"aggregated_train_loss": total_loss / total_n}

    # FedAvg-weighted last-layer sum 
    if total_n > 0:
        out["weighted_mask_lsum"] = total_mask_last_w / total_n

    # FedAvg-weighted *mean per parameter* across all layers
    if total_params_w > 0:
        out["weighted_mask_all_mean"] = total_mask_all_w / total_params_w 

    return out

def aggregate_evaluate_metrics(results):
    total_examples = 0
    total_accuracy = 0.0
    for _, metrics in results:
        num_examples = metrics.get("num_examples", 0)
        accuracy = metrics.get("accuracy", 0.0)
        total_accuracy += accuracy * num_examples
        total_examples += num_examples
    if total_examples == 0:
        return {}
    return {"aggregated_accuracy": np.round((total_accuracy / total_examples) * 100, 2)}


class FedAvgWithDropoutHandling(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(
            fit_metrics_aggregation_fn=aggregate_train_metrics,
            evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            **kwargs,
        )

    def aggregate_fit(self, server_round, results, failures):
        print(f"[ROUND {server_round}] Aggregating {len(results)} results with {len(failures)} failures.")

    
        results = [(cp, res) for cp, res in results if res and res.num_examples > 0]
        if not results:
            print(f"[ROUND {server_round}] No successful results to aggregate.")
            return None, {}

        # --- Order soorted by client id ----------------------------
        results_sorted = sorted(results, key=lambda x: int(x[0].cid))

        # --- Weighted average
        total_n = sum(res.num_examples for _, res in results_sorted)
        first = parameters_to_ndarrays(results_sorted[0][1].parameters)
        agg64 = [np.zeros_like(w, dtype=np.float64) for w in first]

        for cp, res in results_sorted:
            ws = parameters_to_ndarrays(res.parameters)
            wgt = res.num_examples / total_n
            for k in range(len(agg64)):
                agg64[k] += wgt * ws[k].astype(np.float64, copy=False)

         # For debugging -----------------------------------------------------------------------
        print(f"[DEBUG] Round {server_round} Global sum of layer after aggregation: {float(np.sum(agg64[-1])):.6f}")


        # Pull out mask sums from metrics
        mask_entries = [(cp.cid, res.metrics) for cp, res in results_sorted]
        raw_sum = sum(float(m["mask_all_sum"]) * int(m["num_examples"]) for _, m in mask_entries if "mask_all_sum" in m)

        total_n = sum(int(m["num_examples"]) for _, m in mask_entries)
        fedavg_sum = raw_sum / total_n if total_n else 0.0

        total_params = sum(int(m["mask_param_count"]) * int(m["num_examples"]) for _, m in mask_entries if "mask_param_count" in m)
        per_param_mean = raw_sum / total_params if total_params else 0.0

        print(f"[CANCEL] Round {server_round}:")
        print(f"    raw_sum        = {raw_sum:.12f}")
        print(f"    fedavg_sum     = {fedavg_sum:.12f}")
        print(f"    per_param_mean = {per_param_mean:.12e})")

        # Cast back to original dtype ??? I guess not needed if i do everything in float64 ()
        aggregated_parameters = ndarrays_to_parameters(
            [w.astype(first[k].dtype, copy=False) for k, w in enumerate(agg64)]
        )

        # Aggregate metrics
        metrics_aggregated = aggregate_train_metrics([(cp.cid, res.metrics) for cp, res in results_sorted])
        return aggregated_parameters, metrics_aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        num_available_clients = client_manager.num_available()
        sample_size, min_num_clients = self.num_fit_clients(num_available_clients)
        clients = client_manager.sample(sample_size, min_num_clients)

        sampled_ids = [int(c.cid) if str(c.cid).isdigit() else c.cid for c in clients]
        fit_config: List[Tuple[ClientProxy, FitIns]] = []
        total_clients = len(client_manager.all())

        for client in clients:
            config = {
                "true_cid": client.cid,
                "partition_id": int(client.cid) if str(client.cid).isdigit() else client.cid,
                "num_partitions": total_clients,
                "server_round": server_round,
                "sampled_client_ids": ",".join(str(x) for x in sampled_ids),
            }
            fit_config.append((client, FitIns(parameters, config)))

        return fit_config
