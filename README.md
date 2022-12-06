# Fed-NCL
This is the PyTorch implemention of Federated Noisy Client Learning.

# Abstract
> Federated learning (FL) collaboratively trains a shared global model depending on multiple local clients, while keeping the training data decentralized in order to preserve data privacy. However, standard FL methods ignore the noisy client issue, which may harm the overall performance of the shared model. We first investigate critical issue caused by noisy clients in FL and quantify the negative impact of the noisy clients in terms of the representations learned by different layers. We have the following two key observations: (1) the noisy clients can severely impact the convergence and performance of the global model in FL, and (2) the noisy clients can induce greater bias in the deeper layers than the former layers of the global model. Based on the above observations, we propose Fed-NCL, a framework that conducts robust federated learning with noisy clients. Specifically, Fed-NCL first identifies the noisy clients through well estimating the data quality and model divergence. Then robust layer-wise aggregation is proposed to adaptively aggregate the local models of each client to deal with the data heterogeneity caused by the noisy clients. We further perform the label correction on the noisy clients to improve the generalization of the global model. Experimental results on various datasets demonstrate that our algorithm boosts the performances of different state-of-the-art systems with noisy clients.



