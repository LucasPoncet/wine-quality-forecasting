import numpy as np


class EarlyStopper:
    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.max_epoch = self.hyperparameters["max_epoch"]
        self.metric_epochs = []
        self.min_epoch = self.hyperparameters.get("min_epoch_es", 10)
        self.early_stopping_patience = self.hyperparameters.get("patience_es", 10)
        self.best_model_weights = None
        self.wait = 0

    def set(self, model, epoch, metric_epoch):
        keep_training = True
        self.metric_epochs.append(metric_epoch)
        if epoch == 0:
            self.wait = 0
            print("Epoch" + str(epoch) + " - Keeping weights")
            self._keep_weights(model)
        if epoch >= self.min_epoch:
            if metric_epoch > np.max(self.metric_epochs[:-1]):
                self.wait = 0
                print(
                    "Epoch "
                    + str(epoch)
                    + " - Best metrics "
                    + str(metric_epoch)
                    + " - Keeping weights"
                )
                self._keep_weights(model)
            else:
                self.wait += 1
                print("Epoch " + str(epoch) + " - Metrics did not improve, wait:" + str(self.wait))
                if self.wait >= self.early_stopping_patience:
                    print("Epoch " + str(epoch) + " - Patience reached - Restoring weights")
                    keep_training = False
                    self._restore_weights(model)

        if epoch == (self.max_epoch - 1):
            print("Max Epoch reached - Stop training - Restoring weights ")
            keep_training = False
            self._restore_weights(model)
        return keep_training

    def _keep_weights(self, model):
        self.best_model_weights = model.state_dict().copy()

    def _restore_weights(self, model):
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
