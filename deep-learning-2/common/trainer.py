import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_list: list = []
        self.eval_interval = None
        self.current_epoch = 0
