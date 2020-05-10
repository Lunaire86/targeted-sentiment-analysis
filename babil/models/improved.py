#!/usr/bin/env python3
# coding: utf-8

from argparse import Namespace
from dataclasses import dataclass, field
from os.path import join
from typing import List, Tuple

import numpy as np
# import seaborn as sns
import tensorflow.keras as keras
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Embedding
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils.config import PathTracker

METRICS = [
    CategoricalAccuracy(name='accuracy'),
    BinaryAccuracy(name='binary_accuracy'),
    TruePositives(name='tp'),
    TrueNegatives(name='tn'),
    FalsePositives(name='fp'),
    FalseNegatives(name='fn'),
    Precision(name='precision'),
    Recall(name='recall'),
    # BinaryTruePositives(name='btp'),
    # BinaryTrueNegatives(name='btn'),
    # BinaryFalsePositives(name='bfp'),
    # BinaryFalseNegatives(name='bfn')
]


@dataclass
class Baseline:
    # required args
    args: Namespace
    path_to: PathTracker
    weights: np.ndarray

    # default value args
    num_classes: int = 5
    sequence_length: int = 50

    # fields set in __post_init__()
    recurrent_dropout: float = field(init=False)
    checkpoint_path: str = field(init=False)
    callbacks: List[keras.callbacks] = field(init=False, default_factory=list)

    # class output
    model: Model = field(init=False, default=None)
    results: History = field(init=False, default=None)

    def __post_init__(self):
        self.recurrent_dropout = self.args.dropout

        self.checkpoint_path = join(
            self.path_to.models, 'baseline_checkpoint.h5'
        )

        self.callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                verbose=2
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=2,
                verbose=2
            ),
            ModelCheckpoint(
                filepath=self.checkpoint_path,
                monitor='val_loss',
                save_format='h5',
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-3,
                verbose=1
            )
        ]

    def build(self):

        """Build a model using the Keras functional API."""

        text_input = Input(
            shape=(self.sequence_length,),
            name='words'
        )
        embedded = Embedding(
            input_dim=self.weights.shape[0],  # vocab size
            output_dim=self.weights.shape[1],  # embedding dim
            input_length=self.sequence_length,  # num time-steps
            weights=[self.weights],
            mask_zero=True,
            trainable=self.args.train_embeddings,
            name='fastText'
        )(text_input)

        # Fraction of the units to drop for
        # the linear transformation of the
        # -> inputs             (dropout)
        # -> recurrent state    (recurrent dropout)
        lstm_out = Bidirectional(
            LSTM(
                units=self.args.hidden_dim,
                dropout=self.args.dropout,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=True
            ),
            name='BiLSTM'
        )(embedded)

        predicted_labels = Dense(
            self.num_classes + 1,  # add one for padding token
            activation='softmax',
            name='output'
        )(lstm_out)

        self.model = Model(
            inputs=[text_input],
            outputs=[predicted_labels],
            name='baseline'
        )

    def summary(self):
        return self.model.summary()

    def compile(self):
        """Compile model."""
        optim = Adam(learning_rate=self.lr)  # alternatively, try Adamax
        loss = CategoricalCrossentropy()

        self.model.compile(
            optimizer=optim,
            loss=loss,
            metrics=METRICS
        )

    def train(self, X_train, y_train, X_dev, y_dev) -> History:
        """Train model."""
        self.results = self.model.fit(
            X_train, y_train,
            validation_data=(X_dev, y_dev),
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,  # samples per gradient
            shuffle=True,
            callbacks=self.callbacks
        )
        return self.results

    @staticmethod
    def save(model, path) -> None:
        """Save model."""
        model.save(join(path, 'improved.h5'))

    @staticmethod
    def load(path, checkpoint: bool = True) -> Tuple[Model, History]:
        """Loads the pre-trained baseline model.
        Returns both model and its results."""
        if checkpoint:
            model = keras.models.load_model(join(path, 'improved_checkpoint.h5'))
        else:
            model = keras.models.load_model(join(path, 'improved.h5'))

        return model, model.history

    def predict(self):
        """Predict."""
        pass

    def evaluate(self):
        """Evaluate."""
        pass
