import tensorflow as tf
from tqdm import tqdm


# ──────────────────────────────────────────────
# Shared progress callback
# ──────────────────────────────────────────────
class TQDMProgressCallback(tf.keras.callbacks.Callback):
    """Updates a tqdm bar each epoch when using model.fit() in one call."""
    def __init__(self, pbar):
        super().__init__()
        self.pbar = pbar

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0)
        self.pbar.set_postfix({'val_acc': f'{val_acc:.4f}'})
        self.pbar.update(1)


# ──────────────────────────────────────────────
# Secretary stopping callback
# ──────────────────────────────────────────────
class SecretaryStoppingCallback(tf.keras.callbacks.Callback):
    """
    Implements secretary-inspired stopping in a single model.fit() call,
    avoiding the overhead of calling fit() once per epoch.

    Phase 1 (epochs 0 … K-1)  — observation: track best val_accuracy.
    Phase 2 (epochs K … N-1)  — selection:   stop as soon as val_accuracy
                                               exceeds the observation-phase best.
    """

    def __init__(self, observation_ratio: float, max_epochs: int, pbar_observe, pbar_select):
        super().__init__()
        self.K = int(observation_ratio * max_epochs)
        self.best_obs = -1.0
        self.stopped_at = max_epochs          # default: ran to the end
        self.pbar_observe = pbar_observe
        self.pbar_select = pbar_select

    def on_epoch_end(self, epoch, logs=None):
        val_acc = (logs or {}).get('val_accuracy', 0.0)

        if epoch < self.K:
            # ── Observation phase ──
            if val_acc > self.best_obs:
                self.best_obs = val_acc
            self.pbar_observe.set_postfix({'best': f'{self.best_obs:.4f}'})
            self.pbar_observe.update(1)
        else:
            # ── Selection phase ──
            self.pbar_select.set_postfix({
                'val_acc':   f'{val_acc:.4f}',
                'threshold': f'{self.best_obs:.4f}',
            })
            self.pbar_select.update(1)

            if val_acc > self.best_obs:
                self.stopped_at = epoch + 1   # 1-indexed epoch count
                self.model.stop_training = True


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ──────────────────────────────────────────────
# Training strategies
# ──────────────────────────────────────────────
def train_full(x_train, y_train, x_val, y_val, x_test, y_test, max_epochs=100):
    model = build_model()
    with tqdm(total=max_epochs, desc="    Epochs", unit="epoch", leave=False,
              bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=[TQDMProgressCallback(pbar)],
            verbose=0,
        )
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return max_epochs, float(test_acc)


def train_early_stopping(x_train, y_train, x_val, y_val, x_test, y_test, max_epochs=100):
    model = build_model()
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        mode='max',
        restore_best_weights=True,
    )
    with tqdm(total=max_epochs, desc="    Epochs", unit="epoch", leave=False,
              bar_format='{l_bar}{bar:30}{r_bar}') as pbar:
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=[early_stop, TQDMProgressCallback(pbar)],
            verbose=0,
        )
    epochs_used = len(history.history['loss'])
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return epochs_used, float(test_acc)


def train_secretary(x_train, y_train, x_val, y_val, x_test, y_test,
                    observation_ratio=0.37, max_epochs=100):
    K = int(observation_ratio * max_epochs)
    remaining = max_epochs - K
    pct = int(observation_ratio * 100)

    model = build_model()

    with (
        tqdm(total=K,         desc=f"    Observe({pct}%)", unit="epoch",
             leave=False, bar_format='{l_bar}{bar:30}{r_bar}') as pbar_obs,
        tqdm(total=remaining, desc=f"    Select({pct}%)",  unit="epoch",
             leave=False, bar_format='{l_bar}{bar:30}{r_bar}') as pbar_sel,
    ):
        secretary_cb = SecretaryStoppingCallback(
            observation_ratio, max_epochs, pbar_obs, pbar_sel
        )
        model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=max_epochs,
            batch_size=32,
            callbacks=[secretary_cb],
            verbose=0,
        )

    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return secretary_cb.stopped_at, float(test_acc)