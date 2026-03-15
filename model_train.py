import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import average_precision_score
from datetime import datetime

from config import DATASETS_DIR, MODELS_DIR, LOGS_DIR, RANDOM_SEED
from logger_setup import setup_logger

def build_model(T, H, W, C, F_met):
    img_in = layers.Input(shape=(T, H, W, C), name="img_seq")

    x = layers.TimeDistributed(layers.Conv2D(16, 3, activation="relu", padding="same"))(img_in)
    x = layers.TimeDistributed(layers.MaxPool2D())(x)
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu", padding="same"))(x)
    x = layers.TimeDistributed(layers.MaxPool2D())(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)

    met_in = layers.Input(shape=(T, F_met), name="met_seq")
    m = layers.LSTM(32)(met_in)
    m = layers.Dropout(0.2)(m)

    z = layers.Concatenate()([x, m])
    z = layers.Dense(64, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    out = layers.Dense(1, activation="sigmoid")(z)

    model = tf.keras.Model(inputs=[img_in, met_in], outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model

def train() -> None:
    logger, _ = setup_logger()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X_img_tr = np.load(DATASETS_DIR / "X_img_train.npy")
    X_met_tr = np.load(DATASETS_DIR / "X_met_train.npy")
    y_tr = np.load(DATASETS_DIR / "y_train.npy")

    X_img_va = np.load(DATASETS_DIR / "X_img_val.npy")
    X_met_va = np.load(DATASETS_DIR / "X_met_val.npy")
    y_va = np.load(DATASETS_DIR / "y_val.npy")

    T, H, W, C = X_img_tr.shape[1:]
    F_met = X_met_tr.shape[-1]

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    if pos == 0:
        raise ValueError("Training has 0 positive samples. Increase sample size or adjust sampling rules.")
    class_weight = {0: 1.0, 1: float(neg / pos)}
    logger.info(f"Training shapes: X_img={X_img_tr.shape}, X_met={X_met_tr.shape}, y={y_tr.shape}")
    logger.info(f"Class weights: {class_weight}")

    model = build_model(T, H, W, C, F_met)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tb_dir = LOGS_DIR / f"train_{run_id}"
    tb_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = MODELS_DIR / f"best_cnn_lstm_{run_id}.keras"
    csv_log_path = tb_dir / "training.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc_pr",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc_pr",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc_pr",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(csv_log_path)),
        tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir))
    ]

    model.fit(
        {"img_seq": X_img_tr, "met_seq": X_met_tr}, y_tr,
        validation_data=({"img_seq": X_img_va, "met_seq": X_met_va}, y_va),
        epochs=25,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    p_va = model.predict({"img_seq": X_img_va, "met_seq": X_met_va}, batch_size=128).ravel()
    ap = average_precision_score(y_va, p_va)
    logger.info(f"Validation PR-AUC: {ap:.4f}")

    final_path = MODELS_DIR / "cnn_lstm_himawari_era5.keras"
    model.save(final_path)
    logger.info(f"Saved final model: {final_path}")
    logger.info(f"Saved best checkpoint: {checkpoint_path}")
    logger.info(f"Training logs saved: {csv_log_path} | TensorBoard dir: {tb_dir}")
