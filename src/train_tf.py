import argparse
import tensorflow as tf

from data_loader import Dataset, VOCAB
from models.SimpleBiLSTM import SeqCuDNNBiLSTM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--max_epoch", type=int, default=35)
    parser.add_argument("-v", "--val-at", type=int, default=100)
    parser.add_argument("-i", "--hidden-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=76)

    args = parser.parse_args()
     
    with tf.device("/"+args.device):
        bilstm_sequential = SeqCuDNNBiLSTM(len(VOCAB), 16, args.hidden_size)
        model = bilstm_sequential.model

        dataset = Dataset(
            "data/data_dict4.pth",
            args.device,
            val_size=args.val_size,
            test_path="data/test_dict.pth",
        )

        train_data, train_labels = dataset.get_train_data()
        train_labels = tf.keras.utils.to_categorical(train_labels)

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', verbose=1, patience=50, restore_best_weights=True)
        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode="temporal")
        model.fit(train_data, train_labels, batch_size=args.batch_size, epochs=args.max_epoch, class_weight=[0.1, 1, 1.2, 0.8, 1.5], callbacks=[es])
        
        tf.keras.models.save_model(model, "model.model", overwrite=True, include_optimizer=True)
        dataset.save_dataset("dataset.pkl")

