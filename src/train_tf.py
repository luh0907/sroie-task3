import argparse
import json

import tensorflow as tf

#from my_data import VOCAB, color_print
from data_loader import Dataset, VOCAB, color_print
#from my_models import MyModel0
from models.SimpleBiLSTM import SimpleBiLSTM, SeqBiLSTM
from my_utils import pred_to_dict
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("-e", "--max_epoch", type=int, default=35)
    parser.add_argument("-v", "--val-at", type=int, default=100)
    parser.add_argument("-i", "--hidden-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=76)
    parser.add_argument("--val-only", dest='val_only', action='store_true')
    parser.set_defaults(val_only=False)

    args = parser.parse_args()
    #args.device = torch.device(args.device)
     
    with tf.device("/"+args.device):
        #model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)
        #model = SimpleBiLSTM(len(VOCAB), 16, args.hidden_size)
        bilstm_sequential = SeqBiLSTM(len(VOCAB), 16, args.hidden_size)
        model = bilstm_sequential.model

        dataset = Dataset(
            "data/data_dict4.pth",
            args.device,
            val_size=args.val_size,
            test_path="data/test_dict.pth",
        )

        if args.val_only:
            model = tf.keras.models.load_model("model.model")
            validate(model, dataset)
            exit(0)

        train_data, train_labels = dataset.get_train_data()
        print(train_data.shape, train_labels.shape)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model.fit(train_data, train_labels, batch_size=args.batch_size, epochs=args.max_epoch)
        
        '''
        model_json = model.to_json()
        with open("model.json", 'w') as json_file:
            json_file.write(model_json)

        model.save_weights("model.h5")
        '''
        tf.keras.models.save_model(model, "model.model", overwrite=True, include_optimizer=True)

        val_keys, val_data, val_labels = dataset.get_val_data(batch_size=76)
        with open("val_keys.pkl", 'wb') as val_keys_file:
            pickle.dump(val_keys, val_keys_file)
        with open("val_data.pkl", 'wb') as val_data_file:
            pickle.dump(val_data, val_data_file)
        with open("val_labels.pkl", 'wb') as val_labels_file:
            pickle.dump(val_labels, val_labels_file)
        

    '''
    model.eval()
    with torch.no_grad():
        for key in dataset.test_dict.keys():
            text_tensor = dataset.get_test_data(key)

            oupt = model(text_tensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)

            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

            real_text = dataset.test_dict[key]
            result = pred_to_dict(real_text, pred, prob)

            with open("results/" + key + ".json", "w", encoding="utf-8") as json_opened:
                json.dump(result, json_opened, indent=4)

            print(key)
    '''


def get_val_data_from_pickle(keys_filename, data_filename, labels_filename):
    with open(keys_filename, 'rb') as keys_file:
        keys = pickle.load(keys_file)
    with open(data_filename, 'rb') as data_file:
        data = pickle.load(data_file)
    with open(labels_filename, 'rb') as labels_file:
        labels = pickle.load(labels_file)

    return keys, data, labels


def validate(model, dataset, batch_size=1):
    #keys, text, truth = dataset.get_val_data(batch_size=batch_size)
    keys, text, truth = get_val_data_from_pickle("val_keys.pkl", "val_data.pkl", "val_labels.pkl")
    pred = model.predict_classes(text)

    for text_item, pred_item in zip(text, pred):
        #text_item, _ = dataset.val_dict[key]
        real_text = [VOCAB[char_idx] for char_idx in text_item]
        color_print(real_text, pred_item)

'''
def validate(model, dataset, batch_size=1):
    model.eval()
    with torch.no_grad():
        keys, text, truth = dataset.get_val_data(batch_size=batch_size)

        oupt = model(text)
        prob = torch.nn.functional.softmax(oupt, dim=2)
        prob, pred = torch.max(prob, dim=2)

        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()

        for i, key in enumerate(keys):
            real_text, _ = dataset.val_dict[key]
            result = pred_to_dict(real_text, pred[:, i], prob[:, i])

            for k, v in result.items():
                print(f"{k:>8}: {v}")

            color_print(real_text, pred[:, i])
'''

def train(model, dataset, criterion, optimizer, epoch_range, batch_size):
    model.train()

    for epoch in range(*epoch_range):
        optimizer.zero_grad()

        text, truth = dataset.get_train_data(batch_size=batch_size)
        pred = model(text)

        loss = criterion(pred.view(-1, 5), truth.view(-1))
        loss.backward()

        optimizer.step()

        print(f"#{epoch:04d} | Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
