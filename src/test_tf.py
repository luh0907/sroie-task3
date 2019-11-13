import argparse
import tensorflow as tf

from data_loader import Dataset, VOCAB, color_print
from my_utils import pred_to_dict, truth_to_dict, calc_accuracy, compare_truth

def inference(model, dataset, print_size=10, validation_mode=False):
    if validation_mode:
        keys, text, truth = dataset.get_val_data()
        val_size = len(keys)
    else:
        text = dataset.get_test_data()

    pred = model.predict_classes(text)
    prob = model.predict_proba(text)

    class_acc = 0.0
    char_acc = 0.0

    for i, text_item in enumerate(text):
        real_text = "".join([VOCAB[char_idx] for char_idx in text_item])
        result = pred_to_dict(real_text, pred[i], prob[i])
        if validation_mode:
            ground_truth = truth_to_dict(real_text, truth[i])
            class_acc_unit = calc_accuracy(result, ground_truth)
            char_acc_unit = compare_truth(result, ground_truth)
            class_acc += class_acc_unit
            char_acc += char_acc_unit

        if i < print_size:
            print("====== Inference #%d ======" % i)
            for k, v in result.items():
                print(f"{k:>8}: {v}")
            print()

            if validation_mode:
                for k, v in ground_truth.items():
                    print(f"{k:>8}: {v}")

                print("-ACCURACY(Class): %.2f" % class_acc_unit)
                print("-ACCURACY(Char) : %.2f" % char_acc_unit)
                print()

            color_print(real_text, pred[i])
            print("============================")
            print()

    if validation_mode:
        print("=ACCURACY(Class): %.2f" % (class_acc*100/val_size))
        print("=ACCURACY(Char) : %.2f" % (char_acc*100/val_size))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("--model-path", default="saved/model.model")
    parser.add_argument("--dataset-path", default="saved/dataset.pkl")
    parser.add_argument("--val", dest='validation_mode', action='store_true')
    parser.set_defaults(validation_mode=False)

    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)
    dataset = Dataset(None, args.device, test_path="data/test_dict.pth")
    if args.validation_mode:
        dataset.load_dataset("saved/dataset.pkl")
        dataset.device = args.device

    inference(model, dataset, validation_mode=args.validation_mode)

