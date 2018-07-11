from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import sys
import final_data as data

# 256 8000 75%
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
         my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[50, 50, 50],
        # The model must choose between 3 classes.
        n_classes=397)

    # Train the Model.
    classifier.train(
        input_fn=lambda:data.train_input_fn(train_x, train_y,
                                                 args.batch_size),
        steps=args.train_steps)


    # Evaluate the model.
    eval_result = classifier.evaluate(
        input_fn=lambda:data.eval_input_fn(test_x, test_y,
                                                args.batch_size))
    # train and evaluate need repair



    
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    # expected = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
    # predict_x = {
    #     'SepalLength': [5.1, 5.9, 6.9],
    #     'SepalWidth': [3.3, 3.0, 3.1],
    #     'PetalLength': [1.7, 4.2, 5.4],
    #     'PetalWidth': [0.5, 1.5, 2.1],
    # }
    # predict_x need replace
    # output=sys.stdout
    # outputfile=open("out_raw",'w')
    # sys.stdout=outputfile
    # predict_x = data.load_pred()
    # expected = data.LABEL
    # predictions = classifier.predict(
    #     input_fn=lambda:data.eval_input_fn(predict_x,
    #                                             labels=None,
    #                                             batch_size=args.batch_size))

    # # template = ('Prediction is "{}" ({:.1f}%)')
    # template = ('cls_{}')
    # for pred_dict in predictions:
    #     class_id = pred_dict['class_ids'][0]
    #     probability = pred_dict['probabilities'][class_id]

    #     print(template.format(data.LABEL[class_id],
    #                           100 * probability))

    # outputfile.close()
    # sys.stdout=output

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
