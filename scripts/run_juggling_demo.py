#!/usr/bin/env python
"""
Run a custom classifier that was obtained via the train_classifier script.

Usage:
  run_custom_classifier.py [--custom_classifier=PATH]
                           [--camera_id=CAMERA_ID]
                           [--path_in=FILENAME]
                           [--path_out=FILENAME]
                           [--title=TITLE]
                           [--use_gpu]
  run_custom_classifier.py (-h | --help)

Options:
  --custom_classifier=PATH   Path to the custom classifier to use
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
"""
import math
import os
import json
from collections import deque

import cv2
from docopt import docopt

import realtimenet.display
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
from realtimenet.downstream_tasks.nn_utils import Pipe, LogisticRegression
from realtimenet.downstream_tasks.postprocess import PostprocessClassificationOutput
from realtimenet.engine import InferenceListener


FONT = cv2.FONT_HERSHEY_PLAIN


class JugglingDisplayOp:

    QUAL_UNKNOWN = None
    QUAL_GOOD = 'GOOD'
    QUAL_HINT = 'HINT'

    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    def __init__(self):
        self.counts = deque(maxlen=5)

    def display(self, img, display_data):
        if 'sorted_predictions' not in display_data:
            return

        sorted_predictions = display_data['sorted_predictions']

        top_prediction, top_proba = sorted_predictions[0]

        count = self._get_ball_count(top_prediction, top_proba)
        self.counts.append(count)
        num_objects = self.most_frequent(self.counts)
        top_n = 3
        cv2.putText(img, f"Juggling {num_objects} objects", (100, 100), FONT, 3, (255, 255, 255),
                    2, cv2.LINE_AA)

        juggling_quality, msg = self._get_juggling_quality(sorted_predictions[:top_n])
        color = {
            self.QUAL_UNKNOWN: self.WHITE,
            self.QUAL_GOOD: self.GREEN,
            self.QUAL_HINT: self.BLUE
        }[juggling_quality]
        cv2.putText(img, msg, (100, 180), FONT, 3, color,
                    2, cv2.LINE_AA)

        self.draw_count_confidence_bars(img, sorted_predictions)

        return img

    def most_frequent(self, lst):
        return max(set(lst), key=lst.count)

    def draw_count_confidence_bars(self, img, sorted_predictions, threshold=.7):
        count_probas = self._get_ball_count_probas(sorted_predictions)
        for count, proba in count_probas.items():
            dx = 10
            dy = 400
            bar_width = 200
            bar_height = 20
            bar_spacing = 5
            pt1 = (dx, dy + (bar_height + bar_spacing) * count)
            pt2 = (int(dx + bar_width * proba), dy + (bar_height + bar_spacing) * count + bar_height)
            color = self.GREEN if proba > threshold else self.BLACK
            cv2.rectangle(img, pt1, pt2, color)

            pt3 = (pt1[0] + 5, pt1[1] + bar_height - bar_spacing)
            cv2.putText(img, f"{count} objects", pt3, fontFace=FONT, fontScale=1, color=color,
                        thickness=2 if proba > threshold else 1)

    def _get_ball_count(self, prediction, prob, min_prob=0.7):
        # just holding? no matter how many balls the model predicts
        if 'hold' in prediction or prob < min_prob:
            return 0

        # remaining: all other activities
        if '3b_' in prediction:
            return 3
        elif '2b_' in prediction:
            return 2
        elif '1b_' in prediction:
            return 1
        return 0

    def _get_juggling_quality(self, predictions, min_prob=.7):
        top_prediction, top_proba = predictions[0]

        if top_proba < min_prob:
            return self.QUAL_UNKNOWN, ''

        if 'good' in top_prediction:
            return self.QUAL_GOOD, "Good!"

        if 'too_high' in top_prediction:
            return self.QUAL_HINT, "Go lower"

        if 'too_low' in top_prediction:
            return self.QUAL_HINT, "Go higher"

        if 'shaky' in top_prediction:
            return self.QUAL_HINT, "Steady yourself"

        if "shower" in top_prediction:
            return self.QUAL_GOOD, "Nice trick!"

        return self.QUAL_UNKNOWN, ''

    def _get_ball_count_probas(self, sorted_predictions):
        counts = {count: .0 for count in range(4)}
        for prediction, proba in sorted_predictions:
            count = self._get_ball_count(prediction, prob=1.)
            counts[count] += proba
        return counts


class JugglingApp(InferenceListener):

    def __init__(self, **kwargs):
        super(JugglingApp, self).__init__(**kwargs)

    def on_prediction(self, prediction, post_processed_data, img, numpy_img):


        """
        {'sorted_predictions': [
        ('empty_frame', 0.9996071), ('1b_drop', 0.00033792164), ('1b_hold', 3.32698e-05),
        ('leave_frame', 9.400211e-06), ('distracted', 6.5444187e-06), ('3b_off_pace_cont', 2.5848208e-06),
        ('2b_same_time', 2.3597966e-06), ('1b_too_high_single', 2.953931e-07),
        ('2b_unequal_height_cont', 2.4169472e-07), ('3b_hold', 1.1983255e-07),
        ('2b_hold', 5.0515442e-08), ('enter_frame', 4.723658e-08), ('3b_too_low_cont', 4.684075e-08),
        ('2b_good_single', 2.5485017e-08), ('1b_too_low_single', 1.8398207e-08), ('3b_too_high_cont', 1.798572e-08),
        ('2b_unequal_height_single', 1.5541763e-08), ('3b_shower_cont', 1.2838091e-08),
        ('1b_good_single', 6.1939502e-09), ('2b_good_cont', 5.377808e-09), ('talking', 2.0855708e-09),
        ('3b_good_cont', 2.0805064e-09), ('2b_shower_cont', 5.0557886e-10), ('3b_shaky', 4.868166e-10),
        ('1b_good_cont', 2.4186167e-10), ('0b_pretend', 1.7263455e-10)]}

        """
        # print(post_processed_data)

        pass


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    camera_id = args['--camera_id'] or 0
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    custom_classifier = args['--custom_classifier'] or '/home/florian/projects/20bn-realtimenet/juggling'
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # Load original feature extractor
    feature_extractor = feature_extractors.StridedInflatedEfficientNet()
    checkpoint = engine.load_weights('resources/backbone/strided_inflated_efficientnet.ckpt')

    # Load custom classifier
    checkpoint_classifier = engine.load_weights(os.path.join(custom_classifier, 'classifier.checkpoint'))
    # Update original weights in case some intermediate layers have been finetuned
    name_finetuned_layers = set(checkpoint.keys()).intersection(checkpoint_classifier.keys())
    for key in name_finetuned_layers:
        checkpoint[key] = checkpoint_classifier.pop(key)
    feature_extractor.load_state_dict(checkpoint)
    feature_extractor.eval()

    with open(os.path.join(custom_classifier, 'label2int.json')) as file:
        class2int = json.load(file)
    INT2LAB = {value: key for key, value in class2int.items()}

    gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim,
                                            num_out=len(INT2LAB))
    gesture_classifier.load_state_dict(checkpoint_classifier)
    gesture_classifier.eval()

    # Concatenate feature extractor and met converter
    net = Pipe(feature_extractor, gesture_classifier)

    # Create inference engine, video streaming and display instances
    inference_engine = engine.InferenceEngine(net, use_gpu=use_gpu)

    video_source = camera.VideoSource(camera_id=camera_id,
                                      size=inference_engine.expected_frame_size,
                                      filename=path_in)

    framegrabber = camera.VideoStream(video_source,
                                      inference_engine.fps)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    display_ops = [
        realtimenet.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.1),
        JugglingDisplayOp(),
    ]
    display_results = realtimenet.display.DisplayResults(title=title, display_ops=display_ops)

    # inference_listeners = [
    #     JugglingApp()
    # ]

    engine.run_inference_engine(inference_engine,
                                framegrabber,
                                postprocessor,
                                display_results,
                                path_out)
