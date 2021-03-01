#!/usr/bin/env python3

# Learning and Intelligent Systems Project
# Start socket server, receive np image as pickle dump and perform inference.
# Since ros can sometimes be annoying (with different Ubuntu versions and such),
# this script starts an almost embarrasing server to make predictions.
#
# Marius Montebaur, WS20/21

from maskrcnn_benchmark.config import cfg
from inference.coco_predict import COCODPredictor
import socket
import pickle


def perform_inference(image):
    """ Take an image as np.array in bgr8 format and perform inference. """

    config_file = "/mrcnn_integrate/config/e2e_mask_rcnn_R_50_FPN_1x_caffe2_stick.yaml"

    # update the config options with the config file
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
    cfg.merge_from_list(["MODEL.WEIGHT", "/mrcnn_integrate/train_tools/tmp/model_0380000.pth"])

    # Construct the predict and visualizer
    coco_predict = COCODPredictor(cfg, min_image_size=800, confidence_threshold=0.7)

    predictions_raw = coco_predict.run_on_opencv_image(image)
    return predictions_raw


def socket_server():

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Use 0.0.0.0 from within docker container
        s.bind(("0.0.0.0", 8134))
        s.listen()
        print("Started listening")

        while True:
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                data = []
                ps = 0
                while True:
                    packet = conn.recv(4096)
                    ps += len(packet)
                    if not packet: break
                    data.append(packet)

                    # TODO: Hardcoded for (240, 320, 3) uint8 np array image
                    if ps == 230561:
                        break

                request = pickle.loads(b"".join(data))
                prediction = perform_inference(request)

                response = (
                    prediction.bbox.numpy(),
                    prediction.mode  # str
                )

                pickled_response = pickle.dumps(response)

                conn.sendall(pickled_response)


if __name__ == '__main__':
    socket_server()
