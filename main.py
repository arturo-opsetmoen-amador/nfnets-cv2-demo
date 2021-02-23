import os
import cv2
import ast
import jax
import time
import dill
import base
import nfnet
import haiku as hk
import numpy as np
import jax.numpy as jnp

os.environ['DISPLAY'] = ':1'  # check the output of echo $DISPLAY in console

# Ingest video stream (after warm-up) from video capture device (webcam) 0
cap = cv2.VideoCapture(0)
time.sleep(2.0)

# Set up the nfnets variant
variant = 'F0'

# Load the model parameters from npz file
with open(f'./{variant}_haiku.npz', 'rb') as in_file:
    params = dill.load(in_file)
print(f'Model loaded w/ {hk.data_structures.tree_size(params) / 1e6:.2f}M Params')

# Load the dictionary with imagenet classes
with open(f'./imagenet_classlist.txt', 'r') as classes_file:
    imagenet_classlist = ast.literal_eval(classes_file.read())

# Define the image size from nfnet base
imsize = base.nfnet_params[variant]['test_imsize']

# Convert im to tensor and normalize with channel-wise RGB
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


# Prepare the forward fn
def forward(inputs, is_training):
    model = nfnet.NFNet(num_classes=1000, variant=variant)
    return model(inputs, is_training=is_training)['logits']


net = hk.without_apply_rng(hk.transform(forward))
fwd = jax.jit(lambda inputs: net.apply(params, inputs, is_training=False))

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50, 50)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

if __name__ == '__main__':

    while True:
        time_start = time.time()
        ret, frame_vis = cap.read()
        frame = cv2.resize(frame_vis, (imsize, imsize))
        frame = cv2.resize(frame, (imsize + 32, imsize + 32))

        frame = frame[16:16 + imsize, 16:16 + imsize]
        x = (np.float32(frame) - MEAN_RGB) / STDDEV_RGB

        logits = fwd(x[None])  # Give X a newaxis to make it batch-size-1
        which_class = imagenet_classlist[int(logits.argmax())]
        print(f'ImageNet class: {which_class}')

        time_end = time.time()
        cv2.putText(frame_vis, f'{which_class}',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.putText(frame_vis, f'FPS: {np.round(1 / (time_end - time_start))}',
                    (50, 100),
                    font,
                    fontScale,
                    fontColor,
                    lineType)

        cv2.imshow("NFNETS", frame_vis)

        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
