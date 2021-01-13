#!/usr/bin/env python3

# Standard imports
import sys
import inspect
# External imports
import torch
# Local imports
import data
import models


class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def tab(n):
    return ' ' * 4*n

def fail(msg):
    print(colors.FAIL + tab(1) + f"[FAILED] From {inspect.stack()[1][3]}"
          + msg + colors.ENDC)

def succeed(msg = ""):
    print(colors.OKGREEN + tab(1) + "[PASSED]" + msg + colors.ENDC)

def head(msg):
    print(colors.HEADER + msg + colors.ENDC)

def info(msg):
    print(colors.OKBLUE + tab(1) + msg + colors.ENDC)


def data


def test_model_cnn():
    head("Testing the cnn part")

    try:
        charmap = data.CharMap()
        num_mels = 80
        model = models.CTCModel(charmap,
                                n_mels=num_mels,
                                num_hidden=124,
                                num_layers=3,
                                cell_type='GRU',
                                dropout=0.1)

        T = 124
        B = 10

        cnn_inputs = torch.randn((T, B, num_mels))
        cnn_inputs = cnn_inputs.transpose(0, 1).unsqueeze(dim=1)
        out_cnn = model.cnn(cnn_inputs)

        info(f"Got an output of shape {out_cnn.shape}")
        expected_shape = [10, 32, 31, 40]
        if list(out_cnn.shape) == expected_shape:
            succeed()
        else:
            fail(f"was expecting {expected_shape}")
    except:
        fail(f"{sys.exc_info()[0]}")


if __name__ == '__main__':
    test_model_cnn()
