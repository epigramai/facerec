import os

if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isdir('data/captured'):
    os.mkdir('data/captured')

if not os.path.isdir('data/models'):
    os.mkdir('data/models')

if not os.path.isdir('data/pretrained'):
    os.mkdir('data/pretrained')

if not os.path.isdir('data/detection'):
    os.mkdir('data/detection')