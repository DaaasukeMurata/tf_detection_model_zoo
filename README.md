- google official

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Kimura's page about object detection

https://github.com/HidetoKimura/carnd_object_detection

- labels

https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt



# 環境準備

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath

```sh
mac$ brew install protobuf

# From tensorflow/models/research/
mac$ protoc object_detection/protos/*.proto --python_out=.
```

``` sh
(coco) mac-air:image_test$ python -V
Python 3.5.4
(coco) mac-air:image_test$ pip list --format='columns'
Package                Version
---------------------- ---------
bleach                 1.5.0
cycler                 0.10.0
enum34                 1.1.6
html5lib               0.9999999
Markdown               2.6.9
matplotlib             2.1.0
numpy                  1.13.3
olefile                0.44
Pillow                 4.3.0
pip                    9.0.1
protobuf               3.4.0
pyparsing              2.2.0
python-dateutil        2.6.1
pytz                   2017.3
scipy                  1.0.0
setuptools             36.7.2
six                    1.11.0
tensorflow             1.4.0
tensorflow-tensorboard 0.4.0rc2
Werkzeug               0.12.2
wheel                  0.30.0
```


# 実行

```sh
# 毎回必要。source prepare_source.sh
export PYTHONPATH=$PYTHONPATH:`pwd`/models/research:`pwd`/models/research/slim
```



