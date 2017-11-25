- google official

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Kimura-san's page about object detection

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
ackage                Version  
---------------------- ---------
appnope                0.1.0    
astroid                1.5.3    
bleach                 1.5.0    
cycler                 0.10.0   
decorator              4.0.11   
entrypoints            0.2.3    
enum34                 1.1.6    
html5lib               0.9999999
imageio                2.1.2    
ipykernel              4.6.1    
ipython                6.2.1    
ipython-genutils       0.2.0    
ipywidgets             7.0.5    
isort                  4.2.15   
jedi                   0.11.0   
Jinja2                 2.10     
jsonschema             2.6.0    
jupyter                1.0.0    
jupyter-client         5.1.0    
jupyter-console        5.2.0    
jupyter-core           4.4.0    
lazy-object-proxy      1.3.1    
Markdown               2.6.9    
MarkupSafe             1.0      
matplotlib             2.1.0    
mccabe                 0.6.1    
mistune                0.8.1    
moviepy                0.2.3.2  
nbconvert              5.3.1    
nbformat               4.4.0    
notebook               5.2.1    
numpy                  1.13.3   
olefile                0.44     
opencv-python          3.3.0.10 
pandocfilters          1.4.2    
parso                  0.1.0    
pexpect                4.3.0    
pickleshare            0.7.4    
Pillow                 4.3.0    
pip                    9.0.1    
progressbar2           3.34.3   
prompt-toolkit         1.0.15   
protobuf               3.4.0    
ptyprocess             0.5.2    
Pygments               2.2.0    
pyparsing              2.2.0    
python-dateutil        2.6.1    
python-utils           2.2.0    
pytz                   2017.3   
pyzmq                  16.0.3   
qtconsole              4.3.1    
scipy                  1.0.0    
setuptools             36.7.2   
simplegeneric          0.8.1    
six                    1.11.0   
tensorflow             1.4.0    
tensorflow-tensorboard 0.4.0rc2 
terminado              0.7      
testpath               0.3.1    
tornado                4.5.2    
tqdm                   4.11.2   
traitlets              4.3.2    
wcwidth                0.1.7    
Werkzeug               0.12.2   
wheel                  0.30.0   
widgetsnbextension     3.0.8    
wrapt                  1.10.11  
```


# 実行

```sh
# 毎回必要。source prepare_source.sh
export PYTHONPATH=$PYTHONPATH:`pwd`/models/research:`pwd`/models/research/slim
```



