# Deep Short text clustering

DSTC is a method that aims to cluster short text messages using [CLIP](https://openai.com/blog/clip/) and deep auto-encoder.

If you use it please cite it correctly

### Pre-requisites ###

> pip install -r requirements.txt 




#### Reproduce results  ###
run clip-server
<pre> python -m clip_server </pre>

than

<pre>python DSTC.py --maxiter 1500 --pretrain_epochs 200 --ae_weights results/snippets/ae_weights.h5 --save_dir results/snippets/ --dataset search_snippets</pre>

<pre>python DSTC.py --maxiter 1500 --pretrain_epochs 200 --ae_weights results/biomedical/ae_weights.h5 --save_dir results/biomedical/ --dataset biomedical</pre>

<pre>python DSTC.py --maxiter 1500 --pretrain_epochs 200 --ae_weights results/stackoverflow/ae_weights.h5 --save_dir results/stackoverflow/</pre>





#### Acknowledge

This code is based on repo from [here](https://github.com/hadifar/stc_clustering).