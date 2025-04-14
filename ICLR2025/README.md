## Weight Scaling Initialization for SNFs

### Image regression
For running a code, you need to install Pytorch, Numpy and Matplotlib.

```
### Weight scaling (Ours)
CUDA_VISIBLE_DEVICES='gpu_number' python weight_scaling.py --nonlinearity=sine --width=512 --scale=2.5

### Other nonlinearities (choose one)
CUDA_VISIBLE_DEVICES='gpu_number' weight_scaling.py --nonlinearity={sinc, gauss, gabor} --width=512
```
