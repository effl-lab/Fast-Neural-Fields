## Weight Scaling Initialization for SNFs

### Image regression
To run the code, you need to install PyTorch, NumPy, and Matplotlib.

```
### Weight scaling (Ours)
CUDA_VISIBLE_DEVICES='gpu_number' python weight_scaling.py --nonlinearity=sine --width=512 --scale=2.5

### Other nonlinearities (choose one)
CUDA_VISIBLE_DEVICES='gpu_number' weight_scaling.py --nonlinearity={sinc, gauss, gabor} --width=512
```
