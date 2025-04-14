## Weight Scaling Initialization for SNFs

### Image regression

```
### Weight scaling (Ours)
$ python weight_scaling.py --nonlinearity=sine --width=512 --scale=2.5

### Other nonlinearities (choose one)
$ python weight_scaling.py --nonlinearity={sinc, gauss, gabor} --width=512
```
