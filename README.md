# The result of the experiments

## K-Fold only use train sets and validation sets
 The MUTAG dataset was used with 10-fold cross-validation. Using the method in sklearn, random_state is fixed to 1. The highest validation accuracy is extracted from each fold as the accuracy of that fold, and finally the model's accuracy is calculated by averaging.
## Update 2023/9/13
1. Do all the experiments on MUTAG, and create tables.  :fast_forward:
2. Using faster kernel. :fast_forward: 
3. Early stop
4. Trying different attentions.
5. Positional encoding
6. Multi head attention
### 1 Layer 2 Hops Subgraph Accuracy
|  | SP | WL | WLSP | Graphlet_3 |
| :-----:| :------: | :------: | :----: | :---:|
| MUTAG | 0.8518+/-0.0363 <br>5102s | 0.8678+/-0.0402 <br>43710s |0.8728+/-0.0374<br>44751s|0.8512+/-0.0375<br>14300s |

### 1 Layer 3 Hops Subgraph Accuracy
|  | SP | WL | WLSP | Graphlet_3 |
| :-----:| :------: | :------: | :----: | :---:|
| MUTAG | 0.8623+/-0.0367 <br>8525s |0.8675+/-0.0417<br>44947s|0.8570+/-0.0422 <br>47898s|0.8567+/-0.0417<br>44947s |

### 3 Layers 2 Hop Subgraph Accuracy
|  | SP | WL | WLSP | Graphlet_3 |
| :-----:| :------: | :------: | :----: | :---:|
| MUTAG | 0.8781+/-0.0373<br>16850|0.8784+/-0.0372<br>44945s|0.8784+/-0.0349<br>44298s| 0.8833+/-0.0379<br>44945s|

## MUTAG 2023/9/13 
### Shortestpath kernel
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | 0.8000 +/- 0.0439<br> 4674s|0.8111 +/- 0.0274<br>6038s | 0.8278 +/- 0.0228<br> 7995s| | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

### Graphlet kernel
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

### WLSP
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

### WL_GPU
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

## GraphiT method test 
SP MUTAG 0.8444 +/- 0.0233 9850s

SP PTC 0.6059 +/- 0.0217 10845s

However, comparing with the GraphiT results, although our results for validation and test accuracy are not very bad, the train loss is still very high and it is difficult to reach near 0 case in GraphiT. And the validation loss curve stabilizes and does not move at the end.


