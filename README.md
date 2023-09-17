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
<font color=Yellow>Bold plus italic is the highest accuracy, bold is the second highest, and italic is the third highest.</font>
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | 0.8000 +/- 0.0439<br> 4674s|0.8111 +/- 0.0274<br>6038s | 0.8278 +/- 0.0228<br> 7995s| 0.7944 +/- 0.0377<br>10348s|0.7944 +/- 0.0334<br>12488s |
| 2 |0.8333 +/- 0.0324<br>4742s | 0.8389 +/- 0.0337<br>6155s | 0.7944 +/- 0.0385<br> 8061s|0.8333 +/- 0.0283<br>10315| |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |


### WL_GPU
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | 0.8425+/-0.0587<br>5663s| 0.8375+/-0.0585<br>6275s|0.8320 +/-0.0580<br>6814s |0.8490+/-0.0522<br>7117s|0.8545+/-0.0500<br>7703s |
| 2 |0.8369+/-0.0562<br>5830s| 0.8424 +/-0.0544<br>6200|0.8486+/-0.0525<br>6730s |0.8320+/-0.0580<br>7297s |0.8490+/-0.0522<br>7693s|
| 3 |0.8369+/-0.0562<br>5869s |0.8313+/-0.0498<br>6524s|0.8431+/-0.0498<br>6941s | ***0.8653+/-0.5621<br>7522s*** |0.8490+/-0.0522<br>7854s|
| 4 |0.8428+/-0.0502<br>6003s |0.8316+/-0.0584<br>6482s |*0.8493+/-0.0471<br>6968s* |0.8271+/-0.0525<br>7508s |0.8490+/-0.0522<br>8023s |
| 5 |**0.8539+/-0.0520<br>6032s** |0.8369+/-0.0583<br>6516s |0.8323+/-0.0416<br>7073s |0.8434+/-0.0553<br>7703s| 0.8435 +/- 0.0495br<br>8200s |
| 6 | 0.8483+/-0.0468 6150s|0.8372+/-0.0485<br>6648s |0.8486+/-0.0502<br>7149s |0.8543+/-0.0528<br>7637s |0.8379+/-0.0550<br>8277s |

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



## GraphiT method test 
SP MUTAG 0.8444 +/- 0.0233 9850s

SP PTC 0.6059 +/- 0.0217 10845s

However, comparing with the GraphiT results, although our results for validation and test accuracy are not very bad, the train loss is still very high and it is difficult to reach near 0 case in GraphiT. And the validation loss curve stabilizes and does not move at the end.


