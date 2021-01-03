import numpy as np
'''
Five tries =>
1st try: tensor(4.0499, grad_fn=<MeanBackward0>)
2nd try: tensor(3.8310, grad_fn=<MeanBackward0>)
3rd try: tensor(3.8973, grad_fn=<MeanBackward0>)
4th try: tensor(3.4679, grad_fn=<MeanBackward0>)
5th try: tensor(3.7530, grad_fn=<MeanBackward0>)
'''

loss_results_five_iterations = [4.0499,3.8310, 3.8973, 3.8973, 3.4679]

print("Mean = ", np.mean(loss_results_five_iterations))
print("STD = ", np.std(loss_results_five_iterations))