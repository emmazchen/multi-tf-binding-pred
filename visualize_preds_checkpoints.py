
#%%
""" Compare different nums of epochs """

#import actual vs prediction data obtained by running test_step using model check points after various epochs
import torch
pred10=torch.load("test_results/preds_epoch=10.pt").cpu().numpy()
targ10=torch.load("test_results/targets_epoch=10.pt").cpu().numpy()
pred15=torch.load("test_results/preds_epoch=15.pt")
targ15=torch.load("test_results/targets_epoch=15.pt")
pred20=torch.load("test_results/preds_epoch=20.pt")
targ20=torch.load("test_results/targets_epoch=20.pt")

#get r^2
import scipy.stats as stats
r10, _ = stats.pearsonr(targ10, pred10)
r15, _ = stats.pearsonr(targ15, pred15)
r20, _ = stats.pearsonr(targ20, pred20)
r_squared10 = r10**2
r_squared15 = r15**2
r_squared20 = r20**2


# combined plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x=targ10, y=pred10, color='purple',alpha=.3)
ax.scatter(x=targ15, y=pred15, color='blue',alpha=.3)
ax.scatter(x=targ20, y=pred20, color='green',alpha=.3)


ax.set_title("Performance on test set after different numbers of epochs")
legend = ax.legend([f"10 epochs, R^2={r_squared10:.4f}", f"15 epochs, R^2={r_squared15:.4f}",f"20 epochs, R^2={r_squared20:.4f}"], loc='lower right')
ax.add_artist(legend)
ax.set_xlabel("Target negative loss probability")
ax.set_ylabel("Predicted negative loss probability")
fig.show()


# separate plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12),sharex=True, sharey=True)
ax1.scatter(targ10, pred10, color='purple')
ax1.set_title(f'10 epochs, R^2={r_squared10:.4f}')

ax2.scatter(targ15, pred15, color='blue')
ax2.set_title(f'15 epochs, R^2={r_squared15:.4f}')

ax3.scatter(targ20, pred20, color='green')
ax3.set_title(f'20 epochs, R^2={r_squared20:.4f}')
ax3.set_xlabel('Target negative loss probability')

fig.text(0.04, 0.5, 'Predicted negative loss probability', va='center', rotation='vertical')
fig.suptitle('Performance on test set after different numbers of epochs')



# hexbins
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), sharex=True, sharey=True,gridspec_kw={'width_ratios': [1, 1, 1.3]})

hexbin1 = ax1.hexbin(targ10, pred10, gridsize=(7, 7), cmap='Blues')
hexbin2 = ax2.hexbin(targ15, pred15, gridsize=(7, 7), cmap='Blues')
hexbin3 = ax3.hexbin(targ20, pred20, gridsize=(7, 7), cmap='Blues')

ax1.margins(0) 
ax2.margins(0) 
ax3.margins(0) 

cbar3 = fig.colorbar(hexbin3, orientation='vertical')
cbar3.set_label('Frequency')


ax1.set_title(f'10 epochs, R^2={r_squared10:.4f}')
ax2.set_title(f'15 epochs, R^2={r_squared15:.4f}')
ax3.set_title(f'20 epochs, R^2={r_squared20:.4f}')

fig.text(0.04, 0.5, 'Predicted negative loss probability', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Target negative loss probability', ha='center')
fig.suptitle('Performance on test set after different numbers of epochs')

plt.subplots_adjust(wspace=0.1)

plt.show()

# %%
""" Compare seen and unseen tf """
pred20=torch.load("test_results/preds_epoch=20.pt")
targ20=torch.load("test_results/targets_epoch=20.pt")

pred20_unseen=torch.load("test_results/preds_epoch=20_unseentf.pt")
targ20_unseen=torch.load("test_results/targets_epoch=20_unseentf.pt")

import scipy.stats as stats
r20, _ = stats.pearsonr(targ20, pred20)
r20_unseen, _ = stats.pearsonr(targ20_unseen, pred20_unseen)
r_squared20 = r20**2
r_squared20_unseen = r20_unseen**2

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x=targ20, y=pred20, color='green',alpha=.3)
ax.scatter(x=targ20_unseen, y=pred20_unseen, color='orange',alpha=.3)

ax.set_title("Performance on test sets of seen vs unseen TFs after 20 epochs")
legend = ax.legend([f"Seen TFs, R^2={r_squared20:.4f}", f"Unseen TFs, R^2={r_squared20_unseen:.4f}"], loc='lower right')
ax.add_artist(legend)
ax.set_xlabel("Target negative loss probability")
ax.set_ylabel("Predicted negative loss probability")
fig.show()

# %%
""" Compare ablations"""
import torch
pred20=torch.load("test_results/preds_epoch=20.pt")
targ20=torch.load("test_results/targets_epoch=20.pt")
pred20_ablateself=torch.load("test_results/preds_ablateself.pt")
targ20_ablateself=torch.load("test_results/targets_ablateself.pt")
pred20_ablatecross=torch.load("test_results/preds_ablatecross.pt")
targ20_ablatecross=torch.load("test_results/targets_ablatecross.pt")


import scipy.stats as stats
r20, _ = stats.pearsonr(targ20, pred20)
r20_ablateself, _ = stats.pearsonr(targ20_ablateself, pred20_ablateself)
r20_ablatecross, _ = stats.pearsonr(targ20_ablatecross, pred20_ablatecross)
r_squared20 = r20**2
r_squared20_ablateself = r20_ablateself**2
r_squared20_ablatecross = r20_ablatecross**2

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(x=targ20, y=pred20, color='green',alpha=.2)
ax.scatter(x=targ20_ablateself, y=pred20_ablateself, color='orange',alpha=.2)
ax.scatter(x=targ20_ablatecross, y=pred20_ablatecross, color='tomato',alpha=.2)

ax.set_title("Ablations (lr=1e-4, n_epoch=20 for all)")
legend = ax.legend([f"Full model, R^2={r_squared20:.4f}", f"Ablate post-cross attention self attention, R^2={r_squared20_ablateself:.4f}", f"Ablate cross and self attention, R^2={r_squared20_ablatecross:.4f}"], loc='lower right')
ax.add_artist(legend)
ax.set_xlabel("Target negative loss probability")
ax.set_ylabel("Predicted negative loss probability")
fig.show()



# separate plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 30),sharex=True, sharey=True)
ax1.scatter(targ20, pred20, color='green')
ax1.set_title(f'Full model, R^2={r_squared20:.4f}')

ax2.scatter(targ20_ablateself, pred20_ablateself, color='orange')
ax2.set_title(f'Ablate post-cross attention self attention, R^2={r_squared20_ablateself:.4f}')

ax3.scatter(targ20_ablatecross, pred20_ablatecross, color='orangered')
ax3.set_title(f'Ablate cross and self attention, R^2={r_squared20_ablatecross:.4f}')
ax3.set_xlabel('Target negative loss probability')

fig.text(0.04, 0.5, 'Predicted negative loss probability', va='center', rotation='vertical')
fig.suptitle('Performance on test set after various ablations (lr=1e-4, n_epoch=20 for all)')




# hexbins
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), sharex=True, sharey=True,gridspec_kw={'width_ratios': [1, 1, 1.3]})

hexbin1 = ax1.hexbin(targ20, pred20, gridsize=(7, 7), cmap='Greens')
hexbin2 = ax2.hexbin(targ20_ablateself, pred20_ablateself, gridsize=(7, 7), cmap='Greens')
hexbin3 = ax3.hexbin(targ20_ablatecross, pred20_ablatecross, gridsize=(7, 7), cmap='Greens')

ax1.margins(0) 
ax2.margins(0) 
ax3.margins(0) 

cbar3 = fig.colorbar(hexbin3, orientation='vertical')
cbar3.set_label('Frequency')


ax1.set_title(f'Full model, R^2={r_squared20:.4f}')
ax2.set_title(f'Ablate (post-cross attention) self attention, R^2={r_squared20_ablateself:.4f}')
ax3.set_title(f'Ablate cross and self attention, R^2={r_squared20_ablatecross:.4f}')

fig.text(0.04, 0.5, 'Predicted negative loss probability', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Target negative loss probability', ha='center')
fig.suptitle('Performance on test set after various ablations (lr=1e-4, n_epoch=20 for all)')

plt.subplots_adjust(wspace=0.1)

plt.show()

# %%
