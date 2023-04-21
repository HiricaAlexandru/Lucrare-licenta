import matplotlib.pyplot as plt

name = "23"
f = open(f"drawing\LSTM_shallow_23_batch_128_{name}.txt")
path_to_save = "drawing\\"
line = f.readline()

epoch = []
train_loss = []
validation_loss = []
precision1 = []
precision2 = []

while line != "" and line is not None:
    line = line.split()
    epoch.append(int(line[0]))
    train_loss.append(float(line[1]))
    validation_loss.append(float(line[2]))
    precision1.append(float(line[3]))
    precision2.append(float(line[4]))
    line = f.readline()

f.close()

ymax_precision1 = max(precision1)
xpos_precision1 = precision1.index(ymax_precision1)

ymax_precision2 = max(precision2)
xpos_precision2 = precision2.index(ymax_precision2)

plt.title(f"Losses with {name} sequence number")
plt.plot(epoch, train_loss, color = 'red', label = "train loss")
plt.plot(epoch, validation_loss, color = 'green', label = "validation loss")
plt.ylim(0, 3)
plt.xlim(0,70)
plt.legend()
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.savefig(f"{path_to_save}Losses with {name} sequence number")
#plt.show()
plt.cla()
plt.title(f"Accuracies with {name} sequence number")
plt.plot(epoch, precision1, color = 'red', label = "Precision1 metric")
plt.plot(epoch, precision2, color = 'green', label = "Precision2 metric")
plt.ylim(0, 1.5)
plt.legend()
plt.xlabel("Epoch") 
plt.ylabel("Accuracy") 
plt.annotate(f'{"{:.2f}".format(ymax_precision1)}', xy=(xpos_precision1, ymax_precision1), xytext=(xpos_precision1, ymax_precision1-0.2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.annotate(f'{"{:.2f}".format(ymax_precision2)}', xy=(xpos_precision2, ymax_precision2), xytext=(xpos_precision2, ymax_precision2+0.2),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.savefig(f"{path_to_save}Accuracies with {name} sequence number")
#plt.show()


