import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# final_csv = "final_training_log.csv"

# df_final = pd.read_csv(final_csv)

# df_acc_final = df_final[["epoch", "accuracy"]]
# df_loss_final = df_final[["epoch", "loss"]]

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle("Accuracy and Loss Over 100 Epochs")
# ax1.plot(
#     df_acc_final["epoch"],
#     df_acc_final["accuracy"],
#     color="blue",
#     label="accuracy",
# )
# ax1.set_title("Accuracy")
# ax1.set_xlabel("Epochs")
# ax2.set_ylabel("Accuracy")
# ax1.legend()
# ax1.grid(True)

# ax2.plot(
#     df_loss_final["epoch"], df_loss_final["loss"], color="red", label="loss"
# )
# ax2.set_title("Loss")
# ax2.set_xlabel("Epochs")
# ax2.set_ylabel("Loss")
# ax2.legend()
# ax2.grid(True)

# plt.tight_layout()
# plt.savefig("final_accuracy_vs_loss")

model_1_path = "LSTMLab-model1.csv"
model_2_path = "LSTMLab-model2.csv"
model_3_path = "LSTMLab-model3.csv"
model_4_path = "LSTMLab-Sheet3.csv"

model_1 = pd.read_csv(model_1_path)
model_2 = pd.read_csv(model_2_path)
model_3 = pd.read_csv(model_3_path)
model_4 = pd.read_csv(model_4_path)

fig, (ax1, ax2) = plt.subplots(1, 2)


ax1.plot(model_1["epoch"], model_1["accuracy"], color="blue", label="model_1")
ax1.plot(model_2["epoch"], model_2["accuracy"], color="red", label="model_2")
ax1.plot(model_3["epoch"], model_3["accuracy"], color="green", label="model_3")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)


ax2.plot(model_1["epoch"], model_1["loss"], color="blue", label="model_1")
ax2.plot(model_2["epoch"], model_2["loss"], color="red", label="model_2")
ax2.plot(model_3["epoch"], model_3["loss"], color="green", label="model_3")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

fig.suptitle("Accuracy and Loss of Initial Models")
fig.tight_layout()
fig.savefig("initial_models")
