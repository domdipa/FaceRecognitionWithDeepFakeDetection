import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

csv_path = input("Enter path of csv file: ")
similarity_column = input("Enter name of similarity column: ")

df = pd.read_csv(csv_path, sep=';', decimal=',')
df["variant_label"] = df["Test_Variant"].apply(lambda x: 1 if x == 0 else 0)

# used as doc for youden index: https://gist.github.com/twolodzko/4fae2980a1f15f8682d243808e5859bb
fpr, tpr, thresholds = metrics.roc_curve(df["variant_label"], df[similarity_column])
best_index = np.argmax(tpr - fpr)
best_threshold = thresholds[best_index]

print(f"best threshold: {best_threshold}")

plt.plot(fpr, tpr, label="ROC Curve")
plt.scatter(fpr[best_index], tpr[best_index], color='red',
            label=f"Best Threshold: {best_threshold:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='No skill')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()