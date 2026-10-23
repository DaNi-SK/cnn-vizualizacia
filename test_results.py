import json

with open("results.json", "r", encoding="utf-8") as file:
    history = json.load(file)

print("Počet zaznamenaných epoch:", len(history["train_loss"]))
print("Tréningová strata:", history["train_loss"])
print("Tréningová presnosť:", history["train_accuracy"])
print("Testovacia strata:", history["test_loss"])
print("Testovacia presnosť:", history["test_accuracy"])
