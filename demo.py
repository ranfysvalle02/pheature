import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate synthetic shape data
def generate_shapes(n_samples=100, img_size=28):
    print("\n[INFO] Generating synthetic shape data...")
    images, labels = [], []
    for _ in range(n_samples):
        img = np.zeros((img_size, img_size), dtype=np.float32)
        shape_type = np.random.randint(2)
        if shape_type:  # Circle
            rr, cc = np.ogrid[:img_size, :img_size]
            mask = (rr - img_size // 2) ** 2 + (cc - img_size // 2) ** 2 < (img_size // 4) ** 2
            img[mask] = 1.0
        else:  # Square
            img[img_size // 4: -img_size // 4, img_size // 4: -img_size // 4] = 1.0
        images.append(img)
        labels.append(shape_type)
    print(f"[INFO] Generated {n_samples} images (0 = square, 1 = circle).")
    return torch.tensor(images).unsqueeze(1), torch.tensor(labels)

# Step 2: Define the CNN model
class ShapeClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        print("[INFO] Initializing the CNN model for shape classification...")
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(16 * 11 * 11, 2)
        )
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, _):
        x, y = batch
        preds = self(x)
        loss = nn.CrossEntropyLoss()(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print("[INFO] Configuring the optimizer...")
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Step 3: Prepare the data
print("[INFO] Preparing the training data...")
x_train, y_train = generate_shapes(500)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
print("[INFO] Data loaded. Training data contains 500 samples.")

# Step 4: Train the model
print("\n[INFO] Starting model training...")
model = ShapeClassifier()
trainer = pl.Trainer(max_epochs=5, enable_checkpointing=False, logger=False)

print("\n[TRAINING] Model training in progress...\n")
trainer.fit(model, train_loader)
print("[TRAINING COMPLETE] Model finished training.")

# Step 5: Test model performance and visualize predictions
print("\n[TESTING] Evaluating model on new data...")
x_test, y_test = generate_shapes(10)
model.eval()  # Set the model to evaluation mode

# Perform predictions and visualize
with torch.no_grad():
    preds = model(x_test).argmax(dim=1)

    # Plot test images with predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle("Model Predictions on Test Images")
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i][0], cmap="gray")
        pred_label = "Circle" if preds[i].item() == 1 else "Square"
        actual_label = "Circle" if y_test[i].item() == 1 else "Square"
        ax.set_title(f"Pred: {pred_label}\nActual: {actual_label}")
        ax.axis("off")
    plt.show()

# Final summary of accuracy
correct = (preds == y_test).sum().item()
accuracy = correct / len(y_test) * 100
print(f"\n[SUMMARY] Model Accuracy on Test Set: {accuracy}% ({correct}/{len(y_test)} correct predictions)")
print("[DEMO COMPLETE] Check the visualization above for model predictions.")
