import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score

def train(model, 
        train_loader, val_loader,
        num_epochs=10, 
        learning_rate=1e-4, 
        use_amp=True, 
        device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Class-imbalance weight: pos_weight applies to the positive class and grows as hits get rarer
    pos_weight = torch.tensor([3.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=7)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # early stopping
    best_val_f1 = 0.0
    patience = 20
    no_improve_count = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training", leave=False)
        for inputs, labels in loop:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)  # to float, with an extra dimension

            # ============================== TODO [Ex 2.3] ==============================
            # One optimisation step on (inputs, labels), storing the model output in
            # `outputs` and the loss in `loss`:
            #   - zero the gradients
            #   - use_amp=True : forward + loss under torch.cuda.amp.autocast(), then
            #                    scaler.scale(loss).backward(), scaler.step(optimizer),
            #                    scaler.update()
            #   - use_amp=False: plain forward, loss.backward(), optimizer.step()
            # ===========================================================================
            raise NotImplementedError("TODO [Ex 2.3]")

            total_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_loader.dataset)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"[Epoch {epoch+1}]")
        print(f"\tTrain Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")

        # Validation
        _, val_f1 = validate(model, val_loader)

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_count = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

def validate(model, val_loader, threshold=0.5, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            preds = (torch.sigmoid(outputs) >= threshold).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"\t  Val Loss: {val_loss:.4f} |   Val F1: {val_f1:.4f}\n")

    return all_preds, val_f1

def predict(model, loader, threshold=0.5, device=None):
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            outputs = model(inputs)

            preds = (torch.sigmoid(outputs) >= threshold).int()
            all_preds.extend(preds.cpu().numpy())

    return all_preds