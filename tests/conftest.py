from tqdm import tqdm


def train(clf, dataloader, criterion, optimizer):
    clf.train()
    pb = tqdm(dataloader, leave=False)
    total_loss, total_correct, batch_total, label_total = 0., 0., 0., 0.
    for x, y in pb:
        optimizer.zero_grad()
        y_pred = clf(x)
        loss = criterion(y_pred, y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        total_correct += ((y_pred > 0.5) == y).sum().data.item()
        batch_total += len(x)
        label_total += len(y) * len(y[0])

        pb.set_postfix({
            'Train Accuracy': total_correct / label_total,
            'Train Loss': total_loss / batch_total
        })
    return total_loss / batch_total, total_correct / label_total


def infer(clf, dataloader, criterion):
    clf.eval()
    pb = tqdm(dataloader, leave=False)
    total_loss, total_correct, batch_total, label_total = 0., 0., 0., 0.
    for x, y in pb:
        y_pred = clf(x)
        loss = criterion(y_pred, y.float())

        total_loss += loss.data.item()
        total_correct += ((y_pred > 0.5) == y).sum().data.item()
        batch_total += len(x)
        label_total += len(y) * len(y[0])

        pb.set_postfix({
            'Test Accuracy': total_correct / label_total,
            'Test Loss': total_loss / batch_total
        })
    return total_loss / batch_total, total_correct / label_total
