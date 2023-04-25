
from utils.utils import AverageMeter, DiceAccuracy, IoUAccuracy, save_checkpoint
from tqdm  import tqdm
import torch
import os

def train_epoch(model=None, write_iter_num=5, trainloader=None, validloader=None, optimizer=None, scheduler=None, device=None, 
                criterion=None, start_epoch=0, end_epoch=None, log_path="./log", model_path="./weight", best_loss = 0):
    for epoch in range(start_epoch, end_epoch):
        is_best = False
        file = open(os.path.join(log_path, f'{epoch}_log.txt'), 'a')
        scaler = torch.cuda.amp.GradScaler()
        assert trainloader is not None, print("train_dataset is none")
        model.train()        
        ave_accuracy = AverageMeter()
        #scaler = torch.cuda.amp.GradScaler()
        for idx, (Image, Label) in enumerate(tqdm(trainloader)):
            #model input data
            Input = Image.to(device, non_blocking=True)
            label = Label.to(device, non_blocking=True)
            Output = model(Input)
            loss = criterion(Output, label)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accuracy = DiceAccuracy(Output, label, n_classes=1, thresh=0.5, sigmoid=True)
            ave_accuracy.update(accuracy)
            if idx % write_iter_num == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(trainloader)} '
                        f'Loss : {loss :.4f} '
                        f'Accuracy : {accuracy :.2f} ')
            if idx % (2*write_iter_num) == 0:
                tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(trainloader)} '
                        f'Loss : {loss :.4f} '
                        f'Accuracy : {accuracy :.2f} ', file=file)
        tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n')
        tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.4f} \n\n', file=file)
        ave_accuracy = AverageMeter()
        assert validloader is not None, print("train_dataset is none")
        model.eval()
        with torch.no_grad():
            for idx, (Image, Label) in enumerate(tqdm(validloader)):
                #model input data
                Input = Image.to(device, non_blocking=True)
                label = Label.to(device, non_blocking=True)
                Output = model(Input)
                loss = criterion(Output, label)
                accuracy = DiceAccuracy(Output, label, n_classes=1, thresh=0.5, sigmoid=True)
                ave_accuracy.update(accuracy)
                if idx % write_iter_num == 0:
                    tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(validloader)} '
                            f'Loss : {loss :.4f} '
                            f'Accuracy : {accuracy :.2f} ')
                if idx % (2*write_iter_num) == 0:
                    tqdm.write(f'Epoch : {epoch} Iter : {idx}/{len(validloader)} '
                            f'Loss : {loss :.4f} '
                            f'Accuracy : {accuracy :.2f} ', file=file)
            tqdm.write(f'Average Accuracy : {ave_accuracy.average() :.2f} ', file=file)
        accuracy = ave_accuracy.average()
        scheduler.step()
        is_best = accuracy > best_loss
        best_loss = max(best_loss, accuracy)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_loss,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best=is_best, path=model_path)
        file.close()