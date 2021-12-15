import neptune.new as neptune
import json

with open('/home/david/mae_results/log.txt', 'r') as file:
    results = file.readlines()
    results = [line.rstrip() for line in results]

run = neptune.init(
    project='eiphodos/Asparagus'
)
for r in results:
    train_stats = eval(r)
    epoch = train_stats['epoch']
    run['train/loss'].log(train_stats['train_loss'], epoch)
    run['train/loss_scale'].log(train_stats['train_loss_scale'], epoch)
    run['train/weight_decay'].log(train_stats['train_weight_decay'], epoch)
    run['train/grad_norm'].log(train_stats['train_grad_norm'], epoch)
    run['train/min_lr'].log(train_stats['train_min_lr'], epoch)
    run['train/lr'].log(train_stats['train_lr'], epoch)

