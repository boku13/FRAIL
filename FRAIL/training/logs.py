import wandb
import tqdm as tqdm

#move the wandb logging here.
def log(*args, metrics):
    wandb.init(
        project="FRAIL",
        name = f"{args.model.capitalize().replace('rnn', 'RNN')} {args.dataset.capitalize()}",
        config = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        settings=wandb.Settings(start_method="thread")
    )
    config = wandb.config
    
    wandb.log({"Train Loss":metrics.loss.item(),"Train Accuracy":metrics.accuracy, 'Test Loss':metrics.test_losses, "Test Accuracy":metrics.test_accuracies})
