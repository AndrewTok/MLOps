import dataset
import train
import matplotlib.pyplot as plt



def main():

    data = dataset.IrisData(test_size=0.2)
    trainer = train.TrainRunner(data)

    loss_history, acc_history = trainer.train(batch_size=64, epch_num=32)

    print("test accuracy: " + str(trainer.test_current_model()))

    plt.plot(loss_history)
    plt.show()

    pass


if __name__ == '__main__':

    main()