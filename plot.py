from matplotlib import pyplot as plt
import numpy as np  
import csv

def load(fn):
    reader = csv.reader(open(fn))
    iters, acc, loss = [], [], []
    val_acc, val_loss = [], []

    names = next(reader)
    for row in reader:        
        loss.append(float(row[1]))
        val_loss.append(float(row[2]))
        
        acc.append(float(row[3]))
        val_acc.append(float(row[4]))

    return np.array(loss), np.array(val_loss), np.array(acc), np.array(val_acc)


def watch(fn, acfunc = lambda x: x, lsfunc = lambda x: x):
    loss, val_loss, acc, val_acc = load(fn)
    
    print("last acc/val_acc: %.4f %.4f" %( acc[-1], val_acc[-1]))
    print("mean acc/val_acc: %.4f %.4f" %( np.mean(acc[-10:]), np.mean(val_acc[-10:])))
    print("max acc/val_acc: %.4f %.4f" %( max(acc), max(val_acc)))

    plt.subplot(211)
    plt.plot(acfunc(acc), 'b', label = "acc")
    plt.plot(acfunc(val_acc), 'r', label = "val_acc")
    plt.ylim(0.7, 1.01)
    plt.legend()

    plt.subplot(212)
    plt.plot(lsfunc(loss[1:]), 'b', label = "loss")
    plt.plot(lsfunc(val_loss[1:]), 'r', label = "val_loss")
    plt.ylim(-0.01, 0.8)
    plt.legend()
    plt.show()


def compare(fp1, fp2, acfunc = lambda x: x, lsfunc = lambda x: x, 
    labels1 = ["acc1", "val_acc1", "loss1", "val_loss1"], labels2 = ["acc2", "val_acc2", "loss2", "val_loss2"]):
    loss1, val_loss1, acc1, val_acc1 = load(fp1)
    loss2, val_loss2, acc2, val_acc2 = load(fp2)

    print("weigt_decay vs no_weight_decay:")
    print("max val_acc: \t%.4f vs %.4f" %(val_acc1.max(), val_acc2.max()))
    print("mean val_acc: \t%.4f vs %.4f" %(val_acc1[-10:].mean(), val_acc2[-10:].mean()))
    print("last val_acc: \t%.4f vs %.4f" %(val_acc1[-1], val_acc2[-1]))
    
    plt.subplot(211)
    plt.plot(acfunc(acc1), 'lightpink', label = labels1[0]) 
    plt.plot(acfunc(val_acc1), 'r', label = labels1[1])
    plt.plot(acfunc(acc2), 'lightblue', label = labels2[0])
    plt.plot(acfunc(val_acc2), 'b', label = labels2[1])
    plt.ylabel("")
    plt.xlabel("iteration")
    plt.ylabel("log")
    plt.ylim(-3.5, -1)

    plt.legend()
    
    plt.subplot(212)
    plt.plot(lsfunc(loss1[1:]), 'lightpink', label = labels1[2])
    plt.plot(lsfunc(val_loss1[1:]), 'r', label = labels1[3])
    plt.plot(lsfunc(loss2[1:]), 'lightblue', label = labels2[2])
    plt.plot(lsfunc(val_loss2[1:]), 'b', label = labels2[3])
    plt.xlabel("iteration")
    plt.ylabel("log")
    # plt.ylim(-2, 0)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    # compare("bak/log_resnext.csv", "log.csv", acfunc = lambda x: np.log(1 - x), lsfunc = np.log,
        # labels1 = ["err", "val_err", "loss", "val_loss"], labels2 = ["err no_weight_decay", "val_err no_weight_decay", "loss no_weight_decay", "val_loss no_weight_decay"])
    # compare("bak/log_resnext.csv", "log.csv")
    watch("bak/log.csv")