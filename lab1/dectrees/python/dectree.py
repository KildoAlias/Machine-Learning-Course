from dtree import *
import monkdata as m 
from drawtree_qt5 import drawTree as draw
import random
# import matplotlib.pyplot as plt 



def totallyPruned(bestperfomance,trainingset,validationset):
    bestTree = buildTree(trainingset,m.attributes)
    while True:
        pruned = allPruned(bestTree)
        currentperfomance = 0
        for i in range(len(pruned)):
            performance = check(pruned[i],validationset)
            if performance >= currentperfomance:
                currentTree = pruned[i]
                currentperfomance = performance
        if currentperfomance >= bestperfomance:
            bestperfomance = currentperfomance
            bestTree = currentTree
        else:
            return bestTree



def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]
    

def pruningTestError(trainingset,testset,itr,fraction):
    Performance = []
    spread = []
    for j in range(len(fraction)):
        tempspread = []
        tempPerformance = 0
        for i in range(itr):
            train,val = partition(trainingset,fraction[j])
            parentTree = buildTree(train,m.attributes)
            parentPerformance = check(parentTree,val)
            prunedTree = totallyPruned(parentPerformance,train,val)
            tempPerformance += check(prunedTree,testset)
            tempspread.append(check(prunedTree,testset))
        Performance.append(tempPerformance/itr)
        spread.append(max(tempspread)-min(tempspread))
    return Performance, spread
    

def main():

    # random.seed(100)
    datasets = [m.monk1,m.monk2,m.monk3]

    # Entropy 
    print('')
    s = 1
    for dataset in datasets:
        ent = entropy(dataset)
        print('The entropy of Monk-%s is %s' % (s,ent))
        print('')
        s += 1
    print('')


    # Information gain 
    s = 1
    for dataset in datasets:
        for a in range(0,6):
            gain = averageGain(dataset,m.attributes[a])
            print('The information gain for attribute %s in Monk-%s is %s' % (a+1,s,gain))
            print('')
        s += 1
        print('')


    # Best attribute 
    s = 1
    for dataset in datasets:
        best = bestAttribute(dataset,m.attributes)
        print('The best attribute for Monk-%s is %s' % (s,best))
        print('')
        s += 1
    print('')



    # Spilt datasets 
    splitsets = []
    for attr in range(1,5):
        splitsets.append(select(m.monk1,m.attributes[4],attr))


    n = 1
    for dataset in splitsets:
        best = bestAttribute(dataset,m.attributes)
        print('The best attribute for Monk-1, Node %s is %s' % (n,best))
        print('')
        n += 1

    # print(allPositive(splitsets[0]))

    common = []
    for dataset in splitsets: 
        common.append(mostCommon(dataset))
    print('The most common in each leafnode after a5 is: %s' % common)
    print('')


    ########################################################################

    # Building the trees
    Tree1 = buildTree(datasets[0],m.attributes)
    Tree2 = buildTree(datasets[1],m.attributes)
    Tree3 = buildTree(datasets[2],m.attributes)


    # Check performance on test data
    # print('Percentage of correctly classified in test data for MONK-1 is: %s' % check(Tree1, m.monk1test))
    # print('')
    # print('Percentage of correctly classified in test data for MONK-2 is: %s' % check(Tree2, m.monk2test))
    # print('')
    # print('Percentage of correctly classified in test data for MONK-3 is: %s' % check(Tree3, m.monk3test))
    # print('')
    # print('Percentage of correctly classified in training data for MONK-1 is: %s' % check(Tree1, m.monk1))
    # print('')
    # print('Percentage of correctly classified in training data for MONK-2 is: %s' % check(Tree2, m.monk2))
    # print('')
    # print('Percentage of correctly classified in training data for MONK-3 is: %s' % check(Tree3, m.monk3))


    # Error Test
    errtest1 = 1-check(Tree1, m.monk1test)
    errtest2 = 1-check(Tree2, m.monk2test)
    errtest3 = 1-check(Tree3, m.monk3test)

    # Error Train
    errtrain1 = 1-check(Tree1, m.monk1)
    errtrain2 = 1-check(Tree2, m.monk2)
    errtrain3 = 1-check(Tree3, m.monk3)
    print('')

    print('Error in training data set 1-3:')
    print(errtrain1)
    print(errtrain2)
    print(errtrain3)
    print('')

    print('Error in test data set 1-3:')
    print(errtest1)
    print(errtest2)
    print(errtest3)
    print('')


    # Split into training and validation sets 
    monk1train, monk1val = partition(m.monk1, 0.6)
    monk2train, monk2val = partition(m.monk2, 0.6)
    monk3train, monk3val = partition(m.monk3, 0.6)


    fraction = [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
    fractionVector1,spread1 = pruningTestError(m.monk1,m.monk1test,100,fraction)
    fractionVector3,spread3 = pruningTestError(m.monk3,m.monk3test,100,fraction)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(fraction,fractionVector1)
    # plt.title('Mean performance after pruning with differnt fractions')
    # plt.legend(['MONK-1'])
    # plt.xlabel('Fraction')
    # plt.ylabel('Performance')
    # plt.subplot(212)
    # plt.plot(fraction,fractionVector3)
    # plt.legend(['MONK-3'])
    # plt.xlabel('Fraction')
    # plt.ylabel('Performance')
    # plt.show()

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(fraction,spread1)
    # plt.title('Spread of performance with differnt fractions')
    # plt.legend(['MONK-1'])
    # plt.xlabel('Fraction')
    # plt.ylabel('Spread')
    # plt.subplot(212)
    # plt.plot(fraction,spread3)
    # plt.legend(['MONK-3'])
    # plt.xlabel('Fraction')
    # plt.ylabel('Spread')
    # plt.show()




    # orgtrad = buildTree(monk1train,m.attributes)

    # originalPerfomance = check(orgtrad, monk1val)



    # totprune = totallyPruned(originalPerfomance,monk1train,monk1val)


    # draw(totprune)
    # draw(orgtrad)



if __name__ == "__main__":
    main()













