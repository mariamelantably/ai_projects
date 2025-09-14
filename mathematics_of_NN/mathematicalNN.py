from math import e

def sigmoid(x):
    return 1/(1+e**(-1*x))

w = [[0 for i in range(9)] for i in range(9)]
w[0][2], w[0][3], w[0][4] = 0.2, 0.4, 0.6
w[1][2], w[1][3], w[1][4] = 0.5, 0.2, 0.8
w[2][5], w[2][6] = 0.7, 0.8
w[3][5], w[3][6] = 0.9, 0.1
w[4][5], w[4][6] = 0.2, 0.5
w[5][7], w[5][8] = 0.3, 0.9
w[6][7], w[6][8] = 0.6, 0.7

m = [0 for i in range(9)]


n = 1 #learning rate
m[0] = 0.85
m[1] = 0.5
n7 = 0.75
n8 = 0.3
for x in range(15):

    for i in range(2, 9):
        s = 0
        for j in range(9):
            s += w[j][i]*m[j]
        m[i] = sigmoid(s)


    err = [0 for i in range(9)]
    MSE = 1/2*((n7 - m[7])**2 + (n8 - m[8])**2)
    #error of output layers
    err[7] = (m[7] - n7)*m[7]*(1-m[7])
    err[8] = (m[8] - n8)*m[8]*(1-m[8])
    err[5] = (w[5][7]*err[7] + w[5][8]*err[8])*m[5]*(1 - m[5])
    err[6] = (w[6][7]*err[7] + w[6][8]*err[8])*m[6]*(1 - m[6])
    err[2] = (w[2][5]*err[5] + w[2][6]*err[6])*m[2]*(1 - m[2])
    err[3] = (w[3][5]*err[5] + w[3][6]*err[6])*m[3]*(1 - m[3])
    err[4] = (w[4][5]*err[5] + w[4][6]*err[6])*m[4]*(1 - m[4])
    print("\hline")
    print(f"{x+1} & {m[7]} & {m[8]} & {MSE}")
    

    w[5][7] -= n*err[7]*m[5]
    w[6][7] -= n*err[7]*m[6]
    w[5][8] -= n*err[8]*m[5]
    w[6][8] -= n*err[8]*m[6]
    w[3][5] -= n*err[5]*m[3]
    w[4][5] -= n*err[5]*m[4]
    w[3][6] -= n*err[6]*m[3]
    w[4][6] -= n*err[6]*m[4]
    w[2][5] -= n*err[5]*m[2]
    w[2][6] -= n*err[6]*m[2]
    w[0][2] -= n*err[2]*m[0]
    w[0][3] -= n*err[3]*m[0]
    w[0][4] -= n*err[4]*m[0]
    w[1][2] -= n*err[2]*m[1]
    w[1][3] -= n*err[3]*m[1]
    w[1][4] -= n*err[4]*m[1]




    
