import numpy as np
from scipy.special import erf
from numba import jit

born_to_angstrom = 0.529177210903

# In atomic unit, the distance between H2 is
R = 0.7414
R = R/born_to_angstrom
# Configuration of H2 molecule(x axis)
# H1 = np.array([-R/2, 0.0, 0.0]) 
# H2 = np.array([R/2, 0.0, 0.0])


# Define the basis set
from parameter import STO_3G
STO_1s = STO_3G['1s']
STO_2s = STO_3G['2s']

num_basis = 4 # 2 for H1 and 2 for H2


alpha = [STO_1s['alpha'], STO_2s['alpha']]
coeff = [STO_1s['coeff'], STO_2s['coeff']]


# basis_set = []
# for i in range(2):
#     for j in range(2):
#         basis_set.append([alpha[i], coeff[i], [H1, H2][j]])

def Get_basis_set(H1,H2,alpha,coeff):
    basis_set = []
    for i in range(2):
        for j in range(2):
            basis_set.append([alpha[i], coeff[i], [H1, H2][j]])
    return basis_set

def Calculate_overlap_GTO(alpha1,coor1,alpha2,coor2):
    #First calculate the product of two GTO function
    pre_coeff = alpha1*alpha2/(alpha1+alpha2)
    Kab = (2*pre_coeff/np.pi)**(3/4)*np.exp(-pre_coeff*np.linalg.norm(coor1-coor2)**2)
    Rp = (alpha1*coor1+alpha2*coor2)/(alpha1+alpha2)

    # Then integrate exp(-\alpha (r-Rp)**2)
    alphap = alpha1+alpha2
    Inte = (2*np.pi/alphap)**(3/4)
    return Kab*Inte
def Calculate_overlap_STO(orbit1,orbit2):
    alpha1,coeff1,coor1 = orbit1
    alpha2,coeff2,coor2 = orbit2
    overlap = 0
    for i in range(len(coeff1)):
        for j in range(len(coeff2)):
            overlap += coeff1[i]*coeff2[j]*Calculate_overlap_GTO(alpha1[i],coor1,alpha2[j],coor2)
    return overlap

# Return S_matrix(overlap matrix)
def Calculate_S_matrix(basis_set):
    S = np.zeros((len(basis_set),len(basis_set)))
    for i in range(len(basis_set)):
        for j in range(len(basis_set)):
            S[i,j] = Calculate_overlap_STO(basis_set[i],basis_set[j])
    return S
# Kinetic part for GTO
def Kinetic_GTO(alpha1,coor1,alpha2,coor2):
    #First calculate the product of two GTO function    
    Rp = (alpha1*coor1+alpha2*coor2)/(alpha1+alpha2)
    distance = np.linalg.norm(Rp-coor2)

    part2 = Calculate_overlap_GTO(alpha1,coor1,alpha2,coor2)*(3*alpha1*alpha2/(alpha1+alpha2)-2*alpha2**2*distance**2)
    return part2



# External potential for two orbits
def V_ext(alpha1,coor1,alpha2,coor2,Nuc):
    gamma = alpha1 + alpha2
    K = np.exp(-alpha1 * alpha2 * np.linalg.norm(coor1 - coor2) ** 2 / gamma)
    Rp = (alpha1 * coor1 + alpha2 * coor2) / gamma

    distance = np.linalg.norm(Rp-Nuc)

    if distance ==0:
        value = 2*K*np.pi/gamma
    else:
        prefactor = K*(np.pi / gamma) ** (3 / 2)/distance
        value = prefactor*erf(np.sqrt(gamma)*distance)
    return -value*(4*alpha1*alpha2/(np.pi**2))**(3/4)

def H_STO(orbit1,orbit2,H1,H2,Z1=1,Z2=1):
    alpha1,coeff1,coor1 = orbit1
    alpha2,coeff2,coor2 = orbit2
    overlap = 0
    for i in range(len(coeff1)):
        for j in range(len(coeff2)):
            overlap += coeff1[i]*coeff2[j]*(Kinetic_GTO(alpha1[i],coor1,alpha2[j],coor2)+Z1*V_ext(alpha1[i],coor1,alpha2[j],coor2,H1)+Z2*V_ext(alpha1[i],coor1,alpha2[j],coor2,H2))
    return overlap

# 1body part (h\mu v)
def Calculate_H_matrix(basis_set,H1,H2,Z1=1,Z2=1):
    H = np.zeros((len(basis_set),len(basis_set)))
    for i in range(len(basis_set)):
        for j in range(len(basis_set)):
            H[i,j] = H_STO(basis_set[i],basis_set[j],H1,H2,Z1,Z2)
    return H

# Boys function F0(x)
def boys_function(x):
    if x < 1e-8:
        return 1.0
    else:
        return (0.5 * np.sqrt(np.pi / x) * erf(np.sqrt(x)))


def TE_value(alpha1,A,alpha3,C,alpha2,B,alpha4,D):
    p = alpha1 + alpha2
    q = alpha3 + alpha4
    P = (alpha1 * A + alpha2 * B) / p
    Q = (alpha3 * C + alpha4 * D) / q
    RPQ = np.linalg.norm(P - Q)
    
    boys_arg = (p * q * RPQ**2) / (p + q)
    boys_val = boys_function(boys_arg)
    
    prefactor = 2 * (np.pi ** 2.5) / (p * q * np.sqrt(p + q))*(16*alpha1*alpha2*alpha3*alpha4/(np.pi**4))**(3/4)
    exp_factor = np.exp(- (alpha1 * alpha2 * np.linalg.norm(A - B)**2) / p)
    exp_factor *= np.exp(- (alpha3 * alpha4 * np.linalg.norm(C - D)**2) / q)
    
    return prefactor * exp_factor * boys_val

def TE_tensor_value(orbit1,orbit2,orbit3,orbit4):
    alpha1,coeff1,coor1 = orbit1
    alpha2,coeff2,coor2 = orbit2
    alpha3,coeff3,coor3 = orbit3
    alpha4,coeff4,coor4 = orbit4
    integral = 0

    for i in range(len(coeff1)):
        for j in range(len(coeff2)):
            for k in range(len(coeff3)):
                for l in range(len(coeff4)):
                    

                    integral += coeff1[i]*coeff2[j]*coeff3[k]*coeff4[l]*TE_value(alpha1[i],coor1,alpha2[j],coor2,alpha3[k],coor3,alpha4[l],coor4)
    return integral

def TE_tensor(basis_set):
    datalen = len(basis_set)
    tensor = np.zeros((datalen,datalen,datalen,datalen))
    for i in range(datalen):
        for j in range(datalen):
            for k in range(datalen):
                for l in range(datalen):
                    tensor[i,j,k,l] = TE_tensor_value(basis_set[i],basis_set[j],basis_set[k],basis_set[l])
    return tensor


# X^{\dagger} S X = I, examined
def Calculate_X_matrix(S):
    eigval,eigvec = np.linalg.eig(S)
    X = np.zeros((len(S),len(S)))
    for i in range(len(S)):
        X[:,i] = eigvec[:,i]/np.sqrt(eigval[i])
    return X

def Calculate_F_element(i,j,P,tensor):
    datalen = P.shape[0]
    ele = 0
    for k in range(datalen):
        for l in range(datalen):
            ele += P[k,l]*(tensor[i,l,j,k]-0.5*tensor[i,l,k,j])
    return ele

# 结构优化会停在1.39附近
def Stucture_optimization(h,R_init,delta=0.05,Z1=1,Z2=1):
    index = 0
    R_history = []
    while True:
        basis_set = []
        H1 = np.array([-(R_init+h)/2, 0.0, 0.0])
        H2 = np.array([(R_init+h)/2, 0.0, 0.0])
        basis_set = []
        for i in range(2):
            for j in range(2):
                basis_set.append([alpha[i], coeff[i], [H1, H2][j]])

        E_ph,_,_ = Iteration(basis_set,H1,H2)
        E_ph += 1/(R_init+h)
        H1 = np.array([-(R_init-h)/2, 0.0, 0.0])
        H2 = np.array([(R_init-h)/2, 0.0, 0.0])
        basis_set = []
        for i in range(2):
            for j in range(2):
                basis_set.append([alpha[i], coeff[i], [H1, H2][j]])
        E_mh,_,_ = Iteration(basis_set,H1,H2)
        E_mh += 1/(R_init-h)
        gradient = (E_ph-E_mh)/(2*h)
        R_new = R_init-delta*gradient
        if np.abs(R_new-R_init)<1e-6 or index > 1000:
            print("Optimization finished! The optimized distance is:",R_new)
            break
        else:
            index += 1
            print("Optimization step:",index,"Current distance:",R_new,'Current energy:',E_ph)
            R_init = R_new
            R_history.append(R_new)
    return R_history


def Mulliken_population(basis_set,P_matrix,Nuc1,Nuc2,Z1,Z2):
    S_matrix = Calculate_S_matrix(basis_set)
    info_dict = {"Nuc1":{"Coordinate":Nuc1}, "Nuc2":{"Coordinate":Nuc2}}
    index_dict = {"Nuc1":[],"Nuc2":[],"Others":[]}
    for i in range(len(basis_set)):
        coordiante = basis_set[i][2]
        if coordiante[0] == Nuc1[0]:
            index_dict["Nuc1"].append(i)
        elif coordiante[0] == Nuc2[0]:
            index_dict["Nuc2"].append(i)
        else:
            index_dict["Others"].append(i)
    for key in index_dict.keys():
        if key == "Others":
            continue
        else:
            index_list = index_dict[key]
            ni = 0
            for index in index_list:
                for j in range(len(basis_set)):
                    ni += P_matrix[index,j]*S_matrix[j,index]
            info_dict[key]["Population"] = ni
            info_dict[key]["Charge"] = Z1-ni if key == "Nuc1" else Z2-ni
    bond = 0
    PS = P_matrix@S_matrix
    for i in index_dict["Nuc1"]:
        for j in index_dict["Nuc2"]:
            bond += PS[i,j]*PS[j,i]
    info_dict["Bond"] = bond
    return info_dict    


def Density(input_coor,basis_set,Coeff_matrix,index=0):
    x,y,z = input_coor
    density = 0
    for i in range(len(basis_set)):
        alpha,coeff,coor = basis_set[i]
        for j in range(len(coeff)):
            density += coeff[j]*Coeff_matrix[i,index]*np.exp(-alpha[j]*((x-coor[0])**2+(y-coor[1])**2+(z-coor[2])**2))
    return density**2

def Wavefunction(input_coor,basis_set,Coeff_matrix,index=0):
    x,y,z = input_coor
    wavefunction = 0
    for i in range(len(basis_set)):
        alpha,coeff,coor = basis_set[i]
        for j in range(len(coeff)):
            wavefunction += coeff[j]*Coeff_matrix[i,index]*np.exp(-alpha[j]*((x-coor[0])**2+(y-coor[1])**2+(z-coor[2])**2))
    return wavefunction


def Iteration(basis_set,H1,H2,Z1=1,Z2=1,max=1000):
    # Initial guess for P matrix 
    # print("Begin calculation...")
    S_matrix = Calculate_S_matrix(basis_set)
    X_matrix = Calculate_X_matrix(S_matrix)
    H_matrix = Calculate_H_matrix(basis_set,H1,H2,Z1,Z2)
    tensor = TE_tensor(basis_set)
    # print("Finish calculating the integrals...")
    P_initial = np.zeros((len(basis_set),len(basis_set)))
    P_matrix = P_initial
    # print("Begin SCF iteration...")
    index = 0
    while True and index < max:
        
        F_matrix  = H_matrix.copy()
        # F_{ij} = H_{ij}+\sum_{\eta \lambda}P_{\eta \lambda}((i\lambda|j\eta)-(i\lambda|\eta j))
        for i in range(len(basis_set)):
            for j in range(len(basis_set)):
                num = Calculate_F_element(i,j,P_matrix,tensor)
                F_matrix[i,j] += num
        F_prime = X_matrix.T@F_matrix@X_matrix
        eigval,eigvec = np.linalg.eigh(F_prime)
        E_matrix = np.diag(eigval)

        Coeff_matrix_prime = eigvec
        Coeff_matrix = X_matrix@Coeff_matrix_prime


        # Get P matrix
        P_matrix_new = np.zeros((len(basis_set),len(basis_set)))
        for i in range(len(basis_set)):
            for j in range(len(basis_set)):
                P_matrix_new[i,j] += Coeff_matrix[i,0]*Coeff_matrix[j,0]*2
        
        if np.linalg.norm(P_matrix_new-P_matrix)<1e-8:
            break
        else:
            P_matrix = P_matrix_new
            Energy = np.trace(P_matrix@(H_matrix+F_matrix))*0.5
            # print('Iteration:',index,'Energy:',Energy)
            index+=1
    return Energy,Coeff_matrix,P_matrix
def Decompositon_Curve(start,end,points=600):
    R = np.linspace(start,end,points)
    E = []
    for r in R:
        H1 = np.array([-r/2, 0.0, 0.0])
        H2 = np.array([r/2, 0.0, 0.0])
        basis_set = []
        for i in range(2):
            for j in range(2):
                basis_set.append([alpha[i], coeff[i], [H1, H2][j]])
        E_ph,_,_ = Iteration(basis_set,H1,H2)
        E_ph += 1/r
        E.append(E_ph)
    return R,E

def Decompositon_Curve_CID(start,end,points=600):
    R = np.linspace(start,end,points)
    E = []
    for r in R:
        H1 = np.array([-r/2, 0.0, 0.0])
        H2 = np.array([r/2, 0.0, 0.0])
        basis_set = []
        for i in range(2):
            for j in range(2):
                basis_set.append([alpha[i], coeff[i], [H1, H2][j]])
        E_cid = CID2(basis_set,H1,H2)
        E_cid += 1/r
        E.append(E_cid)
    return R,E


def Decompositon_Curve_MP2(start,end,points=600):
    R = np.linspace(start,end,points)
    E = []
    for r in R:
        H1 = np.array([-r/2, 0.0, 0.0])
        H2 = np.array([r/2, 0.0, 0.0])
        basis_set = []
        for i in range(2):
            for j in range(2):
                basis_set.append([alpha[i], coeff[i], [H1, H2][j]])
        E_mp2,_,_ = MP2(basis_set,H1,H2)
        E_mp2 += 1/r
        E.append(E_mp2)
    return R,E



def CID_H_ij(Coeff,tensor,H_HF,i,j):
    num = 0
    size = Coeff.shape[0]
    for m in range(size):
        for v in range(size):
            if i==j:
                num += 2*Coeff[m,i]*Coeff[v,i]*H_HF[m,v] 
            for eta in range(size):
                for lamda in range(size):
                    num += Coeff[m,i]*Coeff[v,i]*Coeff[eta,j]*Coeff[lamda,j]*tensor[m,v,eta,lamda]

    return num

def CID_S_ij(Coeff,S_basis,i,j):
    num = 0
    size = Coeff.shape[0]
    for m in range(size):
        for v in range(size):
            num += Coeff[m,i]*Coeff[v,j]*S_basis[m,v]
    return num**2



# post-HF method
def CID(basis_set,H1,H2,Z1=1,Z2=1,max=1000):
    # First get HF output
    E_HF,Coeff,_ = Iteration(basis_set,H1,H2,Z1,Z2,max)
    S_basis = Calculate_S_matrix(basis_set)
    tensor = TE_tensor(basis_set)
    H_HF = Calculate_H_matrix(basis_set,H1,H2,Z1,Z2)

    H_matrix = Coeff.copy()
    S_matrix = Coeff.copy()
    size = Coeff.shape[0]
    # Only consider double excitation here.
    for i in range(size):
        for j in range(size):
            H_matrix[i,j] = CID_H_ij(Coeff,tensor,H_HF,i,j)
            S_matrix[i,j] = CID_S_ij(Coeff,S_basis,i,j)
    X_matrix = Calculate_X_matrix(S_matrix)
    H_prime = X_matrix.T@H_matrix@X_matrix
    eigval,eigvec = np.linalg.eigh(H_prime)

    Coeff_matrix_prime = eigvec
    Coeff_matrix = X_matrix@Coeff_matrix_prime
    print("The ground energy after CID is:",eigval[0]," Original HF is:",E_HF," Correlation energy:",eigval[0]-E_HF)

def CID2(basis_set,H1,H2,Z1=1,Z2=1,max=1000):
    # First get HF output
    E_HF,Coeff,_ = Iteration(basis_set,H1,H2,Z1,Z2,max)
    S_basis = Calculate_S_matrix(basis_set)
    tensor = TE_tensor(basis_set)
    # Hatree-Fock H matrix(one body term)
    H_HF = Calculate_H_matrix(basis_set,H1,H2,Z1,Z2)
    H_matrix = np.zeros((2,2))
    S_matrix = H_matrix.copy()
    size = 2
    # Only consider double excitation here.
    for i in range(size):
        for j in range(size):
            H_matrix[i,j] = CID_H_ij(Coeff,tensor,H_HF,i,j)
            S_matrix[i,j] = CID_S_ij(Coeff,S_basis,i,j)
    X_matrix = Calculate_X_matrix(S_matrix)
    H_prime = X_matrix.T@H_matrix@X_matrix
    eigval,eigvec = np.linalg.eigh(H_prime)

    Coeff_matrix_prime = eigvec
    Coeff_matrix = X_matrix@Coeff_matrix_prime
    # print("The ground energy after CID is:",eigval[0]," Original HF is:",E_HF," Correlation energy:",eigval[0]-E_HF)
    return eigval[0]
    
def MP2(basis_set,H1,H2,Z1=1,Z2=1,max=1000):
    S_matrix = Calculate_S_matrix(basis_set)
    X_matrix = Calculate_X_matrix(S_matrix)
    H_matrix = Calculate_H_matrix(basis_set,H1,H2,Z1,Z2)
    tensor = TE_tensor(basis_set)
    # print("Finish calculating the integrals...")
    P_initial = np.zeros((len(basis_set),len(basis_set)))
    P_matrix = P_initial
    # print("Begin SCF iteration...")
    index = 0
    while True and index < max:
        
        F_matrix  = H_matrix.copy()
        # F_{ij} = H_{ij}+\sum_{\eta \lambda}P_{\eta \lambda}((i\lambda|j\eta)-(i\lambda|\eta j))
        for i in range(len(basis_set)):
            for j in range(len(basis_set)):
                num = Calculate_F_element(i,j,P_matrix,tensor)
                F_matrix[i,j] += num
        F_prime = X_matrix.T@F_matrix@X_matrix
        eigval,eigvec = np.linalg.eigh(F_prime)
        E_matrix = np.diag(eigval)

        Coeff_matrix_prime = eigvec
        Coeff_matrix = X_matrix@Coeff_matrix_prime


        # Get P matrix
        P_matrix_new = np.zeros((len(basis_set),len(basis_set)))
        for i in range(len(basis_set)):
            for j in range(len(basis_set)):
                P_matrix_new[i,j] += Coeff_matrix[i,0]*Coeff_matrix[j,0]*2
        
        if np.linalg.norm(P_matrix_new-P_matrix)<1e-8:
            break
        else:
            P_matrix = P_matrix_new
            Energy = np.trace(P_matrix@(H_matrix+F_matrix))*0.5
            # print('Iteration:',index,'Energy:',Energy)
            index+=1
    K12 = 0
    size = Coeff_matrix.shape[0]
    for mu in range(size):
        for v in range(size):
            for lamda in range(size):
                for eta in range(size):
                    K12 += Coeff_matrix[mu,0]*Coeff_matrix[v,0]*Coeff_matrix[lamda,1]*Coeff_matrix[eta,1]*tensor[mu,v,lamda,eta]
    Energy_corr = K12**2/((eigval[0]-eigval[1])*2)
    return Energy_corr+Energy,eigval,K12



# E,Coeff_matrix,P_matrix = Iteration(basis_set)


# R_New = Stucture_optimization(0.01,1.3909,5)

if __name__ == '__main__':
        
    R,E = Decompositon_Curve()
    np.savetxt('Energy_curve.txt',np.array([R,E]).T)
    import matplotlib.pyplot as plt
    plt.plot(R,E)
    plt.xlabel('Distance(au)')
    plt.ylabel('Energy(Hartree)')
    plt.title('Energy curve of H2 molecule')
    plt.savefig('Energy_curve.png')