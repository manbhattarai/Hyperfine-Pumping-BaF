import numpy as np
from sympy.physics.wigner import wigner_6j,wigner_9j,wigner_3j,clebsch_gordan
from states import SigmaLevel,PiLevelParity,PiLevelOmega,Superposition
from spin_params import *

#conditional import
from molecular_parameters import *


def kdel(x,y):
    if x == y:
        return 1
    else:
        return 0

def reduced(x):
    return np.sqrt(x*(x+1)*(2*x+1))

def nreduced(x,y):
    return np.sqrt((2*x+1)*(2*y+1))


#### Dipole matrix element between Sigma and Pi states #######################

def H_int(state1,state2,pol=0):
    """state1 is either SigmaLevel or a superposition of SigmaLevel's and 
    state2 is either PiLevelParity or a superposition of PiLevelParity's"""
    
    if type(state1) == SigmaLevel:
        state1 = Superposition([1],[state1])
    if type(state2) == PiLevelParity:
        state2 = Superposition([1],[state2])
    val = 0
    
    for i in range(len(state1.amplitude)):
        for j in range(len(state2.amplitude)):
            #convert parity basis to omega basis
            ket2 = state2.states[j].parity_to_omega()
            for jj in range(len(ket2.amplitude)):
                val += H_int_omega(state1.states[i],ket2.states[jj],pol)* \
                       ket2.amplitude[jj]*state2.amplitude[j]* \
                       np.conj(state1.amplitude[i])
    
    return val    
    
def H_int_omega(state1:SigmaLevel, state2:PiLevelOmega, pol=0):    
    
    G,N,F1,F,mF=state1.G,state1.N,state1.F1,state1.F,state1.mF
    Lambda,Sigma,Omega,Jex,F1p,Fp,mFp = state2.Lambda, \
                                        state2.Sigma, \
                                        state2.Omega, \
                                        state2.parity_state.J, \
                                        state2.parity_state.F1, \
                                        state2.parity_state.F, \
                                        state2.parity_state.mF
    val=0
    for J in np.arange(np.abs(N-S),N+S+1,1):
        for sigma in [-1.0/2,1.0/2]:
            omega = sigma
            for q in np.arange(-1,1+1):
                val += (nreduced(J,G)*(-1)**(G+S+I1)*
                        (-1)**(N-S+omega)*np.sqrt(2*N+1)*wigner_3j(J,S,N,omega,-sigma,0)*
                        wigner_6j(F1,G,N,S,J,I1)*
                        (-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,pol,mFp)*
                        (-1)**(Fp+I2+F1+1)*nreduced(F,Fp)*
                        wigner_6j(F1p,Fp,I2,F,F1,1)*
                        (-1)**(F1p+I1+J+1)*nreduced(F1,F1p)*wigner_6j(Jex,F1p,I1,F1,J,1)*
                        (-1)**(J-omega)*nreduced(J,Jex)*
                        (kdel(sigma,Sigma)*wigner_3j(J,1,Jex,-omega,q,Omega)
                        )
                       )
    return val
    #
    #(-1)**(J+omega)*np.sqrt(2*N+1)*wigner_3j(S,N,J,sigma,0,-omega)* #John Barry

####################################################################################################
    
        
def H0_sigma(state1: SigmaLevel,state2: SigmaLevel):
    state = (
            state1.G,state1.N,state1.F1,state1.F,state1.mF,
            state2.G,state2.N,state2.F1,state2.F,state2.mF
            )
    return  HN(state)+       \
            HNS(state)+     \
            HFBa(state)+    \
            HFF(state)+     \
            HCBa(state)+    \
            HCF(state)+     \
            HQ(state)+      \
            HNI(state) #considered for SrF

def HZeeman_sigma(state1: SigmaLevel,state2: SigmaLevel):
    state = (
            state1.G,state1.N,state1.F1,state1.F,state1.mF,
            state2.G,state2.N,state2.F1,state2.F,state2.mF
            )
    return HgrZS(state)+ \
            HgrZI1(state)+ \
            HgrZI2(state)+ \
            HgrZN(state)+ \
            HgrZgl(state)
     
def H0_pi_parity_basis(state1:PiLevelParity, state2:PiLevelParity):
   
    val = 0    

    #convert parity basis to omega basis
    ket1 = state1.parity_to_omega() #returns a Superposition in omega basis
    ket2 = state2.parity_to_omega() #returns a Superposition in omega basis
    for i in range(len(ket1.amplitude)):
        for j in range(len(ket2.amplitude)):
            val += H0_pi_omega_basis(ket1.states[i],ket2.states[j])* \
                   np.conj(ket1.amplitude[i])*ket2.amplitude[j] ###conjugate added here
                    
    return val                       


                                
def H0_pi_omega_basis(state1: PiLevelOmega,state2: PiLevelOmega):
    state = (
            state1.Lambda,state1.Sigma,state1.Omega,
            state1.parity_state.J,
            state1.parity_state.F1,
            state1.parity_state.F,
            state1.parity_state.mF,
            state2.Lambda,state2.Sigma,state2.Omega,
            state2.parity_state.J,
            state2.parity_state.F1,
            state2.parity_state.F,
            state2.parity_state.mF
            )
    
    return  HF_h(state)+        \
            HF_d(state)+        \
            HexLS(state)+       \
            Hexpq(state)+       \
            HexR_cor(state)+    \
            HBa_h(state)+              \
            HBa_d(state)+              \
            HexQ(state)

                                
def HZeeman_pi_parity_basis(state1,state2):
    val = 0    

    #convert parity basis to omega basis
    ket1 = state1.parity_to_omega() #returns a Superposition in omega basis
    ket2 = state2.parity_to_omega() #returns a Superposition in omega basis
    for i in range(len(ket1.amplitude)):
        for j in range(len(ket2.amplitude)):
            val += HZeeman_pi_omega_basis(ket1.states[i],ket2.states[j])* \
                   np.conj(ket1.amplitude[i])*ket2.amplitude[j] ###conjugate added here
                    
    return val                                

def HZeeman_pi_omega_basis(state1: PiLevelOmega,state2: PiLevelOmega):
    state = (
            state1.Lambda,state1.Sigma,state1.Omega,
            state1.parity_state.J,
            state1.parity_state.F1,
            state1.parity_state.F,
            state1.parity_state.mF,
            state2.Lambda,state2.Sigma,state2.Omega,
            state2.parity_state.J,
            state2.parity_state.F1,
            state2.parity_state.F,
            state2.parity_state.mF
            )
    
    return  HZL(state)+ \
            HexZS(state)+ \
            HZglp(state)+ \
            HexZI2(state)+ \
            HexZI1(state)                               
                                

############################################# Bare Hamiltonian Sigma ################################################## ########################################################################################################################    
def HN(state):
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*kdel(N,Np)*kdel(G,Gp)*
            (BN*N*(N+1)-DN*N**2*(N+1)**2)
            )

def HNS(state):
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return ((gamma+delta_gamma*N*(N+1))*kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*kdel(N,Np)*
            reduced(N)*reduced(S)*nreduced(G,Gp)*
            (-1)**(F1+I1+N+S+1+Gp+Gp)*
            wigner_6j(Np,Gp,F1,G,N,1)*wigner_6j(S,Gp,I1,G,S,1))

def HFBa(state):
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return bFBa/2*kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*kdel(N,Np)*kdel(G,Gp)* \
           (G*(G+1)-S*(S+1)-I1*(I1+1))

def HCBa(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    val=0.0
    for J in np.arange(np.abs(N-S),N+S+1):
        for Jp in np.arange(np.abs(Np-S),Np+S+1):
            val += ((-1)**(G+Gp+S+S+I1+I1+Jp+F1+I1+N)*
                    nreduced(J,Jp)**2*
                    wigner_6j(F1,G,N,S,J,I1)*wigner_6j(F1p,Gp,Np,S,Jp,I1)*
                    wigner_6j(I1,Jp,F1,J,I1,1)*wigner_9j(J,Jp,1,N,Np,2,S,S,1))
    return (-cBa*np.sqrt(30)/3*
            kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*
            reduced(I1)*reduced(S)*nreduced(G,Gp)*nreduced(N,Np)*
            wigner_3j(N,2,Np,0,0,0)*val)

def HQ(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp) = state
    val = 0
    if wigner_3j(I1,2,I1,-I1,0,I1) == 0:
        return 0
    else:
        for J in np.arange(np.abs(N-S),N+S+1):
            for Jp in np.arange(np.abs(Np-S),Np+S+1):
                val += ((-1)**(G+Gp+S+I1+S+I1+Jp+F1+I1+S+Jp)*nreduced(J,Jp)**2*
                        wigner_6j(F1,G,N,S,J,I1)*wigner_6j(F1p,Gp,Np,S,Jp,I1)*
                        wigner_6j(I1,Jp,F1,J,I1,2)*wigner_6j(Np,Jp,S,J,N,2))
        return (eq0Q/4*kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*nreduced(G,Gp)*nreduced(N,Np)*
                wigner_3j(N,2,Np,0,0,0)*1/(wigner_3j(I1,2,I1,-I1,0,I1))*val)
    

def HFF(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (bFF*kdel(mF,mFp)*kdel(F,Fp)*kdel(N,Np)*
            (-1)**(F+I2+N+G+Gp+I1+S+F1p+F1p)*
            nreduced(G,Gp)*nreduced(F1,F1p)*reduced(I2)*reduced(S)*
            wigner_6j(I2,F1p,F,F1,I2,1)*wigner_6j(Gp,F1p,N,F1,G,1)*
            wigner_6j(S,Gp,I1,G,S,1))

def HCF(state): #updated
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    val=0
    for J in np.arange(np.abs(N-S),N+S+1):
        for Jp in np.arange(np.abs(Np-S),Np+S+1):
            val += (
                    (-1)**(G+Gp+S+S+I1+I1+F1p+F1p+F+I2+I1+1+N+J)*
                    nreduced(J,Jp)**2*
                    wigner_6j(F1,G,N,S,J,I1)*wigner_6j(F1p,Gp,Np,S,Jp,I1)*
                    wigner_6j(I2,F1p,F,F1,I2,1)*wigner_6j(Jp,F1p,I1,F1,J,1)*
                    wigner_9j(J,Jp,1,N,Np,2,S,S,1)
                    )
    return (-cF*np.sqrt(30)/3*kdel(mF,mFp)*kdel(F,Fp)*reduced(I2)*
            reduced(S)*nreduced(G,Gp)*
            nreduced(N,Np)*nreduced(F1,F1p)*
            wigner_3j(N,2,Np,0,0,0)*val)

#Considered for SrF
def HNI(state):
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (cI*kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*kdel(N,Np)*reduced(N)*reduced(I1)*nreduced(G,Gp)*(-1)**(G+Gp+F1+N+S+I1+1)
            *wigner_6j(Np,Gp,F1,G,N,1)*wigner_6j(I1,Gp,S,G,I1,1))

####################################### Zeeman Hamiltonian sigma state ##########################################################
def HgrZS(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (gS*uB*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1+1)*nreduced(F,Fp)*nreduced(F1,F1p)*nreduced(G,Gp)*
            (-1)**(F1p+N+G+1)*kdel(N,Np)*wigner_6j(F1p,Fp,I2,F,F1,1)*
            wigner_6j(Gp,F1p,N,F1,G,1)*
            (-1)**(Gp+I1+S+1)*
            wigner_6j(S,Gp,I1,G,S,1)*reduced(S))

def HgrZI1(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (-gI1*uN*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1+1)*nreduced(F,Fp)*nreduced(F1,F1p)*nreduced(G,Gp)*
            (-1)**(F1p+N+G+1)*kdel(N,Np)*wigner_6j(F1p,Fp,I2,F,F1,1)*wigner_6j(Gp,F1p,N,F1,G,1)*
            (-1)**(G+I1+S+1)*
            wigner_6j(I1,Gp,S,G,I1,1)*reduced(I1))

def HgrZI2(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (-gI2*uN*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(F+I2+F1+1)*nreduced(F,Fp)*kdel(F1,F1p)*kdel(N,Np)*kdel(G,Gp)
            *wigner_6j(I2,Fp,F1,F,I2,1)*reduced(I2))

def HgrZN(state): #checked
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    return (-grot*uN*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1+1)*nreduced(F,Fp)*nreduced(F1,F1p)
            *wigner_6j(F1p,Fp,I2,F,F1,1)*kdel(G,Gp)*
            (-1)**(F1+Np+G+1)*wigner_6j(Np,F1p,G,F1,N,1)*reduced(N)*kdel(N,Np))

def HgrZgl(state): #updated
    (G,N,F1,F,mF,Gp,Np,F1p,Fp,mFp)=state
    val=0
    for J in np.arange(np.abs(N-S),N+S+1):
        for Jp in np.arange(np.abs(Np-S),Np+S+1):
            for sigma in np.arange(-S,S+1):
                for sigmap in np.arange(-S,S+1):
                    for q in [-1,1]:
                        val += ((-1)**(G+Gp+S+I1+S+I1)*
                                wigner_6j(F1,G,N,S,J,I1)*wigner_6j(F1p,Gp,Np,S,Jp,I1)*
                                nreduced(J,Jp)**2*nreduced(G,Gp)*
                                (-1)**(N-S+Np-S+sigma+sigmap)*nreduced(N,Np)*
                                wigner_3j(J,S,N,sigma,-sigma,0)*wigner_3j(Jp,S,Np,sigmap,-sigmap,0)*
                                (-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
                                 (-1)**(Fp+I2+F1+1)*nreduced(F,Fp)*wigner_6j(F1p,Fp,I2,F,F1,1)*
                                 (-1)**(F1p+J+I1+1)*nreduced(F1,F1p)*wigner_6j(Jp,F1p,I1,F1,J,1)*
                                 (-1)**(J+S-2*sigma)*wigner_3j(J,1,Jp,-sigma,q,sigmap)*
                                 wigner_3j(S,1,S,-sigma,q,sigmap)*reduced(S))
    return val*gl*uB

################################################################################################################################
#################################################################################################################################


################################################Bare Hamiltonian Pi State#######################################################
#################################################################################################################################

h_Ba  = 1*183.54#205.59
d_Ba  = 1*230.13#256.98
eq0Q1 = 1*61.65#-100.67

def HBa_h(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val = kdel(Lambda,Lambdap)*kdel(Omega,Omegap)*kdel(Sigma,Sigmap)*1/2*h_Ba
    return (kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*1/2*(F1*(F1+1)-I1*(I1+1)-J*(J+1))*
            val/(J*(J+1))
            )

def HBa_d(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val = (-kdel(Lambda,Lambdap+2)*kdel(Omega,Omegap+1)*kdel(Sigma,Sigmap-1)*
           np.sqrt(S*(S+1)-Sigmap*(Sigmap-1))*np.sqrt(J*(J+1)-Omegap*(Omegap+1))-
           kdel(Lambda,Lambdap-2)*kdel(Omega,Omegap-1)*kdel(Sigma,Sigmap+1)*
           np.sqrt(S*(S+1)-Sigmap*(Sigmap+1))*np.sqrt(J*(J+1)-Omegap*(Omegap-1))
            )
    return (1/2*d_Ba*kdel(mF,mFp)*kdel(F,Fp)*kdel(F1,F1p)*1/2*(F1*(F1+1)-I1*(I1+1)-J*(J+1))*
            val/(J*(J+1))
            )

def HexQ(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    if J == 1/2:
        return 0
    else:
        return (eq0Q1/4*kdel(F,Fp)*kdel(mF,mFp)*kdel(F1,F1p)*kdel(Sigma,Sigmap)*(-1)**(J+F1+I1+J-Omega)*nreduced(J,J)*
                wigner_6j(I1,J,F1,J,I1,2)*1/wigner_3j(I1,2,I1,-I1,0,I1)*wigner_3j(J,2,J,-Omega,0,Omegap)
               )

############### $$$$$$$$$$$$$$$$ Alternative expression for the exxcited state hyperfines

h_F = (a_ex-1/2*(b_ex+c_ex))*1
d2 = 3.58*1

def HF_h(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    #val = (kdel(Lambda,Lambdap)*kdel(Omega,Omegap)*kdel(Sigma,Sigmap)*Omegap*
    #        (a_ex*Lambdap+Sigmap*(b_ex+c_ex))
    #       )
    val = kdel(Lambda,Lambdap)*kdel(Omega,Omegap)*kdel(Sigma,Sigmap)*1/2*h_F
    return (kdel(mF,mFp)*kdel(F,Fp)*(-1)**(2*F1p+F+I2+I1+J+1)*nreduced(F1,F1p)*reduced(J)*reduced(I2)*
            wigner_6j(I2,F1p,F,F1,I2,1)*wigner_6j(J,F1p,I1,F1,J,1)*
            val/(J*(J+1))
            )

def HF_d(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val = (-kdel(Lambda,Lambdap+2)*kdel(Omega,Omegap+1)*kdel(Sigma,Sigmap-1)*
           np.sqrt(S*(S+1)-Sigmap*(Sigmap-1))*np.sqrt(J*(J+1)-Omegap*(Omegap+1))-
           kdel(Lambda,Lambdap-2)*kdel(Omega,Omegap-1)*kdel(Sigma,Sigmap+1)*
           np.sqrt(S*(S+1)-Sigmap*(Sigmap+1))*np.sqrt(J*(J+1)-Omegap*(Omegap-1))
            )
    return (1/2*d2*kdel(mF,mFp)*kdel(F,Fp)*(-1)**(2*F1p+F+I2+I1+J+1)*nreduced(F1,F1p)*
            reduced(J)*reduced(I2)*
            wigner_6j(I2,F1p,F,F1,I2,1)*wigner_6j(J,F1p,I1,F1,J,1)*
            val/(J*(J+1))
            )
############################################################################################################################


def HexLS(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    return A*Lambda*Sigma*kdel(Lambda,Lambdap)*kdel(Sigma,Sigmap)*kdel(F1,F1p)*kdel(F,Fp)*kdel(mF,mFp)

def Hexpq(state):   #checked from Brown and carrington Page 618
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val=0
    for q in [-1,1]:
        val += (kdel(Lambda,-2*q+Lambdap)*wigner_3j(S,1,S,-Sigma,q,Sigmap)
            *wigner_3j(J,1,J,-Omega,-q,Omegap))
    return p2q*(-1)**(S-Sigma+J-Omega)*reduced(S)*reduced(J)*val*kdel(F1,F1p)*kdel(F,Fp)*kdel(mF,mFp)


def HexR_cor(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    valJS=0;valLS=0;valJL=0
    for q in [-1,0,1]:
        valJS += kdel(Lambda,Lambdap)* \
                (-1)**(S-Sigma+J-Omega)*wigner_3j(S,1,S,-Sigma,q,Sigmap)*wigner_3j(J,1,J,-Omega,q,Omegap)* \
                reduced(J)*reduced(S)
        valLS += (-1)**q*kdel(Omega,Omegap)*(-1)**(L-Lambda+S-Sigma)* \
                wigner_3j(S,1,S,-Sigma,-q,Sigmap)*wigner_3j(L,1,L,-Lambda,q,Lambdap)* \
                reduced(L)*reduced(S)
        valJL += kdel(Sigma,Sigmap)*(-1)**(L-Lambda+J-Omega)* \
                wigner_3j(L,1,L,-Lambda,q,Lambdap)*wigner_3j(J,1,J,-Omega,q,Omegap)* \
                reduced(J)*reduced(L)
    return Bex*( ((J*(J+1)+L*(L+0)+S*(S+1))*
                  kdel(Lambda,Lambdap)*kdel(Sigma,Sigmap)*kdel(Omega,Omegap) -2*valJS -2*valJL +2*valLS)
                *kdel(F1,F1p)*kdel(F,Fp)*kdel(mF,mFp))
####################################################################################################################################


################################### Zeeman Hamiltonian Pi state ################################################################
def HZL(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    return (gLp*uB*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1)*nreduced(F,Fp)*wigner_6j(F1p,Fp,I2,F,F1,1)*
           (-1)**(F1p+J+I1)*nreduced(F1,F1p)*wigner_6j(J,F1p,I1,F1,J,1)*
            (-1)**(J-Omega)*wigner_3j(J,1,J,-Omega,0,Omegap)*
           nreduced(J,J)*Lambda*kdel(Lambda,Lambdap)*kdel(Sigma,Sigmap))

def HexZS(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val=0
    for q in [-1,0,1]:
        val += (-1)**(J-Omega)*wigner_3j(J,1,J,-Omega,q,Omegap)* \
                (-1)**(S-Sigma)*wigner_3j(S,1,S,-Sigma,q,Sigmap)
    return (gS*uB*kdel(Lambda,Lambdap)*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1)*nreduced(F,Fp)*wigner_6j(F1p,Fp,I2,F,F1,1)*
           (-1)**(F1p+J+I1)*nreduced(F1,F1p)*wigner_6j(J,F1p,I1,F1,J,1)*
            val*nreduced(J,J)*reduced(S))

def HexZI2(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp) = state
    return (-gI2*uN*kdel(Lambda,Lambdap)*kdel(Sigma,Sigmap)*kdel(Omega,Omegap)*
            (-1)**(F-mF+F+I2+F1+1)*wigner_3j(F,1,Fp,-mF,0,mFp)*nreduced(F,Fp)*
            wigner_6j(I2,Fp,F1,F,I2,1)*reduced(I2))

def HexZI1(state): 
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp) = state
    return (-gI1*uN*kdel(Lambda,Lambdap)*kdel(Sigma,Sigmap)*kdel(Omega,Omegap)*
            (-1)**(F-mF+Fp+I2+F1+1)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            nreduced(F,Fp)*wigner_6j(F1p,Fp,I2,F,F1,1)*
            (-1)**(F1+J+I1+1)*nreduced(F1,F1p)*wigner_6j(I1,F1p,J,F1,I1,1)*reduced(I1))

def HZglp(state):
    (Lambda,Sigma,Omega,J,F1,F,mF,Lambdap,Sigmap,Omegap,Jp,F1p,Fp,mFp)=state
    val=0
    for q in [-1,1]:
        val += (-1)**(J-Omega)*wigner_3j(J,1,J,-Omega,-q,Omegap)* \
                (-1)**(S-Sigma)*wigner_3j(S,1,S,-Sigma,q,Sigmap)* \
                (-1)*kdel(Lambda,Lambdap-2*q)
    return (glp*uB*(-1)**(F-mF)*wigner_3j(F,1,Fp,-mF,0,mFp)*
            (-1)**(Fp+I2+F1)*nreduced(F,Fp)*wigner_6j(F1p,Fp,I2,F,F1,1)*
           (-1)**(F1p+J+I1)*nreduced(F1,F1p)*
            wigner_6j(J,F1p,I1,F1,J,1)*val*nreduced(J,J)*reduced(S))
