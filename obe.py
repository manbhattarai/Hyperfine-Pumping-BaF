#from symengine import *

import sympy as sp
from sympy import I,pprint
#t=symbols('t')
t=sp.symbols('t',real = sp.true)

import numpy as np
#from rate_eq import Excitation
from scipy.integrate import solve_ivp

from molecular_parameters import Gamma; Gamma *= 2*np.pi
#from diffeqpy import de,ode

import time


def heaviside(x,a):
    print(x)
    print(a)
    if np.abs(x-a) <a:
        return 1
    else:
        return 0

class Excitation():
    """ Excitation class defines the properties of the optical field.
    Parameters:
    rabi : float
            Rabi frequency of the field. The actual Rabi frequency is rabi*dipole_matrix_element for the particular transition
    pol : int
            Polarization of the field. The dipole matrix element that is created represents transition from Pi to Sigma level. Thus the sense of circular
            polarization is reversed. A value of +1 represnts matrix element due to sigma minus light representing transition from sigma ground state to
            pi excited state. -1 represents sigma plus transiton. 0 represents z- polarized light-transition.
    
    ground_state: SigmaLevel
                    Ground sigma state for the transition
    
    excited_state: PiLevelParity
                    Excited Pi state for the transition
                    Specifying ground_state and excited state defines the frequency of the light only. It need not represnt the set of
                    physically realizable transition.
    detuning : float
                Detuning from the transition frequency specified by the specified ground and excited states
                
    position: float
                Position in time of the center of the beam.
    diameter : float
                The 1/e^2 diameter (in case of Gaussian beam) and the size of the beam (in case of Uniform beam) specified in units of time.
    shape : String
            "Gaussian" to represent a Gaussian beam
            "Uniform" to represent a uniform intensity beam.
    """
    
    def __init__(self, rabi:float, pol:int, ground_state,excited_state, detuning = 0, position = None, diameter = None, shape = None):
        self.rabi = rabi
        self.pol = pol
        self.ground_state = ground_state
        self.excited_state = excited_state
        self.detuning = detuning
        self.position = position
        self.diameter = diameter
        self.shape    = shape
    
    def __repr__(self):
        return f"rabi = {self.rabi}, pol = {self.pol}, Ground = {self.ground_state}, Excited = {self.excited_state}, detuning  = {self.detuning}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
    def __str__(self):
        return f"rabi = {self.rabi}, pol = {self.pol}, Ground = {self.ground_state}, Excited = {self.excited_state}, detuning  = {self.detuning}, position = {self.position}, diameter = {self.diameter}, shape = "+ self.shape
     
class obe:
    """obe class takes in light atom interaction fields, creates interaction Hamiltonian and solves optical bloch equations
    Parameters:
    E_field : Excitation class or a list of Excitation class
            It describes contain all the light fields interating with the molecule.
    states : list
            Contais the list of the ground (G) and the excited (E) states passed as a list [G,E].
    H0 : numpy.ndarray
        Bare Hamiltonian for the levels considered
    Hint : numpy.ndarray
        Matrix of dipole matrix elements. This matrix is later used to construct the actual interaction hamiltonian with the 
        the light detuning and time dependence, for solving the optical bloch equations.
    br: numpy.ndarray
        Matrix of branching ratio. The matrix has a dimension m x n, where m is the number of ground states and n is the number of 
        excited states.
        The element br[m,n] represents the probability of the excited state n decaying to ground state m by spontaneous emission. The sum
        of elements along the columns is 1.
    Hint_func: None or lambda function
        If None, the solver is called to create an interaction Hamiltonian, consideting all the fields passed to the obe class.
        If passed as a lambda function of time, is used as interacting Hamiltonian by the solver.
    transitions: List of tuples
        Performs rotating frame transformation of the Hamiltonian, using only the transitions included in the list.
    rot: Boolean
        When False, the Hamiltonian is created in what is called the interaction picture.
        When True, using the transitions in the transitions list, a transformation to rotating frame is made.
    
    1. List of excitation
    2. Hamiltonian in the interaction picture
    3. Construct the decay matrix, which should be straight forward
    4. Summing up the hamiltoninan over different excitation may not be as trivial as I imagine, we can expect 
        a lot of sum of rabi freqinecies with the corresponding exponential terms.
    5. Write out the repopulation matrix which should be derivabel from the rate equation class
    """
    def __init__(self,E_field,states,H0,Hint,br,Hint_func = None,transitions=[],rot = False): #transitions added
        
        if type(E_field) == Excitation:
            self.E_field = [E_field]
        else:
            self.E_field = E_field
        
        self.ground_states = states[0]
        self.excited_states = states[1]
        self.n_ground = len(self.ground_states)
        self.n_exec = len(self.excited_states)


        self.H0 = np.round(2*np.pi*H0,2)
        self.rot = rot
        self.U = None
        self.Ud = None
        self.transitions = transitions
        self.generate_unitary_matrices()        
        
        A = np.zeros(np.shape(H0))
        for i in range(len(self.ground_states),len(self.ground_states)+len(self.excited_states)):
            A[i,i] = 1.0
        self.decay_matrix = Gamma*A #scipy.sparse.csr_matrix(A)
        self.A0 = np.zeros(np.shape(H0))
        self.br=br
        
        if self.rot:
            self.generate_unitary_matrices_rot()
        #self.generate_unitary_matrices_rotFrame()
        if Hint_func == None:
            self.Hint = self.interaction_picture_symbolic(Hint)
        else:
            self.Hint = Hint_func
                
        
    def generate_unitary_matrices(self):
        U  =  sp.eye(np.shape(self.H0)[0])
        Ud =  sp.eye(np.shape(self.H0)[0])
        for i in range(np.shape(self.H0)[0]):
            U[i,i] = sp.exp(-I*t*self.H0[i,i])
            Ud[i,i] = sp.exp(I*t*self.H0[i,i])
        self.U = U
        self.Ud = Ud
        
    def generate_unitary_matrices_rot(self):
        self.U  =  sp.eye(np.shape(self.H0)[0])
        self.Ud =  sp.eye(np.shape(self.H0)[0])
        fields = self.transitions
        temp = []
        for item in fields:
            (i,j) = item
            if not(j in temp):
                temp.append(j)

        temp_field = []

        for i in range(len(temp)):
            rand_field = []
            for item in fields:
                (m,n) = item
                if n == temp[i]:
                    rand_field.append(item)
            temp_field.append(rand_field)
        
        for item_list in temp_field:
            idx_0 = item_list[0][0]
            idx_1 = item_list[0][1]
            idx = self.transitions.index(item_list[0])
            w_p = self.H0[self.n_ground+idx_1,self.n_ground+idx_1]- \
                    self.H0[idx_0,idx_0]+ \
                    self.E_field[idx].detuning*2*np.pi
                
            self.A0[self.n_ground+idx_1,self.n_ground+idx_1]= w_p #- self.H0[n_ground+idx_1,n_ground+idx_1]
            
            self.U[self.n_ground+idx_1,self.n_ground+idx_1]  = sp.exp(-I*t*w_p)
            self.Ud[self.n_ground+idx_1,self.n_ground+idx_1] = sp.exp(I*t*w_p)
            
            self.U[idx_0,idx_0] = 1
            self.Ud[idx_0,idx_0] = 1
                
            for n,item_each in enumerate(item_list):
                if n == 0:
                    continue
                idx = self.transitions.index(item_each) #will have the same index as the field
                (ii,jj) = item_each
                
                w_c = self.H0[self.n_ground+jj,self.n_ground+jj]- \
                        self.H0[ii,ii]+ \
                        self.E_field[idx].detuning*2*np.pi
                self.A0[ii,ii] = w_p - w_c
                self.U[ii,ii] = sp.exp(-I*t*(w_p-w_c))
                self.Ud[ii,ii] = sp.exp(I*t*(w_p-w_c))
                print(ii)
        
    
    def repopulation_matrix(self,R):
        Rm = np.zeros(np.shape(self.H0)).astype(np.complex128) #remove the (1.0+0j)* term
        R_exec = np.array([R[i,i] for i in range(self.n_ground,self.n_ground+self.n_exec)]).astype(np.complex128)
        
        Rm_diag = self.br@R_exec
        
        #Rm_diag = np.append(Rm_diag,np.array([0]*n_exec))
        #Rm = Gamma*np.diag(Rm_diag)
        for i in range(self.n_ground):
            Rm[i,i] = Gamma*Rm_diag[i]
        #print(type(Rm))
        return Rm

    
    def solve(self,npoints,r_init:np.ndarray, max_step_size = 1.0/Gamma, package = 'Python'):
        
        def Rdot_python(T,u):
            start = time.time()
            
            R = np.array(u.reshape(self.n_ground+self.n_exec,self.n_ground+self.n_exec)).astype(np.complex128)
            
            H = self.Hint(T)
            commuter_term = (0.0-1.0j)*(H@R-R@H)
            decay_term = 1/2.0*(self.decay_matrix@R+R@self.decay_matrix)
            repop_term = self.repopulation_matrix(R)
            return_val = (commuter_term-decay_term+repop_term).flatten()
            
            return return_val
        
        def Rdot_julia(u,p,T):
            u = np.array(u)
            R = np.array(u.reshape(self.n_ground+self.n_exec,self.n_ground+self.n_exec))#.astype(np.complex128)
            
            H = self.Hint(T)
            commuter_term = (0.0-1.0j)*(H@R-R@H)
            decay_term = 1/2.0*(self.decay_matrix@R+R@self.decay_matrix)
            repop_term = np.array(self.repopulation_matrix(R))
            return_val = list((commuter_term-decay_term+repop_term).flatten())
           
            return return_val
        

        #extract the max and the min of the interaction time
        tmax = -1e6
        tmin =  1e6
        for E_field in self.E_field:
            t_start = E_field.position - 1.5*E_field.diameter
            t_end   = E_field.position + 1.5*E_field.diameter
            if t_start < tmin:
                tmin = t_start
            if t_end > tmax:
                tmax = t_end
            
        tinterval = np.linspace(tmin,tmax,npoints)

        #start = time.time()
        if package == 'Python':
            result = solve_ivp(Rdot_python,[tinterval[0],tinterval[-1]],r_init.flatten(),
                            t_eval = tinterval,
                            method = 'RK45',max_step = max_step_size,
                            atol = 1e-7,rtol = 1e-4)
            
            result = np.array(result.y).T
        """
        elif package == 'Julia':
            prob = de.ODEProblem(Rdot_julia,r_init.flatten(),(tinterval[0],tinterval[-1]))
            result = de.solve(prob,de.DP5(),saveat=0.01,reltol=1e-3,abstol=1e-6)
        """    
        """
        #list the object properties
        dir_result = dir(result)
        for item in dir_result:
            if item[0] != '_':
                eval_string = "result."+item
                print("######  "+item+"  #########")
                print(eval(eval_string))
        print(result.u[-1])
        """
        
        #start = time.time()
        #result= np.array([list(i) for i in result.u])
        #stop = time.time()
        #print(f"List comprehension : {stop-start} s.")
        #print(result)
        #result = np.transpose(np.array(result_list))


        #stop = time.time()
        #print(f"Solver took : {stop- start}s")
        #print(result)

       
        
        #temp = np.array(self.comparision)
        #print(np.mean(temp,axis = 0))
        #print(self.counter)
        return result
    

    def interaction_picture_symbolic(self,Hint_list):
        """making the interaction Hamiltonian have the time dependence"""
        start =time.time()
        #lets make Hint a list for each polarization
        if type(Hint_list) is np.ndarray:
            Hint_list = [Hint_list]
        N_pols = len(Hint_list)
        #print(N_pols)
        myHint = sp.Matrix(np.zeros(np.shape(Hint_list[0])))
        
        count_Hint = 0
        
        for Hint in Hint_list:
            for E_field in self.E_field:
                t0 = E_field.position
                tsigma =  E_field.diameter/4
                H_temp = sp.Matrix(Hint) 
                rabi = np.round(E_field.rabi*2*np.pi,2) #angular unit
                idx_ground = self.ground_states.index(E_field.ground_state)
                idx_exec = self.excited_states.index(E_field.excited_state)
                E_res = np.round(self.H0[self.n_ground+idx_exec,self.n_ground+idx_exec]- \
                        self.H0[idx_ground,idx_ground]+ \
                        E_field.detuning*2*np.pi,2) #angular
                for i in range(self.n_ground): #index for ground states
                    for j in range(self.n_ground,self.n_ground+self.n_exec): #index for excited states. Looking at the upper triangular region only
                        
                        if H_temp[i,j] == 0:
                            continue

                        E = self.H0[j,j] - self.H0[i,i] #angular
                        if (np.abs(E_res - E) >= 30*((2*rabi*np.abs(Hint[i,j]))**2+Gamma**2)**0.5): 
                            isNearResonant = False
                        else:
                            isNearResonant = True

                        #Introduce pol multiplier that multiplies by rabi frequency of the correct polarization only
                        mF_init = self.ground_states[i].states[0].mF
                        mF_final = self.excited_states[j-self.n_ground].states[0].mF
                        dmF = mF_init-mF_final #its defined this way because all the Hint matrix element are calculated as <ground|Hint|excited>
                        if dmF == E_field.pol:
                            pol_multiplier = 1
                        else:
                            pol_multiplier = 0
                        
                        if isNearResonant:
                                H_temp[i,j] *= 1/2*rabi*sp.exp( I*(E_res-self.H0[j,j]+self.H0[i,i])*t)*pol_multiplier      #Emission

                                H_temp[j,i] *= 1/2*rabi*sp.exp(-I*(E_res-self.H0[j,j]+self.H0[i,i])*t)*pol_multiplier     #Absorption
                        else:
                            #print("Approximation made")
                            H_temp[i,j] *= 0.0
                            H_temp[j,i] *= 0.0
                            """
                            if E_field.shape == "Gaussian":
                                H_temp[i,j] *= 1/2*rabi*sp.exp(-(t-t0)**2/4/tsigma**2)* \
                                               sp.exp(I*E_res*t)*pol_multiplier* \
                                                            1#Emission
                                H_temp[j,i] *= 1/2*rabi*sp.exp(-(t-t0)**2/4/tsigma**2)* \
                                               sp.exp(-I*E_res*t)*pol_multiplier*\
                                                          1#Absorption
                            else:
                                H_temp[i,j] *= 1/2*rabi*sp.exp( I*(E_res-self.H0[j,j]+self.H0[i,i])*t)*pol_multiplier      #Emission

                                H_temp[j,i] *= 1/2*rabi*sp.exp(-I*(E_res-self.H0[j,j]+self.H0[i,i])*t)*pol_multiplier     #Absorption
                        else:
                            #print("Approximation made")
                            H_temp[i,j] *= 0.0
                            H_temp[j,i] *= 0.0
                            """
                myHint += H_temp#*sp.exp(-(t-t0)**2/4/tsigma**2) 
        

        if E_field.shape == "Gaussian":
            myHint *= sp.exp(-(t-t0)**2/4/tsigma**2)
            
        #myHint *= sp.exp(-(t-t0)**2/4/tsigma**2) 
        #if self.rot:
        #    myHint = (self.Ud)*(self.H0+myHint)*(self.U) - self.A0
        #else:
        #    myHint = (self.Ud)*myHint*(self.U) - self.A0
        #myHint = sp.simplify(myHint)

        #myHint = [ufuncify(t,myHint[i,j],backend = 'numpy') for j in range(myHint.cols) for i in range(myHint.rows)]
       
        #print(self.A0)
        #print(self.U)
        #print(myHint)
        #myHint = sp.lambdify(t,myHint,'numpy')
        #myHint = sympify(myHint)
        #print(myHint)
        myHint = sp.lambdify(t,myHint)
        
        #print(type(myHint))
        #myHint = (self.Ud)*myHint*(self.U) - self.A0
        #print(f"Hint time elapsed : {time.time()-start}s")
        #print(type(myHint))
        return myHint
    

    