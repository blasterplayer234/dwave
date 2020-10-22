import neal
import numpy
import pandas as pd
from pyqubo import Array, Placeholder, solve_qubo, Constraint
from pyqubo import Sum, Model, Mul
import math
from dwave.system import LeapHybridSampler

##Input Data
r = pd.read_excel('ret.xlsx')
s = pd.read_excel('corr.xlsx')

returns = r['return']
sigma = s.loc[:,s.columns!='STOCK'];
sigma = sigma.to_numpy();
returns = returns
sigma = (sigma*1000).astype(int);
returns =(returns*100).astype(int);

###Input Parameters
N = 100;
n = 50
R = 0 ; 

#to calculate HQPU time
import time

for n in [20]:
    sum=0;
    for i in range(N):
        sum += returns[i];

    K = math.log2(sum);
    K = math.floor(K);
    K+=1;


    x = Array.create('arr', N+K, 'BINARY')


    # Constraints in our model
    #x' Sigmax
    min_sigx = 0;

    H =0;#temp hamiltonian for the Constraint

    for i in range(N):
        for j in range(N):
            H += x[i]*x[j]*sigma[i][j];

    min_sigx += Constraint(H,label = "min_portfolio");

    H =0 ;#temp hamiltonian for the Constraint
    for i in range(N):
        H+=x[i];
    H-=n;
    H = H**2;

    select_n = Constraint(H , label = "select_n_projects")

    H = 0; #temp hamiltonian for the Constraint

    for i in range(N):
        H += returns[i]*x[i];
    H -= R
    for i in range(0,K):
        H-= (2**i)*(x[i+N]);
    expectedReturn = Constraint(H**2, label = "min_expected_return");



    #Parameters

    lambda1 = 100
    scale_down = 1;
    lambda2 = 0;

    #final hamiltonian 
    H = (min_sigx/scale_down  + lambda1*select_n + lambda2*expectedReturn);

    model = H.compile();    
    qubo, offset = model.to_qubo()
    useQPU = False;

    num_samples_hybrid = 40;#number of times hybrid solver is used 


    if useQPU:
        for _ in range(num_samples_hybrid):
            # start = time.time()
            sampler = LeapHybridSampler()
            response = sampler.sample_qubo(qubo)
            sample = response.first.sample;
            final_sigx=0;
            for i in range(N):
                for j in range(N):
                    final_sigx += sample['arr[{}]'.format(i)]*sample['arr[{}]'.format(j)]*sigma[i][j];
            print(final_sigx)
            
            # end = time.time()
            # print(end - start,"s")


    else:
        t=0
        sampler = neal.SimulatedAnnealingSampler()
        response = sampler.sample_qubo(qubo, num_sweeps=1000, num_reads=1000)
        
        x=response.first.sample
        mn=1e9;

        for sample in response.samples():   
            final_sigx=0; 
            for i in range(N):
                for j in range(N):
                    final_sigx += sample['arr[{}]'.format(i)]*sample['arr[{}]'.format(j)]*sigma[i][j];
            l=0;
            for i in range(N):
                l+= sample['arr[{}]'.format(i)]

            if(final_sigx<=mn and l==n):
                mn=final_sigx;
                x = sample;
                
        print(mn)
